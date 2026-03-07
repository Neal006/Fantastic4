"""
Core inference engine for the XGBoost solar inverter risk model.
Loads the trained model, scaler, label encoder, and feature columns at import time.
Provides predict() and predict_batch() for single and batch inference.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("mlinference")

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "xgb_best.pkl"
ARTIFACTS_PATH = MODELS_DIR / "inference_artifacts.pkl"

# ── Class names (must match training config) ─────────────────────
CLASS_NAMES = ["no_risk", "degradation_risk", "shutdown_risk"]

# ── Fault descriptions by category ──────────────────────────────
FAULT_MAP = {
    "C": [
        "Low Power Output — String Issue",
        "Partial Shading — Performance Loss",
        "Communication Issue — Alarm Code 2010",
        "Minor Voltage Deviation",
    ],
    "D": [
        "String Degradation",
        "Alarm Code 3021 — Operational Fault",
        "MPPT Drift — High String Imbalance",
        "String Voltage Mismatch",
    ],
    "E": [
        "Overheating — Thermal Shutdown Risk",
        "Grid Fault — Frequency Deviation",
        "Ground Fault — Insulation Failure",
        "Inverter Shutdown — Critical",
        "DC Overcurrent Detected",
    ],
}

# ── 6 core readings that map to raw telemetry ────────────────────
CORE_FIELD_MAP = {
    "dc_voltage": "pv1_voltage",
    "dc_current": "pv1_current",
    "ac_power": "power",
    "module_temp": "temp",
    "ambient_temp": "ambient_temp",
    "irradiation": "meter_active_power",
}


class InferenceEngine:
    """Encapsulates model loading, feature alignment, scaling, and prediction."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_cols: list[str] = []
        self._loaded = False

    def load(self):
        """Load model and inference artifacts from disk."""
        if self._loaded:
            return

        logger.info("Loading XGBoost model from %s ...", MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        logger.info("Loading inference artifacts from %s ...", ARTIFACTS_PATH)
        with open(ARTIFACTS_PATH, "rb") as f:
            artifacts = pickle.load(f)

        self.scaler = artifacts["scaler"]
        self.label_encoder = artifacts["label_encoder"]
        self.feature_cols = artifacts["feature_cols"]

        self._loaded = True
        logger.info(
            "Inference engine ready — %d features, %d classes",
            len(self.feature_cols),
            len(CLASS_NAMES),
        )

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    # ── Build feature vector from raw operator input ─────────────
    def _build_feature_vector_from_raw(self, raw: dict) -> np.ndarray:
        """
        Accept the 6 core readings (+ optional extras) and produce a
        183-dim feature vector.  Missing engineered features are zero-filled.
        """
        vec = np.zeros(self.n_features, dtype=np.float32)

        # Map core fields first
        for api_key, feature_key in CORE_FIELD_MAP.items():
            if api_key in raw and feature_key in self.feature_cols:
                idx = self.feature_cols.index(feature_key)
                vec[idx] = float(raw[api_key])

        # Map any extra fields the operator provided (alarm_code, op_state, etc.)
        for key, value in raw.items():
            if key in self.feature_cols:
                idx = self.feature_cols.index(key)
                vec[idx] = float(value)
            # Also try direct match for engineered features operator might send
            elif key not in CORE_FIELD_MAP and key not in ("inverter_id",):
                # Check if a feature column matches
                if key in self.feature_cols:
                    idx = self.feature_cols.index(key)
                    vec[idx] = float(value)

        # Derive some useful defaults from core readings
        if "ac_power" in raw:
            ac = float(raw["ac_power"])
            # Fill power-related rolling features with the current value as baseline
            for col in self.feature_cols:
                if col.startswith("power_rmean_") and vec[self.feature_cols.index(col)] == 0:
                    vec[self.feature_cols.index(col)] = ac
                elif col.startswith("power_rmin_") and vec[self.feature_cols.index(col)] == 0:
                    vec[self.feature_cols.index(col)] = ac
                elif col.startswith("power_rmax_") and vec[self.feature_cols.index(col)] == 0:
                    vec[self.feature_cols.index(col)] = ac

        if "module_temp" in raw:
            temp = float(raw["module_temp"])
            for col in self.feature_cols:
                if col.startswith("temp_rmean_") and vec[self.feature_cols.index(col)] == 0:
                    vec[self.feature_cols.index(col)] = temp

        if "dc_current" in raw:
            dc_i = float(raw["dc_current"])
            for col in self.feature_cols:
                if col.startswith("pv1_current_rmean_") and vec[self.feature_cols.index(col)] == 0:
                    vec[self.feature_cols.index(col)] = dc_i

        # Set op_state derived features
        if "op_state" in raw:
            op = int(raw["op_state"])
            if "is_running" in self.feature_cols:
                vec[self.feature_cols.index("is_running")] = 1 if op == 5120 else 0
            if "is_off" in self.feature_cols:
                vec[self.feature_cols.index("is_off")] = 1 if op == 0 else 0
        else:
            # Default: assume running
            if "is_running" in self.feature_cols:
                vec[self.feature_cols.index("is_running")] = 1

        # Alarm derived features
        if "alarm_code" in raw:
            alarm = int(raw["alarm_code"])
            if "alarm_active" in self.feature_cols:
                vec[self.feature_cols.index("alarm_active")] = 1 if alarm != 0 else 0

        return vec

    # ── Build feature vector from full feature dict ──────────────
    def _build_feature_vector_from_full(self, features: dict) -> np.ndarray:
        """
        Accept a dict with all 183 feature columns (from the data pipeline).
        Missing ones are zero-filled.
        """
        vec = np.zeros(self.n_features, dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            if col in features:
                vec[i] = float(features[col])
        return vec

    # ── Category mapping (3-class → A–E) ────────────────────────
    @staticmethod
    def _map_category(predicted_class: int, probabilities: np.ndarray) -> str:
        """Map 3-class XGBoost output to A–E category using confidence thresholds."""
        # Class indices: 0=no_risk, 1=degradation_risk, 2=shutdown_risk
        if predicted_class == 2:  # shutdown_risk
            return "E"
        elif predicted_class == 1:  # degradation_risk
            prob = probabilities[1]
            return "D" if prob >= 0.70 else "C"
        else:  # no_risk
            prob = probabilities[0]
            if prob >= 0.90:
                return "A"
            elif prob >= 0.70:
                return "B"
            else:
                return "C"  # Low confidence "no_risk" treated as minor issue

    # ── Fault description ────────────────────────────────────────
    @staticmethod
    def _get_fault_description(
        category: str,
        raw_input: dict,
        probabilities: np.ndarray,
    ) -> Optional[str]:
        """Generate a contextual fault description based on category and readings."""
        if category in ("A", "B"):
            return None

        # Pick a fault description based on the most likely root cause
        if category == "E":
            if raw_input.get("module_temp", 0) > 70:
                return "Overheating — Thermal Shutdown Risk"
            if raw_input.get("irradiation", 999) < 300:
                return "Grid Fault — Frequency Deviation"
            if raw_input.get("dc_voltage", 999) == 0 and raw_input.get("dc_current", 999) == 0:
                return "Inverter Shutdown — Critical"
            return "Ground Fault — Insulation Failure"

        if category == "D":
            if raw_input.get("dc_current", 10) < 5:
                return "String Degradation"
            if raw_input.get("alarm_code", 0) != 0:
                return f"Alarm Code {int(raw_input.get('alarm_code', 0))} — Operational Fault"
            return "String Degradation"

        # Category C
        if raw_input.get("ac_power", 10) < 5:
            return "Low Power Output — String Issue"
        if raw_input.get("dc_voltage", 40) < 32:
            return "Partial Shading — Performance Loss"
        if raw_input.get("alarm_code", 0) != 0:
            return f"Communication Issue — Alarm Code {int(raw_input.get('alarm_code', 0))}"
        return "Low Power Output — String Issue"

    # ── Single prediction ────────────────────────────────────────
    def predict(
        self, raw_input: dict, mode: str = "manual"
    ) -> dict:
        """
        Run inference on a single reading.

        Args:
            raw_input: dict of input values (core 6 fields for manual, full 183 for pipeline)
            mode: "manual" (operator input) or "full" (pipeline with all features)

        Returns:
            dict with category, confidence, probabilities, fault, readings
        """
        if mode == "manual":
            vec = self._build_feature_vector_from_raw(raw_input)
        else:
            vec = self._build_feature_vector_from_full(raw_input)

        # Scale
        vec_scaled = self.scaler.transform(vec.reshape(1, -1))

        # Predict
        proba = self.model.predict_proba(vec_scaled)[0]
        predicted_class = int(np.argmax(proba))
        confidence = float(np.max(proba))

        # Map to A–E
        category = self._map_category(predicted_class, proba)

        # Fault description
        fault = self._get_fault_description(category, raw_input, proba)

        # Build probabilities dict
        prob_dict = {}
        for i, name in enumerate(CLASS_NAMES):
            prob_dict[name] = round(float(proba[i]), 4)

        return {
            "category": category,
            "confidence": round(confidence, 4),
            "predicted_class": CLASS_NAMES[predicted_class],
            "probabilities": prob_dict,
            "fault": fault,
        }

    # ── Batch prediction ─────────────────────────────────────────
    def predict_batch(
        self, readings: list[dict], mode: str = "manual"
    ) -> list[dict]:
        """
        Run inference on multiple readings at once.
        Uses vectorized scaling and prediction for performance.
        """
        if not readings:
            return []

        # Build feature matrix
        vecs = []
        for r in readings:
            features = r.get("features", r)
            if mode == "manual":
                vecs.append(self._build_feature_vector_from_raw(features))
            else:
                vecs.append(self._build_feature_vector_from_full(features))

        X = np.vstack(vecs)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        probas = self.model.predict_proba(X_scaled)

        results = []
        for i, r in enumerate(readings):
            proba = probas[i]
            predicted_class = int(np.argmax(proba))
            confidence = float(np.max(proba))
            category = self._map_category(predicted_class, proba)

            features = r.get("features", r)
            fault = self._get_fault_description(category, features, proba)

            prob_dict = {
                name: round(float(proba[j]), 4)
                for j, name in enumerate(CLASS_NAMES)
            }

            results.append({
                "inverter_id": r.get("inverter_id", f"unknown-{i}"),
                "category": category,
                "confidence": round(confidence, 4),
                "predicted_class": CLASS_NAMES[predicted_class],
                "probabilities": prob_dict,
                "fault": fault,
            })

        return results


# ── Module-level singleton ───────────────────────────────────────
engine = InferenceEngine()

"""
Core inference engine for the InverterShield XGBoost models.

Loads three model artifacts from ml/models/:
  - binary_model.joblib     (XGB binary, used for SHAP)
  - multiclass_model.joblib (XGB multi-class, used for A-E classification)
  - anomaly_model.joblib    (IsolationForest, appended as features)
  - feature_names.joblib    (ordered list; last two are always anomaly_score, is_anomaly)

No StandardScaler — XGBoost is tree-based and scale-invariant.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger("mlinference")

# ── Paths ──────────────────────────────────────────────────────────
_BASE          = Path(__file__).resolve().parent          # mlinference/
_ML_ROOT       = _BASE.parent / "ml"                      # ml/
_MODELS_DIR    = _ML_ROOT / "models"                      # ml/models/
_BASELINE_PATH = _ML_ROOT / "healthy_baseline.json"

# ── Class names ────────────────────────────────────────────────────
CLASS_NAMES = ["no_risk", "degradation_risk", "shutdown_risk"]

# ── API field → feature column mapping ────────────────────────────
# Operator submits 6 human-readable fields; map them to training column names.
CORE_FIELD_MAP = {
    "dc_voltage":   "inv_v_ab",       # Phase AB voltage (closest to DC bus)
    "dc_current":   "inv_pv1_power",  # PV string 1 power (closest to DC input signal)
    "ac_power":     "inv_power",      # AC output power (exact match)
    "module_temp":  "module_temp",    # Module temperature (bonus column, exact)
    "ambient_temp": "ambient_temp",   # Ambient temperature (exact)
    "irradiation":  "irradiation",    # Solar irradiation (bonus column, exact)
}

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


class InferenceEngine:
    """Loads and serves all three InverterShield models."""

    def __init__(self):
        self.binary_model     = None   # XGB binary  — for SHAP
        self.multiclass_model = None   # XGB 3-class — for A-E prediction
        self.anomaly_model    = None   # IsolationForest
        self.feature_cols:    list[str] = []
        self.baseline:        dict      = {}
        self._loaded = False

    # ── Backward-compat property (main.py passes engine.model to SHAP) ─
    @property
    def model(self):
        return self.binary_model

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def base_feature_cols(self) -> list[str]:
        """Feature columns before anomaly augmentation (excludes last 2)."""
        if (len(self.feature_cols) >= 2 and
                self.feature_cols[-2:] == ["anomaly_score", "is_anomaly"]):
            return self.feature_cols[:-2]
        return self.feature_cols

    # ── Load artifacts ─────────────────────────────────────────────
    def load(self):
        if self._loaded:
            return

        for name in ("binary_model", "multiclass_model", "anomaly_model", "feature_names"):
            path = _MODELS_DIR / f"{name}.joblib"
            if not path.exists():
                raise FileNotFoundError(
                    f"Model artifact not found: {path}\n"
                    "Train first: cd ml && python run_pipeline.py"
                )
            obj = joblib.load(path)
            if name == "feature_names":
                self.feature_cols = obj
            else:
                setattr(self, name, obj)

        if not _BASELINE_PATH.exists():
            raise FileNotFoundError(f"Baseline not found: {_BASELINE_PATH}")
        self.baseline = json.loads(_BASELINE_PATH.read_text())

        self._loaded = True
        logger.info(
            "Inference engine ready — %d features (%d base + 2 anomaly), %d classes",
            len(self.feature_cols),
            len(self.base_feature_cols),
            len(CLASS_NAMES),
        )

    # ── Feature vector builders ────────────────────────────────────

    def _build_feature_vector_from_raw(self, raw: dict) -> np.ndarray:
        """
        Build augmented feature vector from 6-field operator input.
        1. Start from baseline (pre-computed KPIs are already correct — do NOT clobber them).
        2. Override with operator-provided values.
        3. Recompute ONLY the KPIs whose inputs were actually provided.
        4. Run anomaly model and append anomaly_score + is_anomaly.
        """
        row = dict(self.baseline)

        # Map core operator fields to training column names
        for api_key, feat_key in CORE_FIELD_MAP.items():
            if api_key in raw:
                row[feat_key] = float(raw[api_key])

        # Power factor — accept both API alias and direct name
        pf_provided = False
        if "meter_pf" in raw:
            row["meter_pf"] = float(raw["meter_pf"]); pf_provided = True
        elif "power_factor" in raw:
            row["meter_pf"] = float(raw["power_factor"]); pf_provided = True

        # Frequency — accept both API alias and direct name
        freq_provided = False
        if "meter_freq" in raw:
            row["meter_freq"] = float(raw["meter_freq"]); freq_provided = True
        elif "frequency" in raw:
            row["meter_freq"] = float(raw["frequency"]); freq_provided = True

        # Alarm code → derive alarm-history features
        alarm_code = int(raw.get("alarm_code", 0))
        if alarm_code != 0:
            row["alarm_count_24h"]        = 1.0
            row["alarm_count_7d"]         = 1.0
            row["hours_since_last_alarm"] = 0.0

        # Selectively recompute only the KPIs whose inputs were overridden.
        # Baseline pre-computed values (e.g. pf_deviation=0.0, freq_deviation=0.0)
        # reflect what the model saw during training — keep them intact unless
        # the caller provided fresh meter readings.
        if pf_provided:
            row["pf_deviation"] = abs(1.0 - row["meter_pf"])

        if freq_provided:
            row["freq_deviation"] = abs(row["meter_freq"] - 50.0)

        if "ac_power" in raw:
            inv_power = float(raw["ac_power"])
            baseline_power = self.baseline.get("inv_power", 3.889)
            row["power_ratio_vs_24h"] = inv_power / baseline_power if baseline_power else 0.0

        # Performance ratio — key degradation signal
        if "ac_power" in raw and "irradiation" in raw:
            irr = max(float(raw["irradiation"]), 50)
            pr  = float(raw["ac_power"]) / (irr / 1000)
            row["performance_ratio"] = min(max(pr, 0.0), 2.0)

        return self._make_vector(row)

    def _build_feature_vector_from_full(self, features: dict) -> np.ndarray:
        """
        Build augmented feature vector from a full feature dict (pipeline / batch).
        Provided values take priority; missing columns fall back to baseline.
        Pre-computed KPI values in the dict are used as-is.
        """
        row = dict(self.baseline)
        row.update({k: float(v) for k, v in features.items() if isinstance(v, (int, float))})
        return self._make_vector(row)

    def _make_vector(self, row: dict) -> np.ndarray:
        """Build base vector, run anomaly model, return augmented vector."""
        base_cols = self.base_feature_cols
        base_vec  = np.array(
            [float(row.get(col, 0.0)) for col in base_cols],
            dtype=np.float32,
        )
        return self._augment_with_anomaly(base_vec)

    def _augment_with_anomaly(self, base_vec: np.ndarray) -> np.ndarray:
        """Append anomaly_score and is_anomaly (-1/1) to match training layout."""
        if self.feature_cols[-2:] != ["anomaly_score", "is_anomaly"]:
            return base_vec
        X = base_vec.reshape(1, -1)
        anomaly_score = float(self.anomaly_model.decision_function(X)[0])
        is_anomaly    = float(self.anomaly_model.predict(X)[0])   # -1 or 1 (matches training)
        return np.append(base_vec, [anomaly_score, is_anomaly]).astype(np.float32)

    # ── Category mapping ────────────────────────────────────────────
    @staticmethod
    def _map_category(predicted_class: int, proba: np.ndarray) -> str:
        # Threshold-based classification tuned for extreme class imbalance
        # (~1% degradation in training).  Lowered thresholds prevent the
        # majority class from drowning out the minority signals.
        if proba[0] >= 0.55:    # strong no_risk signal needed before calling safe
            return "no_risk"
        if proba[1] >= 0.06:    # catch even small degradation probability
            return "degradation"
        return "shutdown"       # default to most urgent class when uncertain

    # ── Fault description ───────────────────────────────────────────
    @staticmethod
    def _get_fault_description(
        category: str, raw_input: dict, probabilities: np.ndarray
    ) -> Optional[str]:
        if category == "no_risk":
            return None
        if category == "shutdown":
            if raw_input.get("module_temp", 0) > 70:
                return "Overheating — Thermal Shutdown Risk"
            if raw_input.get("irradiation", 999) < 300:
                return "Grid Fault — Frequency Deviation"
            if raw_input.get("dc_voltage", 999) == 0 and raw_input.get("dc_current", 999) == 0:
                return "Inverter Shutdown — Critical"
            return "Ground Fault — Insulation Failure"
        # degradation
        if raw_input.get("dc_current", 10) < 5:
            return "String Degradation"
        if raw_input.get("alarm_code", 0):
            return f"Alarm Code {int(raw_input['alarm_code'])} — Operational Fault"
        if raw_input.get("ac_power", 10) < 5:
            return "Low Power Output — String Issue"
        if raw_input.get("dc_voltage", 40) < 32:
            return "Partial Shading — Performance Loss"
        return "String Degradation"

    # ── Single prediction ───────────────────────────────────────────
    def predict(self, raw_input: dict, mode: str = "manual") -> dict:
        if mode == "manual":
            vec = self._build_feature_vector_from_raw(raw_input)
        else:
            vec = self._build_feature_vector_from_full(raw_input)

        X = vec.reshape(1, -1)

        proba           = self.multiclass_model.predict_proba(X)[0]
        predicted_class = int(np.argmax(proba))
        confidence      = float(np.max(proba))
        category        = self._map_category(predicted_class, proba)
        fault           = self._get_fault_description(category, raw_input, proba)

        return {
            "category":        category,
            "confidence":      round(confidence, 4),
            "predicted_class": category,
            "_class_idx":      predicted_class,   # internal — used by SHAP lookup
            "probabilities":   {n: round(float(proba[i]), 4) for i, n in enumerate(CLASS_NAMES)},
            "fault":           fault,
        }

    # ── Batch prediction ────────────────────────────────────────────
    def predict_batch(self, readings: list[dict], mode: str = "manual") -> list[dict]:
        if not readings:
            return []

        vecs = []
        for r in readings:
            features = r.get("features", r)
            if mode == "manual":
                vecs.append(self._build_feature_vector_from_raw(features))
            else:
                vecs.append(self._build_feature_vector_from_full(features))

        probas = self.multiclass_model.predict_proba(np.vstack(vecs))

        results = []
        for i, r in enumerate(readings):
            proba           = probas[i]
            predicted_class = int(np.argmax(proba))
            confidence      = float(np.max(proba))
            category        = self._map_category(predicted_class, proba)
            features        = r.get("features", r)
            fault           = self._get_fault_description(category, features, proba)

            results.append({
                "inverter_id":     r.get("inverter_id", f"unknown-{i}"),
                "category":        category,
                "confidence":      round(confidence, 4),
                "predicted_class": category,
                "_class_idx":      predicted_class,
                "probabilities":   {n: round(float(proba[j]), 4) for j, n in enumerate(CLASS_NAMES)},
                "fault":           fault,
            })
        return results


# ── Module-level singleton ──────────────────────────────────────────
engine = InferenceEngine()


# ── KPI helper (module-level so shap_explainer can import if needed) ─
def _recompute_kpis(row: dict, baseline: dict) -> None:
    """Recompute derived KPI features in-place from updated row values."""
    v_r    = row.get("meter_v_r",  baseline.get("meter_v_r",  236.0))
    v_y    = row.get("meter_v_y",  baseline.get("meter_v_y",  236.0))
    v_b    = row.get("meter_v_b",  baseline.get("meter_v_b",  236.0))
    mean_v = (v_r + v_y + v_b) / 3
    row["voltage_imbalance"] = (
        max(abs(v_r - mean_v), abs(v_y - mean_v), abs(v_b - mean_v)) / mean_v
        if mean_v else 0.0
    )

    pf = row.get("meter_pf", baseline.get("meter_pf", 0.98))
    row["pf_deviation"] = abs(1.0 - pf)

    freq = row.get("meter_freq", baseline.get("meter_freq", 50.0))
    row["freq_deviation"] = abs(freq - 50.0)

    inv_power      = row.get("inv_power", 0.0)
    baseline_power = baseline.get("inv_power", 3.889)
    row["power_ratio_vs_24h"] = inv_power / baseline_power if baseline_power else 0.0

    smu_zero  = row.get("smu_num_zero",      baseline.get("smu_num_zero",      0.0))
    smu_total = row.get("smu_total_strings", baseline.get("smu_total_strings", 1.0))
    row["smu_zero_fraction"] = smu_zero / smu_total if smu_total else 0.0

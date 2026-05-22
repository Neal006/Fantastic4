"""
InverterShield Prediction API.

Run:
    uvicorn src.api.main:app --reload --port 8000

Endpoints:
    POST /predict           — single inverter prediction
    POST /predict/batch     — batch predictions
    GET  /model/info        — training metrics
    GET  /model/features    — feature names + healthy baseline
    GET  /health            — liveness check
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Resolve paths relative to this file
_HERE     = Path(__file__).resolve().parent
_ML_ROOT  = _HERE.parent.parent
_MODELS   = _ML_ROOT / "models"
_BASELINE = _ML_ROOT / "healthy_baseline.json"

app = FastAPI(
    title       = "InverterShield Prediction API",
    description = "7-day predictive maintenance for solar inverters",
    version     = "1.0.0",
)

# ── Lazy-loaded model registry ─────────────────────────────────────
_registry: dict = {}


def _load_models():
    if _registry:
        return
    for name in ("binary_model", "multiclass_model", "anomaly_model", "feature_names"):
        path = _MODELS / f"{name}.joblib"
        if not path.exists():
            raise HTTPException(
                status_code = 503,
                detail      = f"Model artifact not found: {path}. Run run_pipeline.py first.",
            )
        _registry[name] = joblib.load(path)

    if not _BASELINE.exists():
        raise HTTPException(status_code=503, detail="healthy_baseline.json not found.")
    _registry["baseline"] = json.loads(_BASELINE.read_text())


# ── Request / response schemas ─────────────────────────────────────

class InverterData(BaseModel):
    features:    dict
    inverter_id: Optional[str] = "unknown"


class PredictionResponse(BaseModel):
    inverter_id:       str
    failure_probability: float
    risk_level:        str
    risk_class:        int
    is_anomaly:        bool
    anomaly_score:     float
    top_risk_factors:  list
    recommendation:    str


# ── Core inference ─────────────────────────────────────────────────

_RISK_MAP = {0: "no_risk", 1: "degradation_risk", 2: "shutdown_risk"}

_RECOMMENDATIONS = {
    "no_risk":          "Continue routine monitoring.",
    "degradation_risk": "Schedule inspection within 7 days. Check PV strings and clean panels.",
    "shutdown_risk":    "Immediate inspection required. Check alarm codes and temperature.",
}


def _prepare_input(user_features: dict, baseline: dict, feature_names: list) -> pd.DataFrame:
    row = dict(baseline)
    row.update(user_features)

    now = datetime.now()
    row.setdefault("hour",       now.hour)
    row.setdefault("month",      now.month)
    row.setdefault("is_daytime", int(6 <= now.hour <= 18))

    # Recompute derived KPIs from provided signals
    v_r     = row.get("meter_v_r",  baseline.get("meter_v_r",  236.0))
    v_y     = row.get("meter_v_y",  baseline.get("meter_v_y",  236.0))
    v_b     = row.get("meter_v_b",  baseline.get("meter_v_b",  236.0))
    mean_v  = (v_r + v_y + v_b) / 3
    row["voltage_imbalance"] = (
        max(abs(v_r - mean_v), abs(v_y - mean_v), abs(v_b - mean_v)) / mean_v
        if mean_v else 0.0
    )

    pf = row.get("meter_pf", baseline.get("meter_pf", 0.98))
    row["pf_deviation"] = abs(1.0 - pf)

    freq = row.get("meter_freq", baseline.get("meter_freq", 50.0))
    row["freq_deviation"] = abs(freq - 50.0)

    inv_power       = row.get("inv_power",  baseline.get("inv_power",  0.0))
    baseline_power  = baseline.get("inv_power", 1.0)
    row["power_ratio_vs_24h"] = inv_power / baseline_power if baseline_power else 0.0

    smu_zero  = row.get("smu_num_zero",      baseline.get("smu_num_zero",      0.0))
    smu_total = row.get("smu_total_strings", baseline.get("smu_total_strings", 1.0))
    row["smu_zero_fraction"] = smu_zero / smu_total if smu_total else 0.0

    df = pd.DataFrame([row])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_names].fillna(0)


def _predict_one(features: dict) -> dict:
    _load_models()
    binary_model     = _registry["binary_model"]
    multiclass_model = _registry["multiclass_model"]
    anomaly_model    = _registry["anomaly_model"]
    feature_names    = _registry["feature_names"]
    baseline         = _registry["baseline"]

    X = _prepare_input(features, baseline, feature_names)

    anomaly_score = float(anomaly_model.decision_function(X)[0])
    is_anomaly    = anomaly_model.predict(X)[0] == -1

    class_probs         = multiclass_model.predict_proba(X)[0]
    risk_class          = int(class_probs.argmax())
    failure_probability = float(1.0 - class_probs[0])

    import shap
    explainer  = shap.TreeExplainer(binary_model)
    shap_vals  = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        sv = shap_vals[0][0]
    elif shap_vals.ndim == 3:
        sv = shap_vals[0, :, 1]
    else:
        sv = shap_vals[0]

    top_idx = sorted(range(len(sv)), key=lambda i: abs(sv[i]), reverse=True)[:5]
    top_factors = [
        {
            "feature":   feature_names[i],
            "value":     float(X.iloc[0, i]),
            "impact":    float(sv[i]),
            "direction": "increases risk" if sv[i] > 0 else "decreases risk",
        }
        for i in top_idx
    ]

    risk_level = _RISK_MAP[risk_class]
    return {
        "failure_probability": round(failure_probability, 4),
        "risk_level":          risk_level,
        "risk_class":          risk_class,
        "is_anomaly":          bool(is_anomaly),
        "anomaly_score":       round(anomaly_score, 4),
        "top_risk_factors":    top_factors,
        "recommendation":      _RECOMMENDATIONS[risk_level],
    }


# ── Endpoints ──────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(data: InverterData):
    result = _predict_one(data.features)
    result["inverter_id"] = data.inverter_id
    return result


@app.post("/predict/batch")
def predict_batch(items: list[InverterData]):
    return [predict_endpoint(item) for item in items]


@app.get("/model/info")
def model_info():
    metrics_path = _MODELS / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found.")
    return json.loads(metrics_path.read_text())


@app.get("/model/features")
def model_features():
    _load_models()
    return {
        "features": _registry["feature_names"],
        "baseline": _registry["baseline"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}

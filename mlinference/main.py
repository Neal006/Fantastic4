"""
FastAPI inference server for XGBoost solar inverter risk prediction.
Provides endpoints for single / batch predictions with SHAP explanations.

Start:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from inference import engine, CLASS_NAMES
from shap_explainer import shap_explainer

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("mlinference")

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="SolarWatch ML Inference API",
    description="XGBoost-based solar inverter risk prediction with SHAP explanations.",
    version="1.0.0",
)

# ── CORS (allow Express.js backend + frontend dev servers) ───────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup event ────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    """Load model and SHAP explainer on server start."""
    engine.load()
    shap_explainer.initialize(engine.model)
    logger.info("Server startup complete.")


# =====================================================================
#  Pydantic schemas
# =====================================================================

class ManualPredictRequest(BaseModel):
    """Schema for manual operator input (6 core fields + optional extras)."""
    inverter_id: str = Field(..., min_length=1, max_length=50, description="Inverter identifier, e.g. INV-P1-L1-0")
    dc_voltage: float = Field(..., ge=0, le=2000, description="DC voltage (V)")
    dc_current: float = Field(..., ge=0, le=100, description="DC current (A)")
    ac_power: float = Field(..., ge=0, le=500, description="AC power output (kW)")
    module_temp: float = Field(..., ge=-40, le=150, description="Module temperature (°C)")
    ambient_temp: float = Field(..., ge=-40, le=80, description="Ambient temperature (°C)")
    irradiation: float = Field(..., ge=0, le=1500, description="Solar irradiation (W/m²)")
    # Optional extras
    alarm_code: Optional[int] = Field(0, ge=0, description="Alarm code (0 = no alarm)")
    op_state: Optional[int] = Field(5120, description="Operational state (5120 = running, 0 = off)")
    power_factor: Optional[float] = Field(None, ge=0, le=1, description="Power factor (pf)")
    frequency: Optional[float] = Field(None, ge=0, le=100, description="Grid frequency (Hz)")
    include_shap: Optional[bool] = Field(True, description="Include SHAP explanation in response")
    include_plot: Optional[bool] = Field(True, description="Include SHAP bar chart as base64 PNG")

    @field_validator("inverter_id")
    @classmethod
    def sanitize_inverter_id(cls, v: str) -> str:
        """Basic XSS / injection sanitization."""
        return v.strip().replace("<", "").replace(">", "").replace("&", "")


class BatchReading(BaseModel):
    """A single reading in a batch request."""
    inverter_id: str = Field(..., min_length=1, max_length=50)
    features: dict = Field(..., description="Feature dict — either core 6 fields or full 183-column dict")

    @field_validator("inverter_id")
    @classmethod
    def sanitize_inverter_id(cls, v: str) -> str:
        return v.strip().replace("<", "").replace(">", "").replace("&", "")

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: dict) -> dict:
        if not v:
            raise ValueError("features dict must not be empty")
        # Ensure all values are numeric
        for key, val in v.items():
            if not isinstance(val, (int, float)):
                try:
                    v[key] = float(val)
                except (TypeError, ValueError):
                    raise ValueError(f"Feature '{key}' must be numeric, got {type(val).__name__}")
        return v


class BatchPredictRequest(BaseModel):
    """Schema for batch predictions from Express.js or data pipeline."""
    readings: list[BatchReading] = Field(..., min_length=1, max_length=100)
    mode: Optional[str] = Field("manual", pattern="^(manual|full)$", description="'manual' for core fields, 'full' for all 183 features")
    include_shap: Optional[bool] = Field(False, description="Include SHAP values (slower for large batches)")
    include_plot: Optional[bool] = Field(False, description="Include SHAP plots (much slower)")


class PredictionResponse(BaseModel):
    """Single prediction response."""
    inverter_id: str
    category: str
    confidence: float
    predicted_class: str
    probabilities: dict
    fault: Optional[str]
    readings: dict
    shap: Optional[dict] = None
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    count: int
    predictions: list[dict]
    timestamp: str


# =====================================================================
#  Endpoints
# =====================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": engine._loaded,
        "shap_ready": shap_explainer.ready,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "SolarWatch ML Inference",
    }


@app.get("/model/info")
def model_info():
    """Return model metadata: feature list, classes, number of features."""
    if not engine._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    return {
        "model_type": "XGBClassifier",
        "n_features": engine.n_features,
        "feature_columns": engine.feature_cols,
        "class_names": CLASS_NAMES,
        "category_mapping": {
            "A": "no_risk (high confidence ≥90%)",
            "B": "no_risk (moderate confidence 70-89%)",
            "C": "degradation_risk (low confidence) or low-confidence no_risk",
            "D": "degradation_risk (high confidence ≥70%)",
            "E": "shutdown_risk (any probability)",
        },
        "core_input_fields": [
            "dc_voltage", "dc_current", "ac_power",
            "module_temp", "ambient_temp", "irradiation",
        ],
        "optional_input_fields": [
            "alarm_code", "op_state", "power_factor", "frequency",
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(req: ManualPredictRequest):
    """
    Single-inverter prediction from manual operator input.
    Accepts 6 core readings + optional extras, returns category + SHAP.
    """
    if not engine._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw_input = {
        "dc_voltage": req.dc_voltage,
        "dc_current": req.dc_current,
        "ac_power": req.ac_power,
        "module_temp": req.module_temp,
        "ambient_temp": req.ambient_temp,
        "irradiation": req.irradiation,
        "alarm_code": req.alarm_code or 0,
        "op_state": req.op_state or 5120,
    }
    if req.power_factor is not None:
        raw_input["pf"] = req.power_factor
    if req.frequency is not None:
        raw_input["freq"] = req.frequency

    try:
        result = engine.predict(raw_input, mode="manual")
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # SHAP explanation
    shap_data = None
    if req.include_shap and shap_explainer.ready:
        try:
            vec = engine._build_feature_vector_from_raw(raw_input)
            vec_scaled = engine.scaler.transform(vec.reshape(1, -1))[0]
            predicted_idx = CLASS_NAMES.index(result["predicted_class"])
            shap_data = shap_explainer.explain(
                feature_vector=vec_scaled,
                feature_cols=engine.feature_cols,
                class_names=CLASS_NAMES,
                predicted_class_idx=predicted_idx,
                generate_plot=bool(req.include_plot),
                top_n=10,
            )
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)
            shap_data = {"error": str(e)}

    return PredictionResponse(
        inverter_id=req.inverter_id,
        category=result["category"],
        confidence=result["confidence"],
        predicted_class=result["predicted_class"],
        probabilities=result["probabilities"],
        fault=result["fault"],
        readings={
            "dc_voltage": req.dc_voltage,
            "dc_current": req.dc_current,
            "ac_power": req.ac_power,
            "module_temp": req.module_temp,
            "ambient_temp": req.ambient_temp,
            "irradiation": req.irradiation,
        },
        shap=shap_data,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictRequest):
    """
    Batch prediction for multiple inverters.
    Designed for real-time data from Express.js or the data pipeline.
    """
    if not engine._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    mode = req.mode or "manual"
    readings_dicts = []
    for r in req.readings:
        d = {"inverter_id": r.inverter_id, "features": r.features, **r.features}
        readings_dicts.append(d)

    try:
        results = engine.predict_batch(readings_dicts, mode=mode)
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

    # Optionally add SHAP to each result
    if req.include_shap and shap_explainer.ready:
        for i, r in enumerate(req.readings):
            try:
                features = r.features
                if mode == "manual":
                    vec = engine._build_feature_vector_from_raw(features)
                else:
                    vec = engine._build_feature_vector_from_full(features)
                vec_scaled = engine.scaler.transform(vec.reshape(1, -1))[0]
                predicted_idx = CLASS_NAMES.index(results[i]["predicted_class"])
                shap_data = shap_explainer.explain(
                    feature_vector=vec_scaled,
                    feature_cols=engine.feature_cols,
                    class_names=CLASS_NAMES,
                    predicted_class_idx=predicted_idx,
                    generate_plot=bool(req.include_plot),
                    top_n=10,
                )
                results[i]["shap"] = shap_data
            except Exception as e:
                logger.warning("SHAP for reading %d failed: %s", i, e)
                results[i]["shap"] = {"error": str(e)}

    # Attach original readings to each result
    for i, r in enumerate(req.readings):
        results[i]["readings"] = {
            k: v for k, v in r.features.items()
            if k in ("dc_voltage", "dc_current", "ac_power", "module_temp", "ambient_temp", "irradiation")
        }

    return BatchPredictionResponse(
        count=len(results),
        predictions=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Global exception handler ────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred",
        },
    )


# ── Run with: python main.py ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class RiskClass(str, Enum):
    NO_RISK = "no_risk"
    DEGRADATION_RISK = "degradation_risk"
    SHUTDOWN_RISK = "shutdown_risk"


class InverterPrediction(BaseModel):
    inverter_id: str
    plant_id: str
    block: str
    logger_mac: str
    inverter_index: int
    timestamp: datetime
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_class: RiskClass
    shap_values: Dict[str, float]
    raw_features: Dict[str, float]


class ExplanationResponse(BaseModel):
    inverter_id: str
    plant_id: str
    risk_score: float
    risk_class: str
    summary: str
    key_factors: List[Dict[str, str]]
    recommended_actions: List[str]
    urgency: str
    generated_at: datetime
    grounded_sources: List[str]
    disclaimer: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources_used: List[str]
    data_referenced: Optional[Dict] = None


class TicketResponse(BaseModel):
    ticket_id: str
    inverter_id: str
    pdf_path: str
    ticket_data: Dict
    generated_at: str


class RiskReportResponse(BaseModel):
    plant_id: str
    report: str
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    llm_connected: bool
    vector_store_loaded: bool
    inverters_monitored: int
    timestamp: datetime

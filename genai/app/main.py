"""
SolarGuard AI – GenAI Explanation Layer for Solar Plant Risk Assessment.

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import LLM_API_KEY
from app.models import (
    ExplanationResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
)
from app.llm import LLMClient
from app.rag import RAGPipeline
from app.explainer import Explainer
from app.agent import SolarAgent
from app.conversation import ConversationManager
from app.prompts import SYSTEM_PROMPT_CHAT, USER_PROMPT_CHAT
from app.guardrails import guardrail_disclaimer
from app.synthetic_data import (
    get_prediction,
    get_all_inverter_ids,
    get_all_predictions,
    get_plant_predictions,
    get_plant_overview,
    PLANTS,
)

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------
llm = LLMClient()
rag = RAGPipeline()
explainer = Explainer(llm, rag)
agent = SolarAgent(llm, rag)
conv_mgr = ConversationManager()


# ---------------------------------------------------------------------------
# Lifespan – initialise RAG on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not LLM_API_KEY:
        print(
            "\n⚠  LLM_API_KEY is not set. Copy .env.example → .env and add "
            "your Groq key (free at https://console.groq.com).\n"
        )
    try:
        rag.initialize()
    except Exception as exc:
        print(f"[WARN] RAG init failed ({exc}). Running without manual context.")
    yield
    print("SolarGuard AI shutting down.")


app = FastAPI(
    title="SolarGuard AI",
    description=(
        "GenAI explanation layer that converts ML risk predictions into "
        "human-readable guidance for solar plant operators."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
#  HEALTH
# =====================================================================
@app.get("/health", response_model=HealthResponse, tags=["Core"])
async def health():
    return HealthResponse(
        status="healthy",
        llm_connected=llm.check_connection(),
        vector_store_loaded=rag.ready,
        inverters_monitored=len(get_all_inverter_ids()),
        timestamp=datetime.utcnow(),
    )


# =====================================================================
#  EXPLANATION
# =====================================================================
@app.get(
    "/explanation/{inverter_id}",
    response_model=ExplanationResponse,
    tags=["Core"],
)
async def get_explanation(inverter_id: str):
    """Generate a plain-English explanation for a single inverter prediction."""
    try:
        return explainer.explain(inverter_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# =====================================================================
#  MULTI-TURN CHAT  (RAG-augmented, context memory)
# =====================================================================
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """Ask natural-language questions grounded in prediction data and the manual."""
    session_id = conv_mgr.get_or_create_session(req.session_id)

    # --- build data context for this query ---
    # Include all predictions so the LLM can answer cross-plant questions
    data_lines: list[str] = []
    for p in get_all_predictions():
        top_shap = sorted(
            p.shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]
        shap_short = ", ".join(f"{k}: {v:+.3f}" for k, v in top_shap)
        data_lines.append(
            f"{p.inverter_id} | {p.plant_id} | {p.block} | "
            f"risk={p.risk_score:.2f} | class={p.risk_class.value} | "
            f"SHAP=[{shap_short}]"
        )
    data_context = "\n".join(data_lines)

    # --- RAG: retrieve manual chunks relevant to the question ---
    manual_chunks = rag.retrieve(req.message)
    manual_context = (
        "\n---\n".join(c["text"] for c in manual_chunks[:3])
        if manual_chunks
        else "No manual context available."
    )

    # --- build message history ---
    conv_mgr.add_message(session_id, "user", req.message)
    history = conv_mgr.get_history(session_id)

    system = SYSTEM_PROMPT_CHAT.format(plant_overview=get_plant_overview())

    # Append a user message with full context for this turn
    contextual_msg = USER_PROMPT_CHAT.format(
        query=req.message,
        data_context=data_context,
        manual_context=manual_context,
    )

    # Replace the last user message with the context-enriched version
    msgs = history[:-1] + [{"role": "user", "content": contextual_msg}]

    response_text = llm.generate_with_history(system, msgs)
    conv_mgr.add_message(session_id, "assistant", response_text)

    sources = ["ML prediction data"]
    if manual_chunks:
        sources.append("Inverter technical manual (RAG)")

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        sources_used=sources,
    )


# =====================================================================
#  AGENTIC: MAINTENANCE TICKET  (returns JSON + PDF download)
# =====================================================================
@app.post("/agent/maintenance-ticket/{inverter_id}", tags=["Agent"])
async def create_ticket(inverter_id: str):
    """Agentic workflow: retrieve data → assess risk → draft ticket → PDF."""
    try:
        return agent.generate_maintenance_ticket(inverter_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/agent/maintenance-ticket/{inverter_id}/pdf", tags=["Agent"])
async def download_ticket_pdf(inverter_id: str):
    """Generate and download the maintenance ticket as a PDF file."""
    try:
        result = agent.generate_maintenance_ticket(inverter_id)
        return FileResponse(
            result["pdf_path"],
            media_type="application/pdf",
            filename=f"{result['ticket_id']}.pdf",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# =====================================================================
#  AGENTIC: PLANT-WIDE RISK REPORT
# =====================================================================
@app.get("/agent/risk-report/{plant_id}", tags=["Agent"])
async def risk_report(plant_id: str):
    """Generate a narrative risk report covering every inverter in a plant."""
    try:
        report = agent.generate_risk_report(plant_id)
        return {
            "plant_id": plant_id,
            "report": report,
            "generated_at": datetime.utcnow().isoformat(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# =====================================================================
#  UTILITY ENDPOINTS  (useful for frontend / testing)
# =====================================================================
@app.get("/inverters", tags=["Utility"])
async def list_inverters():
    """List all monitored inverter IDs grouped by plant."""
    result = {}
    for pid, pdata in PLANTS.items():
        result[pid] = {
            "name": pdata["name"],
            "blocks": {
                ldata["block"]: ldata["inverters"]
                for ldata in pdata["loggers"].values()
            },
        }
    return result


@app.get("/predictions", tags=["Utility"])
async def list_predictions(
    plant_id: str = Query(None, description="Filter by plant ID"),
):
    """Return raw prediction data (risk scores + SHAP values)."""
    if plant_id:
        preds = get_plant_predictions(plant_id)
        if not preds:
            raise HTTPException(404, f"No predictions for '{plant_id}'")
    else:
        preds = get_all_predictions()
    return [p.model_dump(mode="json") for p in preds]


@app.get("/predictions/{inverter_id}", tags=["Utility"])
async def get_single_prediction(inverter_id: str):
    """Return prediction data for a single inverter."""
    p = get_prediction(inverter_id)
    if not p:
        raise HTTPException(404, f"No prediction for '{inverter_id}'")
    return p.model_dump(mode="json")

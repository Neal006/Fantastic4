"""
Agentic workflow for LUMIN.AI.

The agent autonomously:
  1. Retrieves inverter prediction data
  2. Runs risk assessment
  3. Retrieves relevant troubleshooting from the inverter manual (RAG)
  4. Drafts a maintenance ticket via LLM
  5. Generates a professional PDF
"""

import re
import json
import uuid
from datetime import datetime

from app.llm import LLMClient
from app.rag import RAGPipeline
from app.prompts import (
    SYSTEM_PROMPT_TICKET,
    USER_PROMPT_TICKET,
    SYSTEM_PROMPT_RISK_REPORT,
    USER_PROMPT_RISK_REPORT,
)
from app.guardrails import validate_shap_features
from app.synthetic_data import (
    get_prediction,
    get_plant_predictions,
    PLANTS,
)
from app.ticket import generate_ticket_pdf


class SolarAgent:
    def __init__(self, llm: LLMClient, rag: RAGPipeline):
        self.llm = llm
        self.rag = rag

    # ------------------------------------------------------------------
    # Maintenance ticket  (full agentic pipeline)
    # ------------------------------------------------------------------
    def generate_maintenance_ticket(self, inverter_id: str) -> dict:
        # Step 1 – autonomously retrieve prediction
        prediction = get_prediction(inverter_id)
        if not prediction:
            raise ValueError(f"No prediction data for '{inverter_id}'")

        # Step 2 – validate SHAP features (input guardrail)
        valid_shap = validate_shap_features(prediction.shap_values)

        # Step 3 – RAG: fetch troubleshooting context from manual
        rag_q = self._troubleshoot_query(prediction)
        chunks = self.rag.retrieve(rag_q)
        manual_ctx = (
            "\n---\n".join(c["text"] for c in chunks[:4])
            if chunks
            else "No manual context available."
        )

        # Step 4 – format data for LLM
        shap_str = "\n".join(
            f"  • {f}: {v:+.4f}  (value: {prediction.raw_features.get(f, 'N/A')})"
            for f, v in sorted(
                valid_shap.items(), key=lambda x: abs(x[1]), reverse=True
            )[:10]
        )
        raw_str = "\n".join(
            f"  • {f}: {v}"
            for f, v in list(prediction.raw_features.items())[:15]
        )

        plant_info = self._plant_meta(inverter_id)

        user_prompt = USER_PROMPT_TICKET.format(
            inverter_id=inverter_id,
            plant_id=prediction.plant_id,
            plant_name=plant_info["name"],
            block=plant_info["block"],
            risk_score=prediction.risk_score,
            risk_class=prediction.risk_class.value,
            shap_features=shap_str,
            raw_features=raw_str,
            manual_context=manual_ctx,
        )

        # Step 5 – LLM generates ticket content
        raw = self.llm.generate(SYSTEM_PROMPT_TICKET, user_prompt, temperature=0.2)

        # Step 6 – parse JSON (with fallback)
        ticket_data = self._parse_ticket_json(raw, prediction, inverter_id)

        # Step 7 – generate PDF
        ticket_id = (
            f"TKT-{datetime.utcnow().strftime('%Y%m%d')}-"
            f"{uuid.uuid4().hex[:6].upper()}"
        )
        now = datetime.utcnow()
        pdf_path = generate_ticket_pdf(
            ticket_id=ticket_id,
            inverter_id=inverter_id,
            plant_id=prediction.plant_id,
            plant_name=plant_info["name"],
            block=plant_info["block"],
            risk_score=prediction.risk_score,
            risk_class=prediction.risk_class.value,
            ticket_data=ticket_data,
            timestamp=now,
        )

        return {
            "ticket_id": ticket_id,
            "inverter_id": inverter_id,
            "pdf_path": pdf_path,
            "ticket_data": ticket_data,
            "generated_at": now.isoformat(),
        }

    # ------------------------------------------------------------------
    # Plant-wide risk report
    # ------------------------------------------------------------------
    def generate_risk_report(self, plant_id: str) -> str:
        predictions = get_plant_predictions(plant_id)
        if not predictions:
            raise ValueError(f"No predictions for plant '{plant_id}'")

        plant_name = PLANTS.get(plant_id, {}).get("name", plant_id)

        pred_lines: list[str] = []
        for p in predictions:
            top = sorted(
                p.shap_values.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]
            shap_summary = ", ".join(f"{k}: {v:+.4f}" for k, v in top)
            power = p.raw_features.get(
                "inverters[0].power",
                p.raw_features.get("inverters[1].power", "N/A"),
            )
            temp = p.raw_features.get(
                "inverters[0].temp",
                p.raw_features.get("inverters[1].temp", "N/A"),
            )
            pred_lines.append(
                f"Inverter: {p.inverter_id}  |  Block: {p.block}\n"
                f"  Risk Score: {p.risk_score:.4f}  |  Class: {p.risk_class.value}\n"
                f"  Top SHAP: {shap_summary}\n"
                f"  Power: {power}W  |  Temp: {temp}°C\n"
            )

        user_prompt = USER_PROMPT_RISK_REPORT.format(
            plant_id=plant_id,
            plant_name=plant_name,
            predictions="\n".join(pred_lines),
        )

        return self.llm.generate(
            SYSTEM_PROMPT_RISK_REPORT, user_prompt, temperature=0.3
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _plant_meta(inverter_id: str) -> dict:
        for pid, pdata in PLANTS.items():
            for ldata in pdata["loggers"].values():
                if inverter_id in ldata["inverters"]:
                    return {
                        "name": pdata["name"],
                        "block": ldata["block"],
                        "plant_id": pid,
                    }
        return {"name": "Unknown", "block": "Unknown", "plant_id": "unknown"}

    @staticmethod
    def _troubleshoot_query(prediction) -> str:
        top = sorted(
            prediction.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        parts = ["troubleshooting maintenance"]
        for feat, _ in top:
            if "temp" in feat:
                parts.append("overheating thermal shutdown cooling")
            elif "alarm" in feat:
                parts.append("alarm fault error code reset")
            elif "current" in feat:
                parts.append("string current low degradation")
            elif "voltage" in feat:
                parts.append("voltage drop panel string")
            elif "power" in feat:
                parts.append("power reduction derating output")
            elif "freq" in feat:
                parts.append("grid frequency deviation protection")
            elif "pf" in feat:
                parts.append("power factor reactive grid")
        return " ".join(parts)

    @staticmethod
    def _parse_ticket_json(raw: str, prediction, inverter_id: str) -> dict:
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence:
            text = fence.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "title": f"Risk Alert – {inverter_id}",
                "priority": (
                    "P1-Critical"
                    if prediction.risk_score > 0.85
                    else "P2-High"
                    if prediction.risk_score > 0.6
                    else "P3-Medium"
                ),
                "description": raw[:600],
                "root_cause_analysis": "See description above.",
                "recommended_actions": ["Manual review required."],
                "estimated_downtime": "TBD",
                "parts_needed": [],
                "safety_notes": ["Follow standard safety procedures."],
                "escalation_needed": prediction.risk_score > 0.8,
            }

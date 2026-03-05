"""
Explanation engine – converts ML predictions into operator-friendly narratives.
"""

from datetime import datetime

from app.llm import LLMClient
from app.rag import RAGPipeline
from app.prompts import SYSTEM_PROMPT_EXPLANATION, USER_PROMPT_EXPLANATION
from app.guardrails import (
    validate_shap_features,
    parse_llm_json,
    validate_explanation_output,
    guardrail_disclaimer,
)
from app.synthetic_data import get_prediction, PLANTS
from app.models import ExplanationResponse


class Explainer:
    def __init__(self, llm: LLMClient, rag: RAGPipeline):
        self.llm = llm
        self.rag = rag

    def explain(self, inverter_id: str) -> ExplanationResponse:
        # 1. Fetch prediction
        prediction = get_prediction(inverter_id)
        if not prediction:
            raise ValueError(f"No prediction data for inverter '{inverter_id}'")

        # 2. Input guardrail – validate SHAP features
        valid_shap = validate_shap_features(prediction.shap_values)

        # 3. RAG – retrieve relevant manual context
        rag_query = self._rag_query(prediction)
        manual_chunks = self.rag.retrieve(rag_query)
        manual_context = (
            "\n---\n".join(c["text"] for c in manual_chunks[:3])
            if manual_chunks
            else "No inverter-manual context available."
        )

        # 4. Format SHAP features for prompt (sorted by |importance|)
        sorted_shap = sorted(
            valid_shap.items(), key=lambda x: abs(x[1]), reverse=True
        )[:10]
        shap_str = "\n".join(
            f"  • {feat}: {val:+.4f}  (raw value: "
            f"{prediction.raw_features.get(feat, 'N/A')})"
            for feat, val in sorted_shap
        )

        raw_str = "\n".join(
            f"  • {feat}: {val}"
            for feat, val in list(prediction.raw_features.items())[:15]
        )

        # 5. Resolve plant metadata
        plant_name, block = self._plant_meta(inverter_id)

        # 6. Call LLM
        user_prompt = USER_PROMPT_EXPLANATION.format(
            inverter_id=inverter_id,
            plant_id=prediction.plant_id,
            plant_name=plant_name,
            block=block,
            timestamp=prediction.timestamp.isoformat(),
            risk_score=prediction.risk_score,
            risk_class=prediction.risk_class.value,
            shap_features=shap_str,
            raw_features=raw_str,
            manual_context=manual_context,
        )

        raw_response = self.llm.generate(SYSTEM_PROMPT_EXPLANATION, user_prompt)

        # 7. Output guardrail – parse and validate
        parsed, parse_err = parse_llm_json(raw_response)
        if parsed is None:
            # Graceful fallback when JSON parsing fails
            parsed = {
                "summary": raw_response[:500],
                "key_factors": [],
                "recommended_actions": ["Review raw model output manually."],
                "urgency": "scheduled",
                "estimated_impact": "Unable to parse structured output.",
            }

        violations = validate_explanation_output(parsed, prediction.shap_values)
        if violations:
            parsed["_guardrail_warnings"] = violations

        # 8. Build response
        return ExplanationResponse(
            inverter_id=inverter_id,
            plant_id=prediction.plant_id,
            risk_score=prediction.risk_score,
            risk_class=prediction.risk_class.value,
            summary=parsed.get("summary", ""),
            key_factors=parsed.get("key_factors", []),
            recommended_actions=parsed.get("recommended_actions", []),
            urgency=parsed.get("urgency", "scheduled"),
            generated_at=datetime.utcnow(),
            grounded_sources=[
                "ML model prediction (SHAP analysis)",
                "Inverter sensor telemetry",
                *(
                    ["Inverter technical manual (RAG)"]
                    if manual_chunks
                    else []
                ),
            ],
            disclaimer=guardrail_disclaimer(prediction.risk_class.value),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _rag_query(self, prediction) -> str:
        top = sorted(
            prediction.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        parts = [f"inverter {prediction.risk_class.value}"]
        for feat, _ in top:
            if "temp" in feat:
                parts.append("temperature overheating thermal protection")
            elif "alarm" in feat:
                parts.append("alarm code fault error troubleshooting")
            elif "current" in feat:
                parts.append("string current mismatch degradation")
            elif "voltage" in feat:
                parts.append("string voltage panel issue")
            elif "power" in feat:
                parts.append("power output reduction derating")
            elif "freq" in feat:
                parts.append("grid frequency deviation anti-islanding")
            elif "pf" in feat:
                parts.append("power factor grid connection reactive")
            elif "string" in feat:
                parts.append("string monitoring current imbalance")
        return " ".join(parts)

    @staticmethod
    def _plant_meta(inverter_id: str):
        for pdata in PLANTS.values():
            for ldata in pdata["loggers"].values():
                if inverter_id in ldata["inverters"]:
                    return pdata["name"], ldata["block"]
        return "Unknown", "Unknown"

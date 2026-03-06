"""
Evaluation metrics for the ablation study.
Scores each model response on multiple dimensions relevant to the LUMIN.AI use case.
"""

import json
import re
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# JSON validity
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Optional[dict]:
    """Try to extract JSON from raw LLM output, tolerating markdown fences."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def score_json_validity(response: str, task: str) -> dict:
    """
    Check if the response is valid JSON and contains the required schema fields.
    Returns: {"score": 0-1, "valid_json": bool, "has_required_fields": bool, "details": str}
    """
    if not response:
        return {"score": 0.0, "valid_json": False, "has_required_fields": False, "details": "Empty response"}

    parsed = _extract_json(response)
    if parsed is None:
        return {"score": 0.0, "valid_json": False, "has_required_fields": False, "details": "Not valid JSON"}

    if task == "explanation":
        required = {"summary", "key_factors", "recommended_actions", "urgency"}
    elif task == "ticket":
        required = {"title", "priority", "description", "root_cause_analysis", "recommended_actions"}
    elif task == "chat":
        return {"score": 1.0, "valid_json": True, "has_required_fields": True, "details": "Chat does not require JSON"}
    else:
        required = set()

    present = set(parsed.keys())
    missing = required - present
    has_all = len(missing) == 0
    field_ratio = (len(required) - len(missing)) / len(required) if required else 1.0

    score = 0.5 + (0.5 * field_ratio)  # 0.5 for valid JSON, up to 1.0 for all fields
    details = f"Missing fields: {missing}" if missing else "All required fields present"

    return {"score": round(score, 3), "valid_json": True, "has_required_fields": has_all, "details": details}


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------
def score_hallucination(response: str, shap_features: dict, raw_features: dict) -> dict:
    """
    Check if the model only references features that exist in the provided data.
    Lower hallucination = higher score.
    Returns: {"score": 0-1, "hallucinated_values": list, "details": str}
    """
    if not response:
        return {"score": 0.0, "hallucinated_values": [], "details": "Empty response"}

    hallucinated = []
    all_feature_names = set(shap_features.keys()) | set(raw_features.keys())

    # Extract feature-like patterns from response
    # Look for patterns like inverters[0].temp, meters[0].freq, etc.
    feature_pattern = re.compile(r'(?:inverters|meters|sensors|smu)\[\d+\]\.\w+')
    mentioned_features = set(feature_pattern.findall(response))

    for feat in mentioned_features:
        if feat not in all_feature_names:
            hallucinated.append(feat)

    # Also check for fabricated numeric values with units
    # Look for temperature values not in raw_features
    temp_pattern = re.compile(r'(\d+\.?\d*)\s*°C')
    mentioned_temps = {float(t) for t in temp_pattern.findall(response)}
    actual_temps = set()
    for k, v in raw_features.items():
        if "temp" in k:
            actual_temps.add(float(v))

    fabricated_temps = []
    for t in mentioned_temps:
        if not any(abs(t - at) < 0.5 for at in actual_temps) and actual_temps:
            fabricated_temps.append(t)

    total_issues = len(hallucinated) + len(fabricated_temps)
    score = max(0.0, 1.0 - (total_issues * 0.2))

    details_parts = []
    if hallucinated:
        details_parts.append(f"Unknown features: {hallucinated}")
    if fabricated_temps:
        details_parts.append(f"Possibly fabricated temps: {fabricated_temps}")
    details = "; ".join(details_parts) if details_parts else "No hallucinations detected"

    return {
        "score": round(score, 3),
        "hallucinated_values": hallucinated + [f"{t}°C" for t in fabricated_temps],
        "details": details,
    }


# ---------------------------------------------------------------------------
# Urgency appropriateness
# ---------------------------------------------------------------------------
EXPECTED_URGENCY = {
    "no_risk": ["routine", "scheduled"],
    "degradation_risk": ["within_24h", "scheduled"],
    "shutdown_risk": ["immediate", "within_24h"],
}


def score_urgency(response: str, risk_class: str, task: str) -> dict:
    """
    Check if the urgency classification matches the risk class.
    Only applies to explanation task.
    Returns: {"score": 0-1, "predicted_urgency": str, "expected": list, "details": str}
    """
    if task != "explanation":
        return {"score": 1.0, "predicted_urgency": "N/A", "expected": [], "details": "Not applicable"}

    if not response:
        return {"score": 0.0, "predicted_urgency": None, "expected": EXPECTED_URGENCY.get(risk_class, []), "details": "Empty response"}

    parsed = _extract_json(response)
    if parsed is None:
        return {"score": 0.0, "predicted_urgency": None, "expected": EXPECTED_URGENCY.get(risk_class, []), "details": "Could not parse JSON"}

    urgency = parsed.get("urgency", "").lower().strip()
    expected = EXPECTED_URGENCY.get(risk_class, [])

    if urgency in expected:
        score = 1.0
        details = f"Correct: '{urgency}' matches {risk_class}"
    elif urgency in ["immediate", "within_24h", "scheduled", "routine"]:
        score = 0.3
        details = f"Valid but wrong: '{urgency}' (expected {expected} for {risk_class})"
    else:
        score = 0.0
        details = f"Invalid urgency: '{urgency}'"

    return {"score": round(score, 3), "predicted_urgency": urgency, "expected": expected, "details": details}


# ---------------------------------------------------------------------------
# Technical completeness
# ---------------------------------------------------------------------------
def score_completeness(response: str, task: str) -> dict:
    """
    Measure how complete and actionable the response is.
    Checks for presence of specific technical elements.
    Returns: {"score": 0-1, "checks": dict, "details": str}
    """
    if not response:
        return {"score": 0.0, "checks": {}, "details": "Empty response"}

    checks = {}
    text_lower = response.lower()
    parsed = _extract_json(response)

    if task == "explanation":
        checks["has_summary"] = parsed is not None and bool(parsed.get("summary"))
        checks["has_factors"] = parsed is not None and len(parsed.get("key_factors", [])) > 0
        checks["has_actions"] = parsed is not None and len(parsed.get("recommended_actions", [])) > 0
        checks["has_urgency"] = parsed is not None and bool(parsed.get("urgency"))
        checks["has_impact"] = parsed is not None and bool(parsed.get("estimated_impact"))
        checks["mentions_sensor_values"] = bool(re.search(r'\d+\.?\d*', response))
        checks["mentions_shap"] = any(w in text_lower for w in ["shap", "feature", "impact", "factor"])

    elif task == "ticket":
        checks["has_title"] = parsed is not None and bool(parsed.get("title"))
        checks["has_priority"] = parsed is not None and bool(parsed.get("priority"))
        checks["has_description"] = parsed is not None and len(str(parsed.get("description", ""))) > 20
        checks["has_root_cause"] = parsed is not None and bool(parsed.get("root_cause_analysis"))
        checks["has_actions"] = parsed is not None and len(parsed.get("recommended_actions", [])) > 0
        checks["has_safety"] = parsed is not None and len(parsed.get("safety_notes", [])) > 0
        checks["has_parts"] = parsed is not None and "parts_needed" in (parsed or {})

    elif task == "chat":
        checks["is_substantive"] = len(response) > 50
        checks["references_data"] = bool(re.search(r'\d+\.?\d*', response))
        checks["is_actionable"] = any(w in text_lower for w in ["recommend", "suggest", "should", "action", "check", "inspect"])
        checks["is_grounded"] = any(w in text_lower for w in ["risk", "score", "sensor", "inverter"])
        checks["is_concise"] = len(response) < 2000

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    score = passed / total if total > 0 else 0.0

    return {"score": round(score, 3), "checks": checks, "details": f"{passed}/{total} checks passed"}


# ---------------------------------------------------------------------------
# Response quality (length, coherence heuristics)
# ---------------------------------------------------------------------------
def score_response_quality(response: str) -> dict:
    """
    Basic quality heuristics: appropriate length, no error messages, no refusals.
    Returns: {"score": 0-1, "word_count": int, "details": str}
    """
    if not response:
        return {"score": 0.0, "word_count": 0, "details": "Empty response"}

    word_count = len(response.split())
    issues = []

    # Check for common failure patterns
    refusal_patterns = ["i cannot", "i'm sorry", "as an ai", "i don't have access", "i am unable"]
    if any(p in response.lower() for p in refusal_patterns):
        issues.append("Contains refusal/hedging language")

    error_patterns = ["error", "exception", "traceback", "undefined"]
    if any(p in response.lower()[:100] for p in error_patterns):
        issues.append("Starts with error-like text")

    # Ideal length range: 100-800 words
    if word_count < 30:
        issues.append(f"Too short ({word_count} words)")
    elif word_count > 1500:
        issues.append(f"Excessively long ({word_count} words)")

    deductions = len(issues) * 0.25
    score = max(0.0, 1.0 - deductions)

    return {
        "score": round(score, 3),
        "word_count": word_count,
        "details": "; ".join(issues) if issues else "Good quality",
    }


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------
def evaluate_response(
    response: str,
    task: str,
    risk_class: str,
    shap_features: dict,
    raw_features: dict,
    latency_seconds: float,
) -> dict:
    """
    Run all evaluation metrics on a single response.
    Returns a dict with per-metric scores and an overall weighted score.
    """
    json_result = score_json_validity(response, task)
    hallucination_result = score_hallucination(response, shap_features, raw_features)
    urgency_result = score_urgency(response, risk_class, task)
    completeness_result = score_completeness(response, task)
    quality_result = score_response_quality(response)

    # Weighted overall score
    weights = {
        "json_validity": 0.25,
        "hallucination": 0.25,
        "urgency_accuracy": 0.10,
        "completeness": 0.25,
        "response_quality": 0.15,
    }

    overall = (
        weights["json_validity"] * json_result["score"]
        + weights["hallucination"] * hallucination_result["score"]
        + weights["urgency_accuracy"] * urgency_result["score"]
        + weights["completeness"] * completeness_result["score"]
        + weights["response_quality"] * quality_result["score"]
    )

    return {
        "overall_score": round(overall, 3),
        "json_validity": json_result,
        "hallucination": hallucination_result,
        "urgency_accuracy": urgency_result,
        "completeness": completeness_result,
        "response_quality": quality_result,
        "latency_seconds": latency_seconds,
    }

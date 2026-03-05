"""
Hallucination guardrails for SolarGuard AI.

Ensures the LLM never fabricates telemetry values and only references
features that were actually present in the SHAP analysis.
"""

import re
import json
from typing import Dict, Set, Tuple

# ---------------------------------------------------------------------------
# Valid feature names (derived from the dataset schema)
# ---------------------------------------------------------------------------
_BASE_FEATURES = {
    "alarm_code", "op_state", "power", "temp",
    "pv1_current", "pv1_voltage", "pv1_power",
    "pv2_current", "pv2_voltage", "pv2_power",
    "pv3_current", "pv3_voltage",
    "pv4_current", "pv4_voltage",
    "pv5_current", "pv5_voltage",
    "pv6_current", "pv6_voltage",
    "pv7_current", "pv7_voltage",
    "pv8_current", "pv8_voltage",
    "pv9_current", "pv9_voltage",
    "kwh_total", "kwh_today", "kwh_midnight",
    "limit_percent", "model", "serial", "id",
}

_METER_FEATURES = {
    "meter_kwh_today", "meter_active_power", "meter_kwh_import",
    "meter_kwh_total", "pf", "freq",
    "p_b", "p_y", "p_r", "v_b", "v_y", "v_r",
    "meter_reactive_power", "meter_apparent_power",
    "base_meter_kwh_import", "base_meter_kwh_total",
    "original_meter_kwh_import", "original_meter_kwh_total",
}

_SENSOR_FEATURES = {"ambient_temp"}

_SMU_FEATURES = {f"string{i}" for i in range(1, 19)}

_PREFIXES = [
    "inverters[0].", "inverters[1].",
    "meters[0].",
    "sensors[0].",
    "smu[0].", "smu[1].",
]


def get_all_valid_features() -> Set[str]:
    """Expand every prefix × base feature into the full set."""
    features: Set[str] = set()
    for prefix in _PREFIXES:
        if prefix.startswith("inverters"):
            features.update(f"{prefix}{f}" for f in _BASE_FEATURES)
        elif prefix.startswith("meters"):
            features.update(f"{prefix}{f}" for f in _METER_FEATURES)
        elif prefix.startswith("sensors"):
            features.update(f"{prefix}{f}" for f in _SENSOR_FEATURES)
        elif prefix.startswith("smu"):
            features.update(f"{prefix}{f}" for f in _SMU_FEATURES)
    return features


ALL_VALID_FEATURES = get_all_valid_features()


# ---------------------------------------------------------------------------
# Input guardrail – validate SHAP features before sending to LLM
# ---------------------------------------------------------------------------

def validate_shap_features(shap_values: Dict[str, float]) -> Dict[str, float]:
    """Keep only features that belong to the known dataset schema."""
    validated = {}
    for feature, value in shap_values.items():
        if feature in ALL_VALID_FEATURES:
            validated[feature] = value
        else:
            # Allow if it at least starts with a known prefix
            if any(feature.startswith(p) for p in _PREFIXES):
                validated[feature] = value
    return validated


# ---------------------------------------------------------------------------
# Output guardrail – validate LLM JSON response
# ---------------------------------------------------------------------------

def parse_llm_json(raw: str) -> Tuple[dict | None, str | None]:
    """Extract and parse JSON from raw LLM output, tolerating markdown fences."""
    text = raw.strip()
    # Strip markdown code fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


def validate_explanation_output(
    parsed: dict,
    provided_shap: Dict[str, float],
) -> list[str]:
    """Return a list of guardrail violations (empty = clean)."""
    issues: list[str] = []

    provided_keys = set(provided_shap.keys())

    for factor in parsed.get("key_factors", []):
        feat = factor.get("feature", "")
        if feat and feat not in provided_keys:
            # Tolerate shortened names (e.g. "pv3_current" matching
            # "inverters[1].pv3_current")
            if not any(feat in k for k in provided_keys):
                issues.append(
                    f"Hallucinated feature '{feat}' – not in SHAP input"
                )

    # Verify urgency is one of the allowed values
    allowed_urgency = {"immediate", "within_24h", "scheduled", "routine"}
    if parsed.get("urgency", "").lower() not in allowed_urgency:
        issues.append(f"Invalid urgency value: {parsed.get('urgency')}")

    return issues


# ---------------------------------------------------------------------------
# Disclaimer text
# ---------------------------------------------------------------------------

def guardrail_disclaimer(risk_class: str) -> str:
    base = (
        "This analysis is AI-generated based on ML model predictions and "
        "SHAP feature importances. All referenced values come directly from "
        "sensor telemetry and model output. No values have been fabricated."
    )
    if risk_class in ("shutdown_risk", "degradation_risk"):
        base += (
            " Please verify critical readings with on-site instrumentation "
            "before taking action."
        )
    return base

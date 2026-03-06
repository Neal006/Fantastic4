"""
Test scenarios for the ablation study.
Uses the actual LUMIN.AI prompts with synthetic inverter data
covering all 3 risk classes: no_risk, degradation_risk, shutdown_risk.
"""

import sys
from pathlib import Path

# Allow importing from parent genai/app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.prompts import (
    SYSTEM_PROMPT_EXPLANATION,
    USER_PROMPT_EXPLANATION,
    SYSTEM_PROMPT_CHAT,
    USER_PROMPT_CHAT,
    SYSTEM_PROMPT_TICKET,
    USER_PROMPT_TICKET,
    SYSTEM_PROMPT_RISK_REPORT,
    USER_PROMPT_RISK_REPORT,
)

# ---------------------------------------------------------------------------
# 3 representative inverter scenarios (one per risk class)
# ---------------------------------------------------------------------------

SCENARIO_NORMAL = {
    "inverter_id": "INV-P1-L1-0",
    "plant_id": "plant_1",
    "plant_name": "Plant 1 - Celestical",
    "block": "Block A",
    "timestamp": "2026-03-05T15:30:00Z",
    "risk_score": 0.12,
    "risk_class": "no_risk",
    "shap_values": {
        "inverters[0].power": -0.04,
        "inverters[0].temp": -0.03,
        "sensors[0].ambient_temp": -0.02,
        "inverters[0].pv1_current": -0.01,
        "meters[0].pf": -0.01,
        "inverters[0].kwh_today": -0.01,
    },
    "raw_features": {
        "inverters[0].power": 8750.0,
        "inverters[0].temp": 42.3,
        "inverters[0].alarm_code": 0,
        "inverters[0].op_state": 1,
        "inverters[0].pv1_current": 9.8,
        "inverters[0].pv1_voltage": 38.2,
        "inverters[0].pv2_current": 9.7,
        "inverters[0].pv2_voltage": 38.0,
        "inverters[0].pv3_current": 9.6,
        "inverters[0].pv3_voltage": 37.9,
        "inverters[0].kwh_today": 42.5,
        "inverters[0].limit_percent": 100,
        "meters[0].pf": 0.98,
        "meters[0].freq": 50.01,
        "sensors[0].ambient_temp": 34.2,
    },
}

SCENARIO_DEGRADATION = {
    "inverter_id": "INV-P1-L1-1",
    "plant_id": "plant_1",
    "plant_name": "Plant 1 - Celestical",
    "block": "Block A",
    "timestamp": "2026-03-05T15:30:00Z",
    "risk_score": 0.65,
    "risk_class": "degradation_risk",
    "shap_values": {
        "inverters[1].pv3_current": +0.18,
        "smu[1].string3": +0.14,
        "inverters[1].power": +0.12,
        "inverters[1].pv3_voltage": +0.08,
        "sensors[0].ambient_temp": +0.04,
        "inverters[1].kwh_today": +0.05,
        "inverters[1].pv4_current": +0.03,
    },
    "raw_features": {
        "inverters[1].power": 6200.0,
        "inverters[1].temp": 44.1,
        "inverters[1].alarm_code": 0,
        "inverters[1].op_state": 1,
        "inverters[1].pv1_current": 9.5,
        "inverters[1].pv1_voltage": 37.8,
        "inverters[1].pv2_current": 9.4,
        "inverters[1].pv2_voltage": 37.6,
        "inverters[1].pv3_current": 3.1,
        "inverters[1].pv3_voltage": 28.4,
        "inverters[1].pv4_current": 7.2,
        "inverters[1].pv4_voltage": 36.1,
        "inverters[1].kwh_today": 28.3,
        "inverters[1].limit_percent": 100,
        "meters[0].pf": 0.96,
        "meters[0].freq": 50.00,
        "sensors[0].ambient_temp": 35.8,
        "smu[1].string3": 3.1,
        "smu[1].string4": 7.2,
    },
}

SCENARIO_SHUTDOWN = {
    "inverter_id": "INV-P1-L2-0",
    "plant_id": "plant_1",
    "plant_name": "Plant 1 - Celestical",
    "block": "Block B",
    "timestamp": "2026-03-05T15:30:00Z",
    "risk_score": 0.89,
    "risk_class": "shutdown_risk",
    "shap_values": {
        "inverters[0].temp": +0.35,
        "sensors[0].ambient_temp": +0.15,
        "inverters[0].alarm_code": +0.12,
        "inverters[0].power": +0.10,
        "inverters[0].limit_percent": +0.08,
        "inverters[0].pv1_current": +0.04,
    },
    "raw_features": {
        "inverters[0].power": 4100.0,
        "inverters[0].temp": 78.6,
        "inverters[0].alarm_code": 4003,
        "inverters[0].op_state": 2,
        "inverters[0].pv1_current": 8.1,
        "inverters[0].pv1_voltage": 36.5,
        "inverters[0].pv2_current": 7.9,
        "inverters[0].pv2_voltage": 36.2,
        "inverters[0].pv3_current": 7.8,
        "inverters[0].pv3_voltage": 36.0,
        "inverters[0].kwh_today": 18.7,
        "inverters[0].limit_percent": 42,
        "meters[0].pf": 0.94,
        "meters[0].freq": 50.02,
        "sensors[0].ambient_temp": 47.3,
    },
}

ALL_SCENARIOS = {
    "normal_operation": SCENARIO_NORMAL,
    "degradation_risk": SCENARIO_DEGRADATION,
    "shutdown_risk": SCENARIO_SHUTDOWN,
}


# ---------------------------------------------------------------------------
# Prompt builders  –  format prompts exactly as the production system does
# ---------------------------------------------------------------------------

def _format_shap(shap_values: dict, raw_features: dict) -> str:
    sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    return "\n".join(
        f"  \u2022 {feat}: {val:+.4f}  (raw value: {raw_features.get(feat, 'N/A')})"
        for feat, val in sorted_shap
    )


def _format_raw(raw_features: dict) -> str:
    return "\n".join(
        f"  \u2022 {feat}: {val}"
        for feat, val in list(raw_features.items())[:15]
    )


def build_explanation_prompts(scenario: dict) -> tuple:
    """Returns (system_prompt, user_prompt) for explanation task."""
    shap_str = _format_shap(scenario["shap_values"], scenario["raw_features"])
    raw_str = _format_raw(scenario["raw_features"])
    user_prompt = USER_PROMPT_EXPLANATION.format(
        inverter_id=scenario["inverter_id"],
        plant_id=scenario["plant_id"],
        plant_name=scenario["plant_name"],
        block=scenario["block"],
        timestamp=scenario["timestamp"],
        risk_score=scenario["risk_score"],
        risk_class=scenario["risk_class"],
        shap_features=shap_str,
        raw_features=raw_str,
        manual_context="No inverter-manual context available.",
    )
    return SYSTEM_PROMPT_EXPLANATION, user_prompt


def build_ticket_prompts(scenario: dict) -> tuple:
    """Returns (system_prompt, user_prompt) for ticket generation task."""
    shap_str = _format_shap(scenario["shap_values"], scenario["raw_features"])
    raw_str = _format_raw(scenario["raw_features"])
    user_prompt = USER_PROMPT_TICKET.format(
        inverter_id=scenario["inverter_id"],
        plant_id=scenario["plant_id"],
        plant_name=scenario["plant_name"],
        block=scenario["block"],
        risk_score=scenario["risk_score"],
        risk_class=scenario["risk_class"],
        shap_features=shap_str,
        raw_features=raw_str,
        manual_context="No inverter-manual context available.",
    )
    return SYSTEM_PROMPT_TICKET, user_prompt


def build_chat_prompts(scenario: dict) -> tuple:
    """Returns (system_prompt, user_prompt) for chat Q&A task."""
    plant_overview = (
        f"{scenario['plant_name']} ({scenario['plant_id']}):\n"
        f"  {scenario['block']}:\n"
        f"    {scenario['inverter_id']}: risk={scenario['risk_score']:.2f} "
        f"class={scenario['risk_class']}"
    )
    data_context = (
        f"{scenario['inverter_id']} | {scenario['plant_id']} | {scenario['block']} | "
        f"risk={scenario['risk_score']:.2f} | class={scenario['risk_class']} | "
        f"SHAP=[{', '.join(f'{k}: {v:+.3f}' for k, v in list(scenario['shap_values'].items())[:3])}]"
    )
    system = SYSTEM_PROMPT_CHAT.format(plant_overview=plant_overview)
    user = USER_PROMPT_CHAT.format(
        query=f"What is the current status of {scenario['inverter_id']} and what should I do?",
        data_context=data_context,
        manual_context="No manual context available.",
    )
    return system, user


# ---------------------------------------------------------------------------
# All test cases
# ---------------------------------------------------------------------------

def get_all_test_cases() -> list:
    """
    Returns a list of dicts, each with:
      - task: str (explanation / ticket / chat)
      - scenario_name: str
      - risk_class: str
      - system_prompt: str
      - user_prompt: str
    """
    cases = []
    for name, scenario in ALL_SCENARIOS.items():
        sys_e, usr_e = build_explanation_prompts(scenario)
        cases.append({
            "task": "explanation",
            "scenario_name": name,
            "risk_class": scenario["risk_class"],
            "system_prompt": sys_e,
            "user_prompt": usr_e,
        })

        sys_t, usr_t = build_ticket_prompts(scenario)
        cases.append({
            "task": "ticket",
            "scenario_name": name,
            "risk_class": scenario["risk_class"],
            "system_prompt": sys_t,
            "user_prompt": usr_t,
        })

        sys_c, usr_c = build_chat_prompts(scenario)
        cases.append({
            "task": "chat",
            "scenario_name": name,
            "risk_class": scenario["risk_class"],
            "system_prompt": sys_c,
            "user_prompt": usr_c,
        })

    return cases

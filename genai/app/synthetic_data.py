"""
Synthetic ML prediction data for testing.
Replace with real ML backend API calls when the model is ready.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from app.models import InverterPrediction, RiskClass


# ---------------------------------------------------------------------------
# Plant / inverter topology  (matches the actual dataset structure)
# ---------------------------------------------------------------------------
PLANTS = {
    "plant_1": {
        "name": "Plant 1 - Celestical",
        "loggers": {
            "ICR2-LT1": {
                "mac": "ICR2-LT1-Celestical-10000.73",
                "block": "Block A",
                "inverters": ["INV-P1-L1-0", "INV-P1-L1-1"],
            },
            "ICR2-LT2": {
                "mac": "ICR2-LT2-Celestical-10000.73",
                "block": "Block B",
                "inverters": ["INV-P1-L2-0", "INV-P1-L2-1"],
            },
        },
    },
    "plant_2": {
        "name": "Plant 2",
        "loggers": {
            "Logger-AC12": {
                "mac": "80-1F-12-0F-AC-12",
                "block": "Block A",
                "inverters": ["INV-P2-L1-0", "INV-P2-L1-1"],
            },
            "Logger-ACBB": {
                "mac": "80-1F-12-0F-AC-BB",
                "block": "Block B",
                "inverters": ["INV-P2-L2-0", "INV-P2-L2-1"],
            },
        },
    },
    "plant_3": {
        "name": "Plant 3",
        "loggers": {
            "Logger-1469": {
                "mac": "54-10-EC-8C-14-69",
                "block": "Block A",
                "inverters": ["INV-P3-L1-0", "INV-P3-L1-1"],
            },
            "Logger-146E": {
                "mac": "54-10-EC-8C-14-6E",
                "block": "Block B",
                "inverters": ["INV-P3-L2-0", "INV-P3-L2-1"],
            },
        },
    },
}


def _inv_meta(inverter_id: str) -> dict:
    """Return plant_id, block, mac, index for a given inverter_id."""
    for pid, pdata in PLANTS.items():
        for lid, ldata in pdata["loggers"].items():
            for idx, inv_id in enumerate(ldata["inverters"]):
                if inv_id == inverter_id:
                    return {
                        "plant_id": pid,
                        "block": ldata["block"],
                        "mac": ldata["mac"],
                        "index": idx,
                        "plant_name": pdata["name"],
                    }
    return None


# ---------------------------------------------------------------------------
# Synthetic predictions  –  diverse failure scenarios for demo / testing
# ---------------------------------------------------------------------------
_NOW = datetime.utcnow()

SYNTHETIC_PREDICTIONS: Dict[str, InverterPrediction] = {}

_SCENARIOS = [
    # ---- Plant 1 ----
    {
        "id": "INV-P1-L1-0",
        "risk_score": 0.12,
        "risk_class": RiskClass.NO_RISK,
        "scenario": "normal_operation",
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
    },
    {
        "id": "INV-P1-L1-1",
        "risk_score": 0.65,
        "risk_class": RiskClass.DEGRADATION_RISK,
        "scenario": "string_degradation",
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
    },
    {
        "id": "INV-P1-L2-0",
        "risk_score": 0.89,
        "risk_class": RiskClass.SHUTDOWN_RISK,
        "scenario": "overheating",
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
    },
    {
        "id": "INV-P1-L2-1",
        "risk_score": 0.08,
        "risk_class": RiskClass.NO_RISK,
        "scenario": "normal_operation",
        "shap_values": {
            "inverters[1].power": -0.03,
            "inverters[1].temp": -0.02,
            "inverters[1].pv1_current": -0.01,
            "sensors[0].ambient_temp": +0.01,
            "meters[0].pf": -0.01,
        },
        "raw_features": {
            "inverters[1].power": 9100.0,
            "inverters[1].temp": 40.5,
            "inverters[1].alarm_code": 0,
            "inverters[1].op_state": 1,
            "inverters[1].pv1_current": 9.9,
            "inverters[1].pv1_voltage": 38.5,
            "inverters[1].pv2_current": 9.8,
            "inverters[1].pv2_voltage": 38.3,
            "inverters[1].kwh_today": 44.1,
            "inverters[1].limit_percent": 100,
            "meters[0].pf": 0.99,
            "meters[0].freq": 50.00,
            "sensors[0].ambient_temp": 33.8,
        },
    },
    # ---- Plant 2 ----
    {
        "id": "INV-P2-L1-0",
        "risk_score": 0.72,
        "risk_class": RiskClass.DEGRADATION_RISK,
        "scenario": "alarm_triggered",
        "shap_values": {
            "inverters[0].alarm_code": +0.28,
            "inverters[0].op_state": +0.18,
            "inverters[0].power": +0.12,
            "inverters[0].pv1_current": +0.06,
            "meters[0].meter_active_power": +0.05,
            "inverters[0].kwh_today": +0.04,
        },
        "raw_features": {
            "inverters[0].power": 5500.0,
            "inverters[0].temp": 51.2,
            "inverters[0].alarm_code": 3021,
            "inverters[0].op_state": 3,
            "inverters[0].pv1_current": 6.8,
            "inverters[0].pv1_voltage": 35.1,
            "inverters[0].pv2_current": 6.5,
            "inverters[0].pv2_voltage": 34.8,
            "inverters[0].kwh_today": 22.1,
            "inverters[0].limit_percent": 60,
            "meters[0].meter_active_power": 5480.0,
            "meters[0].pf": 0.93,
            "meters[0].freq": 50.01,
            "sensors[0].ambient_temp": 36.5,
        },
    },
    {
        "id": "INV-P2-L1-1",
        "risk_score": 0.15,
        "risk_class": RiskClass.NO_RISK,
        "scenario": "normal_operation",
        "shap_values": {
            "inverters[1].power": -0.05,
            "inverters[1].temp": -0.03,
            "inverters[1].pv1_current": -0.02,
            "meters[0].pf": -0.02,
            "sensors[0].ambient_temp": -0.01,
        },
        "raw_features": {
            "inverters[1].power": 8900.0,
            "inverters[1].temp": 41.8,
            "inverters[1].alarm_code": 0,
            "inverters[1].op_state": 1,
            "inverters[1].pv1_current": 9.6,
            "inverters[1].pv1_voltage": 38.0,
            "inverters[1].pv2_current": 9.5,
            "inverters[1].pv2_voltage": 37.8,
            "inverters[1].kwh_today": 41.2,
            "inverters[1].limit_percent": 100,
            "meters[0].pf": 0.97,
            "meters[0].freq": 50.00,
            "sensors[0].ambient_temp": 35.1,
        },
    },
    {
        "id": "INV-P2-L2-0",
        "risk_score": 0.58,
        "risk_class": RiskClass.DEGRADATION_RISK,
        "scenario": "low_power_output",
        "shap_values": {
            "inverters[0].power": +0.22,
            "inverters[0].pv5_current": +0.09,
            "inverters[0].pv6_current": +0.08,
            "inverters[0].kwh_today": +0.07,
            "meters[0].pf": +0.05,
            "inverters[0].pv5_voltage": +0.04,
        },
        "raw_features": {
            "inverters[0].power": 5100.0,
            "inverters[0].temp": 43.7,
            "inverters[0].alarm_code": 0,
            "inverters[0].op_state": 1,
            "inverters[0].pv1_current": 9.3,
            "inverters[0].pv1_voltage": 37.5,
            "inverters[0].pv5_current": 4.2,
            "inverters[0].pv5_voltage": 30.1,
            "inverters[0].pv6_current": 4.8,
            "inverters[0].pv6_voltage": 31.5,
            "inverters[0].kwh_today": 24.6,
            "inverters[0].limit_percent": 100,
            "meters[0].pf": 0.91,
            "meters[0].freq": 49.98,
            "sensors[0].ambient_temp": 36.0,
        },
    },
    {
        "id": "INV-P2-L2-1",
        "risk_score": 0.91,
        "risk_class": RiskClass.SHUTDOWN_RISK,
        "scenario": "grid_fault",
        "shap_values": {
            "meters[0].freq": +0.30,
            "meters[0].v_r": +0.20,
            "inverters[1].alarm_code": +0.15,
            "inverters[1].op_state": +0.12,
            "meters[0].pf": +0.08,
            "meters[0].v_y": +0.05,
        },
        "raw_features": {
            "inverters[1].power": 0.0,
            "inverters[1].temp": 38.2,
            "inverters[1].alarm_code": 5001,
            "inverters[1].op_state": 0,
            "inverters[1].pv1_current": 0.0,
            "inverters[1].pv1_voltage": 38.1,
            "inverters[1].kwh_today": 12.4,
            "inverters[1].limit_percent": 0,
            "meters[0].meter_active_power": 0.0,
            "meters[0].pf": 0.42,
            "meters[0].freq": 51.8,
            "meters[0].v_r": 268.5,
            "meters[0].v_y": 265.2,
            "meters[0].v_b": 241.0,
            "sensors[0].ambient_temp": 34.5,
        },
    },
    # ---- Plant 3 ----
    {
        "id": "INV-P3-L1-0",
        "risk_score": 0.05,
        "risk_class": RiskClass.NO_RISK,
        "scenario": "normal_operation",
        "shap_values": {
            "inverters[0].power": -0.03,
            "inverters[0].temp": -0.02,
            "sensors[0].ambient_temp": -0.01,
            "meters[0].pf": -0.01,
        },
        "raw_features": {
            "inverters[0].power": 9200.0,
            "inverters[0].temp": 39.8,
            "inverters[0].alarm_code": 0,
            "inverters[0].op_state": 1,
            "inverters[0].pv1_current": 9.9,
            "inverters[0].pv1_voltage": 38.6,
            "inverters[0].kwh_today": 45.2,
            "inverters[0].limit_percent": 100,
            "meters[0].pf": 0.99,
            "meters[0].freq": 50.00,
            "sensors[0].ambient_temp": 32.1,
        },
    },
    {
        "id": "INV-P3-L1-1",
        "risk_score": 0.11,
        "risk_class": RiskClass.NO_RISK,
        "scenario": "normal_operation",
        "shap_values": {
            "inverters[1].power": -0.04,
            "inverters[1].temp": -0.02,
            "inverters[1].pv1_current": -0.02,
            "sensors[0].ambient_temp": -0.01,
            "meters[0].pf": -0.01,
        },
        "raw_features": {
            "inverters[1].power": 8800.0,
            "inverters[1].temp": 41.0,
            "inverters[1].alarm_code": 0,
            "inverters[1].op_state": 1,
            "inverters[1].pv1_current": 9.7,
            "inverters[1].pv1_voltage": 38.1,
            "inverters[1].kwh_today": 43.8,
            "inverters[1].limit_percent": 100,
            "meters[0].pf": 0.98,
            "meters[0].freq": 50.01,
            "sensors[0].ambient_temp": 32.5,
        },
    },
    {
        "id": "INV-P3-L2-0",
        "risk_score": 0.45,
        "risk_class": RiskClass.DEGRADATION_RISK,
        "scenario": "partial_shading",
        "shap_values": {
            "smu[0].string7": +0.12,
            "smu[0].string8": +0.10,
            "inverters[0].pv7_current": +0.08,
            "inverters[0].pv8_current": +0.06,
            "inverters[0].power": +0.05,
            "inverters[0].pv7_voltage": +0.03,
        },
        "raw_features": {
            "inverters[0].power": 7100.0,
            "inverters[0].temp": 42.0,
            "inverters[0].alarm_code": 0,
            "inverters[0].op_state": 1,
            "inverters[0].pv1_current": 9.4,
            "inverters[0].pv1_voltage": 37.8,
            "inverters[0].pv7_current": 4.5,
            "inverters[0].pv7_voltage": 31.2,
            "inverters[0].pv8_current": 5.0,
            "inverters[0].pv8_voltage": 32.0,
            "inverters[0].kwh_today": 33.1,
            "inverters[0].limit_percent": 100,
            "meters[0].pf": 0.95,
            "meters[0].freq": 50.00,
            "sensors[0].ambient_temp": 33.5,
            "smu[0].string7": 4.5,
            "smu[0].string8": 5.0,
        },
    },
    {
        "id": "INV-P3-L2-1",
        "risk_score": 0.52,
        "risk_class": RiskClass.DEGRADATION_RISK,
        "scenario": "communication_issue",
        "shap_values": {
            "inverters[1].op_state": +0.18,
            "inverters[1].alarm_code": +0.12,
            "inverters[1].power": +0.10,
            "smu[1].string1": +0.05,
            "smu[1].string2": +0.04,
            "inverters[1].kwh_today": +0.03,
        },
        "raw_features": {
            "inverters[1].power": 6800.0,
            "inverters[1].temp": 43.5,
            "inverters[1].alarm_code": 2010,
            "inverters[1].op_state": 4,
            "inverters[1].pv1_current": 8.8,
            "inverters[1].pv1_voltage": 37.0,
            "inverters[1].pv2_current": 8.6,
            "inverters[1].pv2_voltage": 36.8,
            "inverters[1].kwh_today": 30.5,
            "inverters[1].limit_percent": 75,
            "meters[0].pf": 0.94,
            "meters[0].freq": 50.01,
            "sensors[0].ambient_temp": 34.0,
            "smu[1].string1": 8.8,
            "smu[1].string2": 8.6,
        },
    },
]


def _build_predictions():
    for s in _SCENARIOS:
        meta = _inv_meta(s["id"])
        if not meta:
            continue
        SYNTHETIC_PREDICTIONS[s["id"]] = InverterPrediction(
            inverter_id=s["id"],
            plant_id=meta["plant_id"],
            block=meta["block"],
            logger_mac=meta["mac"],
            inverter_index=meta["index"],
            timestamp=_NOW - timedelta(minutes=5),
            risk_score=s["risk_score"],
            risk_class=s["risk_class"],
            shap_values=s["shap_values"],
            raw_features=s["raw_features"],
        )


_build_predictions()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_prediction(inverter_id: str) -> Optional[InverterPrediction]:
    return SYNTHETIC_PREDICTIONS.get(inverter_id)


def get_plant_predictions(plant_id: str) -> List[InverterPrediction]:
    return [p for p in SYNTHETIC_PREDICTIONS.values() if p.plant_id == plant_id]


def get_all_predictions() -> List[InverterPrediction]:
    return list(SYNTHETIC_PREDICTIONS.values())


def get_all_inverter_ids() -> List[str]:
    return list(SYNTHETIC_PREDICTIONS.keys())


def update_prediction(inverter_id: str, ml_result: dict) -> Optional[InverterPrediction]:
    """
    Update or create a prediction from ML inference result.
    ml_result should match the response schema from the ML inference server.
    """
    meta = _inv_meta(inverter_id)
    if not meta:
        return None

    # Map ML category (A-E) + predicted_class to RiskClass
    predicted_class = ml_result.get("predicted_class", "no_risk")
    risk_class_map = {
        "no_risk": RiskClass.NO_RISK,
        "degradation_risk": RiskClass.DEGRADATION_RISK,
        "shutdown_risk": RiskClass.SHUTDOWN_RISK,
    }
    risk_class = risk_class_map.get(predicted_class, RiskClass.NO_RISK)

    # Build SHAP values dict from ML response
    shap_data = ml_result.get("shap") or {}
    shap_values = {}
    if isinstance(shap_data, dict) and "top_features" in shap_data:
        for feat in shap_data["top_features"]:
            shap_values[feat["feature"]] = feat.get("shap_value", feat.get("value", 0.0))
    elif isinstance(shap_data, dict) and "all_values" in shap_data:
        shap_values = shap_data["all_values"]

    # Build raw_features from readings
    readings = ml_result.get("readings", {})
    raw_features = {}
    feature_map = {
        "dc_voltage": f"inverters[{meta['index']}].pv1_voltage",
        "dc_current": f"inverters[{meta['index']}].pv1_current",
        "ac_power": f"inverters[{meta['index']}].power",
        "module_temp": f"inverters[{meta['index']}].temp",
        "ambient_temp": "sensors[0].ambient_temp",
        "irradiation": "meters[0].meter_active_power",
    }
    for api_key, feature_key in feature_map.items():
        if api_key in readings:
            raw_features[feature_key] = readings[api_key]

    pred = InverterPrediction(
        inverter_id=inverter_id,
        plant_id=meta["plant_id"],
        block=meta["block"],
        logger_mac=meta["mac"],
        inverter_index=meta["index"],
        timestamp=datetime.utcnow(),
        risk_score=ml_result.get("confidence", 0.5),
        risk_class=risk_class,
        shap_values=shap_values if shap_values else {"model_output": 0.0},
        raw_features=raw_features if raw_features else {"power": 0.0},
    )
    SYNTHETIC_PREDICTIONS[inverter_id] = pred
    return pred


def update_predictions_batch(ml_results: list[dict]) -> list[InverterPrediction]:
    """Update multiple predictions from ML inference batch results."""
    updated = []
    for result in ml_results:
        inv_id = result.get("inverter_id")
        if inv_id:
            pred = update_prediction(inv_id, result)
            if pred:
                updated.append(pred)
    return updated


def get_plant_overview() -> str:
    """Return a concise text summary of all plants for chat context."""
    lines = []
    for pid, pdata in PLANTS.items():
        lines.append(f"\n{pdata['name']} ({pid}):")
        for lid, ldata in pdata["loggers"].items():
            lines.append(f"  {ldata['block']} (Logger: {ldata['mac']}):")
            for inv_id in ldata["inverters"]:
                pred = SYNTHETIC_PREDICTIONS.get(inv_id)
                if pred:
                    lines.append(
                        f"    {inv_id}: risk={pred.risk_score:.2f} "
                        f"class={pred.risk_class.value} "
                        f"power={pred.raw_features.get('inverters[0].power', pred.raw_features.get('inverters[1].power', 'N/A'))}W"
                    )
    return "\n".join(lines)

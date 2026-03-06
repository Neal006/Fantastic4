"""Per-plant column registry & op_state→class mapping tables."""


def get_inverter_count(plant_name):
    return {"Plant1_LT1": 12, "Plant1_LT2": 11, "Plant2_AC12": 2,
            "Plant2_ACBB": 5, "Plant3_1469": 1, "Plant3_146E": 1}.get(plant_name, 1)


def get_inverter_columns(plant_name, idx):
    """Standardised column dict for one inverter."""
    p = f"inverters[{idx}]"
    cols = {"power": f"{p}.power", "op_state": f"{p}.op_state",
            "kwh_today": f"{p}.kwh_today", "kwh_total": f"{p}.kwh_total"}
    from .settings import PLANT_TYPE
    pt = PLANT_TYPE.get(plant_name)
    if pt == "celestical":
        cols.update({"v_ab": f"{p}.v_ab", "v_bc": f"{p}.v_bc", "v_ca": f"{p}.v_ca",
                     "freq": f"{p}.freq", "temp": f"{p}.temp", "pv1_power": f"{p}.pv1_power"})
        cols["alarm_code"] = None
    elif pt == "sungrow":
        cols["alarm_code"] = f"{p}.alarm_code"
        for pv in range(1, 10):
            cols[f"pv{pv}_current"] = f"{p}.pv{pv}_current"
            cols[f"pv{pv}_voltage"] = f"{p}.pv{pv}_voltage"
        cols.update({"pv1_power": f"{p}.pv1_power", "pv2_power": f"{p}.pv2_power",
                     "temp": f"{p}.temp"})
    elif pt == "plant3":
        cols.update({"alarm_code": f"{p}.alarm_code", "pv1_current": f"{p}.pv1_current",
                     "pv1_voltage": f"{p}.pv1_voltage", "pv1_power": f"{p}.pv1_power",
                     "temp": f"{p}.temp"})
    return cols


def get_meter_columns():
    return {k: f"meters[0].{k}" for k in
            ["meter_active_power", "meter_kwh_import", "meter_kwh_total", "pf", "freq",
             "v_r", "v_y", "v_b", "p_r", "p_y", "p_b"]}


def get_smu_columns(df_columns):
    import re
    return [c for c in df_columns if re.match(r"^smu\[\d+\]\.string\d+$", c)]


def get_sensor_columns(df_columns):
    names = ["sensors[0].irradiation", "sensors[0].irradiation_kwh",
             "sensors[0].irradiation_temp", "sensors[0].wind_speed"]
    return [c for c in names if c in df_columns]


# op-state mapping: value → class or "check_power"/"check_alarm"
CELESTICAL_OP_MAP = {-1: "check_power", 0: 1}
SUNGROW_OP_MAP = {0: "check_power", 5120: 1, 33280: 3, 5632: "check_alarm",
                  4608: 4, 4864: 4, 37120: "check_alarm", 21760: 5, 33024: 3}
PLANT3_OP_MAP = {4: "check_power", 0: 1, 3: 1, 5: 3, 8: 4, 7: 5}

GRID_ALARM_CODES = {548, 558, 563, 581, 557, 534, 556}
SEVERE_ALARM_CODES = {39, 8, 10, 12, 463, 464}

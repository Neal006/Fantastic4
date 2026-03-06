"""KPI features: specific yield, string imbalance, DC/AC ratio, power deviation."""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ROLLING_WINDOWS, PLANT_TYPE
from config.column_registry import (get_inverter_count, get_inverter_columns,
                                     get_smu_columns, get_sensor_columns)


def add_kpi_features(df, plant_name):
    n_inv = get_inverter_count(plant_name)
    feats = {}

    for i in range(n_inv):
        inv = get_inverter_columns(plant_name, i)
        pfx = f"inv{i}"
        pc = inv["power"]
        if pc not in df.columns: continue
        power = df[pc]

        # specific yield
        kc = inv.get("kwh_today")
        if kc and kc in df.columns:
            cap = max(power.quantile(0.99), 1)
            feats[f"{pfx}_specific_yield"] = (df[kc] / cap).astype(np.float32)

        # power deviation from 7d median
        w7 = ROLLING_WINDOWS["7d"]
        med = power.rolling(w7, min_periods=288).median()
        feats[f"{pfx}_power_deviation"] = (power - med).fillna(0).astype(np.float32)

        # energy plateau (stuck kwh_total)
        ktc = inv.get("kwh_total")
        if ktc and ktc in df.columns:
            kd = df[ktc].diff().fillna(0)
            plateau = (kd == 0).astype(int)
            groups = (plateau != plateau.shift()).cumsum()
            feats[f"{pfx}_energy_plateau"] = plateau.groupby(groups).cumsum().astype(np.float32)

        # DC/AC ratio
        pv1 = inv.get("pv1_power")
        if pv1 and pv1 in df.columns:
            dc = df[pv1].copy()
            pv2 = inv.get("pv2_power")
            if pv2 and pv2 in df.columns: dc += df[pv2]
            ac = power.replace(0, np.nan)
            feats[f"{pfx}_dc_ac_ratio"] = (dc / ac).fillna(0).replace([np.inf, -np.inf], 0).clip(0, 10).astype(np.float32)

    # string imbalance (CV of SMU strings)
    smu = get_smu_columns(df.columns)
    if len(smu) >= 2:
        sd = df[smu].astype(np.float32)
        mu = sd.mean(axis=1).replace(0, np.nan)
        feats["string_imbalance_idx"] = (sd.std(axis=1) / mu).fillna(0).replace([np.inf, -np.inf], 0).clip(0, 10).astype(np.float32)
        feats["string_max_deviation"] = sd.sub(sd.mean(axis=1), axis=0).abs().max(axis=1).astype(np.float32)

    # performance ratio (Plant2_ACBB has irradiation)
    sens = get_sensor_columns(df.columns)
    if "sensors[0].irradiation" in sens:
        irr = df["sensors[0].irradiation"].replace(0, np.nan)
        tp = sum(df[get_inverter_columns(plant_name, j)["power"]]
                 for j in range(n_inv) if get_inverter_columns(plant_name, j)["power"] in df.columns)
        cap = max(tp.quantile(0.99), 1)
        feats["performance_ratio"] = (tp / (irr * cap)).fillna(0).replace([np.inf, -np.inf], 0).clip(0, 2).astype(np.float32)

    if feats:
        return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
    return df

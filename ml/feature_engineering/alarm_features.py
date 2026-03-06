"""Alarm-derived features: counts, diversity, time since alarm, fault history."""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ROLLING_WINDOWS, PLANT_TYPE
from config.column_registry import get_inverter_count, get_inverter_columns


def add_alarm_features(df, plant_name):
    ptype = PLANT_TYPE[plant_name]
    w24 = ROLLING_WINDOWS["24h"]
    w7d = ROLLING_WINDOWS["7d"]

    if ptype == "celestical":
        # Plant1 has no alarm columns
        for c in ["alarm_count_24h", "alarm_count_7d", "alarm_diversity_24h",
                   "alarm_active", "fault_state_count_7d"]:
            df[c] = np.float32(0)
        df["time_since_alarm"] = np.float32(999999)
        return df

    n_inv = get_inverter_count(plant_name)
    alarm_active_list = []
    for i in range(n_inv):
        inv = get_inverter_columns(plant_name, i)
        ac = inv.get("alarm_code")
        if ac and ac in df.columns:
            alarm_active_list.append((df[ac] != 0).astype(np.float32))

    if not alarm_active_list:
        for c in ["alarm_count_24h", "alarm_count_7d", "alarm_diversity_24h",
                   "alarm_active", "fault_state_count_7d"]:
            df[c] = np.float32(0)
        df["time_since_alarm"] = np.float32(999999)
        return df

    combined = pd.concat(alarm_active_list, axis=1).max(axis=1)
    df["alarm_active"] = combined
    df["alarm_count_24h"] = combined.rolling(w24, min_periods=1).sum().astype(np.float32)
    df["alarm_count_7d"] = combined.rolling(w7d, min_periods=1).sum().astype(np.float32)

    # alarm diversity in 24h
    first_ac = None
    for i in range(n_inv):
        inv = get_inverter_columns(plant_name, i)
        c = inv.get("alarm_code")
        if c and c in df.columns:
            first_ac = c; break
    if first_ac:
        df["alarm_diversity_24h"] = df[first_ac].rolling(w24, min_periods=1).apply(
            lambda w: len(set(w[w != 0])) if any(w != 0) else 0, raw=True
        ).fillna(0).astype(np.float32)

    # time since last alarm
    vals = combined.values
    ts = np.full(len(df), 999999, dtype=np.float32)
    last = -999999
    for j in range(len(df)):
        if vals[j] > 0: last = j
        ts[j] = min(j - last, 999999)
    df["time_since_alarm"] = ts

    # fault state count 7d
    if "label_instant" in df.columns:
        df["fault_state_count_7d"] = (df["label_instant"] >= 2).astype(np.float32).rolling(w7d, min_periods=1).sum().astype(np.float32)
    else:
        df["fault_state_count_7d"] = np.float32(0)

    return df

"""Rolling telemetry features: power, voltage, current, frequency, temperature."""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ROLLING_WINDOWS
from config.column_registry import get_inverter_count, get_inverter_columns, get_meter_columns


def _rolling_stats(s, w, pfx):
    r = s.rolling(w, min_periods=1)
    return {f"{pfx}_mean_{w}": r.mean(), f"{pfx}_std_{w}": r.std().fillna(0),
            f"{pfx}_min_{w}": r.min(), f"{pfx}_max_{w}": r.max(),
            f"{pfx}_slope_{w}": (s - s.shift(w).fillna(s)) / max(w, 1)}


def add_telemetry_features(df, plant_name):
    n_inv = get_inverter_count(plant_name)
    feats = {}
    windows = [ROLLING_WINDOWS["1h"], ROLLING_WINDOWS["6h"]]

    for i in range(n_inv):
        inv = get_inverter_columns(plant_name, i)
        pfx = f"inv{i}"

        # power (most important)
        pc = inv["power"]
        if pc in df.columns:
            s = pd.to_numeric(df[pc], errors='coerce').fillna(0).astype(np.float32)
            for w in windows:
                feats.update(_rolling_stats(s, w, f"{pfx}_power"))
            feats[f"{pfx}_power_diff"] = s.diff().fillna(0)
            feats[f"{pfx}_power_pct"] = s.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            # 24h trend
            w24 = ROLLING_WINDOWS["24h"]
            r24 = s.rolling(w24, min_periods=1)
            feats[f"{pfx}_power_mean_{w24}"] = r24.mean()
            feats[f"{pfx}_power_std_{w24}"] = r24.std().fillna(0)

        # temperature
        tc = inv.get("temp")
        if tc and tc in df.columns:
            s = pd.to_numeric(df[tc], errors='coerce').fillna(0).astype(np.float32)
            for w in windows:
                r = s.rolling(w, min_periods=1)
                feats[f"{pfx}_temp_mean_{w}"] = r.mean()
                feats[f"{pfx}_temp_max_{w}"] = r.max()
            feats[f"{pfx}_temp_diff"] = s.diff().fillna(0)

        # frequency
        fc = inv.get("freq")
        if fc and fc in df.columns:
            s = pd.to_numeric(df[fc], errors='coerce').fillna(0).astype(np.float32)
            w1h = ROLLING_WINDOWS["1h"]
            r = s.rolling(w1h, min_periods=1)
            feats[f"{pfx}_freq_mean_{w1h}"] = r.mean()
            feats[f"{pfx}_freq_std_{w1h}"] = r.std().fillna(0)

    # meter-level
    mc = get_meter_columns()
    for key in ["meter_active_power", "pf", "freq"]:
        col = mc.get(key)
        if col and col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
            w1h = ROLLING_WINDOWS["1h"]
            r = s.rolling(w1h, min_periods=1)
            feats[f"meter_{key}_mean_{w1h}"] = r.mean()
            feats[f"meter_{key}_std_{w1h}"] = r.std().fillna(0)

    fd = pd.DataFrame(feats, index=df.index)
    for c in fd.columns:
        fd[c] = fd[c].astype(np.float32)
    return pd.concat([df, fd], axis=1)

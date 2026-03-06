"""Statistical anomaly: Z-score on power residuals vs time-of-day expectation."""
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.column_registry import get_inverter_count, get_inverter_columns


def add_statistical_anomaly_features(df, plant_name):
    n_inv = get_inverter_count(plant_name)
    ts = pd.to_datetime(df["timestamp"], format='mixed', errors='coerce').ffill() if "timestamp" in df.columns else df.index.to_series()
    hour = ts.dt.hour
    for i in range(n_inv):
        inv = get_inverter_columns(plant_name, i)
        pc = inv["power"]
        if pc not in df.columns: continue
        power = df[pc]
        # Rolling 30 days (12 hourly 5-min intervals * 30 = 360) rather than global to prevent future leakage
        med = power.groupby(hour).transform(lambda x: x.rolling(360, min_periods=1).median())
        std = power.groupby(hour).transform(lambda x: x.rolling(360, min_periods=1).std()).fillna(1).replace(0, 1)
        z = ((power - med) / std).fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)
        df[f"inv{i}_power_zscore"] = z.astype(np.float32)
        df[f"inv{i}_power_zscore_abs"] = np.abs(z).astype(np.float32)
    return df

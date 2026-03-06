"""Time features: hour, DOW, month, daytime flag, cyclical encoding."""
import numpy as np
import pandas as pd


def add_time_features(df):
    ts = pd.to_datetime(df["timestamp"], format='mixed', errors='coerce').ffill() if "timestamp" in df.columns else df.index.to_series()
    df["hour"] = ts.dt.hour.astype(np.float32)
    df["day_of_week"] = ts.dt.dayofweek.astype(np.float32)
    df["month"] = ts.dt.month.astype(np.float32)
    df["is_daytime"] = ((ts.dt.hour >= 6) & (ts.dt.hour < 18)).astype(np.float32)
    df["minutes_since_midnight"] = (ts.dt.hour * 60 + ts.dt.minute).astype(np.float32)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    return df

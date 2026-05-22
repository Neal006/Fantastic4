"""
Feature engineering and target variable construction.

Call build_features(df) on a raw hourly per-inverter DataFrame to get a
fully featured DataFrame ready for training.

If you already have a featured parquet with target columns, skip this module.
"""

import numpy as np
import pandas as pd

ROLLING_COLS = [
    "inv_power", "inv_temp", "ambient_temp", "meter_freq",
    "meter_pf", "meter_v_r", "meter_v_y", "meter_v_b", "smu_string_mean",
]
WINDOWS = [6, 24, 168]
HORIZON = 168   # 7-day forward label window


# ── Public entry point ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering and target construction."""
    df = df.sort_values("datetime").copy()
    df = _time_features(df)
    df = _rolling_features(df)
    df = _alarm_features(df)
    df = _derived_kpis(df)
    df = _build_targets(df)
    return df


# ── Private helpers ────────────────────────────────────────────────

def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_daytime"]  = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
    return df


def _rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ROLLING_COLS:
        if col not in df.columns:
            continue
        for w in WINDOWS:
            min_p = max(1, w // 4)
            df[f"{col}_roll{w}h_mean"] = df[col].rolling(w, min_periods=min_p).mean()
            df[f"{col}_roll{w}h_std"]  = df[col].rolling(w, min_periods=min_p).std()
    return df


def _alarm_features(df: pd.DataFrame) -> pd.DataFrame:
    alarm = df["inv_alarm_code"].fillna(0)
    df["alarm_active"]         = (alarm > 0).astype(int)
    df["alarm_count_24h"]      = df["alarm_active"].rolling(24,  min_periods=1).sum()
    df["alarm_count_7d"]       = df["alarm_active"].rolling(168, min_periods=1).sum()
    df["hours_since_last_alarm"] = _hours_since_last_alarm(alarm)
    return df


def _hours_since_last_alarm(alarm_series: pd.Series) -> list:
    result, last = [], 9999
    for val in alarm_series:
        if val > 0:
            last = 0
        else:
            last = min(last + 1, 9999)
        result.append(last)
    return result


def _derived_kpis(df: pd.DataFrame) -> pd.DataFrame:
    v_cols = ["meter_v_r", "meter_v_y", "meter_v_b"]
    if all(c in df.columns for c in v_cols):
        voltages = df[v_cols]
        mean_v   = voltages.mean(axis=1)
        max_dev  = voltages.sub(mean_v, axis=0).abs().max(axis=1)
        df["voltage_imbalance"] = (max_dev / mean_v.replace(0, float("nan"))).fillna(0)

    if "meter_pf" in df.columns:
        df["pf_deviation"] = (1 - df["meter_pf"]).abs()

    if "meter_freq" in df.columns:
        df["freq_deviation"] = (df["meter_freq"] - 50).abs()

    if "inv_power" in df.columns:
        rolling_mean_24h = df["inv_power"].rolling(24, min_periods=6).mean()
        df["power_ratio_vs_24h"] = (
            df["inv_power"] / rolling_mean_24h.replace(0, float("nan"))
        ).fillna(0)

    if "smu_num_zero" in df.columns and "smu_total_strings" in df.columns:
        df["smu_zero_fraction"] = (
            df["smu_num_zero"] / df["smu_total_strings"].replace(0, float("nan"))
        ).fillna(0)

    return df


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["shutdown_event"] = (
        (df["inv_alarm_code"] > 0) |
        ((df["inv_power"] == 0) & (df["is_daytime"] == 1))
    ).astype(int)

    avg_24h = df["inv_power"].rolling(24, min_periods=6).mean()
    df["degradation_event"] = (
        (df["inv_power"] > 0) &
        (df["inv_power"] < 0.5 * avg_24h) &
        (df["is_daytime"] == 1) &
        (avg_24h > 0.1)
    ).astype(int)

    future_shutdown    = df["shutdown_event"].iloc[::-1].rolling(HORIZON, min_periods=1).sum().iloc[::-1]
    future_degradation = df["degradation_event"].iloc[::-1].rolling(HORIZON, min_periods=1).sum().iloc[::-1]

    df["target_binary"]     = ((future_shutdown > 0) | (future_degradation > 0)).astype(int)
    df["target_multiclass"] = 0
    df.loc[future_degradation > 0, "target_multiclass"] = 1
    df.loc[future_shutdown    > 0, "target_multiclass"] = 2

    return df.dropna(subset=["target_binary", "target_multiclass"])

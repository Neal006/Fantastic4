"""
Stage 3 – Feature Engineering
==============================
Build predictive features from raw telemetry:
- Rolling statistics (mean, std, min, max) on key signals.
- Computed KPIs: performance ratio, power variability, string imbalance,
  voltage deviation, daily energy delta.
- Alarm / op-state derived features.
- Cyclical time encodings (hour, day-of-week) + is_daytime flag.
- Lag features (1-day, 7-day).

Input  → ``processed/cleaned.parquet``
Output → ``processed/featured.parquet``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_DIR, TIMESTAMP_COL, INVERTER_ID_COL, PLANT_ID_COL, MAC_COL,
    ROLLING_WINDOWS, SAMPLES_PER_DAY, PV_STRING_CURRENT_COLS, SEED,
)
from utils import log_section, log_step, save_parquet, load_parquet, Timer


# ── Rolling statistics ───────────────────────────────────────────
ROLLING_TARGETS = ["power", "temp", "pv1_current", "pf", "v_r"]


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean / std / min / max for key signals."""
    grp_cols = [c for c in [PLANT_ID_COL, MAC_COL, INVERTER_ID_COL] if c in df.columns]
    for label, win in ROLLING_WINDOWS.items():
        for col in ROLLING_TARGETS:
            if col not in df.columns:
                continue
            grp = df.groupby(grp_cols)[col]
            df[f"{col}_rmean_{label}"] = grp.transform(
                lambda s: s.rolling(win, min_periods=1).mean()
            )
            df[f"{col}_rstd_{label}"] = grp.transform(
                lambda s: s.rolling(win, min_periods=1).std().fillna(0)
            )
            if label == "24h":
                df[f"{col}_rmin_{label}"] = grp.transform(
                    lambda s: s.rolling(win, min_periods=1).min()
                )
                df[f"{col}_rmax_{label}"] = grp.transform(
                    lambda s: s.rolling(win, min_periods=1).max()
                )
    log_step("Rolling features added")
    return df


# ── Computed KPIs ────────────────────────────────────────────────
def _add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Derive operational KPIs."""
    # Power variability: rolling_std / rolling_mean (24 h)
    mean24 = df.get("power_rmean_24h", pd.Series(0, index=df.index))
    std24 = df.get("power_rstd_24h", pd.Series(0, index=df.index))
    df["power_variability"] = np.where(mean24 > 0, std24 / mean24, 0)

    # String current imbalance (max – min across available PV strings)
    pv_cols = [c for c in PV_STRING_CURRENT_COLS if c in df.columns]
    if pv_cols:
        pv_mat = df[pv_cols].values
        df["string_imbalance"] = np.nanmax(pv_mat, axis=1) - np.nanmin(pv_mat, axis=1)
    else:
        df["string_imbalance"] = 0

    # Voltage deviation: inverter pv1_voltage vs grid v_r (if both exist)
    if "pv1_voltage" in df.columns and "v_r" in df.columns:
        df["voltage_deviation"] = (df["pv1_voltage"] - df["v_r"]).abs()
    else:
        df["voltage_deviation"] = 0

    # Daily energy delta ── diff of kwh_total within inverter per day
    if "kwh_total" in df.columns:
        grp_cols = [c for c in [PLANT_ID_COL, MAC_COL, INVERTER_ID_COL] if c in df.columns]
        df["kwh_daily_delta"] = df.groupby(grp_cols)["kwh_total"].diff().fillna(0).clip(lower=0)
    else:
        df["kwh_daily_delta"] = 0

    # Total PV power (sum of string powers)
    pwr_cols = [c for c in ["pv1_power", "pv2_power"] if c in df.columns]
    if pwr_cols:
        df["total_pv_power"] = df[pwr_cols].sum(axis=1)
    else:
        df["total_pv_power"] = df.get("power", 0)

    log_step("KPIs added")
    return df


# ── Alarm & op-state features ───────────────────────────────────
def _add_alarm_features(df: pd.DataFrame) -> pd.DataFrame:
    if "alarm_code" in df.columns:
        df["alarm_active"] = (df["alarm_code"] != 0).astype(np.int8)
        # Rolling count of consecutive alarm rows (per inverter)
        grp_cols = [c for c in [PLANT_ID_COL, MAC_COL, INVERTER_ID_COL] if c in df.columns]
        df["alarm_streak"] = df.groupby(grp_cols)["alarm_active"].transform(
            lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
        ) * df["alarm_active"]
    else:
        df["alarm_active"] = 0
        df["alarm_streak"] = 0

    if "op_state" in df.columns:
        df["is_running"] = (df["op_state"] == 5120).astype(np.int8)
        df["is_off"] = (df["op_state"] == 0).astype(np.int8)
    else:
        df["is_running"] = 1
        df["is_off"] = 0

    log_step("Alarm & op-state features added")
    return df


# ── Time encodings ───────────────────────────────────────────────
def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df[TIMESTAMP_COL]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_daytime"] = ((ts.dt.hour >= 6) & (ts.dt.hour < 18)).astype(np.int8)
    log_step("Time encodings added")
    return df


# ── Lag features ─────────────────────────────────────────────────
LAG_TARGETS = ["power", "temp", "string_imbalance", "power_variability"]
LAG_PERIODS = {"1d": SAMPLES_PER_DAY, "7d": SAMPLES_PER_DAY * 7}


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = [c for c in [PLANT_ID_COL, MAC_COL, INVERTER_ID_COL] if c in df.columns]
    for label, lag in LAG_PERIODS.items():
        for col in LAG_TARGETS:
            if col not in df.columns:
                continue
            df[f"{col}_lag_{label}"] = df.groupby(grp_cols)[col].shift(lag)
    # Fill NaN from lags with 0
    lag_cols = [c for c in df.columns if "_lag_" in c]
    df[lag_cols] = df[lag_cols].fillna(0)
    log_step("Lag features added")
    return df


# ── main ─────────────────────────────────────────────────────────
def run():
    log_section("Stage 3 · Feature Engineering")

    df = load_parquet(PROCESSED_DIR / "cleaned.parquet")

    # Ensure sorted by group + time
    grp_cols = [c for c in [PLANT_ID_COL, MAC_COL, INVERTER_ID_COL] if c in df.columns]
    df.sort_values(grp_cols + [TIMESTAMP_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)

    with Timer():
        df = _add_rolling_features(df)
        df = _add_kpis(df)
        df = _add_alarm_features(df)
        df = _add_time_features(df)
        df = _add_lag_features(df)

    log_step(f"Final shape: {df.shape}")
    save_parquet(df, PROCESSED_DIR / "featured.parquet", "featured data")
    return df


if __name__ == "__main__":
    run()

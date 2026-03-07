"""
Stage 2 – Data Cleaning
========================
- Parse timestamps (epoch ms → datetime).
- Drop bogus timestamps (before 2024 or after 2026).
- Remove exact-duplicate rows.
- Forward-fill NaN for slow-moving columns within each inverter group.
- Cap extreme outliers using IQR on numeric columns.

Input  → ``processed/ingested.parquet``
Output → ``processed/cleaned.parquet``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_DIR, TIMESTAMP_COL, INVERTER_ID_COL, PLANT_ID_COL, MAC_COL,
)
from utils import log_section, log_step, save_parquet, load_parquet, Timer


# ── helpers ──────────────────────────────────────────────────────
SLOW_FILL_COLS = [
    "kwh_total", "kwh_today", "kwh_midnight",
    "model", "serial", "limit_percent",
    "meter_kwh_import", "meter_kwh_total",
    "base_meter_kwh_import", "base_meter_kwh_total",
    "original_meter_kwh_import", "original_meter_kwh_total",
]

IQR_MULTIPLIER = 3.0  # generous to keep real signal


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert epoch-ms timestamp to datetime and filter sane range."""
    df[TIMESTAMP_COL] = pd.to_numeric(df[TIMESTAMP_COL], errors="coerce")
    df.dropna(subset=[TIMESTAMP_COL], inplace=True)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], unit="ms", utc=True)
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.tz_localize(None)  # naive UTC
    # Filter sane range
    mask = (df[TIMESTAMP_COL] >= "2024-01-01") & (df[TIMESTAMP_COL] <= "2026-01-01")
    before = len(df)
    df = df.loc[mask].copy()
    log_step(f"Timestamp filter: kept {len(df):,} / {before:,} rows")
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df.drop_duplicates(inplace=True)
    log_step(f"Duplicates removed: {before - len(df):,}")
    return df


def _forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill slow-moving columns within each inverter group."""
    grp_cols = [PLANT_ID_COL, MAC_COL, INVERTER_ID_COL]
    grp_cols = [c for c in grp_cols if c in df.columns]
    fill_cols = [c for c in SLOW_FILL_COLS if c in df.columns]
    if fill_cols and grp_cols:
        df.sort_values(grp_cols + [TIMESTAMP_COL], inplace=True)
        df[fill_cols] = df.groupby(grp_cols)[fill_cols].ffill()
        log_step(f"Forward-filled {len(fill_cols)} slow-moving columns")
    return df


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise extreme values using IQR on numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifier / id columns
    skip = {INVERTER_ID_COL, "id", "__v", "dataLoggerModelId"}
    numeric_cols = [c for c in numeric_cols if c not in skip]
    clipped = 0
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        n = mask.sum()
        if n > 0:
            df[col] = df[col].clip(lower, upper)
            clipped += n
    log_step(f"Outlier capping: {clipped:,} values clipped across {len(numeric_cols)} columns")
    return df


# ── main ─────────────────────────────────────────────────────────
def run():
    log_section("Stage 2 · Data Cleaning")

    with Timer():
        df = load_parquet(PROCESSED_DIR / "ingested.parquet")

    log_step(f"Starting shape: {df.shape}")

    with Timer():
        df = _parse_timestamps(df)
        df = _remove_duplicates(df)
        df = _forward_fill(df)
        df = _cap_outliers(df)

    # Drop non-useful columns
    drop_cols = [c for c in ("_id", "createdAt", "fromServer", "dataLoggerModelId",
                              "__v", "timestampDate", "batteries", "grid_master") if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    log_step(f"Dropped metadata columns: {drop_cols}")

    # Fill remaining NaN in numerics with 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    log_step(f"Final shape: {df.shape}")
    save_parquet(df, PROCESSED_DIR / "cleaned.parquet", "cleaned data")
    return df


if __name__ == "__main__":
    run()

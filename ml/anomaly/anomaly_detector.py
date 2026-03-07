"""
Stage 5 – Anomaly Detection (Complementary Signal Layer)
=========================================================
Two unsupervised methods produce extra features for the prediction models:
  1. Isolation Forest anomaly score.
  2. Rolling z-score flags (power & temp >3σ from 24-h mean).

Input  → ``processed/labeled.parquet``
Output → ``processed/anomaly_enriched.parquet``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_DIR, TIMESTAMP_COL, INVERTER_ID_COL, PLANT_ID_COL, MAC_COL,
    SAMPLES_PER_DAY, SEED,
)
from utils import log_section, log_step, save_parquet, load_parquet, Timer


# ── Isolation Forest ─────────────────────────────────────────────
ISO_FEATURES = [
    "power", "temp", "string_imbalance", "pf", "v_r",
    "power_variability", "alarm_active",
]


def _add_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """Fit Isolation Forest on available ISO_FEATURES; add anomaly score."""
    cols = [c for c in ISO_FEATURES if c in df.columns]
    if not cols:
        log_step("Isolation Forest: no valid columns found — skipping")
        df["iso_forest_score"] = 0.0
        return df

    X = df[cols].fillna(0).values
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=SEED,
        n_jobs=-1,
    )
    log_step(f"Fitting Isolation Forest on {len(cols)} features …")
    iso.fit(X)
    # decision_function: the lower, the more anomalous (negative = anomaly)
    df["iso_forest_score"] = iso.decision_function(X)
    # Binary flag
    df["iso_forest_anomaly"] = (iso.predict(X) == -1).astype(np.int8)
    n_anomalies = df["iso_forest_anomaly"].sum()
    log_step(f"Isolation Forest: {n_anomalies:,} anomalies flagged ({100 * n_anomalies / len(df):.2f}%)")
    return df


# ── Rolling z-score ──────────────────────────────────────────────
ZSCORE_TARGETS = ["power", "temp"]
ZSCORE_THRESHOLD = 3.0


def _add_zscore_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where signal is >3σ from 24-h rolling mean."""
    grp_cols = [c for c in [PLANT_ID_COL, MAC_COL, INVERTER_ID_COL] if c in df.columns]
    window = SAMPLES_PER_DAY  # 288

    any_flag = np.zeros(len(df), dtype=np.int8)
    for col in ZSCORE_TARGETS:
        if col not in df.columns:
            continue
        grp = df.groupby(grp_cols)[col]
        rmean = grp.transform(lambda s: s.rolling(window, min_periods=1).mean())
        rstd = grp.transform(lambda s: s.rolling(window, min_periods=1).std().fillna(1))
        zscore = ((df[col] - rmean) / rstd.replace(0, 1)).abs()
        flag = (zscore > ZSCORE_THRESHOLD).astype(np.int8)
        df[f"{col}_zscore"] = zscore
        df[f"{col}_zscore_flag"] = flag
        any_flag |= flag.values
        log_step(f"  z-score ({col}): {flag.sum():,} flags")

    df["zscore_anomaly"] = any_flag
    return df


# ── main ─────────────────────────────────────────────────────────
def run():
    log_section("Stage 5 · Anomaly Detection")

    df = load_parquet(PROCESSED_DIR / "labeled.parquet")

    with Timer():
        df = _add_isolation_forest(df)
        df = _add_zscore_anomalies(df)

    log_step(f"Final shape: {df.shape}")
    save_parquet(df, PROCESSED_DIR / "anomaly_enriched.parquet", "anomaly-enriched data")
    return df


if __name__ == "__main__":
    run()

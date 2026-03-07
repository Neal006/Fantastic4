"""
Stage 4 – Label Creation (3-class)
====================================
For each row, look **forward** 7 days and assign:
  0 – no_risk        : no abnormal event in the lookahead window.
  1 – degradation_risk: sustained power drop >30 % or long alarm streaks.
  2 – shutdown_risk   : op_state indicates fault/shutdown.

Input  → ``processed/featured.parquet``
Output → ``processed/labeled.parquet``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_DIR, TIMESTAMP_COL, INVERTER_ID_COL, PLANT_ID_COL, MAC_COL,
    LOOKAHEAD_SAMPLES, LABEL_COL, POWER_DROP_THRESHOLD,
    ALARM_DURATION_THRESH, SAMPLES_PER_DAY,
)
from utils import log_section, log_step, save_parquet, load_parquet, Timer


def _create_event_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-row binary indicators for shutdown and degradation events.
    
    Conservative definitions to avoid label saturation:
    
    Shutdown event  = FAULT alarm codes (>= 400) during operations.
                      These are genuine hardware/grid faults vs normal standby.
    Degradation     = non-ideal op_state during daytime (not running=5120, not off=0)
                      which indicates the inverter is in a transitional/error state
                      OR sustained power drops >30% from rolling mean during daytime
    """
    is_daytime = df.get("is_daytime", pd.Series(1, index=df.index)).values.astype(bool)
    alarm_code = df.get("alarm_code", pd.Series(0, index=df.index)).values
    op_state = df.get("op_state", pd.Series(5120, index=df.index)).values

    # ═══════════════ Shutdown event ═══════════════
    # Real fault alarm codes (Sungrow codes >= 400 are actual faults: 
    # 464, 534, 548, 549, 557, 558, 559, 563, 581 in our data)
    FAULT_ALARM_CODES = {464, 534, 548, 549, 557, 558, 559, 563, 581}
    has_fault_alarm = np.isin(alarm_code, list(FAULT_ALARM_CODES))
    shutdown_event = has_fault_alarm.astype(np.int8)

    # ═══════════════ Degradation event ═══════════════
    degradation_event = np.zeros(len(df), dtype=np.int8)

    # 1) Transitional / error op_states during daytime
    #    op_state values like 3, 4, 5, 7, 8, 4608, 4864, 5632, 20480
    #    indicate the inverter is NOT in normal running or off states
    NORMAL_STATES = {0.0, 5120.0}
    abnormal_opstate = (~np.isin(op_state, list(NORMAL_STATES))) & is_daytime
    degradation_event |= abnormal_opstate.astype(np.int8)

    # 2) Minor alarm codes during daytime (codes 8, 10, 39 = warnings)
    WARNING_ALARM_CODES = {8, 10, 39}
    has_warning = np.isin(alarm_code, list(WARNING_ALARM_CODES)) & is_daytime
    degradation_event |= has_warning.astype(np.int8)

    # 3) Power drop from rolling mean during daytime
    if "power" in df.columns and "power_rmean_24h" in df.columns:
        rolling_mean = df["power_rmean_24h"].values
        current_power = df["power"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            drop_ratio = np.where(
                rolling_mean > 100,  # meaningful production
                (rolling_mean - current_power) / rolling_mean,
                0,
            )
        power_drop = (drop_ratio > POWER_DROP_THRESHOLD) & is_daytime
        degradation_event |= power_drop.astype(np.int8)

    # Shutdown takes priority over degradation
    degradation_event = degradation_event & (~shutdown_event.astype(bool))
    degradation_event = degradation_event.astype(np.int8)

    df["_shutdown_event"] = shutdown_event
    df["_degradation_event"] = degradation_event

    log_step(f"Shutdown events: {shutdown_event.sum():,} / {len(df):,} ({100*shutdown_event.sum()/len(df):.2f}%)")
    log_step(f"Degradation events: {degradation_event.sum():,} / {len(df):,} ({100*degradation_event.sum()/len(df):.2f}%)")
    return df


def _assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-rolling max to propagate events into a 7-day lookahead label."""
    grp_cols = [c for c in [PLANT_ID_COL, MAC_COL, INVERTER_ID_COL] if c in df.columns]
    window = LOOKAHEAD_SAMPLES  # 2016 samples ≈ 7 days

    log_step("Computing forward-rolling labels (this may take a moment) …")

    # For each group, compute forward-looking rolling max
    # We reverse, do a backward rolling, then reverse back
    def _forward_rolling_max(series: pd.Series) -> pd.Series:
        return series[::-1].rolling(window, min_periods=1).max()[::-1]

    for col in ("_shutdown_event", "_degradation_event"):
        target = f"{col}_lookahead"
        df[target] = df.groupby(grp_cols)[col].transform(_forward_rolling_max)

    # Assign label: shutdown > degradation > no_risk
    conditions = [
        df["_shutdown_event_lookahead"] >= 1,
        df["_degradation_event_lookahead"] >= 1,
    ]
    choices = [2, 1]
    df[LABEL_COL] = np.select(conditions, choices, default=0).astype(np.int8)

    # Drop helper columns
    df.drop(columns=[
        "_shutdown_event", "_degradation_event",
        "_shutdown_event_lookahead", "_degradation_event_lookahead",
    ], inplace=True)

    return df


# ── main ─────────────────────────────────────────────────────────
def run():
    log_section("Stage 4 · Label Creation")

    df = load_parquet(PROCESSED_DIR / "featured.parquet")

    with Timer():
        df = _create_event_indicators(df)
        df = _assign_labels(df)

    # Report class distribution
    dist = df[LABEL_COL].value_counts().sort_index()
    total = len(df)
    log_step("Class distribution:")
    for cls, cnt in dist.items():
        log_step(f"  {cls}: {cnt:>10,}  ({100 * cnt / total:5.1f}%)")

    log_step(f"Final shape: {df.shape}")
    save_parquet(df, PROCESSED_DIR / "labeled.parquet", "labeled data")
    return df


if __name__ == "__main__":
    run()

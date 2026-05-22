"""
Recompute targets on an already-featured parquet using strict definitions.

Why this exists:
  The original featured_data.parquet labels ~67% of rows as shutdown_risk
  because any alarm_code > 0 (including minor codes) triggers a shutdown event.
  This floods the model with false positives and makes it collapse to always
  predicting shutdown.

  This script replaces targets in-place using:
    - Shutdown: critical alarms (code >= 10) OR 3+ consecutive daytime zero-power hours
    - Degradation: performance_ratio < 75% of 7-day rolling median for 2+ hours
    - Labels computed per-inverter group so rolling windows don't bleed across inverters

Usage:
    python src/data_processing/retarget.py
    python src/data_processing/retarget.py --input processed/featured_data.parquet
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import log_section, log_step, Timer

SHUTDOWN_HORIZON    = 48    # 2-day forward window  — rows before a shutdown = shutdown_risk
DEGRADATION_HORIZON = 168   # 7-day forward window  — rows before degradation (but not shutdown) = degradation_risk


def _retarget_group(group: pd.DataFrame) -> pd.DataFrame:
    """Recompute events and targets for a single inverter's sorted time series."""
    g = group.copy()

    has_irr = "irradiation" in g.columns
    irr = g["irradiation"].clip(lower=50) if has_irr else pd.Series(50.0, index=g.index)

    # Performance ratio — no clip so relative comparisons work correctly.
    # (Clipping at 2.0 makes healthy ~6.0 and mildly-degraded ~4.0 look identical.)
    pr_raw = (g["inv_power"] / (irr / 1000)).replace([np.inf, -np.inf], np.nan)
    g["performance_ratio"] = pr_raw.fillna(0)

    # Determine daytime
    is_daytime = g["is_daytime"].astype(int) if "is_daytime" in g.columns else (
        ((g["hour"] >= 6) & (g["hour"] <= 18)).astype(int)
        if "hour" in g.columns else pd.Series(1, index=g.index)
    )

    # Add PR rolling features per-inverter (no cross-inverter window bleed)
    for w, min_p in [(6, 2), (24, 6), (168, 24)]:
        g[f"performance_ratio_roll{w}h_mean"] = g["performance_ratio"].rolling(w, min_periods=min_p).mean()
        g[f"performance_ratio_roll{w}h_std"]  = g["performance_ratio"].rolling(w, min_periods=min_p).std()

    # ── Shutdown events ──────────────────────────────────────────────
    # Critical alarms (code >= 10) OR 3+ consecutive daytime zero-power hours.
    alarm_col = "inv_alarm_code" if "inv_alarm_code" in g.columns else None
    g["critical_alarm"] = (g[alarm_col] >= 10).astype(int) if alarm_col else 0

    zero_daytime = ((g["inv_power"] == 0) & (is_daytime == 1)).astype(int)
    g["sustained_shutdown"] = (zero_daytime.rolling(3, min_periods=3).sum() >= 3).astype(int)
    g["shutdown_event"] = ((g["sustained_shutdown"] == 1) | (g["critical_alarm"] == 1)).astype(int)

    # ── Degradation events ───────────────────────────────────────────
    # A degradation event is any of these three conditions during daytime:
    #
    # 1. PR-based: power/irradiation ratio drops to < 55% of 7-day daytime median
    #    (nighttime zeros are masked out of the median so they don't suppress the baseline)
    #
    # 2. Ratio-based: inv_power drops to < 50% of its own 24-hour mean (recomputed
    #    per-inverter here) while irradiation is substantial
    #
    # 3. Alarm-assisted: minor alarms (code 1-9) combined with low power output
    #
    # Any of the three sustained for 2+ consecutive hours = degradation_event.
    daytime_sun = (is_daytime == 1) & (irr > 100)

    # Condition 1 — PR vs 7d daytime baseline
    pr_daytime = g["performance_ratio"].where(daytime_sun)
    pr_median_7d = pr_daytime.rolling(168, min_periods=6).median().ffill().bfill()
    cond1 = (
        (g["performance_ratio"] < 0.55 * pr_median_7d) &
        daytime_sun &
        (pr_median_7d > 0.1)
    ).astype(int)

    # Condition 2 — per-inverter recomputed power ratio vs 24h mean
    rolling_24h = g["inv_power"].rolling(24, min_periods=6).mean()
    cond2 = (
        (g["inv_power"] < 0.5 * rolling_24h) &
        daytime_sun &
        (rolling_24h > 0.2)
    ).astype(int)

    # Condition 3 — minor alarm AND output below 70% of rolling mean
    if "inv_alarm_code" in g.columns:
        minor_alarm = ((g["inv_alarm_code"] > 0) & (g["inv_alarm_code"] < 10)).astype(int)
        cond3 = (
            minor_alarm &
            (g["inv_power"] < 0.7 * rolling_24h) &
            daytime_sun &
            (rolling_24h > 0.2)
        ).astype(int)
    else:
        cond3 = pd.Series(0, index=g.index)

    combined = np.maximum(np.maximum(cond1, cond2), cond3)
    g["degradation_event"] = (combined.rolling(3, min_periods=2).sum() >= 2).astype(int)

    # ── Forward label windows ────────────────────────────────────────
    # Use separate horizons: shutdown gets a 2-day urgency window;
    # degradation uses a 7-day window.  A row labeled shutdown_risk means
    # "a critical failure is expected within 48 h".  degradation_risk means
    # "performance degradation is expected within 7 days but NOT an imminent shutdown".
    future_shutdown    = g["shutdown_event"].iloc[::-1].rolling(SHUTDOWN_HORIZON,    min_periods=1).sum().iloc[::-1]
    future_degradation = g["degradation_event"].iloc[::-1].rolling(DEGRADATION_HORIZON, min_periods=1).sum().iloc[::-1]

    g["target_binary"]     = ((future_shutdown > 0) | (future_degradation > 0)).astype(int)
    g["target_multiclass"] = 0
    g.loc[future_degradation > 0, "target_multiclass"] = 1   # degradation first
    g.loc[future_shutdown    > 0, "target_multiclass"] = 2   # shutdown overwrites (higher priority)

    return g


def retarget(parquet_path: Path) -> None:
    log_section("Retarget — Strict Label Recomputation")

    log_step(f"Loading {parquet_path.name} ...")
    df = pd.read_parquet(parquet_path)
    log_step(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    # Show old class distribution
    if "target_multiclass" in df.columns:
        old_counts = Counter(df["target_multiclass"].astype(int))
        total = len(df)
        log_step("Old label distribution:")
        for cls, name in [(0, "no_risk"), (1, "degradation_risk"), (2, "shutdown_risk")]:
            cnt = old_counts.get(cls, 0)
            log_step(f"  {name:<20s}  {cnt:>7,}  ({cnt/total:.1%})")

    # Determine grouping column
    group_col = None
    for candidate in ("inverter_id", "inv_id", "inverter_idx"):
        if candidate in df.columns:
            group_col = candidate
            break

    log_step(f"Grouping by: {group_col or 'none (single inverter)'}")

    with Timer():
        if group_col:
            n_groups = df[group_col].nunique()
            log_step(f"Processing {n_groups} inverter groups ...")
            parts = []
            for inv_id, grp in df.groupby(group_col):
                grp_sorted = grp.sort_values("datetime") if "datetime" in grp.columns else grp
                parts.append(_retarget_group(grp_sorted))
            df = pd.concat(parts)
            if "datetime" in df.columns:
                df = df.sort_values(["datetime"]).reset_index(drop=True)
        else:
            if "datetime" in df.columns:
                df = df.sort_values("datetime")
            df = _retarget_group(df).reset_index(drop=True)

    # Show new class distribution
    new_counts = Counter(df["target_multiclass"].astype(int))
    total = len(df)
    log_step("New label distribution:")
    for cls, name in [(0, "no_risk"), (1, "degradation_risk"), (2, "shutdown_risk")]:
        cnt = new_counts.get(cls, 0)
        log_step(f"  {name:<20s}  {cnt:>7,}  ({cnt/total:.1%})")

    # Backup original and save
    backup = parquet_path.with_suffix(".parquet.bak")
    if not backup.exists():
        import shutil
        shutil.copy2(parquet_path, backup)
        log_step(f"Backup saved -> {backup.name}")

    df.to_parquet(parquet_path, index=False)
    log_step(f"Retargeted parquet saved -> {parquet_path}")
    log_section("Retarget complete")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import FEATURED_PARQUET

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FEATURED_PARQUET)
    args = parser.parse_args()
    retarget(args.input)

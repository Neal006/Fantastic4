"""
Raw data loading and hourly aggregation.

Use load_featured_parquet() directly if you already have a featured parquet.
Use load_plant_data() + aggregate_hourly() + compute_smu_features() if you
need to process raw CSVs from scratch.
"""

import re
from pathlib import Path

import pandas as pd


def load_featured_parquet(path: Path) -> pd.DataFrame:
    """Load a pre-built featured parquet (skips all preprocessing)."""
    df = pd.read_parquet(path)
    print(f"  [LOAD] {Path(path).name}  [{len(df):,} rows x {len(df.columns)} cols]")
    return df


def load_plant_data(csv_path: str, plant_id: str, logger_mac: str):
    """Load a raw plant CSV and detect inverter indices."""
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    inv_indices = sorted({
        int(m.group(1))
        for col in df.columns
        for m in [re.match(r"inverters\[(\d+)\]", col)]
        if m
    })
    df["plant_id"] = plant_id
    df["logger_mac"] = logger_mac
    return df, inv_indices


def aggregate_hourly(df_inverter: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a per-inverter 5-min DataFrame to hourly resolution."""
    agg_rules = {
        "inv_alarm_code":              "max",
        "inv_kwh_today":               "last",
        "inv_power":                   "mean",
        "inv_temp":                    "mean",
        "inv_freq":                    "mean",
        "inv_v_ab":                    "mean",
        "inv_v_bc":                    "mean",
        "inv_v_ca":                    "mean",
        "inv_pv1_power":               "mean",
        "meter_pf":                    "mean",
        "meter_freq":                  "mean",
        "meter_v_r":                   "mean",
        "meter_v_y":                   "mean",
        "meter_v_b":                   "mean",
        "meter_meter_active_power":    "mean",
        "ambient_temp":                "mean",
    }
    existing = {k: v for k, v in agg_rules.items() if k in df_inverter.columns}
    return (
        df_inverter
        .set_index("datetime")
        .resample("1h")
        .agg(existing)
        .reset_index()
    )


def compute_smu_features(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise SMU string-current columns into aggregate features."""
    smu_cols = [c for c in df.columns if c.startswith("smu.smu_string_current")]
    if not smu_cols:
        df["smu_string_mean"] = 0.0
        df["smu_string_std"]  = 0.0
        df["smu_num_zero"]    = 0
        df["smu_total_strings"] = 0
        return df
    df["smu_string_mean"]   = df[smu_cols].mean(axis=1)
    df["smu_string_std"]    = df[smu_cols].std(axis=1)
    df["smu_num_zero"]      = (df[smu_cols] == 0).sum(axis=1)
    df["smu_total_strings"] = len(smu_cols)
    return df

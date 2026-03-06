"""
Data preprocessing pipeline for solar inverter telemetry CSVs.
10-step cleaning: NaN removal, metadata drop, timestamp conversion,
dedup, sort, fill, dtype conversion.
"""

import pandas as pd
import numpy as np


# metadata / non-ML columns to always drop
METADATA_COLS = [
    "_id", "createdAt", "timestampDate", "dataLoggerModelId",
    "__v", "grid_master", "fromServer", "mac", "model", "serial",
]


def _log(step: int, msg: str, df: pd.DataFrame) -> None:
    print(f"[Step {step:>2}] {msg}  →  shape {df.shape}")


# -- individual steps --------------------------------------------------------

def remove_high_nan_cols(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Step 1 – drop columns with >threshold fraction NaN."""
    nan_frac = df.isna().mean()
    drop = nan_frac[nan_frac > threshold].index.tolist()
    df = df.drop(columns=drop)
    _log(1, f"Dropped {len(drop)} cols with >{threshold*100:.0f}% NaN", df)
    return df


def remove_constant_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 – drop columns whose unique non-NaN count ≤ 1."""
    nuniq = df.nunique(dropna=True)
    drop = nuniq[nuniq <= 1].index.tolist()
    df = df.drop(columns=drop)
    _log(2, f"Dropped {len(drop)} constant-value cols", df)
    return df


def drop_metadata_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 – drop non-ML metadata columns."""
    drop = [c for c in METADATA_COLS if c in df.columns]
    # also catch any .id sub-columns (e.g. inverters[0].id, smu[3].id)
    id_cols = [c for c in df.columns if c.endswith(".id")]
    drop = list(set(drop + id_cols))
    df = df.drop(columns=drop)
    _log(3, f"Dropped {len(drop)} metadata cols", df)
    return df


def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4 – epoch ms → IST datetime, set as index."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    df = df.set_index("timestamp")
    _log(4, "Converted timestamp to IST datetime index", df)
    return df


def remove_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5 – keep first occurrence of each timestamp."""
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    _log(5, f"Removed {before - len(df)} duplicate-timestamp rows", df)
    return df


def sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6 – sort by timestamp index."""
    df = df.sort_index()
    _log(6, "Sorted by timestamp", df)
    return df


def drop_mid_collection_nan_cols(
    df: pd.DataFrame, threshold: float = 0.4
) -> pd.DataFrame:
    """Step 7 – drop columns with >threshold fraction NaN (e.g. sensors added mid-collection)."""
    nan_frac = df.isna().mean()
    drop = nan_frac[nan_frac > threshold].index.tolist()
    df = df.drop(columns=drop)
    _log(7, f"Dropped {len(drop)} cols with >{threshold*100:.0f}% NaN", df)
    return df


def handle_remaining_nan(
    df: pd.DataFrame, max_gap: str = "30min"
) -> pd.DataFrame:
    """Step 8 – forward-fill capped at max_gap, then backward-fill edges."""
    # infer frequency to compute limit
    freq = pd.infer_freq(df.index)
    if freq is not None:
        period = pd.tseries.frequencies.to_offset(freq)
        limit = int(pd.Timedelta(max_gap) / pd.Timedelta(period))
    else:
        # fallback: median inter-sample gap
        diffs = df.index.to_series().diff().dropna()
        median_gap = diffs.median()
        limit = max(1, int(pd.Timedelta(max_gap) / median_gap))

    df = df.ffill(limit=limit).bfill()
    _log(8, f"Forward-fill (limit={limit}) + backward-fill done", df)
    return df


def drop_remaining_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Step 9 – drop any rows still containing NaN."""
    before = len(df)
    df = df.dropna()
    _log(9, f"Dropped {before - len(df)} rows still with NaN", df)
    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Step 10 – numeric columns to float32 for memory efficiency."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)
    _log(10, f"Converted {len(num_cols)} numeric cols to float32", df)
    return df


# -- orchestrator ------------------------------------------------------------

STEPS = [
    remove_high_nan_cols,        # 1
    remove_constant_cols,        # 2
    drop_metadata_cols,          # 3
    convert_timestamp,           # 4
    remove_duplicate_timestamps, # 5
    sort_by_timestamp,           # 6
    drop_mid_collection_nan_cols,# 7
    handle_remaining_nan,        # 8
    drop_remaining_nan_rows,     # 9
    convert_dtypes,              # 10
]


def preprocess_pipeline(csv_path: str) -> pd.DataFrame:
    """Run all 10 preprocessing steps on a raw CSV file."""
    print(f"\n{'='*60}")
    print(f"Preprocessing: {csv_path}")
    print(f"{'='*60}")
    df = pd.read_csv(csv_path)
    print(f"Raw shape: {df.shape}")

    for step_fn in STEPS:
        df = step_fn(df)

    print(f"{'='*60}")
    print(f"Final shape: {df.shape}  |  NaN remaining: {df.isna().sum().sum()}")
    print(f"{'='*60}\n")
    return df


# quick run when executed directly
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        r"data/Plant 3/Copy of 54-10-EC-8C-14-69.raws.csv"
    )
    df = preprocess_pipeline(path)
    print(df.head())
    print(df.dtypes.value_counts())

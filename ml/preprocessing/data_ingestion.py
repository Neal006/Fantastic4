"""
Stage 1 – Data Ingestion
========================
Load all raw CSV files from ``data/Plant */`` and melt the wide
inverter columns into long format: one row per (timestamp, inverter_id).
Shared columns (meter, sensor, SMU) are repeated for each inverter row.

Output → ``processed/ingested.parquet``
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, PROCESSED_DIR, TIMESTAMP_COL, INVERTER_ID_COL, PLANT_ID_COL, MAC_COL
from utils import log_section, log_step, save_parquet, Timer


# ── helpers ──────────────────────────────────────────────────────
_INV_PATTERN = re.compile(r"^inverters\[(\d+)\]\.(.+)$")


def _detect_inverter_columns(columns):
    """Return {inv_index: {field_name: original_col_name}}."""
    mapping: dict[int, dict[str, str]] = {}
    for col in columns:
        m = _INV_PATTERN.match(col)
        if m:
            idx, field = int(m.group(1)), m.group(2)
            mapping.setdefault(idx, {})[field] = col
    return mapping


def _melt_inverters(df: pd.DataFrame, inv_map: dict, shared_cols: list[str]):
    """Melt wide inverter columns into long format."""
    frames = []
    for inv_idx, field_map in sorted(inv_map.items()):
        # Subset: shared cols + this inverter's cols
        sub = df[shared_cols].copy()
        for field, raw_col in field_map.items():
            sub[field] = df[raw_col].values
        sub[INVERTER_ID_COL] = inv_idx + 1  # 1-based
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


# ── main ─────────────────────────────────────────────────────────
def run(sample_frac: float = 1.0):
    log_section("Stage 1 · Data Ingestion")

    all_frames = []
    plant_dirs = sorted(DATA_DIR.glob("Plant *"))
    if not plant_dirs:
        raise FileNotFoundError(f"No 'Plant *' directories found in {DATA_DIR}")

    for pdir in plant_dirs:
        plant_id = pdir.name  # e.g. "Plant 1"
        csv_files = sorted(pdir.glob("*.csv"))
        log_step(f"{plant_id}: found {len(csv_files)} CSV file(s)")

        for csv_path in csv_files:
            log_step(f"  Reading {csv_path.name} ...")
            with Timer():
                df = pd.read_csv(csv_path, low_memory=False)
                if 0 < sample_frac < 1.0:
                    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

            log_step(f"    Raw shape: {df.shape}")

            # Identify inverter columns
            inv_map = _detect_inverter_columns(df.columns)
            n_inv = len(inv_map)
            log_step(f"    Detected {n_inv} inverter(s)")

            # Identify shared columns (non-inverter)
            inv_cols_flat = {c for fm in inv_map.values() for c in fm.values()}
            shared_cols = [c for c in df.columns if c not in inv_cols_flat]

            # Rename meter / sensor / smu cols (strip prefix)
            rename = {}
            for c in shared_cols:
                for prefix in ("meters[0].", "sensors[0].", "smu[0].", "smu[1].",
                               "smu[2].", "smu[3].", "smu[4]."):
                    if c.startswith(prefix):
                        rename[c] = c.replace(prefix, "")
                        break
            df.rename(columns=rename, inplace=True)
            shared_cols = [rename.get(c, c) for c in shared_cols]

            # Deduplicate shared columns (e.g. smu[0].string1 and smu[1].string1
            # both become string1 -- keep only unique names)
            seen = set()
            deduped = []
            for c in shared_cols:
                if c not in seen:
                    seen.add(c)
                    deduped.append(c)
            shared_cols = deduped
            # Also remove duplicate columns from df itself
            df = df.loc[:, ~df.columns.duplicated()]

            # Melt
            melted = _melt_inverters(df, inv_map, shared_cols)
            melted[PLANT_ID_COL] = plant_id
            all_frames.append(melted)
            log_step(f"    Melted shape: {melted.shape}")

    result = pd.concat(all_frames, ignore_index=True, sort=False)
    log_step(f"Combined shape: {result.shape}")

    # Basic dtype fixes
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_numeric(result["timestamp"], errors="coerce")

    # Coerce object columns that are actually numeric
    for col in result.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(result[col], errors="coerce")
        # If more than 50% of non-null values converted successfully, treat as numeric
        if converted.notna().sum() > result[col].notna().sum() * 0.5:
            result[col] = converted

    save_parquet(result, PROCESSED_DIR / "ingested.parquet", "ingested data")
    return result


if __name__ == "__main__":
    run()

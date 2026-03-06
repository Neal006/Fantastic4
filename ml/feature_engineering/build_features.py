"""Feature pipeline orchestrator: cleaned CSV → engineered parquet."""
import sys, os, time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DATA_CLEANED_DIR, FEATURES_DIR, PLANT_FILES, NUM_CLASSES, CLASS_NAMES
from feature_engineering.target_labeling import label_dataset
from feature_engineering.telemetry_features import add_telemetry_features
from feature_engineering.kpi_features import add_kpi_features
from feature_engineering.alarm_features import add_alarm_features
from feature_engineering.time_features import add_time_features


def build_features_for_plant(plant_name, verbose=True):
    fname = PLANT_FILES[plant_name]
    fpath = os.path.join(DATA_CLEANED_DIR, fname)
    t0 = time.time()
    if verbose: print(f"\n{'='*60}\n  Building features: {plant_name}\n{'='*60}")

    df = pd.read_csv(fpath, low_memory=False)
    if "timestamp" not in df.columns and df.index.name and "time" in df.index.name.lower():
        df = df.reset_index()
    if verbose: print(f"  Loaded: {len(df):,} rows × {len(df.columns)} cols")

    # Force coerce all non-timestamp/metadata columns to numeric to drop string errors ('-')
    import numpy as np
    for c in df.columns:
        if c not in ["timestamp", "plant_name"] and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)

    df = label_dataset(df, plant_name)
    if verbose: print(f"  ✓ Target labels ({len(df.columns)} cols)")
    df = add_telemetry_features(df, plant_name)
    if verbose: print(f"  ✓ Telemetry features ({len(df.columns)} cols)")
    df = add_kpi_features(df, plant_name)
    if verbose: print(f"  ✓ KPI features ({len(df.columns)} cols)")
    df = add_alarm_features(df, plant_name)
    if verbose: print(f"  ✓ Alarm features ({len(df.columns)} cols)")
    df = add_time_features(df)
    if verbose: print(f"  ✓ Time features ({len(df.columns)} cols)")

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out = os.path.join(FEATURES_DIR, f"{plant_name}_features.parquet")
    df.to_parquet(out, index=False, engine="pyarrow")
    if verbose:
        print(f"  ✓ Saved: {out}")
        print(f"  Final: {len(df):,} rows × {len(df.columns)} cols  ({time.time()-t0:.1f}s)")
        print(f"\n  Target distribution:")
        for c in range(NUM_CLASSES):
            n = (df["target_7d"] == c).sum()
            print(f"    {c} {CLASS_NAMES[c]:>20}: {n:>8,} ({n/len(df)*100:.2f}%)")
    return df


def build_all_features():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    for name in PLANT_FILES:
        build_features_for_plant(name)
    print(f"\n✓ All features saved to {FEATURES_DIR}")


if __name__ == "__main__":
    build_all_features()

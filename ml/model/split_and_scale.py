"""
Stage 6 – Split & Scale
========================
- Chronological hold-out test set (last 20 %).
- Walk-forward (expanding-window) cross-validation folds on the training set.
- StandardScaler fitted on train only.
- SMOTE applied on each training fold to balance the 3 classes.

Input  → ``processed/anomaly_enriched.parquet``
Output → ``processed/splits.pkl``  (dict of folds + test set)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_DIR, MODELS_DIR, TIMESTAMP_COL, INVERTER_ID_COL, PLANT_ID_COL,
    MAC_COL, LABEL_COL, SEED, N_CV_FOLDS, TEST_FRAC, SMOTE_K_NEIGHBORS,
    CLASS_NAMES,
)
from utils import log_section, log_step, save_pickle, load_parquet, Timer


# Columns to exclude from features
_META_COLS = {
    TIMESTAMP_COL, INVERTER_ID_COL, PLANT_ID_COL, MAC_COL, LABEL_COL,
    "model", "serial", "id", "meter_model", "meter_serial", "meter_id",
    "fromServer", "dataLoggerModelId", "__v", "timestampDate",
    "_id", "createdAt", "batteries", "grid_master",
}


def _select_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (exclude meta & label)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return sorted(c for c in num_cols if c not in _META_COLS)


def run():
    log_section("Stage 6 · Split & Scale")

    df = load_parquet(PROCESSED_DIR / "anomaly_enriched.parquet")

    # Ensure sorted chronologically
    df.sort_values(TIMESTAMP_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = _select_feature_cols(df)
    log_step(f"Feature columns: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[LABEL_COL].values.astype(np.int8)

    # Remap labels to 0-indexed consecutive integers (XGBoost requirement)
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int8)
    log_step(f"Label mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    timestamps = df[TIMESTAMP_COL].values

    # -- Replace any remaining NaN / inf with 0 --
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # -- Hold-out test set (last 20 %) --
    n = len(X)
    test_start = int(n * (1 - TEST_FRAC))
    X_trainval, X_test = X[:test_start], X[test_start:]
    y_trainval, y_test = y[:test_start], y[test_start:]
    ts_trainval = timestamps[:test_start]

    log_step(f"Train+Val: {len(X_trainval):,} | Test: {len(X_test):,}")

    # ── Walk-forward CV folds ──
    fold_size = len(X_trainval) // (N_CV_FOLDS + 1)
    folds = []
    for i in range(N_CV_FOLDS):
        train_end = fold_size * (i + 2)
        val_start = train_end
        val_end = min(val_start + fold_size, len(X_trainval))
        if val_end <= val_start:
            continue

        X_tr, y_tr = X_trainval[:train_end], y_trainval[:train_end]
        X_va, y_va = X_trainval[val_start:val_end], y_trainval[val_start:val_end]

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # SMOTE on training fold
        unique, counts = np.unique(y_tr, return_counts=True)
        min_count = counts.min()
        k = min(SMOTE_K_NEIGHBORS, max(min_count - 1, 1))
        if min_count > 1 and len(unique) > 1:
            sm = SMOTE(random_state=SEED, k_neighbors=k)
            X_tr_s, y_tr_sm = sm.fit_resample(X_tr_s, y_tr)
        else:
            y_tr_sm = y_tr

        folds.append({
            "X_train": X_tr_s.astype(np.float32),
            "y_train": y_tr_sm,
            "X_val": X_va_s.astype(np.float32),
            "y_val": y_va,
            "scaler": scaler,
        })
        log_step(
            f"  Fold {i + 1}: train {len(X_tr_s):,} (SMOTE'd) | val {len(X_va):,} "
            f"| classes {dict(zip(*np.unique(y_tr_sm, return_counts=True)))}"
        )

    # ── Scale full train+val and test ──
    final_scaler = StandardScaler()
    X_trainval_s = final_scaler.fit_transform(X_trainval)
    X_test_s = final_scaler.transform(X_test)

    # SMOTE on full training set
    unique, counts = np.unique(y_trainval, return_counts=True)
    min_count = counts.min()
    k = min(SMOTE_K_NEIGHBORS, max(min_count - 1, 1))
    if min_count > 1 and len(unique) > 1:
        sm = SMOTE(random_state=SEED, k_neighbors=k)
        X_trainval_sm, y_trainval_sm = sm.fit_resample(X_trainval_s, y_trainval)
    else:
        X_trainval_sm, y_trainval_sm = X_trainval_s, y_trainval

    log_step(f"Full train (SMOTE'd): {len(X_trainval_sm):,} | Test: {len(X_test_s):,}")

    splits = {
        "feature_cols": feature_cols,
        "folds": folds,
        "X_trainval": X_trainval_sm.astype(np.float32),
        "y_trainval": y_trainval_sm,
        "X_test": X_test_s.astype(np.float32),
        "y_test": y_test,
        "scaler": final_scaler,
        "label_encoder": le,
    }

    save_pickle(splits, PROCESSED_DIR / "splits.pkl", "splits")
    return splits


if __name__ == "__main__":
    run()

"""
One-time utility: extract inference artifacts from ml/processed/splits.pkl
and align them to the actual model's expected feature count.

The model was trained on 145 features but splits.pkl may have 183 columns.
This script extracts only the features the model expects.

Run once:
    python copy_splits.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

SPLITS_SRC = Path(__file__).resolve().parent.parent / "ml" / "processed" / "splits.pkl"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "xgb_best.pkl"
DEST = Path(__file__).resolve().parent / "models" / "inference_artifacts.pkl"


def main():
    if not SPLITS_SRC.exists():
        print(f"ERROR: splits.pkl not found at {SPLITS_SRC}")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"ERROR: xgb_best.pkl not found at {MODEL_PATH}")
        sys.exit(1)

    # Load model to get expected feature count
    print(f"Loading model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    model_n_features = model.n_features_in_
    print(f"  Model expects {model_n_features} features")

    # Load splits
    print(f"Loading splits from {SPLITS_SRC} ...")
    with open(SPLITS_SRC, "rb") as f:
        splits = pickle.load(f)

    full_feature_cols = splits["feature_cols"]
    full_scaler = splits["scaler"]
    print(f"  splits.pkl has {len(full_feature_cols)} feature columns")
    print(f"  Scaler fitted on {full_scaler.n_features_in_} features")

    # Slice to model's expected feature count
    feature_cols = full_feature_cols[:model_n_features]

    # Build a sliced scaler that only transforms the first N features
    sliced_scaler = StandardScaler()
    sliced_scaler.mean_ = full_scaler.mean_[:model_n_features]
    sliced_scaler.var_ = full_scaler.var_[:model_n_features]
    sliced_scaler.scale_ = full_scaler.scale_[:model_n_features]
    sliced_scaler.n_features_in_ = model_n_features
    sliced_scaler.n_samples_seen_ = full_scaler.n_samples_seen_
    # Ensure feature_names_in_ is set correctly (if it exists)
    if hasattr(full_scaler, 'feature_names_in_'):
        sliced_scaler.feature_names_in_ = full_scaler.feature_names_in_[:model_n_features]

    artifacts = {
        "scaler": sliced_scaler,
        "label_encoder": splits["label_encoder"],
        "feature_cols": feature_cols,
    }

    DEST.parent.mkdir(parents=True, exist_ok=True)
    with open(DEST, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved inference artifacts -> {DEST}")
    print(f"  - scaler: StandardScaler (n_features={sliced_scaler.n_features_in_})")
    print(f"  - label_encoder classes: {list(artifacts['label_encoder'].classes_)}")
    print(f"  - feature_cols: {len(feature_cols)} columns (aligned to model)")
    print(f"  - First 10 features: {feature_cols[:10]}")
    print(f"  - Last 5 features: {feature_cols[-5:]}")


if __name__ == "__main__":
    main()

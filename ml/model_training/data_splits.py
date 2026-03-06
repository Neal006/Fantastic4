"""70/30 stratified random split + feature column selection."""
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import TEST_SIZE, RANDOM_STATE, FEATURES_DIR, PLANT_FILES

NON_FEATURE_COLS = {"timestamp", "label_instant", "target_7d", "plant_name"}


def load_feature_data(plant_names=None):
    if plant_names is None: plant_names = list(PLANT_FILES.keys())
    dfs = []
    for n in plant_names:
        fp = os.path.join(FEATURES_DIR, f"{n}_features.parquet")
        if not os.path.exists(fp):
            print(f"  ⚠️ Missing: {fp}"); continue
        df = pd.read_parquet(fp); df["plant_name"] = n; dfs.append(df)
    if not dfs: raise FileNotFoundError("No feature files found!")
    return pd.concat(dfs, ignore_index=True)


def get_feature_columns(df):
    exclude = set(NON_FEATURE_COLS)
    exclude |= {c for c in df.columns if c.startswith("label_inv")}
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in [np.float32, np.float64, np.int32, np.int64]]


def split_data(df, target_col="target_7d"):
    feat_cols = get_feature_columns(df)
    
    # Temporal split per plant to prevent time-series bleeding
    train_dfs, test_dfs = [], []
    for p in df["plant_name"].unique():
        pdf = df[df["plant_name"] == p].copy()
        if "timestamp" in pdf.columns:
            pdf = pdf.sort_values("timestamp")
        
        n_train = int(len(pdf) * (1 - TEST_SIZE))
        train_dfs.append(pdf.iloc[:n_train])
        test_dfs.append(pdf.iloc[n_train:])
        
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    
    X_train = np.nan_to_num(train_df[feat_cols].values.astype(np.float32), nan=0, posinf=0, neginf=0)
    y_train = train_df[target_col].values.astype(np.int32)
    X_test = np.nan_to_num(test_df[feat_cols].values.astype(np.float32), nan=0, posinf=0, neginf=0)
    y_test = test_df[target_col].values.astype(np.int32)
    
    print(f"  Temporal Split: {len(X_train):,} train / {len(X_test):,} test  |  Features: {len(feat_cols)}")
    print(f"  Train dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test dist:  {dict(zip(*np.unique(y_test, return_counts=True)))}")
    return X_train, X_test, y_train, y_test, feat_cols

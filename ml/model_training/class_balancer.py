"""SMOTE oversampling + class weight computation."""
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter


def apply_smote(X, y, random_state=42):
    counter = Counter(y)
    min_c = min(counter.values())
    k = min(5, min_c - 1) if min_c > 1 else 1
    if k < 1:
        print("  ⚠️ Classes too small for SMOTE, skipping")
        return X, y
    try:
        sm = SMOTE(random_state=random_state, k_neighbors=k)
        Xr, yr = sm.fit_resample(X, y)
        print(f"  SMOTE: {dict(Counter(y))} → {dict(Counter(yr))}")
        return Xr, yr
    except ValueError as e:
        print(f"  ⚠️ SMOTE failed ({e}), using original")
        return X, y


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n = len(y); nc = len(classes)
    return {int(c): n / (nc * cnt) for c, cnt in zip(classes, counts)}

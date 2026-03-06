"""Per-class and overall precision, recall, F1, AUC."""
import numpy as np
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                              f1_score, precision_score, recall_score)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import NUM_CLASSES, CLASS_NAMES


def compute_all_metrics(y_true, y_pred, y_proba=None):
    m = {}
    m["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    if y_proba is not None:
        try: m["auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except: m["auc_ovr"] = 0.0
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    for i in range(NUM_CLASSES):
        m[f"precision_class{i}"] = p[i]; m[f"recall_class{i}"] = r[i]
        m[f"f1_class{i}"] = f[i]; m[f"support_class{i}"] = int(s[i])
    return m


def print_metrics(m, name="Model"):
    print(f"\n{'='*60}\n  {name} — Results\n{'='*60}")
    print(f"  Precision: {m['precision_macro']:.4f}  Recall: {m['recall_macro']:.4f}  "
          f"F1: {m['f1_macro']:.4f}  AUC: {m.get('auc_ovr',0):.4f}")
    print(f"\n  {'Class':>25} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>8}")
    print(f"  {'-'*55}")
    for i in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[i]:>25} {m.get(f'precision_class{i}',0):>7.4f} "
              f"{m.get(f'recall_class{i}',0):>7.4f} {m.get(f'f1_class{i}',0):>7.4f} "
              f"{m.get(f'support_class{i}',0):>8}")

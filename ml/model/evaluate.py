"""
Stage 9 – Evaluation
=====================
Comprehensive evaluation of the best ensemble model on the hold-out test set.
Reports per-class & macro precision, recall, F1, AUC.
Generates a confusion matrix plot and saves a classification report CSV.

Input  → ``models/ensemble_meta.pkl``
Output → ``outputs/confusion_matrix.png``, ``outputs/classification_report.csv``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODELS_DIR, OUTPUTS_DIR, CLASS_NAMES
from utils import log_section, log_step, load_pickle, Timer


def run():
    log_section("Stage 9 · Evaluation")

    ens = load_pickle(MODELS_DIR / "ensemble_meta.pkl")
    method = ens["method"]
    y_test = ens["y_test"]

    # Produce final predictions
    if method == "stacking":
        meta_features = np.hstack([ens["xgb_prob"], ens["lstm_prob"]])
        meta_clf = ens["meta_clf"]
        y_prob = meta_clf.predict_proba(meta_features)
        y_pred = meta_clf.predict(meta_features)
    else:
        w = ens["xgb_weight"]
        y_prob = w * ens["xgb_prob"] + (1 - w) * ens["lstm_prob"]
        y_pred = y_prob.argmax(axis=1)

    # ── Metrics ──
    p_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        auc_macro = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = float("nan")

    log_step(f"Ensemble method: {method}")
    log_step(f"Macro  ─  P={p_macro:.4f}  R={r_macro:.4f}  F1={f1_macro:.4f}  AUC={auc_macro:.4f}")

    # Classification report
    report_str = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    print(report_str)

    report_dict = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T
    report_path = OUTPUTS_DIR / "classification_report.csv"
    report_df.to_csv(report_path)
    log_step(f"Saved classification report → {report_path}")

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_title("Confusion Matrix – Ensemble (Hold-out Test)", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    log_step(f"Saved confusion matrix → {cm_path}")

    return {
        "precision": p_macro,
        "recall": r_macro,
        "f1": f1_macro,
        "auc": auc_macro,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    run()

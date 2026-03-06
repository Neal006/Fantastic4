"""Report generation: confusion matrix + ROC curves."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import NUM_CLASSES, CLASS_NAMES, REPORTS_DIR


def plot_confusion_matrix(y_true, y_pred, name="Model"):
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}"); plt.tight_layout()
    p = os.path.join(REPORTS_DIR, f"cm_{name.lower().replace(' ','_')}.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def plot_roc_curves(y_true, y_proba, name="Model"):
    yb = label_binarize(y_true, classes=range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        if yb[:, i].sum() == 0: continue
        fpr, tpr, _ = roc_curve(yb[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {name}"); ax.legend(loc="lower right"); plt.tight_layout()
    p = os.path.join(REPORTS_DIR, f"roc_{name.lower().replace(' ','_')}.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def generate_all_reports(y_true, y_pred, y_proba, name="Model"):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    paths = [plot_confusion_matrix(y_true, y_pred, name)]
    if y_proba is not None: paths.append(plot_roc_curves(y_true, y_proba, name))
    print(f"  Reports → {REPORTS_DIR}"); return paths

"""
Stage 7b -- Train CatBoost
===========================
Multi-class CatBoost with Optuna hyper-parameter tuning on walk-forward CV.
Reports per-fold precision, recall, F1, and AUC.
After training, computes Accuracy, Precision, Recall, F1, AUC on
Train / Validation / Test splits and saves metrics CSV + AUC-curve plot.

Input  -> processed/splits.pkl
Output -> models/cbc_best.pkl, models/cbc_optuna_study.pkl,
          outputs/cbc_metrics.csv, outputs/cbc_auc_curve.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, SEED,
    CBC_SEARCH_SPACE, CBC_OPTUNA_TRIALS, CLASS_NAMES,
)
from utils import log_section, log_step, save_pickle, load_pickle, Timer

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_CLASSES = len(CLASS_NAMES)


def _ensure_all_classes(X, y, n_classes=N_CLASSES):
    """Add one dummy sample per missing class so CatBoost sees all classes."""
    present = set(np.unique(y))
    missing = set(range(n_classes)) - present
    if not missing:
        return X, y
    n_feat = X.shape[1]
    X_extra = np.zeros((len(missing), n_feat), dtype=X.dtype)
    y_extra = np.array(sorted(missing), dtype=y.dtype)
    return np.vstack([X, X_extra]), np.concatenate([y, y_extra])


def _compute_metrics(clf, X, y, label: str):
    """Compute accuracy, precision, recall, F1, AUC for a given split."""
    y_pred = clf.predict(X).astype(int).ravel()
    y_prob = clf.predict_proba(X)
    acc = accuracy_score(y, y_pred)
    p = precision_score(y, y_pred, average="macro", zero_division=0)
    r = recall_score(y, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    try:
        auc_val = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc_val = float("nan")
    log_step(f"  {label}: Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  AUC={auc_val:.4f}")
    return {"split": label, "accuracy": acc, "precision": p, "recall": r, "f1": f1, "auc": auc_val}


def _plot_auc_curve(clf, X_test, y_test, save_path: Path):
    """Plot per-class OvR AUC curves and save."""
    y_prob = clf.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(CLASS_NAMES):
        if y_bin.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("CatBoost -- ROC / AUC Curves (Test Set)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log_step(f"Saved AUC curve -> {save_path}")


def _objective(trial, folds):
    """Optuna objective: macro-F1 averaged across walk-forward folds."""
    params = {
        "depth": trial.suggest_int("depth", *CBC_SEARCH_SPACE["depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *CBC_SEARCH_SPACE["learning_rate"], log=True),
        "iterations": trial.suggest_int("iterations", *CBC_SEARCH_SPACE["iterations"], step=50),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *CBC_SEARCH_SPACE["l2_leaf_reg"]),
        "bagging_temperature": trial.suggest_float("bagging_temperature", *CBC_SEARCH_SPACE["bagging_temperature"]),
        "random_strength": trial.suggest_float("random_strength", *CBC_SEARCH_SPACE["random_strength"]),
        "border_count": trial.suggest_int("border_count", *CBC_SEARCH_SPACE["border_count"]),
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": SEED,
        "verbose": 0,
        "auto_class_weights": "Balanced",
        "thread_count": -1,
    }

    scores = []
    for fold in folds:
        X_tr, y_tr = _ensure_all_classes(fold["X_train"], fold["y_train"])
        X_va, y_va = _ensure_all_classes(fold["X_val"], fold["y_val"])

        clf = CatBoostClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30, verbose=0)
        y_pred = clf.predict(fold["X_val"]).astype(int).ravel()
        f1 = f1_score(fold["y_val"], y_pred, average="macro", zero_division=0)
        scores.append(f1)

    if not scores:
        return 0.0
    return np.mean(scores)


def run():
    log_section("Stage 7b - Train CatBoost (Optuna)")

    splits = load_pickle(PROCESSED_DIR / "splits.pkl")
    folds = splits["folds"]

    log_step(f"Starting Optuna study with {CBC_OPTUNA_TRIALS} trials on {len(folds)} CV folds ...")

    with Timer():
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda t: _objective(t, folds), n_trials=CBC_OPTUNA_TRIALS, show_progress_bar=True)

    best = study.best_params
    log_step(f"Best macro-F1: {study.best_value:.4f}")
    log_step(f"Best params: {best}")

    # Retrain on full train+val with best params
    log_step("Retraining on full training set ...")
    best.update({
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": SEED,
        "verbose": 0,
        "auto_class_weights": "Balanced",
        "thread_count": -1,
    })
    final_clf = CatBoostClassifier(**best)
    X_full, y_full = _ensure_all_classes(splits["X_trainval"], splits["y_trainval"])
    final_clf.fit(X_full, y_full, verbose=0)

    # ── Per-split evaluation ──────────────────────────────────────
    log_step("Per-split evaluation:")
    metrics_rows = []

    # Train
    metrics_rows.append(_compute_metrics(final_clf, splits["X_trainval"], splits["y_trainval"], "Train"))

    # Validation (average across CV folds)
    val_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}
    for i, fold in enumerate(folds):
        m = _compute_metrics(final_clf, fold["X_val"], fold["y_val"], f"Valid-Fold-{i+1}")
        for k in val_metrics:
            val_metrics[k].append(m[k])
    avg_val = {k: np.nanmean(v) for k, v in val_metrics.items()}
    avg_val["split"] = "Valid (CV avg)"
    metrics_rows.append(avg_val)
    log_step(f"  Valid (CV avg): Acc={avg_val['accuracy']:.4f}  "
             f"P={avg_val['precision']:.4f}  R={avg_val['recall']:.4f}  "
             f"F1={avg_val['f1']:.4f}  AUC={avg_val['auc']:.4f}")

    # Cross-validation F1 (per-fold + mean +/- std)
    cv_f1s = val_metrics["f1"]
    log_step(f"  CV F1 per fold: {[f'{v:.4f}' for v in cv_f1s]}")
    log_step(f"  CV F1 mean +/- std: {np.mean(cv_f1s):.4f} +/- {np.std(cv_f1s):.4f}")

    # Test
    metrics_rows.append(_compute_metrics(final_clf, splits["X_test"], splits["y_test"], "Test"))

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = OUTPUTS_DIR / "cbc_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log_step(f"Saved metrics -> {metrics_path}")

    # AUC curve on test set
    _plot_auc_curve(final_clf, splits["X_test"], splits["y_test"], OUTPUTS_DIR / "cbc_auc_curve.png")

    save_pickle(final_clf, MODELS_DIR / "cbc_best.pkl", "CatBoost best model")
    save_pickle(study, MODELS_DIR / "cbc_optuna_study.pkl", "Optuna study")
    return final_clf


if __name__ == "__main__":
    run()

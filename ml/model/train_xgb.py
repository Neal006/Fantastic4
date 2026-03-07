"""
Stage 7a -- Train XGBoost
=========================
Multi-class XGBoost with Optuna hyper-parameter tuning on walk-forward CV.
Reports per-fold precision, recall, F1, and AUC.

Ensures all 3 classes are present by injecting minimal synthetic samples
for any missing class.

Input  -> processed/splits.pkl
Output -> models/xgb_best.pkl, models/xgb_optuna_study.pkl
"""

import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_DIR, MODELS_DIR, SEED, XGB_SEARCH_SPACE, XGB_OPTUNA_TRIALS,
    CLASS_NAMES,
)
from utils import log_section, log_step, save_pickle, load_pickle, Timer

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_CLASSES = len(CLASS_NAMES)


def _ensure_all_classes(X, y, n_classes=N_CLASSES):
    """Add one dummy sample per missing class so XGBoost sees all classes."""
    present = set(np.unique(y))
    missing = set(range(n_classes)) - present
    if not missing:
        return X, y
    n_feat = X.shape[1]
    X_extra = np.zeros((len(missing), n_feat), dtype=X.dtype)
    y_extra = np.array(sorted(missing), dtype=y.dtype)
    return np.vstack([X, X_extra]), np.concatenate([y, y_extra])


def _objective(trial, folds):
    """Optuna objective: macro-F1 averaged across walk-forward folds."""
    params = {
        "max_depth": trial.suggest_int("max_depth", *XGB_SEARCH_SPACE["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *XGB_SEARCH_SPACE["learning_rate"], log=True),
        "n_estimators": trial.suggest_int("n_estimators", *XGB_SEARCH_SPACE["n_estimators"], step=50),
        "subsample": trial.suggest_float("subsample", *XGB_SEARCH_SPACE["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *XGB_SEARCH_SPACE["colsample_bytree"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *XGB_SEARCH_SPACE["min_child_weight"]),
        "gamma": trial.suggest_float("gamma", *XGB_SEARCH_SPACE["gamma"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *XGB_SEARCH_SPACE["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *XGB_SEARCH_SPACE["reg_lambda"]),
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "eval_metric": "mlogloss",
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }

    scores = []
    for fold in folds:
        X_tr, y_tr = _ensure_all_classes(fold["X_train"], fold["y_train"])
        X_va, y_va = _ensure_all_classes(fold["X_val"], fold["y_val"])

        clf = XGBClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        y_pred = clf.predict(fold["X_val"])  # predict on original val (no dummies)
        f1 = f1_score(fold["y_val"], y_pred, average="macro", zero_division=0)
        scores.append(f1)

    if not scores:
        return 0.0
    return np.mean(scores)


def run():
    log_section("Stage 7a - Train XGBoost (Optuna)")

    splits = load_pickle(PROCESSED_DIR / "splits.pkl")
    folds = splits["folds"]

    log_step(f"Starting Optuna study with {XGB_OPTUNA_TRIALS} trials on {len(folds)} CV folds ...")

    with Timer():
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda t: _objective(t, folds), n_trials=XGB_OPTUNA_TRIALS, show_progress_bar=True)

    best = study.best_params
    log_step(f"Best macro-F1: {study.best_value:.4f}")
    log_step(f"Best params: {best}")

    # Retrain on full train+val with best params
    log_step("Retraining on full training set ...")
    best.update({
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "eval_metric": "mlogloss",
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    })
    final_clf = XGBClassifier(**best)
    X_full, y_full = _ensure_all_classes(splits["X_trainval"], splits["y_trainval"])
    final_clf.fit(X_full, y_full, verbose=False)

    # Per-fold metrics
    log_step("Per-fold validation metrics:")
    for i, fold in enumerate(folds):
        y_pred = final_clf.predict(fold["X_val"])
        y_prob = final_clf.predict_proba(fold["X_val"])
        p = precision_score(fold["y_val"], y_pred, average="macro", zero_division=0)
        r = recall_score(fold["y_val"], y_pred, average="macro", zero_division=0)
        f1 = f1_score(fold["y_val"], y_pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(fold["y_val"], y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
        log_step(f"  Fold {i + 1}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

    save_pickle(final_clf, MODELS_DIR / "xgb_best.pkl", "XGBoost best model")
    save_pickle(study, MODELS_DIR / "xgb_optuna_study.pkl", "Optuna study")
    return final_clf


if __name__ == "__main__":
    run()

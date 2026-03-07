"""
Stage 7 -- Train and Evaluate XGBoost
=======================================
Multi-class XGBoost with Optuna hyper-parameter tuning on walk-forward CV.
Reports Train, Val, and Test metrics: Accuracy, Precision, Recall, F1, AUC.
Generates SHAP explainability plots and CSV.

Ensures all 3 classes are present by injecting minimal synthetic samples
for any missing class.

Input  -> processed/splits.pkl
Output -> models/xgb_best.pkl, models/xgb_optuna_study.pkl
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, SEED, XGB_SEARCH_SPACE, XGB_OPTUNA_TRIALS,
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


def _get_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return acc, p, r, f1, auc

def run():
    log_section("Stage 7 - Train and Evaluate XGBoost (Optuna)")

    splits = load_pickle(PROCESSED_DIR / "splits.pkl")
    folds = splits["folds"]

    log_step(f"Starting Optuna study with {XGB_OPTUNA_TRIALS} trials on {len(folds)} CV folds ...")

    with Timer():
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda t: _objective(t, folds), n_trials=XGB_OPTUNA_TRIALS, show_progress_bar=True)

    best = study.best_params
    log_step(f"Best macro-F1: {study.best_value:.4f}")
    log_step(f"Best params: {best}")

    # Cross-validation detailed evaluation on best params
    log_step("Per-fold evaluation with best params:")
    best.update({
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "eval_metric": "mlogloss",
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    })
    
    cv_clf = XGBClassifier(**best)
    for i, fold in enumerate(folds):
        X_tr, y_tr = _ensure_all_classes(fold["X_train"], fold["y_train"])
        cv_clf.fit(X_tr, y_tr, verbose=False)
        
        y_pred_tr = cv_clf.predict(fold["X_train"]) # Original, no dummies
        y_prob_tr = cv_clf.predict_proba(fold["X_train"]) 
        acc_tr, p_tr, r_tr, f1_tr, auc_tr = _get_metrics(fold["y_train"], y_pred_tr, y_prob_tr)
        
        y_pred_va = cv_clf.predict(fold["X_val"])
        y_prob_va = cv_clf.predict_proba(fold["X_val"])
        acc_va, p_va, r_va, f1_va, auc_va = _get_metrics(fold["y_val"], y_pred_va, y_prob_va)

        log_step(f"--- Fold {i + 1} ---")
        log_step(f"  Train: Acc={acc_tr:.3f} | P={p_tr:.3f} | R={r_tr:.3f} | F1={f1_tr:.3f} | AUC={auc_tr:.3f}")
        log_step(f"  Val  : Acc={acc_va:.3f} | P={p_va:.3f} | R={r_va:.3f} | F1={f1_va:.3f} | AUC={auc_va:.3f}")

    # Retrain on full train+val with best params
    log_step("Retraining on full training set ...")
    final_clf = XGBClassifier(**best)
    X_full, y_full = _ensure_all_classes(splits["X_trainval"], splits["y_trainval"])
    final_clf.fit(X_full, y_full, verbose=False)

    y_pred_full = final_clf.predict(splits["X_trainval"])
    y_prob_full = final_clf.predict_proba(splits["X_trainval"])
    acc_trf, p_trf, r_trf, f1_trf, auc_trf = _get_metrics(splits["y_trainval"], y_pred_full, y_prob_full)
    
    log_step("--- Full Train+Val Metrics ---")
    log_step(f"  Acc={acc_trf:.3f} | P={p_trf:.3f} | R={r_trf:.3f} | F1={f1_trf:.3f} | AUC={auc_trf:.3f}")

    # Hold-out Test split evaluation
    log_step("Evaluating on Test Set...")
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    
    y_pred_te = final_clf.predict(X_test)
    y_prob_te = final_clf.predict_proba(X_test)
    acc_te, p_te, r_te, f1_te, auc_te = _get_metrics(y_test, y_pred_te, y_prob_te)

    log_step(f"--- Hold-out Test Metrics ---")
    log_step(f"  Acc={acc_te:.3f} | P={p_te:.3f} | R={r_te:.3f} | F1={f1_te:.3f} | AUC={auc_te:.3f}")

    save_pickle(final_clf, MODELS_DIR / "xgb_best.pkl", "XGBoost best model")
    save_pickle(study, MODELS_DIR / "xgb_optuna_study.pkl", "Optuna study")
    
    # ── SHAP Explainability ──
    log_step("Computing SHAP values for Explainability...")
    feature_cols = splits["feature_cols"]

    # Use a sample of test data for SHAP to save time
    max_samples = min(2000, len(X_test))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_test), max_samples, replace=False)
    X_sample = X_test[idx]

    with Timer():
        explainer = shap.TreeExplainer(final_clf)
        shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=0)  # (n_classes, n_samples, n_features)
        mean_abs = np.mean(np.abs(stacked), axis=(0, 1))  # (n_features,)
    elif len(shap_values.shape) == 3:
        # (n_samples, n_features, n_classes)
        mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        mean_abs = np.mean(np.abs(shap_values), axis=0)

    # ── Top features ──
    top_idx = np.argsort(mean_abs)[::-1][:5]
    log_step("Top 5 features by mean |SHAP|:")
    for rank, i in enumerate(top_idx, 1):
        log_step(f"  {rank}. {feature_cols[i]:40s}  {mean_abs[i]:.4f}")

    # ── Summary bar plot ──
    top10_idx = np.argsort(mean_abs)[::-1][:10]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_cols[i] for i in top10_idx][::-1],
        [mean_abs[i] for i in top10_idx][::-1],
        color="#4C72B0",
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Top 10 Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    bar_path = OUTPUTS_DIR / "shap_summary.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    log_step(f"Saved SHAP summary bar → {bar_path}")

    # ── Beeswarm plot ──
    try:
        if isinstance(shap_values, list):
            sv = shap_values[0] # use first class for beeswarm
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 0]
        else:
            sv = shap_values

        fig2 = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            sv, X_sample,
            feature_names=feature_cols,
            max_display=15,
            show=False,
        )
        plt.title(f"SHAP Beeswarm – Class: {CLASS_NAMES[0]}", fontsize=14)
        plt.tight_layout()
        bee_path = OUTPUTS_DIR / "shap_beeswarm.png"
        fig2.savefig(bee_path, dpi=150)
        plt.close(fig2)
        log_step(f"Saved SHAP beeswarm → {bee_path}")
    except Exception as e:
        log_step(f"Beeswarm plot skipped: {e}")

    top_df = pd.DataFrame({
        "feature": [feature_cols[i] for i in top_idx],
        "mean_abs_shap": [mean_abs[i] for i in top_idx],
    })
    top_df.to_csv(OUTPUTS_DIR / "shap_top5.csv", index=False)
    log_step(f"Saved top-5 CSV → {OUTPUTS_DIR / 'shap_top5.csv'}")

    return final_clf


if __name__ == "__main__":
    run()

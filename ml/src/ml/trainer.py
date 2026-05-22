"""
Train all three InverterShield models and save artifacts.

Input:  processed/featured_data.parquet  (with target_binary + target_multiclass)
Output: models/binary_model.joblib
        models/multiclass_model.joblib
        models/anomaly_model.joblib
        models/feature_names.joblib
        models/metrics.json
        healthy_baseline.json
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    FEATURED_PARQUET, MODELS_DIR, OUTPUTS_DIR, BASE_DIR,
    SEED, TRAIN_FRAC, N_CV_FOLDS,
    ISOLATION_CONTAMINATION, ISOLATION_N_ESTIMATORS,
    XGB_PARAMS_SHARED, EXCLUDE_COLS, KEY_FEATURES,
)
from utils import log_section, log_step, Timer


def run():
    log_section("InverterShield — Model Training")

    # ── 1. Load featured data ──────────────────────────────────────
    log_step(f"Loading {FEATURED_PARQUET.name} ...")
    df = pd.read_parquet(FEATURED_PARQUET)
    log_step(f"Loaded {len(df):,} rows x {len(df.columns)} cols")

    _check_required_columns(df)

    # ── 2. Feature columns ─────────────────────────────────────────
    exclude = set(EXCLUDE_COLS)
    feature_cols = [c for c in df.columns if c not in exclude]
    log_step(f"Feature columns: {len(feature_cols)}")

    # ── 3. Time-aware train/test split ────────────────────────────
    df_sorted  = df.sort_values("datetime" if "datetime" in df.columns else df.columns[0])
    split_idx  = int(len(df_sorted) * TRAIN_FRAC)
    train_df   = df_sorted.iloc[:split_idx]
    test_df    = df_sorted.iloc[split_idx:]
    log_step(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["target_binary"].values
    X_test  = test_df[feature_cols].fillna(0).values
    y_test  = test_df["target_binary"].values

    # ── 4. Anomaly model (IsolationForest) ────────────────────────
    log_step("Training Isolation Forest ...")
    with Timer():
        anomaly_model = IsolationForest(
            n_estimators  = ISOLATION_N_ESTIMATORS,
            contamination = ISOLATION_CONTAMINATION,
            random_state  = SEED,
            n_jobs        = -1,
        )
        anomaly_model.fit(X_train)

    anomaly_score_train = anomaly_model.decision_function(X_train)
    is_anomaly_train    = anomaly_model.predict(X_train)          # -1 or 1

    # Augment training features with anomaly signals
    X_train_aug = np.column_stack([X_train, anomaly_score_train, is_anomaly_train])
    X_test_aug  = np.column_stack([
        X_test,
        anomaly_model.decision_function(X_test),
        anomaly_model.predict(X_test),
    ])
    feature_names_aug = feature_cols + ["anomaly_score", "is_anomaly"]

    # ── 5. Binary XGBoost with TimeSeriesSplit CV ─────────────────
    log_step("Training Binary XGBoost (5-fold TimeSeriesSplit CV) ...")
    neg_count       = (y_train == 0).sum()
    pos_count       = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    binary_params = {
        **XGB_PARAMS_SHARED,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
    }

    tscv      = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    cv_metrics = []

    with Timer():
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_aug), 1):
            X_tr, X_val = X_train_aug[tr_idx], X_train_aug[val_idx]
            y_tr, y_val = y_train[tr_idx],     y_train[val_idx]

            clf = XGBClassifier(**binary_params)
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = clf.predict(X_val)
            y_prob = clf.predict_proba(X_val)[:, 1]
            cv_metrics.append({
                "fold":      fold,
                "f1":        round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
                "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
                "recall":    round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
                "roc_auc":   round(float(roc_auc_score(y_val, y_prob)), 4),
            })
            log_step(
                f"  Fold {fold}: F1={cv_metrics[-1]['f1']:.3f}  "
                f"P={cv_metrics[-1]['precision']:.3f}  "
                f"R={cv_metrics[-1]['recall']:.3f}  "
                f"AUC={cv_metrics[-1]['roc_auc']:.3f}"
            )

    log_step("Retraining binary model on full training set ...")
    binary_model = XGBClassifier(**binary_params)
    binary_model.fit(X_train_aug, y_train, verbose=False)

    # Test-set evaluation
    y_pred_te = binary_model.predict(X_test_aug)
    y_prob_te = binary_model.predict_proba(X_test_aug)[:, 1]
    test_metrics = {
        "f1":        round(float(f1_score(y_test, y_pred_te, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred_te, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred_te, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_prob_te)), 4),
    }
    log_step(
        f"Test: F1={test_metrics['f1']:.3f}  P={test_metrics['precision']:.3f}  "
        f"R={test_metrics['recall']:.3f}  AUC={test_metrics['roc_auc']:.3f}"
    )

    # ── 6. Multi-class XGBoost ─────────────────────────────────────
    log_step("Training Multi-class XGBoost ...")
    y_multi  = train_df["target_multiclass"].values
    weights  = compute_sample_weight("balanced", y_multi)

    multi_params = {
        **XGB_PARAMS_SHARED,
        "objective":   "multi:softprob",
        "num_class":   3,
        "eval_metric": "mlogloss",
    }

    multiclass_model = XGBClassifier(**multi_params)
    with Timer():
        multiclass_model.fit(X_train_aug, y_multi, sample_weight=weights, verbose=False)

    y_pred_multi = multiclass_model.predict(X_test_aug)
    y_test_multi = test_df["target_multiclass"].values
    multi_f1     = round(float(f1_score(y_test_multi, y_pred_multi, average="weighted", zero_division=0)), 4)
    log_step(f"Multi-class weighted F1 (test): {multi_f1:.3f}")

    # ── 7. SHAP explainability ─────────────────────────────────────
    log_step("Computing SHAP values ...")
    max_samples = min(5000, len(X_test_aug))
    rng         = np.random.RandomState(SEED)
    idx_sample  = rng.choice(len(X_test_aug), max_samples, replace=False)
    X_shap      = X_test_aug[idx_sample]

    with Timer():
        explainer   = shap.TreeExplainer(binary_model)
        shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        mean_abs = np.mean(np.abs(np.stack(shap_values)), axis=(0, 1))
    elif shap_values.ndim == 3:
        mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        mean_abs = np.mean(np.abs(shap_values), axis=0)

    top10_idx        = np.argsort(mean_abs)[::-1][:10]
    top_shap_features = [feature_names_aug[i] for i in top10_idx]

    log_step("Top 10 features by mean |SHAP|:")
    for rank, i in enumerate(top10_idx, 1):
        log_step(f"  {rank:2d}. {feature_names_aug[i]:45s}  {mean_abs[i]:.4f}")

    # SHAP bar chart
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names_aug[i] for i in top10_idx][::-1],
        [mean_abs[i] for i in top10_idx][::-1],
        color="#4C72B0",
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Top 10 Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    bar_path = OUTPUTS_DIR / "shap_summary.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    log_step(f"Saved SHAP bar chart -> {bar_path}")

    # ── 8. Save artifacts ─────────────────────────────────────────
    log_step("Saving model artifacts ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(binary_model,     MODELS_DIR / "binary_model.joblib")
    joblib.dump(multiclass_model, MODELS_DIR / "multiclass_model.joblib")
    joblib.dump(anomaly_model,    MODELS_DIR / "anomaly_model.joblib")
    joblib.dump(feature_names_aug, MODELS_DIR / "feature_names.joblib")

    metrics = {
        "cv_binary":          cv_metrics,
        "test_binary":        test_metrics,
        "test_multiclass_f1": multi_f1,
        "top_shap_features":  top_shap_features,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log_step(f"Models saved to {MODELS_DIR}")

    # ── 9. Healthy baseline ────────────────────────────────────────
    _save_healthy_baseline(df, BASE_DIR)

    log_section("Training complete")
    return binary_model, multiclass_model, anomaly_model


def _check_required_columns(df: pd.DataFrame):
    required = {"target_binary", "target_multiclass"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Featured parquet is missing columns: {missing}\n"
            "Run src/data_processing/features.py to build targets, "
            "or provide a parquet that already includes them."
        )


def _save_healthy_baseline(df: pd.DataFrame, base_dir: Path):
    from config import KEY_FEATURES
    healthy   = df[df["target_binary"] == 0]
    available = [c for c in KEY_FEATURES if c in healthy.columns]
    baseline  = healthy[available].median().round(5).to_dict()
    path      = base_dir / "healthy_baseline.json"
    path.write_text(json.dumps(baseline, indent=2))
    log_step(f"Healthy baseline saved -> {path}")


if __name__ == "__main__":
    run()

"""LightGBM + Optuna + MLflow."""
import sys, os
import numpy as np
import lightgbm as lgb
import optuna, mlflow, joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import N_OPTUNA_TRIALS, OPTUNA_TIMEOUT, CV_FOLDS, RANDOM_STATE, NUM_CLASSES, MODELS_DIR
from model_training.class_balancer import apply_smote, compute_class_weights
from mlops.mlflow_setup import init_mlflow
from mlops.tracking import log_optuna_study, log_model


def _objective(trial, X, y):
    params = {
        "objective": "multiclass", "num_class": NUM_CLASSES, "metric": "multi_logloss",
        "boosting_type": "gbdt", "verbosity": -1, "n_jobs": -1, "random_state": RANDOM_STATE,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    cw = compute_class_weights(y)
    sw = np.array([cw[int(v)] for v in y])
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        Xtr, Xv = X[ti], X[vi]
        ytr, yv = y[ti], y[vi]
        Xtr, ytr = apply_smote(Xtr, ytr, RANDOM_STATE + fold)
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        scores.append(f1_score(yv, m.predict(Xv), average="macro"))
    return np.mean(scores)


def train_lightgbm(X_train, y_train, X_test, y_test, feature_names):
    init_mlflow()
    with mlflow.start_run(run_name="lightgbm_optuna"):
        print(f"\n{'='*60}\n  LightGBM — Optuna ({N_OPTUNA_TRIALS} trials)\n{'='*60}")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: _objective(t, X_train, y_train),
                       n_trials=N_OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)
        print(f"\n  Best F1: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")
        log_optuna_study(study)

        bp = {**study.best_params, "objective": "multiclass", "num_class": NUM_CLASSES,
              "metric": "multi_logloss", "boosting_type": "gbdt", "verbosity": -1,
              "n_jobs": -1, "random_state": RANDOM_STATE}
        # Split a temporal 10% validation set for final early stopping before SMOTE to keep validation pure
        from sklearn.model_selection import train_test_split
        X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train)
        
        Xs, ys = apply_smote(X_tr_f, y_tr_f)
        model = lgb.LGBMClassifier(**bp)
        model.fit(Xs, ys, eval_set=[(X_val_f, y_val_f)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        log_model(model, "lightgbm_model", "lightgbm")
        p = os.path.join(MODELS_DIR, "lightgbm_best.pkl")
        joblib.dump(model, p); mlflow.log_artifact(p)
        mlflow.log_metric("best_cv_f1_macro", study.best_value)

        imp = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
        print("\n  Top 10 features:")
        for fn, iv in imp[:10]: print(f"    {fn:>40}: {iv:>6}")
    return model, study

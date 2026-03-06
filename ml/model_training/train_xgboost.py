"""XGBoost + Optuna + MLflow."""
import sys, os
import numpy as np
import xgboost as xgb
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
        "objective": "multi:softprob", "num_class": NUM_CLASSES, "eval_metric": "mlogloss",
        "tree_method": "hist", "verbosity": 0, "n_jobs": -1, "random_state": RANDOM_STATE,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }
    cw = compute_class_weights(y)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        Xtr, ytr = apply_smote(X[ti], y[ti], RANDOM_STATE + fold)
        m = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        m.fit(Xtr, ytr, eval_set=[(X[vi], y[vi])], verbose=False)
        scores.append(f1_score(y[vi], m.predict(X[vi]), average="macro"))
    return np.mean(scores)


def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    init_mlflow()
    with mlflow.start_run(run_name="xgboost_optuna"):
        print(f"\n{'='*60}\n  XGBoost — Optuna ({N_OPTUNA_TRIALS} trials)\n{'='*60}")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: _objective(t, X_train, y_train),
                       n_trials=N_OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)
        print(f"\n  Best F1: {study.best_value:.4f}")
        log_optuna_study(study)

        bp = {**study.best_params, "objective": "multi:softprob", "num_class": NUM_CLASSES,
              "eval_metric": "mlogloss", "tree_method": "hist", "verbosity": 0,
              "n_jobs": -1, "random_state": RANDOM_STATE}
        # Split a 10% validation set for final early stopping before SMOTE
        from sklearn.model_selection import train_test_split
        X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train)

        Xs, ys = apply_smote(X_tr_f, y_tr_f)
        model = xgb.XGBClassifier(**bp, early_stopping_rounds=50)
        model.fit(Xs, ys, eval_set=[(X_val_f, y_val_f)], verbose=False)
        log_model(model, "xgboost_model", "xgboost")
        p = os.path.join(MODELS_DIR, "xgboost_best.pkl")
        joblib.dump(model, p); mlflow.log_artifact(p)
        mlflow.log_metric("best_cv_f1_macro", study.best_value)
    return model, study

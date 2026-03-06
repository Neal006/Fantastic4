"""MLflow tracking helpers."""
import mlflow, mlflow.sklearn
import numpy as np, os


def log_training_params(params, prefix=""):
    for k, v in params.items(): mlflow.log_param(f"{prefix}{k}", v)

def log_cv_metrics(metrics, step=None):
    for k, v in metrics.items():
        if isinstance(v, (list, np.ndarray)):
            mlflow.log_metric(f"{k}_mean", float(np.mean(v)), step=step)
            mlflow.log_metric(f"{k}_std", float(np.std(v)), step=step)
        else:
            mlflow.log_metric(k, float(v), step=step)

def log_classification_metrics(metrics, prefix=""):
    for k, v in metrics.items(): mlflow.log_metric(f"{prefix}{k}", float(v))

def log_model(model, name, model_type="sklearn"):
    if model_type == "lightgbm": import mlflow.lightgbm; mlflow.lightgbm.log_model(model, name)
    elif model_type == "xgboost": import mlflow.xgboost; mlflow.xgboost.log_model(model, name)
    elif model_type == "pytorch": import mlflow.pytorch; mlflow.pytorch.log_model(model, name)
    else: mlflow.sklearn.log_model(model, name)

def log_artifact_file(fp):
    if os.path.exists(fp): mlflow.log_artifact(fp)

def log_optuna_study(study):
    mlflow.log_param("optuna_n_trials", len(study.trials))
    mlflow.log_param("optuna_best_trial", study.best_trial.number)
    mlflow.log_metric("optuna_best_value", study.best_value)
    for k, v in study.best_params.items(): mlflow.log_param(f"best_{k}", v)

"""MLflow setup & experiment config."""
import mlflow
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT


def init_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"  MLflow: {MLFLOW_EXPERIMENT} (ID: {exp.experiment_id})")
    return exp

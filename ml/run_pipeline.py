"""
Solar Inverter ML Pipeline Orchestrator.
Usage:
  python run_pipeline.py --stage features
  python run_pipeline.py --stage train
  python run_pipeline.py --stage evaluate
  python run_pipeline.py --stage explain
  python run_pipeline.py --stage all
"""
import argparse, sys, os, time, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import MODELS_DIR, FEATURES_DIR, PLANT_FILES

def stage_features(plants=None):
    from feature_engineering.build_features import build_features_for_plant
    from anomaly_detection.isolation_forest import add_isolation_forest_features
    from anomaly_detection.statistical_detector import add_statistical_anomaly_features
    plants = plants or list(PLANT_FILES.keys())
    for n in plants:
        df = build_features_for_plant(n)
        print("  Adding anomaly features...")
        df = add_isolation_forest_features(df, n)
        df = add_statistical_anomaly_features(df, n)
        df.to_parquet(os.path.join(FEATURES_DIR, f"{n}_features.parquet"), index=False, engine="pyarrow")
        print(f"  ✓ Updated with anomaly features: {len(df.columns)} cols")

def stage_train(plants=None):
    from model_training.data_splits import load_feature_data, split_data
    from model_training.train_lightgbm import train_lightgbm
    from model_training.train_xgboost import train_xgboost
    from model_training.train_lstm import train_lstm
    from model_training.ensemble import train_ensemble
    print("\n  Loading data..."); df = load_feature_data(plants)
    X_train, X_test, y_train, y_test, feats = split_data(df)
    joblib.dump({"feature_names": feats, "X_test": X_test, "y_test": y_test}, os.path.join(MODELS_DIR, "split_info.pkl"))
    lgbm, _ = train_lightgbm(X_train, y_train, X_test, y_test, feats)
    xgb_m, _ = train_xgboost(X_train, y_train, X_test, y_test, feats)
    lstm, _ = train_lstm(X_train, y_train, X_test, y_test, feats)
    train_ensemble(lgbm, xgb_m, lstm, X_train, y_train, X_test, y_test, feats)

def stage_evaluate():
    from evaluation.metrics import compute_all_metrics, print_metrics
    from evaluation.reports import generate_all_reports
    from mlops.tracking import log_classification_metrics, log_artifact_file
    from mlops.mlflow_setup import init_mlflow
    import mlflow, torch, numpy as np
    from config.settings import NUM_CLASSES, LSTM_SEQ_LENGTH
    from model_training.train_lstm import create_sequences, InverterLSTM
    
    info = joblib.load(os.path.join(MODELS_DIR, "split_info.pkl"))
    init_mlflow()
    
    models = {}
    for name, path in {"LightGBM": "lightgbm_best.pkl", "XGBoost": "xgboost_best.pkl", "Ensemble": "ensemble_meta.pkl"}.items():
        fp = os.path.join(MODELS_DIR, path)
        if os.path.exists(fp): models[name] = joblib.load(fp)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_fp = os.path.join(MODELS_DIR, "lstm_best.pt")
    lstm_cfg_fp = os.path.join(MODELS_DIR, "lstm_config.pkl")
    if os.path.exists(lstm_fp) and os.path.exists(lstm_cfg_fp):
        cfg = joblib.load(lstm_cfg_fp)
        lstm = InverterLSTM(cfg["input_size"], cfg["hidden_size"], cfg["num_layers"], cfg["num_classes"], cfg["dropout"])
        lstm.load_state_dict(torch.load(lstm_fp, map_location=device))
        lstm.to(device)
        models["LSTM"] = lstm
        
    def _get_lstm_proba(m, X):
        Xs, _ = create_sequences(X, np.zeros(len(X)), LSTM_SEQ_LENGTH)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(Xs)), batch_size=256)
        m.eval(); all_p = []
        with torch.no_grad():
            for (b,) in loader: all_p.append(torch.softmax(m(b.to(device)), 1).cpu().numpy())
        p = np.vstack(all_p)
        pad = len(X) - len(p)
        if pad > 0: p = np.vstack([np.full((pad, NUM_CLASSES), 1/NUM_CLASSES, np.float32), p])
        return p

    probas = {}
    for name in ["LightGBM", "XGBoost", "LSTM", "Ensemble"]:  # specific order for ensemble
        if name not in models: continue
        m = models[name]
        if name == "Ensemble":
            if len(probas) < 3:
                print("  Skipping Ensemble eval (missing base models)"); continue
            meta_Xt = np.hstack([probas["LightGBM"], probas["XGBoost"], probas["LSTM"]])
            yp = m.predict(meta_Xt); yprob = m.predict_proba(meta_Xt)
        elif name == "LSTM":
            yprob = _get_lstm_proba(m, info["X_test"]); yp = yprob.argmax(axis=1)
        else:
            yp = m.predict(info["X_test"]); yprob = m.predict_proba(info["X_test"])
        
        probas[name] = yprob
        mets = compute_all_metrics(info["y_test"], yp, yprob)
        print_metrics(mets, name)
        with mlflow.start_run(run_name=f"eval_{name.lower()}"):
            log_classification_metrics(mets, f"{name.lower()}_")
            for rp in generate_all_reports(info["y_test"], yp, yprob, name): log_artifact_file(rp)

def stage_explain():
    from explainability.shap_analysis import run_shap_analysis
    from mlops.mlflow_setup import init_mlflow
    from mlops.tracking import log_artifact_file
    import mlflow
    info = joblib.load(os.path.join(MODELS_DIR, "split_info.pkl"))
    init_mlflow()
    for name, path in {"LightGBM": "lightgbm_best.pkl", "XGBoost": "xgboost_best.pkl"}.items():
        fp = os.path.join(MODELS_DIR, path)
        if not os.path.exists(fp): continue
        with mlflow.start_run(run_name=f"shap_{name.lower()}"):
            paths, _ = run_shap_analysis(joblib.load(fp), info["X_test"], info["feature_names"], name, 8)
            for p in paths: log_artifact_file(p)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["features", "train", "evaluate", "explain", "all"])
    p.add_argument("--plant", default=None)
    args = p.parse_args(); t0 = time.time()
    print(f"\n{'='*60}\n  Pipeline: {args.stage}\n{'='*60}")
    plants = [args.plant] if args.plant else None
    if args.stage in ("features", "all"): stage_features(plants)
    if args.stage in ("train", "all"): stage_train(plants)
    if args.stage in ("evaluate", "all"): stage_evaluate()
    if args.stage in ("explain", "all"): stage_explain()
    print(f"\n{'='*60}\n  Complete! ({time.time()-t0:.0f}s)\n{'='*60}")

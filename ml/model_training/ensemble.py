"""Stacking ensemble: LightGBM + XGBoost + LSTM → LogReg meta-learner."""
import sys, os
import numpy as np
import joblib, mlflow
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import NUM_CLASSES, RANDOM_STATE, CV_FOLDS, MODELS_DIR, LSTM_SEQ_LENGTH
from model_training.train_lstm import InverterLSTM, create_sequences
from mlops.mlflow_setup import init_mlflow
from mlops.tracking import log_model


def _oof_tree(model, X, y):
    from model_training.class_balancer import apply_smote
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(X), NUM_CLASSES), dtype=np.float32)
    for ti, vi in skf.split(X, y):
        clone = model.__class__(**model.get_params())
        X_sm, y_sm = apply_smote(X[ti], y[ti], RANDOM_STATE)
        if hasattr(clone, "early_stopping_rounds") or "xgb" in str(type(clone)).lower():
            clone.fit(X_sm, y_sm, eval_set=[(X[vi], y[vi])], verbose=False)
        else:
             import lightgbm as lgb
             clone.fit(X_sm, y_sm, eval_set=[(X[vi], y[vi])], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[vi] = clone.predict_proba(X[vi])
    return oof


def _oof_lstm(base_model, X, y, device):
    skf = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(X), NUM_CLASSES), dtype=np.float32)
    hs = base_model.lstm.hidden_size
    nl = base_model.lstm.num_layers
    do = base_model.drop.p
    ins = base_model.lstm.input_size
    
    Xs_all, ys_all = create_sequences(X, y, LSTM_SEQ_LENGTH)
    for ti, vi in skf.split(Xs_all, ys_all):
        m = InverterLSTM(ins, hs, nl, NUM_CLASSES, do).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        tl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.FloatTensor(Xs_all[ti]), torch.LongTensor(ys_all[ti])), 
            batch_size=256, shuffle=True)
        m.train()
        for _ in range(5):  # 5 epochs for OOF proxy
            for Xb, yb in tl:
                opt.zero_grad(); crit(m(Xb.to(device)), yb.to(device)).backward(); opt.step()
        m.eval()
        vl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(Xs_all[vi])), batch_size=256)
        all_p = []
        with torch.no_grad():
            for (b,) in vl: all_p.append(torch.softmax(m(b.to(device)), 1).cpu().numpy())
        idx_mapped = vi + LSTM_SEQ_LENGTH - 1
        oof[idx_mapped] = np.vstack(all_p)
        
    oof[:LSTM_SEQ_LENGTH-1] = 1.0 / NUM_CLASSES
    return oof


def train_ensemble(lgbm, xgb_m, lstm, X_train, y_train, X_test, y_test, feat_names):
    init_mlflow()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with mlflow.start_run(run_name="stacking_ensemble"):
        print(f"\n{'='*60}\n  Stacking Ensemble\n{'='*60}")
        print("  OOF: LightGBM..."); lg_oof = _oof_tree(lgbm, X_train, y_train)
        print("  OOF: XGBoost..."); xg_oof = _oof_tree(xgb_m, X_train, y_train)
        print("  OOF: LSTM (5 epochs)..."); ls_oof = _oof_lstm(lstm, X_train, y_train, device)
        meta_X = np.hstack([lg_oof, xg_oof, ls_oof])
        print(f"  Meta features: {meta_X.shape}")

        meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                   multi_class="multinomial", C=0.1, class_weight="balanced")  # Reg C=0.1 to avoid overfit
        meta.fit(meta_X, y_train)
        print(f"  Train F1: {f1_score(y_train, meta.predict(meta_X), average='macro'):.4f}")

        # For test, we use the fully trained LSTM directly via a fast inference wrapper
        def _lstm_infer(m, X, dev):
            Xs, _ = create_sequences(X, np.zeros(len(X)), LSTM_SEQ_LENGTH)
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(Xs)), batch_size=256)
            m.eval(); all_p = []
            with torch.no_grad():
                for (b,) in loader: all_p.append(torch.softmax(m(b.to(dev)), 1).cpu().numpy())
            p = np.vstack(all_p)
            pad = len(X) - len(p)
            if pad > 0: p = np.vstack([np.full((pad, NUM_CLASSES), 1/NUM_CLASSES, np.float32), p])
            return p
            
        meta_Xt = np.hstack([lgbm.predict_proba(X_test), xgb_m.predict_proba(X_test),
                             _lstm_infer(lstm, X_test, device)])
        test_f1 = f1_score(y_test, meta.predict(meta_Xt), average="macro")
        print(f"  Test F1:  {test_f1:.4f}")
        mlflow.log_metric("test_f1_macro", test_f1)

        p = os.path.join(MODELS_DIR, "ensemble_meta.pkl")
        joblib.dump(meta, p); mlflow.log_artifact(p)
        log_model(meta, "ensemble_meta", "sklearn")
    return meta

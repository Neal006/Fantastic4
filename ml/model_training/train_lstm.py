"""LSTM temporal model + Optuna + MLflow."""
import sys, os
import numpy as np
import optuna, mlflow, joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import N_OPTUNA_TRIALS, OPTUNA_TIMEOUT, RANDOM_STATE, NUM_CLASSES, MODELS_DIR, LSTM_SEQ_LENGTH
from mlops.mlflow_setup import init_mlflow
from mlops.tracking import log_optuna_study


class InverterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


def create_sequences(X, y, seq_len=LSTM_SEQ_LENGTH):
    n, nf = X.shape
    ns = n - seq_len + 1
    if ns <= 0: raise ValueError(f"Too few samples ({n}) for seq_len={seq_len}")
    Xs = np.zeros((ns, seq_len, nf), dtype=np.float32)
    ys = np.zeros(ns, dtype=np.int64)
    for i in range(ns):
        Xs[i] = X[i:i+seq_len]
        ys[i] = y[i+seq_len-1]
    return Xs, ys


def _train_epoch(model, loader, opt, crit, device):
    model.train(); total = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad(); loss = crit(model(Xb), yb); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total += loss.item()
    return total / len(loader)


def _eval(model, loader, device):
    model.eval(); preds, true = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            p = model(Xb.to(device)).argmax(1).cpu().numpy()
            preds.extend(p); true.extend(yb.numpy())
    return np.array(preds), np.array(true)


def _objective(trial, X, y, input_size):
    hs = trial.suggest_int("hidden_size", 32, 256)
    nl = trial.suggest_int("num_layers", 1, 3)
    do = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [64, 128, 256])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xs, ys = create_sequences(X, y)
    Xtr, Xv, ytr, yv = train_test_split(Xs, ys, test_size=0.2, random_state=RANDOM_STATE, stratify=ys)

    cc = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float32)
    cc = np.maximum(cc, 1)
    wt = torch.FloatTensor(len(ytr) / (NUM_CLASSES * cc)).to(device)
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.LongTensor(ytr)), bs, shuffle=True)
    vl = DataLoader(TensorDataset(torch.FloatTensor(Xv), torch.LongTensor(yv)), bs)

    m = InverterLSTM(input_size, hs, nl, NUM_CLASSES, do).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=wt)
    best, patience = 0, 0
    for ep in range(20):
        _train_epoch(m, tl, opt, crit, device)
        p, t = _eval(m, vl, device)
        f1 = f1_score(t, p, average="macro")
        if f1 > best: best = f1; patience = 0
        else:
            patience += 1
            if patience >= 5: break
        trial.report(f1, ep)
        if trial.should_prune(): raise optuna.TrialPruned()
    return best


def train_lstm(X_train, y_train, X_test, y_test, feature_names):
    init_mlflow()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[1]
    with mlflow.start_run(run_name="lstm_optuna"):
        print(f"\n{'='*60}\n  LSTM — Optuna (device: {device})\n{'='*60}")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                     pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
        study.optimize(lambda t: _objective(t, X_train, y_train, input_size),
                       n_trials=min(N_OPTUNA_TRIALS, 30), timeout=OPTUNA_TIMEOUT, show_progress_bar=True)
        print(f"\n  Best F1: {study.best_value:.4f}")
        log_optuna_study(study)

        bp = study.best_params
        Xs, ys = create_sequences(X_train, y_train)
        cc = np.bincount(ys, minlength=NUM_CLASSES).astype(np.float32)
        cc = np.maximum(cc, 1)
        wt = torch.FloatTensor(len(ys) / (NUM_CLASSES * cc)).to(device)
        tl = DataLoader(TensorDataset(torch.FloatTensor(Xs), torch.LongTensor(ys)), bp["batch_size"], shuffle=True)
        model = InverterLSTM(input_size, bp["hidden_size"], bp["num_layers"], NUM_CLASSES, bp["dropout"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=bp["lr"])
        crit = nn.CrossEntropyLoss(weight=wt)
        for ep in range(50):
            loss = _train_epoch(model, tl, opt, crit, device)
            if (ep+1) % 10 == 0: print(f"    Epoch {ep+1}/50 loss: {loss:.4f}")

        p = os.path.join(MODELS_DIR, "lstm_best.pt")
        torch.save(model.state_dict(), p); mlflow.log_artifact(p)
        joblib.dump({"input_size": input_size, "hidden_size": bp["hidden_size"],
                     "num_layers": bp["num_layers"], "num_classes": NUM_CLASSES,
                     "dropout": bp["dropout"], "seq_length": LSTM_SEQ_LENGTH},
                    os.path.join(MODELS_DIR, "lstm_config.pkl"))
        mlflow.log_metric("best_cv_f1_macro", study.best_value)
    return model, study

"""
Stage 7b – Train LSTM
======================
Sequence-based LSTM model for multi-class risk prediction.
Reshapes flat features into sliding windows of 288 steps (24 h).

Input  → ``processed/splits.pkl``
Output → ``models/lstm_best.keras``
"""

import sys, os
from pathlib import Path

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_DIR, MODELS_DIR, SEED,
    LSTM_SEQ_LEN, LSTM_UNITS_1, LSTM_UNITS_2, LSTM_DROPOUT,
    LSTM_DENSE, LSTM_EPOCHS, LSTM_BATCH, LSTM_PATIENCE,
    CLASS_NAMES,
)
from utils import log_section, log_step, save_pickle, load_pickle, Timer


tf.random.set_seed(SEED)


# ── Sequence helpers ─────────────────────────────────────────────
def _to_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Convert flat (N, F) to (N-seq_len+1, seq_len, F) sliding windows."""
    if len(X) <= seq_len:
        # Not enough data; pad with zeros
        pad_len = seq_len - len(X) + 1
        X = np.vstack([np.zeros((pad_len, X.shape[1]), dtype=X.dtype), X])
        y = np.concatenate([np.zeros(pad_len, dtype=y.dtype), y])
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys)


def _build_model(n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(LSTM_SEQ_LEN, n_features)),
        keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True),
        keras.layers.Dropout(LSTM_DROPOUT),
        keras.layers.LSTM(LSTM_UNITS_2),
        keras.layers.Dropout(LSTM_DROPOUT),
        keras.layers.Dense(LSTM_DENSE, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── main ─────────────────────────────────────────────────────────
def run():
    log_section("Stage 7b · Train LSTM")

    splits = load_pickle(PROCESSED_DIR / "splits.pkl")
    n_classes = len(CLASS_NAMES)
    n_features = splits["X_trainval"].shape[1]

    # Use a shorter sequence length if dataset is small
    seq_len = min(LSTM_SEQ_LEN, len(splits["X_trainval"]) // 4)
    seq_len = max(seq_len, 16)  # minimum 16 steps
    log_step(f"Sequence length: {seq_len} | Features: {n_features}")

    # Convert to sequences
    log_step("Building sequences for train …")
    with Timer():
        X_train_seq, y_train_seq = _to_sequences(splits["X_trainval"], splits["y_trainval"], seq_len)
    log_step(f"Train sequences: {X_train_seq.shape}")

    log_step("Building sequences for test …")
    with Timer():
        X_test_seq, y_test_seq = _to_sequences(splits["X_test"], splits["y_test"], seq_len)
    log_step(f"Test sequences: {X_test_seq.shape}")

    # Build & train
    model = _build_model(n_features, n_classes)
    model.summary(print_fn=lambda x: log_step(x))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=LSTM_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    log_step("Training LSTM …")
    with Timer():
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH,
            callbacks=callbacks,
            verbose=1,
        )

    # Evaluate
    y_prob = model.predict(X_test_seq, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    p = precision_score(y_test_seq, y_pred, average="macro", zero_division=0)
    r = recall_score(y_test_seq, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test_seq, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_test_seq, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    log_step(f"LSTM Test ─ P={p:.3f}  R={r:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

    # Save
    model_path = MODELS_DIR / "lstm_best.keras"
    model.save(model_path)
    log_step(f"Saved LSTM model → {model_path}")

    # Save predictions for ensemble
    save_pickle(
        {"y_prob": y_prob, "y_true": y_test_seq, "seq_len": seq_len},
        MODELS_DIR / "lstm_test_preds.pkl",
        "LSTM test predictions",
    )
    return model


if __name__ == "__main__":
    run()

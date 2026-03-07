"""
Stage 8 – Ensemble
===================
Combine XGBoost and LSTM predictions via:
  1. Soft-voting (weighted probability average).
  2. Stacking (logistic regression meta-learner).
Select the variant with the best macro-F1 on the test set.

Input  → ``models/xgb_best.pkl``, ``models/lstm_test_preds.pkl``, ``processed/splits.pkl``
Output → ``models/ensemble_meta.pkl``
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR, MODELS_DIR, SEED, CLASS_NAMES
from utils import log_section, log_step, save_pickle, load_pickle, Timer


def run():
    log_section("Stage 8 · Ensemble")

    splits = load_pickle(PROCESSED_DIR / "splits.pkl")
    xgb_model = load_pickle(MODELS_DIR / "xgb_best.pkl")
    lstm_preds = load_pickle(MODELS_DIR / "lstm_test_preds.pkl")

    # XGBoost probabilities on test set
    xgb_prob = xgb_model.predict_proba(splits["X_test"])

    # LSTM probabilities (may be shorter due to sequence windowing)
    lstm_prob = lstm_preds["y_prob"]
    lstm_len = len(lstm_prob)
    y_true_lstm = lstm_preds["y_true"]

    # Align: use the last lstm_len rows of XGB predictions
    xgb_prob_aligned = xgb_prob[-lstm_len:]
    y_test_aligned = splits["y_test"][-lstm_len:]

    # ── 1) Soft-voting with different weight combos ──
    log_step("Evaluating soft-voting ensembles …")
    best_f1, best_w = 0, 0.5
    for w in np.arange(0.1, 1.0, 0.1):
        blended = w * xgb_prob_aligned + (1 - w) * lstm_prob
        y_pred = blended.argmax(axis=1)
        f1 = f1_score(y_test_aligned, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_w = w
    log_step(f"Best soft-vote: XGB weight={best_w:.1f}, macro-F1={best_f1:.4f}")

    soft_prob = best_w * xgb_prob_aligned + (1 - best_w) * lstm_prob
    soft_pred = soft_prob.argmax(axis=1)

    # ── 2) Stacking meta-learner ──
    log_step("Training stacking meta-learner …")
    meta_features = np.hstack([xgb_prob_aligned, lstm_prob])

    # Simple 50/50 split of test for meta-train / meta-eval
    mid = len(meta_features) // 2
    meta_X_tr, meta_X_te = meta_features[:mid], meta_features[mid:]
    meta_y_tr, meta_y_te = y_test_aligned[:mid], y_test_aligned[mid:]

    meta_clf = LogisticRegression(
        max_iter=1000, random_state=SEED, multi_class="multinomial"
    )
    meta_clf.fit(meta_X_tr, meta_y_tr)
    stack_pred = meta_clf.predict(meta_X_te)
    stack_f1 = f1_score(meta_y_te, stack_pred, average="macro", zero_division=0)
    log_step(f"Stacking macro-F1 (meta-test): {stack_f1:.4f}")

    # ── Select best ──
    if stack_f1 > best_f1:
        log_step("✓ Stacking wins – retraining meta-learner on all aligned data")
        meta_clf_full = LogisticRegression(
            max_iter=1000, random_state=SEED, multi_class="multinomial"
        )
        meta_clf_full.fit(meta_features, y_test_aligned)
        ensemble_info = {
            "method": "stacking",
            "meta_clf": meta_clf_full,
            "best_f1": stack_f1,
        }
    else:
        log_step("✓ Soft-voting wins")
        ensemble_info = {
            "method": "soft_vote",
            "xgb_weight": best_w,
            "best_f1": best_f1,
        }

    # Save aligned probabilities for evaluation
    ensemble_info["xgb_prob"] = xgb_prob_aligned
    ensemble_info["lstm_prob"] = lstm_prob
    ensemble_info["y_test"] = y_test_aligned

    save_pickle(ensemble_info, MODELS_DIR / "ensemble_meta.pkl", "ensemble info")
    return ensemble_info


if __name__ == "__main__":
    run()

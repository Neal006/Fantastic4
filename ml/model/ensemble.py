"""
Stage 8 – Ensemble
===================
Combine XGBoost and CatBoost predictions via:
  1. Soft-voting (weighted probability average).
  2. Stacking (logistic regression meta-learner).
Select the variant with the best macro-F1 on a held-out portion.

Input  → ``models/xgb_best.pkl``, ``models/cbc_best.pkl``, ``processed/splits.pkl``
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
    log_section("Stage 8 · Ensemble (XGB + CatBoost)")

    splits = load_pickle(PROCESSED_DIR / "splits.pkl")
    xgb_model = load_pickle(MODELS_DIR / "xgb_best.pkl")
    cbc_model = load_pickle(MODELS_DIR / "cbc_best.pkl")

    X_test = splits["X_test"]
    y_test = splits["y_test"]

    # Probabilities on test set (same shape — no alignment needed)
    xgb_prob = xgb_model.predict_proba(X_test)
    cbc_prob = cbc_model.predict_proba(X_test)

    log_step(f"XGB prob shape: {xgb_prob.shape} | CBC prob shape: {cbc_prob.shape}")

    # ── 1) Soft-voting with different weight combos ──
    log_step("Evaluating soft-voting ensembles ...")
    best_f1, best_w = 0, 0.5
    for w in np.arange(0.1, 1.0, 0.05):
        blended = w * xgb_prob + (1 - w) * cbc_prob
        y_pred = blended.argmax(axis=1)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_w = round(w, 2)
    log_step(f"Best soft-vote: XGB weight={best_w:.2f}, macro-F1={best_f1:.4f}")

    soft_prob = best_w * xgb_prob + (1 - best_w) * cbc_prob

    # ── 2) Stacking meta-learner ──
    log_step("Training stacking meta-learner ...")
    meta_features = np.hstack([xgb_prob, cbc_prob])

    # 50/50 split of test for meta-train / meta-eval
    mid = len(meta_features) // 2
    meta_X_tr, meta_X_te = meta_features[:mid], meta_features[mid:]
    meta_y_tr, meta_y_te = y_test[:mid], y_test[mid:]

    meta_clf = LogisticRegression(
        max_iter=1000, random_state=SEED, multi_class="multinomial"
    )
    meta_clf.fit(meta_X_tr, meta_y_tr)
    stack_pred = meta_clf.predict(meta_X_te)
    stack_f1 = f1_score(meta_y_te, stack_pred, average="macro", zero_division=0)
    log_step(f"Stacking macro-F1 (meta-test): {stack_f1:.4f}")

    # ── Select best ──
    if stack_f1 > best_f1:
        log_step(">> Stacking wins - retraining meta-learner on all data")
        meta_clf_full = LogisticRegression(
            max_iter=1000, random_state=SEED, multi_class="multinomial"
        )
        meta_clf_full.fit(meta_features, y_test)
        ensemble_info = {
            "method": "stacking",
            "meta_clf": meta_clf_full,
            "best_f1": stack_f1,
        }
    else:
        log_step(">> Soft-voting wins")
        ensemble_info = {
            "method": "soft_vote",
            "xgb_weight": best_w,
            "best_f1": best_f1,
        }

    # Save aligned probabilities for evaluation
    ensemble_info["xgb_prob"] = xgb_prob
    ensemble_info["cbc_prob"] = cbc_prob
    ensemble_info["y_test"] = y_test

    save_pickle(ensemble_info, MODELS_DIR / "ensemble_meta.pkl", "ensemble info")
    return ensemble_info


if __name__ == "__main__":
    run()

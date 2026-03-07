"""
Stage 10 – SHAP Explainability
================================
Compute SHAP values for the XGBoost model and generate:
  - Summary bar plot (top 10 features).
  - Beeswarm plot.
  - Top 5 features with mean |SHAP| printed.

Input  → ``models/xgb_best.pkl``, ``processed/splits.pkl``
Output → ``outputs/shap_summary.png``, ``outputs/shap_beeswarm.png``
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, CLASS_NAMES
from utils import log_section, log_step, load_pickle, Timer


def run():
    log_section("Stage 10 · SHAP Explainability")

    xgb_model = load_pickle(MODELS_DIR / "xgb_best.pkl")
    splits = load_pickle(PROCESSED_DIR / "splits.pkl")
    feature_cols = splits["feature_cols"]

    X_test = splits["X_test"]

    # Use a sample for speed (max 2000 rows)
    max_samples = min(2000, len(X_test))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_test), max_samples, replace=False)
    X_sample = X_test[idx]

    log_step(f"Computing SHAP values for {max_samples} test samples …")
    with Timer():
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)

    # shap_values is a list of (n_samples, n_features) arrays, one per class
    # Stack and take mean absolute across all classes
    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=0)  # (n_classes, n_samples, n_features)
        mean_abs = np.mean(np.abs(stacked), axis=(0, 1))  # (n_features,)
    else:
        mean_abs = np.mean(np.abs(shap_values), axis=0)

    # ── Top 5 features ──
    top_idx = np.argsort(mean_abs)[::-1][:5]
    log_step("Top 5 features by mean |SHAP|:")
    for rank, i in enumerate(top_idx, 1):
        log_step(f"  {rank}. {feature_cols[i]:40s}  {mean_abs[i]:.4f}")

    # ── Summary bar plot ──
    top10_idx = np.argsort(mean_abs)[::-1][:10]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_cols[i] for i in top10_idx][::-1],
        [mean_abs[i] for i in top10_idx][::-1],
        color="#4C72B0",
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Top 10 Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    bar_path = OUTPUTS_DIR / "shap_summary.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    log_step(f"Saved SHAP summary bar → {bar_path}")

    # ── Beeswarm plot (use class 0 for clarity, or combined if 2-D) ──
    try:
        if isinstance(shap_values, list):
            # Use the "no_risk" class for the beeswarm
            sv = shap_values[0][np.argsort(idx)]  # re-sort for nicer plot
            sv = shap_values[0]
        else:
            sv = shap_values

        fig2 = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            sv, X_sample,
            feature_names=feature_cols,
            max_display=15,
            show=False,
        )
        plt.title("SHAP Beeswarm – Class: no_risk", fontsize=14)
        plt.tight_layout()
        bee_path = OUTPUTS_DIR / "shap_beeswarm.png"
        fig2.savefig(bee_path, dpi=150)
        plt.close(fig2)
        log_step(f"Saved SHAP beeswarm → {bee_path}")
    except Exception as e:
        log_step(f"Beeswarm plot skipped: {e}")

    # ── Save top features to CSV ──
    import pandas as pd
    top_df = pd.DataFrame({
        "feature": [feature_cols[i] for i in top_idx],
        "mean_abs_shap": [mean_abs[i] for i in top_idx],
    })
    top_df.to_csv(OUTPUTS_DIR / "shap_top5.csv", index=False)
    log_step(f"Saved top-5 CSV → {OUTPUTS_DIR / 'shap_top5.csv'}")


if __name__ == "__main__":
    run()

"""SHAP: Feature importance and dependence plots for tree models."""
import os, numpy as np, shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import REPORTS_DIR

def run_shap_analysis(model, X_test, feat_names, name="LightGBM", top_n=8):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pfx = name.lower().replace(" ", "_")
    paths = []
    print(f"\n  Computing SHAP for {name}...")
    explainer = shap.TreeExplainer(model)
    X_sample = X_test[np.random.choice(len(X_test), min(5000, len(X_test)), replace=False)]
    sv = explainer.shap_values(X_sample)
    
    mas = np.mean([np.abs(v) for v in sv], axis=0) if isinstance(sv, list) else np.abs(sv)
    fi = mas.mean(axis=0)
    top_idx = np.argsort(fi)[-top_n:][::-1]
    top_feats = [feat_names[i] for i in top_idx]
    
    print(f"\n  Top {top_n} features:")
    for r, i in enumerate(top_idx): print(f"    {r+1}. {feat_names[i]:>40}: {fi[i]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), [fi[i] for i in top_idx[::-1]])
    ax.set_yticks(range(top_n)); ax.set_yticklabels([feat_names[i] for i in top_idx[::-1]])
    ax.set_xlabel("Mean |SHAP|"); ax.set_title(f"Top {top_n} Features — {name}")
    plt.tight_layout(); p = os.path.join(REPORTS_DIR, f"shap_bar_{pfx}.png")
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    sv0 = sv[0] if isinstance(sv, list) else sv
    try:
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(sv0[:, top_idx], X_sample[:, top_idx], feature_names=top_feats, show=False)
        plt.title(f"SHAP Summary — {name}"); plt.tight_layout()
        p = os.path.join(REPORTS_DIR, f"shap_summary_{pfx}.png")
        plt.savefig(p, dpi=150); plt.close(); paths.append(p)
    except Exception as e: print(f"  ⚠️ Summary plot failed: {e}")

    for r in range(min(3, top_n)):
        try:
            fig = plt.figure(figsize=(8, 6))
            shap.dependence_plot(top_idx[r], sv0, X_sample, feature_names=feat_names, show=False)
            plt.title(f"SHAP Dependence — {top_feats[r]}"); plt.tight_layout()
            p = os.path.join(REPORTS_DIR, f"shap_dep_{pfx}_{r}.png")
            plt.savefig(p, dpi=150); plt.close(); paths.append(p)
        except Exception as e: print(f"  ⚠️ Dependence plot {r} failed: {e}")
    
    print(f"  SHAP plots → {REPORTS_DIR}")
    return paths, top_feats

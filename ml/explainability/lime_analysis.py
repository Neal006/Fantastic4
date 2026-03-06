"""LIME: Instance-level explainability (backup to SHAP)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import REPORTS_DIR, CLASS_NAMES

try: from lime.lime_tabular import LimeTabularExplainer; LIME_OK = True
except ImportError: LIME_OK = False

def explain_single_prediction(model, X_train, X_inst, feat_names, name="Model", idx=0):
    if not LIME_OK: print("  ⚠️ LIME not installed"); return None
    os.makedirs(REPORTS_DIR, exist_ok=True)
    exp = LimeTabularExplainer(X_train, feature_names=feat_names, class_names=CLASS_NAMES, mode="classification")
    ex = exp.explain_instance(X_inst, model.predict_proba, num_features=10, top_labels=3)
    p = os.path.join(REPORTS_DIR, f"lime_{name.lower().replace(' ','_')}_inst{idx}.html")
    ex.save_to_file(p); print(f"  LIME → {p}"); return p

"""Debug SHAP explainer to understand the output format."""

import numpy as np
import pickle
from pathlib import Path
import shap

# Load model
model_path = Path('models/xgb_best.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("Model type:", type(model))
print("Model classes:", model.classes_ if hasattr(model, 'classes_') else "No classes_")

# Create explainer
explainer = shap.TreeExplainer(model)
print("\nExplainer created successfully")

# Test with dummy data
X_test = np.random.randn(1, 145)
print(f"\nInput shape: {X_test.shape}")

# Get SHAP values
shap_vals = explainer.shap_values(X_test)

print(f"\nSHAP values type: {type(shap_vals)}")
if isinstance(shap_vals, list):
    print(f"SHAP values is a list with {len(shap_vals)} elements")
    for i, sv in enumerate(shap_vals):
        print(f"  Element {i} shape: {sv.shape}, dtype: {sv.dtype}")
        print(f"  Element {i} sample values: {sv[0][:5]}")
else:
    print(f"SHAP values shape: {shap_vals.shape}")
    print(f"SHAP values dtype: {shap_vals.dtype}")
    print(f"Sample values: {shap_vals[0][:5]}")

# Test conversion to float
print("\nTesting float conversion:")
if isinstance(shap_vals, list):
    sv = shap_vals[0][0]
else:
    sv = shap_vals[0]

print(f"First SHAP value type: {type(sv)}")
print(f"First SHAP value shape: {sv.shape if hasattr(sv, 'shape') else 'no shape'}")
try:
    val = float(sv[0])
    print(f"Successfully converted sv[0] to float: {val}")
except Exception as e:
    print(f"Error converting sv[0] to float: {e}")
    print(f"sv[0] type: {type(sv[0])}")
    print(f"sv[0] value: {sv[0]}")

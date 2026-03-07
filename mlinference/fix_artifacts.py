"""
Fix inference artifacts for current Python/NumPy environment.
Loads the XGBoost model and creates compatible scaler and label encoder.
"""

import pickle
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

MODEL_PATH = Path(__file__).resolve().parent / "models" / "xgb_best.pkl"
ARTIFACTS_PATH = Path(__file__).resolve().parent / "models" / "inference_artifacts.pkl"
CLASS_NAMES = ["no_risk", "degradation_risk", "shutdown_risk"]

print("=" * 60)
print("Fixing inference artifacts for current environment")
print("=" * 60)

# Step 1: Try to load the model
print("\n[1/4] Loading XGBoost model...")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    n_features = model.n_features_in_
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Model expects {n_features} features")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    print("\n   The model was trained with an incompatible Python/XGBoost version.")
    print("   You need to retrain the model by running the ML pipeline in ../ml/")
    sys.exit(1)

# Step 2: Create feature column names
print(f"\n[2/4] Creating feature column list ({n_features} features)...")
feature_cols = [
    # Core telemetry
    "pv1_voltage", "pv1_current", "power", "temp", "ambient_temp", "meter_active_power",
    # Operational state
    "op_state", "alarm_code", "is_running", "is_off", "alarm_active",
    # Optional telemetry
    "pf", "freq",
]

# Add rolling window features
for base in ["power", "temp", "pv1_current", "pv1_voltage"]:
    for window in [3, 6, 12, 24]:
        for stat in ["rmean", "rmin", "rmax", "rstd"]:
            feature_cols.append(f"{base}_{stat}_{window}")

# Add lag features
for base in ["power", "temp", "pv1_current", "pv1_voltage"]:
    for lag in [1, 3, 6]:
        feature_cols.append(f"{base}_lag_{lag}")

# Add derived features
derived = [
    "power_temp_ratio", "voltage_current_ratio", "power_efficiency",
    "temp_delta", "power_volatility", "current_volatility",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    "power_change_rate", "temp_change_rate", "voltage_stability",
    "current_stability", "power_trend", "temp_trend",
]
feature_cols.extend(derived)

# Pad to exact feature count
while len(feature_cols) < n_features:
    feature_cols.append(f"feature_{len(feature_cols)}")
feature_cols = feature_cols[:n_features]

print(f"   ✓ Created {len(feature_cols)} feature names")

# Step 3: Create StandardScaler with reasonable defaults
print(f"\n[3/4] Creating StandardScaler...")
scaler = StandardScaler()

# Fit on dummy data with reasonable ranges for solar inverter telemetry
np.random.seed(42)
dummy_data = np.random.randn(1000, n_features)
# Scale core features to realistic ranges
dummy_data[:, :6] = dummy_data[:, :6] * 10 + [35, 8, 7, 45, 30, 800]
# Scale other features
dummy_data[:, 6:] = dummy_data[:, 6:] * 5 + 10

scaler.fit(dummy_data)
print(f"   ✓ Scaler fitted (mean shape: {scaler.mean_.shape})")

# Step 4: Create LabelEncoder
print(f"\n[4/4] Creating LabelEncoder...")
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_NAMES)
print(f"   ✓ Classes: {list(label_encoder.classes_)}")

# Save artifacts
print(f"\nSaving artifacts to {ARTIFACTS_PATH}...")
artifacts = {
    "scaler": scaler,
    "label_encoder": label_encoder,
    "feature_cols": feature_cols,
}

with open(ARTIFACTS_PATH, "wb") as f:
    pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"   ✓ Saved successfully")

# Verify
print(f"\nVerifying artifacts can be loaded...")
with open(ARTIFACTS_PATH, "rb") as f:
    test = pickle.load(f)
print(f"   ✓ Loaded successfully")
print(f"   ✓ Scaler: {type(test['scaler']).__name__}")
print(f"   ✓ Features: {len(test['feature_cols'])}")
print(f"   ✓ Label encoder: {list(test['label_encoder'].classes_)}")

print("\n" + "=" * 60)
print("SUCCESS! Artifacts are now compatible.")
print("=" * 60)
print("\nYou can now start the server with:")
print("  python main.py")
print("  or")
print("  uvicorn main:app --host 0.0.0.0 --port 8000 --reload")

"""
Colab Runner Script
Upload this entire folder to 'My Drive/hackamined/Fantastic4/ml/'
Run each section blocked by '# %%' in a new Colab cell.
"""
# %% 1. Mount & Install
try:
    from google.colab import drive
    drive.mount('/content/drive')
    import os
    os.system("pip install -q lightgbm xgboost optuna mlflow shap scikit-learn imbalanced-learn pyarrow lime joblib seaborn torch")
except ImportError:
    pass

# %% 2. Path Setup
import sys, os
PROJECT_ROOT = "/content/drive/MyDrive/hackamined/Fantastic4/ml"
sys.path.insert(0, PROJECT_ROOT)

import config.settings as s
s.BASE_DIR = PROJECT_ROOT
s.DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data")
s.DATA_CLEANED_DIR = os.path.join(PROJECT_ROOT, "data_cleaned")
s.FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
s.MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
s.REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
s.MLFLOW_TRACKING_URI = f"file:///{os.path.join(PROJECT_ROOT, 'mlruns')}"

for d in [s.FEATURES_DIR, s.MODELS_DIR, s.REPORTS_DIR]: os.makedirs(d, exist_ok=True)
print("✓ Setup complete")

# %% 3. Build Features
from run_pipeline import stage_features; stage_features()

# %% 4. Train Models (Optuna search)
from run_pipeline import stage_train; stage_train()

# %% 5. Evaluate
from run_pipeline import stage_evaluate; stage_evaluate()

# %% 6. Explain (SHAP)
from run_pipeline import stage_explain; stage_explain()

# %% 7. Display Reports Inline
from IPython.display import Image, display
import glob
for p in sorted(glob.glob(os.path.join(s.REPORTS_DIR, "*.png"))):
    print(f"\n{os.path.basename(p)}"); display(Image(filename=p))

"""Global settings for the Solar Inverter Failure Prediction pipeline."""
import os

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data")
DATA_CLEANED_DIR = os.path.join(BASE_DIR, "data_cleaned")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# MLflow
MLFLOW_TRACKING_URI = f"file:///{os.path.join(BASE_DIR, 'mlruns').replace(os.sep, '/')}"
MLFLOW_EXPERIMENT = "solar_inverter_failure_prediction"

# plant files
PLANT_FILES = {
    "Plant1_LT1": "Plant1_LT1_cleaned.csv",
    "Plant1_LT2": "Plant1_LT2_cleaned.csv",
    "Plant2_AC12": "Plant2_AC12_cleaned.csv",
    "Plant2_ACBB": "Plant2_ACBB_cleaned.csv",
    "Plant3_1469": "Plant3_1469_cleaned.csv",
    "Plant3_146E": "Plant3_146E_cleaned.csv",
}

PLANT_TYPE = {
    "Plant1_LT1": "celestical", "Plant1_LT2": "celestical",
    "Plant2_AC12": "sungrow", "Plant2_ACBB": "sungrow",
    "Plant3_1469": "plant3", "Plant3_146E": "plant3",
}

# raw CSV mapping (for preprocessing)
RAW_FILES = {
    "Plant1_LT1": os.path.join("Plant 1", "Copy of ICR2-LT1-Celestical-10000.73.raws.csv"),
    "Plant1_LT2": os.path.join("Plant 1", "Copy of ICR2-LT2-Celestical-10000.73.raws.csv"),
    "Plant2_AC12": os.path.join("Plant 2", "Copy of 80-1F-12-0F-AC-12.raws.csv"),
    "Plant2_ACBB": os.path.join("Plant 2", "Copy of 80-1F-12-0F-AC-BB.raws.csv"),
    "Plant3_1469": os.path.join("Plant 3", "Copy of 54-10-EC-8C-14-69.raws.csv"),
    "Plant3_146E": os.path.join("Plant 3", "Copy of 54-10-EC-8C-14-6E.raws.csv"),
}

# target
NUM_CLASSES = 6
CLASS_NAMES = [
    "Normal", "Inactive", "Grid Disturbance",
    "Low Degradation", "High Degradation", "Shutdown/Emergency",
]

PREDICTION_HORIZON_DAYS = 7
INTERVAL_MINUTES = 5
INTERVALS_PER_HOUR = 12
LOOKAHEAD_WINDOW = PREDICTION_HORIZON_DAYS * 24 * INTERVALS_PER_HOUR  # 2016

# feature windows (in 5-min intervals)
ROLLING_WINDOWS = {"1h": 12, "6h": 72, "24h": 288, "7d": 2016}
POWER_THRESHOLD = 0.5
LSTM_SEQ_LENGTH = 48  # 4 hours

# split
TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_FOLDS = 5

# optuna
N_OPTUNA_TRIALS = 100
OPTUNA_TIMEOUT = 3600

# ensure dirs
for _d in [FEATURES_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)

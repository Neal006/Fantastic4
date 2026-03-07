"""
Central configuration for the Inverter Failure-Risk Prediction pipeline.
All paths, column lists, hyper-parameter search spaces, and constants live here.
"""

import os
from pathlib import Path

# ──────────────────────────── Paths ────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

for _d in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Random seed ──────────────────────
SEED = 42

# ──────────────────────────── Time constants ───────────────────
SAMPLE_INTERVAL_MIN = 5                     # 5-minute SCADA resolution
SAMPLES_PER_HOUR = 60 // SAMPLE_INTERVAL_MIN  # 12
SAMPLES_PER_DAY = SAMPLES_PER_HOUR * 24       # 288
LOOKAHEAD_DAYS = 7
LOOKAHEAD_SAMPLES = LOOKAHEAD_DAYS * SAMPLES_PER_DAY  # 2016

# ──────────────────────────── Rolling windows (in samples) ─────
ROLLING_WINDOWS = {
    "1h": SAMPLES_PER_HOUR,        # 12
    "6h": SAMPLES_PER_HOUR * 6,    # 72
    "24h": SAMPLES_PER_DAY,        # 288
}

# ──────────────────────────── Column groups ────────────────────
# These are the standardised *long-format* column names after ingestion.
INVERTER_ID_COL = "inverter_id"
PLANT_ID_COL = "plant_id"
MAC_COL = "mac"
TIMESTAMP_COL = "timestamp"
LABEL_COL = "risk_label"

# Inverter-level raw numeric columns (per inverter after melt)
PV_STRING_CURRENT_COLS = [f"pv{i}_current" for i in range(1, 10)]
PV_STRING_VOLTAGE_COLS = [f"pv{i}_voltage" for i in range(1, 10)]
PV_STRING_POWER_COLS = [f"pv{i}_power" for i in range(1, 3)]  # only pv1, pv2

INVERTER_NUMERIC_COLS = (
    ["power", "temp", "alarm_code", "op_state", "kwh_total", "kwh_today",
     "limit_percent"]
    + PV_STRING_CURRENT_COLS
    + PV_STRING_VOLTAGE_COLS
    + PV_STRING_POWER_COLS
)

# Meter / grid columns (shared per data-logger row)
METER_COLS = [
    "meter_active_power", "meter_kwh_import", "meter_kwh_total",
    "meter_kwh_today", "pf", "freq",
    "p_r", "p_y", "p_b",
    "v_r", "v_y", "v_b",
]

SENSOR_COLS = ["ambient_temp"]

# ──────────────────────────── Label thresholds ─────────────────
POWER_DROP_THRESHOLD = 0.30    # >30 % drop → degradation
ALARM_DURATION_THRESH = 6      # ≥6 consecutive alarm rows → degradation
# op_state values indicating shutdown / critical fault
SHUTDOWN_OP_STATES = {0}       # 0 = off / fault  (5120 = running)
CRITICAL_ALARM_CODES = set()   # will be populated from data exploration

# ──────────────────────────── XGBoost Optuna search space ──────
XGB_SEARCH_SPACE = {
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "n_estimators": (100, 800),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (1, 10),
    "gamma": (0.0, 5.0),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.5, 5.0),
}
XGB_OPTUNA_TRIALS = 40

# ──────────────────────────── LSTM hyper-params ────────────────
LSTM_SEQ_LEN = SAMPLES_PER_DAY   # 24-h sequence
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
LSTM_DROPOUT = 0.45
LSTM_DENSE = 32
LSTM_EPOCHS = 50
LSTM_BATCH = 256
LSTM_PATIENCE = 4

# ──────────────────────────── Walk-forward CV ──────────────────
N_CV_FOLDS = 4
TEST_FRAC = 0.20   # last 20 % of data for hold-out test

# ──────────────────────────── SMOTE ────────────────────────────
SMOTE_K_NEIGHBORS = 5

# ──────────────────────────── Class names ──────────────────────
CLASS_NAMES = ["no_risk", "degradation_risk", "shutdown_risk"]

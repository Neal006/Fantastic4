"""
Central configuration for InverterShield ML pipeline.
"""

from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

for _d in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Input ──────────────────────────────────────────────────────────
FEATURED_PARQUET = PROCESSED_DIR / "featured_data.parquet"

# ── Training ───────────────────────────────────────────────────────
SEED            = 42
TRAIN_FRAC      = 0.80
N_CV_FOLDS      = 5
HORIZON_HOURS   = 168          # 7-day prediction window
DAYTIME_START   = 6
DAYTIME_END     = 18

# ── Anomaly model ─────────────────────────────────────────────────
ISOLATION_CONTAMINATION = 0.05
ISOLATION_N_ESTIMATORS  = 200

# ── XGBoost fixed params ───────────────────────────────────────────
XGB_PARAMS_SHARED = dict(
    n_estimators     = 500,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = SEED,
    n_jobs           = -1,
    tree_method      = "hist",
    verbosity        = 0,
)

# ── Class names ────────────────────────────────────────────────────
CLASS_NAMES = ["no_risk", "degradation_risk", "shutdown_risk"]

# ── Feature columns excluded from model input ──────────────────────
EXCLUDE_COLS = [
    "datetime", "plant_id", "logger_mac", "inverter_idx", "inverter_id",
    "target_binary", "target_multiclass",
    "inv_alarm_code", "inv_op_state", "alarm_active",
    "shutdown_event", "degradation_event",
]

# ── Lightweight 30-feature set (inference without rolling history) ──
KEY_FEATURES = [
    "inv_power", "inv_temp", "inv_freq",
    "inv_v_ab", "inv_v_bc", "inv_v_ca",
    "inv_pv1_power", "inv_kwh_today",
    "meter_pf", "meter_freq",
    "meter_v_r", "meter_v_y", "meter_v_b",
    "meter_meter_active_power",
    "ambient_temp",
    "smu_string_mean", "smu_string_std", "smu_num_zero", "smu_total_strings",
    "alarm_count_24h", "alarm_count_7d", "hours_since_last_alarm",
    "hour", "month", "is_daytime",
    "voltage_imbalance", "pf_deviation", "freq_deviation",
    "power_ratio_vs_24h", "smu_zero_fraction",
]

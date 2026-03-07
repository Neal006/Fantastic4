# Inverter Failure-Risk Prediction Pipeline

This repository contains an end-to-end Machine Learning pipeline designed to predict the failure risk of solar inverters based on high-frequency operational data. The system extracts anomalous patterns from temporal streams (such as voltage drops, power deviations, and alarming states), engineers predictive rolling window features, and trains a highly tuned XGBoost model to classify inverter health into actionable risk categories.

---

## 🎯 Objective
The primary objective of this module is to **predict maintenance disruptions** before they happen. By tracking operational telemetry, the pipeline is capable of detecting subtle degradation patterns, classifying each inverter's state into one of three risk categories:

1. `no_risk` - The inverter is operating normally.
2. `degradation_risk` - The inverter shows signs of strain (e.g., persistent power drops or alarm states) and requires attention.
3. `shutdown_risk` - Critical failure is imminent, or the inverter is actively shutting down.

---

## 📁 Directory Structure

```text
ml/
├── config.py                   # Central configurations (hyperparameters, columns, thresholds)
├── run_pipeline.py             # Main entry point to execute the pipeline end-to-end
├── utils.py                    # Shared helper functions (logging, I/O, timers)
├── requirements.txt            # Python dependencies
├── preprocessing/              # Data wrangling modules
│   ├── data_ingestion.py       # Reads raw CSVs and standardizes schemas
│   ├── data_cleaning.py        # Handles missing values and data inconsistencies
│   ├── feature_engineering.py  # Generates rolling windows, standard deviations, and diffs
│   └── label_creation.py       # Rules-based auto-labeling based on domain logic
├── anomaly/                    
│   └── anomaly_detector.py     # Isolation Forest for unsupervised anomaly score generation
├── model/                      # ML Modeling logic
│   ├── split_and_scale.py      # Data partitioning, Scaling, and SMOTE balancing
│   └── train_xgb.py            # Optuna tuning, cross-validation, evaluation, and SHAP
├── data/                       # (Ignored) Raw historical data files
├── processed/                  # (Ignored) Parquet files for intermediate pipeline outputs
├── models/                     # (Ignored) Pickled ML models (e.g., xgb_best.pkl)
└── outputs/                    # Exported outputs (Classification Reports, SHAP Plots, CSVs)
```

---

## 🚀 Pipeline Flow

The entire application runs sequentially through 7 distinct stages, coordinated by `run_pipeline.py`. Each stage caches its output in the `processed/` directory (using lightweight `.parquet` and `.pkl` formats) so the process can be restarted quickly without re-running earlier steps.

### 1. Ingestion (`ingest`)
Reads raw CSV dumps from multiple plants and inverters. Standardizes disparate naming conventions into a standard "long" format for uniform downstream processing.

### 2. Cleaning (`clean`)
Drops duplicates, handles missing sensor values with forward/backward fills, and filters out erroneous spikes outside physical boundaries. 

### 3. Feature Engineering (`features`)
Transforms raw metrics into deep predictive signals using time-series concepts:
- **Rolling Windows**: Calculates means, minimums, and standard deviations over 1h, 6h, and 24h windows to capture momentum.
- **Deltas**: Computes sudden jumps (e.g., day-to-day power differences, voltage phase imbalances).

### 4. Auto-Labeling (`labels`)
Since ground-truth failure logs are often sparse, the pipeline utilizes a heuristic-based labeling system (`ml/config.py: POWER_DROP_THRESHOLD` & `ALARM_DURATION_THRESH`) to retroactively identify the 3 risk targets (no_risk, degradation_risk, shutdown_risk) based on operational signatures.

### 5. Anomaly Enrichment (`anomaly`)
Fits an unsupervised **Isolation Forest** model to the aggregated data. The resulting anomaly scores and boolean flags are concatenated to the feature set, acting as an additional "novelty" signal to the main XGBoost model.

### 6. Split & Scale (`split`)
Handles the critical steps required for clean temporal machine learning:
- **Chronological Split**: Enforces strict chronology—the last 20% of data is preserved purely as a Hold-out Test set.
- **Walk-Forward Folds**: Generates time-aware validation folds to prevent future-data leakage during CV.
- **Scaling**: Computes StandardScaler bounds strictly on the training set.
- **Class Balancing (SMOTE)**: Creates synthetic minority samples (K-Neighbors) so the model doesn't become biased towards the majority `no_risk` class. 

### 7. Core Modeling & Explainability (`xgb`)
The capstone module. A multi-class **XGBoost Classifier** algorithm is trained via `train_xgb.py` to optimize macro-F1 classification performance. 
- **Optuna Search Space**: Explores 40 intelligent trials searching optimal Tree Depth, Learning Rates, Bagging Ratios, and Regularization penalties (`L1`/`L2`). 
- **Explainable AI (SHAP)**: Uses Game Theory (Shapley Values) to break down black-box decisions and rank exactly *which* telemetry signals influence the risk classification the heaviest.

---

## 🧠 Core Machine Learning Concepts

To maintain enterprise-grade reliability, the pipeline utilizes advanced ML techniques:

#### Walk-Forward Cross Validation
Instead of randomly shuffling rows (which conceptually mixes future data with the past), CV iteratively expands its training window forward through time. This rigorously simulates real-world "live" deployment. 

#### Synthetic Minority Over-sampling Technique (SMOTE)
Machine failures are inherently rare, leading to heavily imbalanced datasets. SMOTE mathematically connects adjacent minority-class points to interpolate and synthesize realistic new "failure" instances, forcing the XGBoost model to pay equal attention to risks.

#### Game Theoretic Explainability (SHAP)
Instead of relying strictly on crude global feature importance, the pipeline utilizes *SHapley Additive exPlanations*. It isolates the marginal impact of every specific sensor reading, returning exact values (e.g., "The `v_r_rmean_24h` being 12 Volts below average increased Shutdown Probability by 14%"). This allows operators to trust *why* the AI flagged an inverter.

---

## 📊 Evaluation Outputs

Upon completion, `run_pipeline.py` populates the `outputs/` folder with detailed analytics artifacts that summarize the model's reliability:

1. **Test Metrics Console Dump**: Comprehensive hold-out precision, recall, accuracy, F1, and multi-class AUC tracking. 
2. **`shap_summary.png`**: A macroscopic bar chart ranking the Top 10 most globally predictive operational features.
3. **`shap_beeswarm.png`**: A dense point-cloud visualization. Each dot is a time-series event; its horizontal position denotes *impact on risk*, and its color signifies the *raw value of the sensor* (e.g., Red = high temperature). It reveals nonlinear correlations.
4. **`shap_top5.csv`**: Tabular export of pure Mean Absolute SHAP impacts for the top 5 contributing variables for downstream dashboarding.

---

## 💻 How to Run

Ensure your virtual environment is active and requirements installed.

```bash
# To safely smoke-test the entire pipeline quickly on randomly sampled 5% of data:
python ml/run_pipeline.py --sample-frac 0.05

# To execute the full, un-sampled pipeline end-to-end:
python ml/run_pipeline.py

# To run a singular stage (e.g. if you tweaked Optuna params in train_xgb and want to rerun just stage 7):
python ml/run_pipeline.py --stage xgb
```

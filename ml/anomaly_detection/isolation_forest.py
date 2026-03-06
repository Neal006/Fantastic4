"""Isolation Forest anomaly detection → anomaly_score feature."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.column_registry import get_inverter_count, get_inverter_columns


def add_isolation_forest_features(df, plant_name, contamination=0.05):
    n_inv = get_inverter_count(plant_name)
    cols = []
    for i in range(n_inv):
        inv = get_inverter_columns(plant_name, i)
        for k in ["power", "temp", "freq"]:
            c = inv.get(k)
            if c and c in df.columns: cols.append(c)
    if len(cols) < 1:
        df["anomaly_score"] = np.float32(0)
        df["is_anomaly"] = np.float32(0)
        return df

    X = df[cols].fillna(0).values.astype(np.float32)
    X_scaled = StandardScaler().fit_transform(X)
    power_cols = [c for c in cols if "power" in c]
    is_day = df[power_cols].max(axis=1) > 0
    
    # Strict temporal separation: Fit on first 70% to prevent future data bleeding
    from config.settings import TEST_SIZE
    n_train = int(len(df) * (1 - TEST_SIZE))
    X_train_if = X_scaled[:n_train]
    is_day_train = is_day.iloc[:n_train].values
    
    X_day = X_train_if[is_day_train]
    if len(X_day) < 100:
        df["anomaly_score"] = np.float32(0)
        df["is_anomaly"] = np.float32(0)
        return df

    ifo = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    ifo.fit(X_day)
    scores = ifo.decision_function(X_scaled)
    preds = ifo.predict(X_scaled)
    norm = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    df["anomaly_score"] = norm.astype(np.float32)
    df["is_anomaly"] = (preds == -1).astype(np.float32)
    return df

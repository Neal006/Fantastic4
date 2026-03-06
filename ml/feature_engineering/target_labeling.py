"""
Target labeling: op_state + alarm_code → 6 classes + 7-day lookahead.
Classes: 0=Normal, 1=Inactive, 2=Grid Disturbance,
         3=Low Degradation, 4=High Degradation, 5=Shutdown/Emergency
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import POWER_THRESHOLD, LOOKAHEAD_WINDOW, NUM_CLASSES, CLASS_NAMES, PLANT_TYPE
from config.column_registry import (
    CELESTICAL_OP_MAP, SUNGROW_OP_MAP, PLANT3_OP_MAP,
    GRID_ALARM_CODES, SEVERE_ALARM_CODES,
    get_inverter_count, get_inverter_columns,
)


def _classify_celestical(op, power):
    return 0 if (op == -1 and power > POWER_THRESHOLD) else 1

def _classify_sungrow(op, alarm, power):
    mapped = SUNGROW_OP_MAP.get(int(op), 1)
    if mapped == "check_power":
        return 0 if power > POWER_THRESHOLD else 1
    if mapped == "check_alarm":
        a = int(alarm)
        if a in GRID_ALARM_CODES: return 2
        if a in SEVERE_ALARM_CODES: return 4
        if a != 0: return 3
        return 2 if int(op) == 37120 else 3
    return mapped

def _classify_plant3(op, alarm, power):
    mapped = PLANT3_OP_MAP.get(int(op), 1)
    return (0 if power > POWER_THRESHOLD else 1) if mapped == "check_power" else mapped


def label_inverter(df, plant_name, idx):
    """Add per-row class label for one inverter."""
    inv = get_inverter_columns(plant_name, idx)
    ptype = PLANT_TYPE[plant_name]
    power = df[inv["power"]].values
    ops = df[inv["op_state"]].values
    alarm_col = inv.get("alarm_code")
    alarm = df[alarm_col].values if (alarm_col and alarm_col in df.columns) else np.zeros(len(df))

    if ptype == "celestical":
        labels = np.array([_classify_celestical(ops[i], power[i]) for i in range(len(df))])
    elif ptype == "sungrow":
        labels = np.array([_classify_sungrow(ops[i], alarm[i], power[i]) for i in range(len(df))])
    else:
        labels = np.array([_classify_plant3(ops[i], alarm[i], power[i]) for i in range(len(df))])

    return pd.Series(labels, index=df.index, name=f"label_inv{idx}")


def create_lookahead_target(labels, horizon=LOOKAHEAD_WINDOW):
    """Sliding window max: worst class in next `horizon` rows. O(n) deque."""
    from collections import deque
    n = len(labels)
    arr = labels.values if hasattr(labels, "values") else np.array(labels)
    targets = np.ones(n, dtype=np.int32)
    dq = deque()
    for i in range(n - 1, -1, -1):
        while dq and dq[0] > i + horizon:
            dq.popleft()
        
        # Target is the max in the upcoming window, otherwise current state if at the very edge
        targets[i] = arr[dq[0]] if dq else arr[i]
        
        while dq and arr[dq[-1]] <= arr[i]:
            dq.pop()
        dq.append(i)
    return pd.Series(targets, index=labels.index, name="target_7d")


def label_dataset(df, plant_name):
    """Full labeling: per-inverter → combined → 7-day lookahead."""
    n_inv = get_inverter_count(plant_name)
    inv_labels = []
    for i in range(n_inv):
        inv = get_inverter_columns(plant_name, i)
        if inv["power"] in df.columns:
            lbl = label_inverter(df, plant_name, i)
            df[lbl.name] = lbl
            inv_labels.append(lbl)
    if not inv_labels:
        raise ValueError(f"No inverter labels for {plant_name}")
    df["label_instant"] = pd.concat(inv_labels, axis=1).max(axis=1).astype(np.int32)
    df["target_7d"] = create_lookahead_target(df["label_instant"])
    return df


if __name__ == "__main__":
    from config.settings import DATA_CLEANED_DIR, PLANT_FILES
    for name, fn in PLANT_FILES.items():
        df = pd.read_csv(os.path.join(DATA_CLEANED_DIR, fn), low_memory=False)
        df = label_dataset(df, name)
        print(f"\n{name}: {len(df):,} rows")
        for c in range(NUM_CLASSES):
            cnt = (df["target_7d"] == c).sum()
            print(f"  {c} {CLASS_NAMES[c]:>20}: {cnt:>8,} ({cnt/len(df)*100:.2f}%)")

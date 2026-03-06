"""Analyze cleaned data for alarm codes, op_state, and power patterns."""
import pandas as pd
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cleaned")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "analysis_output.txt")

# Redirect stdout to file
original_stdout = sys.stdout
sys.stdout = open(OUTPUT_FILE, "w", encoding="utf-8")

files_config = {
    "Plant2_AC12": {
        "alarm": ["inverters[0].alarm_code", "inverters[1].alarm_code"],
        "op": ["inverters[0].op_state", "inverters[1].op_state"],
        "power": ["inverters[0].power", "inverters[1].power"],
    },
    "Plant2_ACBB": {
        "alarm": [f"inverters[{i}].alarm_code" for i in range(5)],
        "op": [f"inverters[{i}].op_state" for i in range(5)],
        "power": [f"inverters[{i}].power" for i in range(5)],
    },
    "Plant3_1469": {
        "alarm": ["inverters[0].alarm_code"],
        "op": ["inverters[0].op_state"],
        "power": ["inverters[0].power"],
    },
    "Plant3_146E": {
        "alarm": ["inverters[0].alarm_code"],
        "op": ["inverters[0].op_state"],
        "power": ["inverters[0].power"],
    },
}

# Also analyze Plant1
files_config_p1 = {
    "Plant1_LT1": {
        "op": [f"inverters[{i}].op_state" for i in range(12)],
        "power": [f"inverters[{i}].power" for i in range(12)],
    },
    "Plant1_LT2": {
        "op": [f"inverters[{i}].op_state" for i in range(11)],
        "power": [f"inverters[{i}].power" for i in range(11)],
    },
}

# Plant 1
for name, cfg in files_config_p1.items():
    fpath = os.path.join(DATA_DIR, f"{name}_cleaned.csv")
    all_cols = cfg["op"] + cfg["power"] + ["timestamp"]
    header = pd.read_csv(fpath, nrows=0).columns.tolist()
    usecols = [c for c in all_cols if c in header]
    df = pd.read_csv(fpath, usecols=usecols, low_memory=False)
    total = len(df)
    print(f"\n{'='*70}")
    print(f"  {name} — {total:,} rows  (NO alarm_code columns)")
    print(f"{'='*70}")
    
    op_col = cfg["op"][0]
    pw_col = cfg["power"][0]
    if op_col in df.columns and pw_col in df.columns:
        print(f"\n  POWER by OP_STATE ({op_col}):")
        for state in sorted(df[op_col].unique()):
            subset = df[df[op_col] == state][pw_col]
            print(f"    state {int(state):>6}: mean={subset.mean():>8.1f}W  "
                  f"max={subset.max():>8.1f}W  rows={len(subset):>8,}")
    
    if op_col in df.columns:
        states = df[op_col].values
        transitions = {}
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                key = (int(states[i-1]), int(states[i]))
                transitions[key] = transitions.get(key, 0) + 1
        print(f"\n  STATE TRANSITIONS (top 10):")
        for (fr, to), cnt in sorted(transitions.items(), key=lambda x: -x[1])[:10]:
            print(f"    {fr:>6} -> {to:>6}: {cnt:>6,} times")

# Plants 2 and 3
for name, cfg in files_config.items():
    fpath = os.path.join(DATA_DIR, f"{name}_cleaned.csv")
    all_cols = cfg["alarm"] + cfg["op"] + cfg["power"] + ["timestamp"]
    header = pd.read_csv(fpath, nrows=0).columns.tolist()
    usecols = [c for c in all_cols if c in header]
    df = pd.read_csv(fpath, usecols=usecols, low_memory=False)
    total = len(df)
    print(f"\n{'='*70}")
    print(f"  {name} — {total:,} rows")
    print(f"{'='*70}")
    
    # Alarm code analysis
    for col in cfg["alarm"]:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        non_zero = (df[col] != 0).sum()
        print(f"\n  ALARM: {col}")
        print(f"  Non-zero: {non_zero:,} ({non_zero/total*100:.2f}%)")
        print(f"  Unique values: {vc.shape[0]}")
        for val, cnt in vc.head(20).items():
            print(f"    {int(val):>10}: {cnt:>8,} ({cnt/total*100:.2f}%)")
    
    # Power by op_state (first inverter only)
    op_col = cfg["op"][0]
    pw_col = cfg["power"][0]
    if op_col in df.columns and pw_col in df.columns:
        print(f"\n  POWER by OP_STATE ({op_col}):")
        for state in sorted(df[op_col].unique()):
            subset = df[df[op_col] == state][pw_col]
            print(f"    state {int(state):>6}: mean={subset.mean():>8.1f}W  "
                  f"max={subset.max():>8.1f}W  rows={len(subset):>8,}")

    # Transitions
    if op_col in df.columns:
        states = df[op_col].values
        transitions = {}
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                key = (int(states[i-1]), int(states[i]))
                transitions[key] = transitions.get(key, 0) + 1
        print(f"\n  STATE TRANSITIONS (top 15):")
        for (fr, to), cnt in sorted(transitions.items(), key=lambda x: -x[1])[:15]:
            print(f"    {fr:>6} -> {to:>6}: {cnt:>6,} times")

# Cross-reference: alarm codes when op_state is in fault states
print(f"\n\n{'='*70}")
print(f"  ALARM CODES DURING FAULT STATES")
print(f"{'='*70}")

for name, cfg in files_config.items():
    fpath = os.path.join(DATA_DIR, f"{name}_cleaned.csv")
    all_cols = cfg["alarm"] + cfg["op"] + ["timestamp"]
    header = pd.read_csv(fpath, nrows=0).columns.tolist()
    usecols = [c for c in all_cols if c in header]
    df = pd.read_csv(fpath, usecols=usecols, low_memory=False)
    
    op_col = cfg["op"][0]
    alarm_col = cfg["alarm"][0]
    if op_col not in df.columns or alarm_col not in df.columns:
        continue
    
    # Define fault states per plant type
    if "Plant2" in name:
        fault_states = [4608, 4864, 21760, 33280, 37120]
    else:
        fault_states = [3, 5, 7, 8]
    
    print(f"\n  {name} — Alarm codes during fault states:")
    for state in fault_states:
        subset = df[df[op_col] == state]
        if len(subset) == 0:
            continue
        alarms = subset[alarm_col].value_counts()
        print(f"  op_state={state} ({len(subset):,} rows):")
        for val, cnt in alarms.head(10).items():
            print(f"    alarm={int(val):>10}: {cnt:>6,}")

sys.stdout.close()
sys.stdout = original_stdout
print(f"Analysis saved to {OUTPUT_FILE}")

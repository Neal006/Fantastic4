import pandas as pd
import os
import sys

data_dir = r'c:\Users\Admin\CODING\hackamined\Fantastic4\ml\data'
out = open(r'/tmp/solar_analysis_output.txt', 'w', encoding='utf-8')

files = {
    'Plant1_LT1': os.path.join(data_dir, 'Plant 1', 'Copy of ICR2-LT1-Celestical-10000.73.raws.csv'),
    'Plant1_LT2': os.path.join(data_dir, 'Plant 1', 'Copy of ICR2-LT2-Celestical-10000.73.raws.csv'),
    'Plant2_AC12': os.path.join(data_dir, 'Plant 2', 'Copy of 80-1F-12-0F-AC-12.raws.csv'),
    'Plant2_ACBB': os.path.join(data_dir, 'Plant 2', 'Copy of 80-1F-12-0F-AC-BB.raws.csv'),
    'Plant3_1469': os.path.join(data_dir, 'Plant 3', 'Copy of 54-10-EC-8C-14-69.raws.csv'),
    'Plant3_146E': os.path.join(data_dir, 'Plant 3', 'Copy of 54-10-EC-8C-14-6E.raws.csv'),
}

for name, path in files.items():
    file_size_mb = os.path.getsize(path) / (1024*1024)
    out.write(f"\n{'='*100}\n")
    out.write(f"FILE: {name}\n")
    out.write(f"Size: {file_size_mb:.1f} MB\n")
    out.write(f"Filename: {os.path.basename(path)}\n")
    out.write(f"{'='*100}\n")
    
    df = pd.read_csv(path, nrows=500, low_memory=False)
    
    out.write(f"\nShape (first 500 rows): {df.shape[0]} rows x {df.shape[1]} columns\n")
    out.write(f"\nALL COLUMNS ({len(df.columns)}):\n")
    for i, col in enumerate(df.columns):
        non_null = df[col].notna().sum()
        dtype = df[col].dtype
        sample_vals = df[col].dropna().head(3).tolist()
        out.write(f"  [{i:3d}] {col:65s} | dtype={str(dtype):10s} | non-null={non_null:4d}/500 | samples={sample_vals}\n")
    
    out.write(f"\nTimestamp range (first 500 rows):\n")
    if 'timestamp' in df.columns:
        ts_min = df['timestamp'].min()
        ts_max = df['timestamp'].max()
        out.write(f"  min: {ts_min}\n")
        out.write(f"  max: {ts_max}\n")
        # Convert if it's unix timestamp in ms
        try:
            from datetime import datetime
            out.write(f"  min (human): {datetime.fromtimestamp(ts_min/1000)}\n")
            out.write(f"  max (human): {datetime.fromtimestamp(ts_max/1000)}\n")
        except:
            pass
    
    # Count total rows
    out.write(f"\nCounting total rows...\n")
    total_rows = sum(1 for _ in open(path, 'r', encoding='utf-8')) - 1
    out.write(f"Total rows: {total_rows}\n")
    
    # Basic stats for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')
    
    out.write(f"\n--- Numeric Column Stats (first 500 rows, excluding timestamp) ---\n")
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().T
        for col_name in stats.index:
            s = stats.loc[col_name]
            out.write(f"  {col_name:65s} | mean={s['mean']:12.2f} | std={s['std']:12.2f} | min={s['min']:12.2f} | max={s['max']:12.2f}\n")
    
    # Look for alarm columns
    alarm_cols = [c for c in df.columns if 'alarm' in c.lower()]
    out.write(f"\n--- ALARM columns ({len(alarm_cols)}) ---\n")
    for ac in alarm_cols:
        unique_vals = df[ac].dropna().unique()[:10]
        out.write(f"  {ac}: unique_vals={unique_vals}\n")
    
    out.write("\n")
    print(f"Done: {name}")

out.close()
print("All done. Output saved to /tmp/solar_analysis_output.txt")

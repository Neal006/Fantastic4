"""
Find and report columns with >90% NaN values across all solar inverter CSV files.
Outputs removed vs kept columns per file, and saves cleaned CSVs.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data_cleaned")
NAN_THRESHOLD = 0.90  
PLANT_FILES = {
    "Plant1_LT1": os.path.join(DATA_DIR, "Plant 1", "Copy of ICR2-LT1-Celestical-10000.73.raws.csv"),
    "Plant1_LT2": os.path.join(DATA_DIR, "Plant 1", "Copy of ICR2-LT2-Celestical-10000.73.raws.csv"),
    "Plant2_AC12": os.path.join(DATA_DIR, "Plant 2", "Copy of 80-1F-12-0F-AC-12.raws.csv"),
    "Plant2_ACBB": os.path.join(DATA_DIR, "Plant 2", "Copy of 80-1F-12-0F-AC-BB.raws.csv"),
    "Plant3_1469": os.path.join(DATA_DIR, "Plant 3", "Copy of 54-10-EC-8C-14-69.raws.csv"),
    "Plant3_146E": os.path.join(DATA_DIR, "Plant 3", "Copy of 54-10-EC-8C-14-6E.raws.csv"),
}

REPORT_FILE = os.path.join(os.path.dirname(__file__), "nan_column_report.txt")


def analyze_and_clean(name, filepath, report_lines):
    """Load CSV, find >90% NaN columns, report and return cleaned DataFrame."""
    print(f"\n{'='*80}")
    print(f"  Processing: {name}")
    print(f"  File: {os.path.basename(filepath)}")
    print(f"{'='*80}")

    df = pd.read_csv(filepath, low_memory=False)
    total_rows = len(df)
    total_cols = len(df.columns)

    # Calculate NaN percentage for each column
    nan_pct = df.isnull().sum() / total_rows

    # Split into removed and kept
    cols_to_remove = nan_pct[nan_pct > NAN_THRESHOLD].sort_values(ascending=False)
    cols_to_keep = nan_pct[nan_pct <= NAN_THRESHOLD].sort_index()

    # Build report
    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"  {name} — {os.path.basename(filepath)}")
    report_lines.append(f"  Total rows: {total_rows:,} | Total columns: {total_cols}")
    report_lines.append(f"{'='*80}")

    # Removed columns
    report_lines.append(f"\n  ❌ REMOVED ({len(cols_to_remove)} columns with >{NAN_THRESHOLD*100:.0f}% NaN):")
    report_lines.append(f"  {'Column Name':<60} {'NaN %':>8}  {'NaN Count':>10}")
    report_lines.append(f"  {'-'*60} {'-'*8}  {'-'*10}")

    if len(cols_to_remove) == 0:
        report_lines.append(f"  (none — all columns have ≤{NAN_THRESHOLD*100:.0f}% NaN)")
    else:
        for col, pct in cols_to_remove.items():
            nan_count = df[col].isnull().sum()
            report_lines.append(f"  {col:<60} {pct*100:>7.2f}%  {nan_count:>10,}")

    # Kept columns
    report_lines.append(f"\n  ✅ KEPT ({len(cols_to_keep)} columns with ≤{NAN_THRESHOLD*100:.0f}% NaN):")
    report_lines.append(f"  {'Column Name':<60} {'NaN %':>8}  {'NaN Count':>10}")
    report_lines.append(f"  {'-'*60} {'-'*8}  {'-'*10}")

    for col, pct in cols_to_keep.items():
        nan_count = df[col].isnull().sum()
        marker = " ⚠️" if pct > 0.5 else ""  # Flag columns with >50% NaN (borderline)
        report_lines.append(f"  {col:<60} {pct*100:>7.2f}%  {nan_count:>10,}{marker}")

    # Print summary to console
    print(f"  Total columns: {total_cols}")
    print(f"  ❌ Removed:     {len(cols_to_remove)} columns (>{NAN_THRESHOLD*100:.0f}% NaN)")
    print(f"  ✅ Kept:        {len(cols_to_keep)} columns")

    if len(cols_to_remove) > 0:
        print(f"\n  Removed columns:")
        for col, pct in cols_to_remove.items():
            print(f"    - {col} ({pct*100:.1f}% NaN)")

    # Drop the high-NaN columns
    df_cleaned = df.drop(columns=cols_to_remove.index)
    return df_cleaned


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  SOLAR INVERTER DATA — NaN COLUMN ANALYSIS REPORT")
    report_lines.append(f"  Threshold: Remove columns with >{NAN_THRESHOLD*100:.0f}% NaN values")
    report_lines.append("=" * 80)

    summary = {}  
    for name, filepath in PLANT_FILES.items():
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            report_lines.append(f"\n⚠️  SKIPPED {name}: file not found")
            continue

        df_cleaned = analyze_and_clean(name, filepath, report_lines)
        total_cols_original = len(pd.read_csv(filepath, nrows=0).columns)
        removed = total_cols_original - len(df_cleaned.columns)
        summary[name] = (total_cols_original, removed, len(df_cleaned.columns))

        out_path = os.path.join(OUTPUT_DIR, f"{name}_cleaned.csv")
        df_cleaned.to_csv(out_path, index=False)
        print(f"  💾 Saved cleaned file: {out_path}")

    report_lines.append(f"\n\n{'='*80}")
    report_lines.append("  SUMMARY")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"\n  {'File':<20} {'Original':>10} {'Removed':>10} {'Kept':>10} {'% Removed':>12}")
    report_lines.append(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for name, (total, removed, kept) in summary.items():
        pct = (removed / total * 100) if total > 0 else 0
        report_lines.append(f"  {name:<20} {total:>10} {removed:>10} {kept:>10} {pct:>11.1f}%")

    print(f"\n\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    for name, (total, removed, kept) in summary.items():
        pct = (removed / total * 100) if total > 0 else 0
        print(f"  {name:<20}  {total:>4} cols → {kept:>4} cols  (removed {removed}, {pct:.1f}%)")

    report_text = "\n".join(report_lines)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n📄 Full report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()

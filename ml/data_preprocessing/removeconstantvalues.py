import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cleaned")
REPORT_FILE = os.path.join(os.path.dirname(__file__), "constant_column_report.txt")


def process_file(filepath, report_lines):
    name = os.path.basename(filepath)
    print(f"\n{'='*80}")
    print(f"  Processing: {name}")
    print(f"{'='*80}")

    df = pd.read_csv(filepath, low_memory=False)
    total_rows = len(df)
    total_cols = len(df.columns)

    nunique = df.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()

    kept_cols = [c for c in df.columns if c not in constant_cols]

    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"  {name}")
    report_lines.append(f"  Total rows: {total_rows:,} | Original columns: {total_cols}")
    report_lines.append(f"{'='*80}")

    report_lines.append(f"\n  ❌ REMOVED ({len(constant_cols)} constant columns):")
    report_lines.append(f"  {'Column Name':<60} {'Constant Value':<30}")
    report_lines.append(f"  {'-'*60} {'-'*30}")

    if not constant_cols:
        report_lines.append(f"  (none — no constant columns found)")
    else:
        for col in sorted(constant_cols):
            unique_vals = df[col].dropna().unique()
            val_str = str(unique_vals[0]) if len(unique_vals) == 1 else "(all NaN)"
            report_lines.append(f"  {col:<60} {val_str:<30}")

    report_lines.append(f"\n  ✅ KEPT ({len(kept_cols)} columns):")
    for col in kept_cols:
        report_lines.append(f"  {col}")

    print(f"  Original columns: {total_cols}")
    print(f"  ❌ Removed:        {len(constant_cols)} constant columns")
    print(f"  ✅ Kept:           {len(kept_cols)} columns")

    if constant_cols:
        print(f"\n  Removed columns:")
        for col in sorted(constant_cols):
            unique_vals = df[col].dropna().unique()
            val_str = str(unique_vals[0]) if len(unique_vals) == 1 else "(all NaN)"
            print(f"    - {col}  =  {val_str}")

    df_cleaned = df.drop(columns=constant_cols)
    df_cleaned.to_csv(filepath, index=False)
    print(f"  💾 Saved: {filepath}")

    return total_cols, len(constant_cols), len(kept_cols)


def main():
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  CONSTANT COLUMN REMOVAL REPORT")
    report_lines.append("  Removes columns where every row has the same value")
    report_lines.append("=" * 80)

    summary = {}

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith("_cleaned.csv"):
            continue
        filepath = os.path.join(DATA_DIR, fname)
        total, removed, kept = process_file(filepath, report_lines)
        summary[fname] = (total, removed, kept)

    report_lines.append(f"\n\n{'='*80}")
    report_lines.append("  SUMMARY")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"\n  {'File':<35} {'Original':>10} {'Removed':>10} {'Kept':>10}")
    report_lines.append(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

    print(f"\n\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")

    for fname, (total, removed, kept) in summary.items():
        report_lines.append(f"  {fname:<35} {total:>10} {removed:>10} {kept:>10}")
        print(f"  {fname:<35}  {total:>4} → {kept:>4} cols  (removed {removed})")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n📄 Full report: {REPORT_FILE}")


if __name__ == "__main__":
    main()

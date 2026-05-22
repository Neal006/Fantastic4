"""
InverterShield — Pipeline Orchestrator
=======================================

Usage:
    python run_pipeline.py                  # train on featured_data.parquet
    python run_pipeline.py --stage train    # same as above
    python run_pipeline.py --stage features # run feature engineering only

Stages:
    features  — build features + targets from raw hourly CSVs (optional)
    train     — train all 3 models, save artifacts, compute healthy baseline

Prerequisites for 'train':
    Place your featured parquet at:
        ml/processed/featured_data.parquet

    It must contain columns: target_binary, target_multiclass, plus all
    feature columns described in ML_INFO.md Section 4-5.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import log_section, log_step


STAGES = ["features", "train"]


def _run_features():
    log_section("Stage: Feature Engineering")
    log_step("Load your hourly per-inverter DataFrame, then call:")
    log_step("  from src.data_processing.features import build_features")
    log_step("  df_featured = build_features(df_hourly)")
    log_step("  df_featured.to_parquet('processed/featured_data.parquet', index=False)")
    log_step("Or provide the parquet directly (see ML_INFO.md Section 2).")


def _run_train():
    from src.ml.trainer import run
    run()


def main():
    parser = argparse.ArgumentParser(description="InverterShield ML Pipeline")
    parser.add_argument(
        "--stage", type=str, default="train",
        choices=STAGES,
        help="Pipeline stage to run (default: train)",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    log_section("InverterShield ML Pipeline")

    if args.stage == "features":
        _run_features()
    elif args.stage == "train":
        _run_train()

    elapsed = time.perf_counter() - t0
    log_section(f"Done  ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()

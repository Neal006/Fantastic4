"""
run_pipeline.py – End-to-end orchestrator
==========================================
Usage:
    python run_pipeline.py                   # run ALL stages
    python run_pipeline.py --stage ingest    # run one stage
    python run_pipeline.py --sample-frac 0.05  # quick smoke-test on 5 %

Stages (in order):
    ingest -> clean -> features -> labels -> anomaly -> split -> xgb -> catboost -> ensemble -> evaluate -> shap
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import log_section, log_step


# ── Stage registry ───────────────────────────────────────────────
STAGES = [
    "ingest",
    "clean",
    "features",
    "labels",
    "anomaly",
    "split",
    "xgb",
    "catboost",
    "ensemble",
    "evaluate",
    "shap",
]


def _run_stage(name: str, sample_frac: float = 1.0):
    """Import and run the requested pipeline stage."""
    if name == "ingest":
        from preprocessing.data_ingestion import run
        return run(sample_frac=sample_frac)
    elif name == "clean":
        from preprocessing.data_cleaning import run
        return run()
    elif name == "features":
        from preprocessing.feature_engineering import run
        return run()
    elif name == "labels":
        from preprocessing.label_creation import run
        return run()
    elif name == "anomaly":
        from anomaly.anomaly_detector import run
        return run()
    elif name == "split":
        from model.split_and_scale import run
        return run()
    elif name == "xgb":
        from model.train_xgb import run
        return run()
    elif name == "catboost":
        from model.train_catboost import run
        return run()
    elif name == "ensemble":
        from model.ensemble import run
        return run()
    elif name == "evaluate":
        from model.evaluate import run
        return run()
    elif name == "shap":
        from model.shap_explain import run
        return run()
    else:
        raise ValueError(f"Unknown stage: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Inverter Failure-Risk Prediction Pipeline"
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        choices=STAGES,
        help="Run a single stage instead of the full pipeline.",
    )
    parser.add_argument(
        "--sample-frac", type=float, default=1.0,
        help="Fraction of raw data to ingest (for smoke-testing). Default: 1.0",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    log_section("Inverter Failure-Risk Prediction Pipeline")

    if args.stage:
        stages_to_run = [args.stage]
    else:
        stages_to_run = STAGES

    for stage in stages_to_run:
        try:
            _run_stage(stage, sample_frac=args.sample_frac)
        except Exception as e:
            print(f"\n[FAIL] Stage '{stage}' failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elapsed = time.perf_counter() - t0
    log_section(f"Pipeline complete  ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()

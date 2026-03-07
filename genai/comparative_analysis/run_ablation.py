"""
Main ablation study runner.
Runs all test scenarios across all models, evaluates responses,
and saves raw results to JSON for report generation.

Usage:
    cd genai
    python -m comparative_analysis.run_ablation
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Ensure genai/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comparative_analysis.config import MODELS, RESULTS_DIR
from comparative_analysis.model_clients import create_client
from comparative_analysis.test_scenarios import get_all_test_cases, ALL_SCENARIOS
from comparative_analysis.evaluate import evaluate_response


def run_ablation():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_cases = get_all_test_cases()

    print("=" * 70)
    print("  LUMIN.AI  –  Ablation Study / Comparative Analysis")
    print("=" * 70)
    print(f"  Models   : {len(MODELS)}")
    print(f"  Tests    : {len(test_cases)}")
    print(f"  Total    : {len(MODELS) * len(test_cases)} API calls")
    print(f"  Started  : {datetime.utcnow().isoformat()}Z")
    print("=" * 70)

    all_results = {}

    for model_key, model_cfg in MODELS.items():
        print(f"\n{'─' * 60}")
        print(f"  Model: {model_cfg['display_name']}")
        print(f"  Provider: {model_cfg['provider']}  |  ID: {model_cfg['model_id']}")
        print(f"{'─' * 60}")

        # Create client
        try:
            client = create_client(model_cfg)
        except Exception as e:
            print(f"  ✗ Failed to create client: {e}")
            all_results[model_key] = {"error": str(e), "results": []}
            continue

        # Health check
        health = client.health_check()
        if not health["ok"]:
            print(f"  ✗ Health check failed: {health['message']}")
            all_results[model_key] = {"error": health["message"], "results": []}
            continue
        print(f"  ✓ Health check passed")

        model_results = []
        for i, tc in enumerate(test_cases):
            label = f"[{i+1}/{len(test_cases)}] {tc['task']}:{tc['scenario_name']}"
            print(f"  {label} … ", end="", flush=True)

            # Call model
            result = client.generate(tc["system_prompt"], tc["user_prompt"])

            if result["error"]:
                print(f"✗ error: {result['error'][:80]}")
                eval_scores = {
                    "overall_score": 0.0,
                    "json_validity": {"score": 0.0},
                    "hallucination": {"score": 0.0},
                    "urgency_accuracy": {"score": 0.0},
                    "completeness": {"score": 0.0},
                    "response_quality": {"score": 0.0},
                    "latency_seconds": 0.0,
                }
            else:
                # Evaluate
                scenario_data = ALL_SCENARIOS[tc["scenario_name"]]
                eval_scores = evaluate_response(
                    response=result["response"],
                    task=tc["task"],
                    risk_class=tc["risk_class"],
                    shap_features=scenario_data["shap_values"],
                    raw_features=scenario_data["raw_features"],
                    latency_seconds=result["latency_seconds"],
                )
                print(
                    f"✓ score={eval_scores['overall_score']:.2f}  "
                    f"latency={result['latency_seconds']:.1f}s  "
                    f"tokens={result.get('output_tokens', '?')}"
                )

            model_results.append({
                "task": tc["task"],
                "scenario_name": tc["scenario_name"],
                "risk_class": tc["risk_class"],
                "response": result["response"],
                "latency_seconds": result["latency_seconds"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "error": result["error"],
                "evaluation": eval_scores,
            })

            # Small delay to avoid rate limits between calls
            time.sleep(1.5)

        all_results[model_key] = {
            "display_name": model_cfg["display_name"],
            "provider": model_cfg["provider"],
            "model_id": model_cfg["model_id"],
            "results": model_results,
        }

        # Rate-limit pause between models
        print(f"  → Pausing 3s before next model …")
        time.sleep(3)

    # ---------------------------------------------------------------------------
    # Save raw results
    # ---------------------------------------------------------------------------
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"ablation_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Raw results saved to: {results_path}")

    # Also save a "latest" symlink-like copy
    latest_path = RESULTS_DIR / "ablation_results_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"✓ Latest results at: {latest_path}")

    # ---------------------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    header = f"{'Model':<35} {'Avg Score':>10} {'Avg Latency':>12} {'JSON Valid':>10} {'Hallu.':>8}"
    print(header)
    print("─" * 90)

    for model_key, data in all_results.items():
        if "error" in data and not data.get("results"):
            print(f"  {model_key:<33} {'ERROR':>10} – {data.get('error', '')[:40]}")
            continue
        results = data["results"]
        valid = [r for r in results if not r["error"]]
        if not valid:
            print(f"  {data.get('display_name', model_key):<33} {'ALL FAILED':>10}")
            continue

        avg_score = sum(r["evaluation"]["overall_score"] for r in valid) / len(valid)
        avg_latency = sum(r["latency_seconds"] for r in valid) / len(valid)
        json_rate = sum(1 for r in valid if r["evaluation"]["json_validity"]["score"] >= 0.5) / len(valid)
        hallu_score = sum(r["evaluation"]["hallucination"]["score"] for r in valid) / len(valid)

        print(
            f"  {data['display_name']:<33} "
            f"{avg_score:>9.3f} "
            f"{avg_latency:>10.1f}s "
            f"{json_rate:>9.0%} "
            f"{hallu_score:>7.2f}"
        )

    print("=" * 90)
    print(f"\nDone! Run 'python -m comparative_analysis.generate_report' to create graphs.\n")


if __name__ == "__main__":
    run_ablation()

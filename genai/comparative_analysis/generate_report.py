"""
Generate comparative analysis report with graphs from ablation study results.

Usage:
    cd genai
    python -m comparative_analysis.generate_report
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comparative_analysis.config import RESULTS_DIR, GRAPHS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
METRIC_LABELS = {
    "overall_score": "Overall Score",
    "json_validity": "JSON Validity",
    "hallucination": "Hallucination Prevention",
    "urgency_accuracy": "Urgency Accuracy",
    "completeness": "Technical Completeness",
    "response_quality": "Response Quality",
}

TASK_COLORS = {
    "explanation": "#4F46E5",
    "ticket": "#0891B2",
    "chat": "#059669",
}

MODEL_COLORS = [
    "#4F46E5", "#E11D48", "#D97706", "#059669",
    "#7C3AED", "#0891B2", "#DC2626",
]


def load_results() -> dict:
    path = RESULTS_DIR / "ablation_results_latest.json"
    if not path.exists():
        print(f"✗ No results found at {path}")
        print("  Run 'python -m comparative_analysis.run_ablation' first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_metrics(data: dict) -> dict:
    """Aggregate metrics per model."""
    metrics = {}
    for model_key, model_data in data.items():
        results = model_data.get("results", [])
        valid = [r for r in results if not r.get("error")]
        if not valid:
            continue

        display_name = model_data.get("display_name", model_key)
        n = len(valid)

        metrics[model_key] = {
            "display_name": display_name,
            "n_valid": n,
            "n_total": len(results),
            "avg_overall": sum(r["evaluation"]["overall_score"] for r in valid) / n,
            "avg_latency": sum(r["latency_seconds"] for r in valid) / n,
            "avg_json": sum(r["evaluation"]["json_validity"]["score"] for r in valid) / n,
            "avg_hallucination": sum(r["evaluation"]["hallucination"]["score"] for r in valid) / n,
            "avg_urgency": sum(r["evaluation"]["urgency_accuracy"]["score"] for r in valid) / n,
            "avg_completeness": sum(r["evaluation"]["completeness"]["score"] for r in valid) / n,
            "avg_quality": sum(r["evaluation"]["response_quality"]["score"] for r in valid) / n,
            "avg_input_tokens": sum(r["input_tokens"] or 0 for r in valid) / n,
            "avg_output_tokens": sum(r["output_tokens"] or 0 for r in valid) / n,
        }

        # Per-task breakdown
        for task in ["explanation", "ticket", "chat"]:
            task_results = [r for r in valid if r["task"] == task]
            if task_results:
                tn = len(task_results)
                metrics[model_key][f"avg_overall_{task}"] = sum(r["evaluation"]["overall_score"] for r in task_results) / tn
                metrics[model_key][f"avg_latency_{task}"] = sum(r["latency_seconds"] for r in task_results) / tn

    return metrics


# ---------------------------------------------------------------------------
# Graph 1: Overall score comparison (bar chart)
# ---------------------------------------------------------------------------
def plot_overall_scores(metrics: dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [m["display_name"] for m in metrics.values()]
    scores = [m["avg_overall"] for m in metrics.values()]
    colors = MODEL_COLORS[:len(names)]

    bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Average Overall Score (0–1)", fontsize=11)
    ax.set_title("Overall Model Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.1%}", va="center", fontsize=10, fontweight="bold")

    ax.invert_yaxis()
    plt.tight_layout()
    path = GRAPHS_DIR / "01_overall_scores.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ---------------------------------------------------------------------------
# Graph 2: Radar chart – multi-metric comparison
# ---------------------------------------------------------------------------
def plot_radar_chart(metrics: dict):
    categories = ["JSON Validity", "Hallucination\nPrevention", "Urgency\nAccuracy",
                   "Completeness", "Response\nQuality"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for i, (model_key, m) in enumerate(metrics.items()):
        values = [m["avg_json"], m["avg_hallucination"], m["avg_urgency"],
                  m["avg_completeness"], m["avg_quality"]]
        values += values[:1]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(angles, values, "o-", linewidth=2, label=m["display_name"], color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Multi-Metric Radar Comparison", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    path = GRAPHS_DIR / "02_radar_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ---------------------------------------------------------------------------
# Graph 3: Latency comparison
# ---------------------------------------------------------------------------
def plot_latency(metrics: dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [m["display_name"] for m in metrics.values()]
    latencies = [m["avg_latency"] for m in metrics.values()]
    colors = MODEL_COLORS[:len(names)]

    bars = ax.barh(names, latencies, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Average Latency (seconds)", fontsize=11)
    ax.set_title("Response Latency Comparison", fontsize=14, fontweight="bold", pad=15)

    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{lat:.1f}s", va="center", fontsize=10, fontweight="bold")

    ax.invert_yaxis()
    plt.tight_layout()
    path = GRAPHS_DIR / "03_latency_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ---------------------------------------------------------------------------
# Graph 4: Per-task performance grouped bar chart
# ---------------------------------------------------------------------------
def plot_per_task_scores(metrics: dict):
    tasks = ["explanation", "ticket", "chat"]
    task_labels = ["Explanation", "Ticket Generation", "Chat Q&A"]
    model_names = [m["display_name"] for m in metrics.values()]

    x = np.arange(len(tasks))
    n_models = len(metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model_key, m) in enumerate(metrics.items()):
        scores = [m.get(f"avg_overall_{t}", 0) for t in tasks]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=m["display_name"],
                      color=MODEL_COLORS[i % len(MODEL_COLORS)], edgecolor="white")

    ax.set_ylabel("Average Score", fontsize=11)
    ax.set_title("Performance by Task Type", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    path = GRAPHS_DIR / "04_per_task_scores.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ---------------------------------------------------------------------------
# Graph 5: Score vs Latency scatter (efficiency frontier)
# ---------------------------------------------------------------------------
def plot_score_vs_latency(metrics: dict):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (model_key, m) in enumerate(metrics.items()):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.scatter(m["avg_latency"], m["avg_overall"], s=200, c=color,
                   edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(m["display_name"], (m["avg_latency"], m["avg_overall"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlabel("Average Latency (seconds)", fontsize=11)
    ax.set_ylabel("Average Overall Score", fontsize=11)
    ax.set_title("Quality vs Speed Trade-off (top-left is best)", fontsize=14, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = GRAPHS_DIR / "05_score_vs_latency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ---------------------------------------------------------------------------
# Graph 6: Metric breakdown heatmap
# ---------------------------------------------------------------------------
def plot_metric_heatmap(metrics: dict):
    metric_keys = ["avg_json", "avg_hallucination", "avg_urgency", "avg_completeness", "avg_quality"]
    metric_labels = ["JSON Validity", "Hallucination\nPrevention", "Urgency\nAccuracy",
                     "Completeness", "Response\nQuality"]
    model_names = [m["display_name"] for m in metrics.values()]

    data = np.array([[m[k] for k in metric_keys] for m in metrics.values()])

    fig, ax = plt.subplots(figsize=(12, max(4, len(model_names) * 0.8 + 2)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)

    for i in range(len(model_names)):
        for j in range(len(metric_labels)):
            val = data[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=11,
                    fontweight="bold", color=color)

    ax.set_title("Detailed Metric Heatmap", fontsize=14, fontweight="bold", pad=15)
    fig.colorbar(im, ax=ax, label="Score", shrink=0.8)
    plt.tight_layout()
    path = GRAPHS_DIR / "06_metric_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ---------------------------------------------------------------------------
# Graph 7: Token usage comparison
# ---------------------------------------------------------------------------
def plot_token_usage(metrics: dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = [m["display_name"] for m in metrics.values()]
    input_tokens = [m["avg_input_tokens"] for m in metrics.values()]
    output_tokens = [m["avg_output_tokens"] for m in metrics.values()]

    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, input_tokens, width, label="Input Tokens", color="#4F46E5", edgecolor="white")
    ax.bar(x + width/2, output_tokens, width, label="Output Tokens", color="#E11D48", edgecolor="white")

    ax.set_ylabel("Average Tokens", fontsize=11)
    ax.set_title("Token Usage Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9, rotation=20, ha="right")
    ax.legend()
    plt.tight_layout()
    path = GRAPHS_DIR / "07_token_usage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def generate_markdown_report(metrics: dict, data: dict):
    lines = [
        "# LUMIN.AI – Comparative Analysis / Ablation Study",
        "",
        "## Executive Summary",
        "",
        "This report compares **{n_models}** LLM models across **3 task types** "
        "(explanation, ticket generation, chat) and **3 risk scenarios** "
        "(no_risk, degradation_risk, shutdown_risk) for the LUMIN.AI solar-plant "
        "diagnostic use case.".format(n_models=len(metrics)),
        "",
        "### Evaluation Criteria",
        "",
        "| Metric | Weight | Description |",
        "|--------|--------|-------------|",
        "| JSON Validity | 25% | Produces valid JSON with all required schema fields |",
        "| Hallucination Prevention | 25% | Only references provided sensor data and SHAP features |",
        "| Urgency Accuracy | 10% | Assigns correct urgency level for risk class |",
        "| Technical Completeness | 25% | Includes all required technical details |",
        "| Response Quality | 15% | Appropriate length, no refusals, coherent output |",
        "",
        "---",
        "",
        "## Overall Rankings",
        "",
        "| Rank | Model | Overall Score | Avg Latency | JSON Rate | Halluc. Score |",
        "|------|-------|--------------|-------------|-----------|---------------|",
    ]

    ranked = sorted(metrics.items(), key=lambda x: x[1]["avg_overall"], reverse=True)
    for rank, (mk, m) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {m['display_name']} | {m['avg_overall']:.1%} "
            f"| {m['avg_latency']:.1f}s | {m['avg_json']:.0%} | {m['avg_hallucination']:.0%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-Task Breakdown",
        "",
    ]

    for task in ["explanation", "ticket", "chat"]:
        lines.append(f"### {task.title()} Task")
        lines.append("")
        lines.append("| Model | Score | Latency |")
        lines.append("|-------|-------|---------|")
        for mk, m in ranked:
            score = m.get(f"avg_overall_{task}", 0)
            latency = m.get(f"avg_latency_{task}", 0)
            lines.append(f"| {m['display_name']} | {score:.1%} | {latency:.1f}s |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Detailed Metric Scores",
        "",
        "| Model | JSON | Hallucination | Urgency | Completeness | Quality |",
        "|-------|------|---------------|---------|--------------|---------|",
    ]

    for mk, m in ranked:
        lines.append(
            f"| {m['display_name']} "
            f"| {m['avg_json']:.0%} "
            f"| {m['avg_hallucination']:.0%} "
            f"| {m['avg_urgency']:.0%} "
            f"| {m['avg_completeness']:.0%} "
            f"| {m['avg_quality']:.0%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Graphs",
        "",
        "![Overall Scores](graphs/01_overall_scores.png)",
        "",
        "![Radar Comparison](graphs/02_radar_comparison.png)",
        "",
        "![Latency Comparison](graphs/03_latency_comparison.png)",
        "",
        "![Per-Task Scores](graphs/04_per_task_scores.png)",
        "",
        "![Score vs Latency](graphs/05_score_vs_latency.png)",
        "",
        "![Metric Heatmap](graphs/06_metric_heatmap.png)",
        "",
        "![Token Usage](graphs/07_token_usage.png)",
        "",
        "---",
        "",
        "## Recommendation",
        "",
    ]

    best_mk, best_m = ranked[0]
    lines.append(
        f"Based on the ablation study, **{best_m['display_name']}** achieves the highest "
        f"overall score of **{best_m['avg_overall']:.1%}** across all evaluation criteria."
    )
    lines.append("")

    if len(ranked) > 1:
        runner_mk, runner_m = ranked[1]
        lines.append(
            f"The runner-up is **{runner_m['display_name']}** with "
            f"**{runner_m['avg_overall']:.1%}**."
        )
        lines.append("")

    # Find fastest model
    fastest = min(metrics.items(), key=lambda x: x[1]["avg_latency"])
    lines.append(
        f"For latency-sensitive deployments, **{fastest[1]['display_name']}** "
        f"offers the fastest response time at **{fastest[1]['avg_latency']:.1f}s** average."
    )

    lines += [
        "",
        "---",
        "",
        f"*Report generated automatically by LUMIN.AI Ablation Study Framework*",
    ]

    report_path = GRAPHS_DIR.parent / "ABLATION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {report_path.name}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    print("=" * 60)
    print("  LUMIN.AI – Generating Comparative Analysis Report")
    print("=" * 60)

    data = load_results()
    metrics = get_model_metrics(data)

    if not metrics:
        print("✗ No valid model results found. Check ablation output.")
        sys.exit(1)

    print(f"\n  Models with valid results: {len(metrics)}\n")
    print("  Generating graphs …")

    plot_overall_scores(metrics)
    plot_radar_chart(metrics)
    plot_latency(metrics)
    plot_per_task_scores(metrics)
    plot_score_vs_latency(metrics)
    plot_metric_heatmap(metrics)
    plot_token_usage(metrics)

    print("\n  Generating markdown report …")
    report_path = generate_markdown_report(metrics, data)

    print(f"\n{'=' * 60}")
    print(f"  Done! Report: {report_path}")
    print(f"  Graphs: {GRAPHS_DIR}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

"""Render a publication-style comparison directly from completed raw results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "result" / "locomo_100_results_v3.json"
OUTPUT = ROOT / "result" / "locomo_100_comparison.png"


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or len(report.get("rows", [])) != 200:
        raise ValueError("A completed 100-question paired run is required")
    systems = report["summary"]["systems"]
    simple, nexus = systems["SimpleMem"], systems["MemNexus"]
    categories = ["single-hop", "temporal", "multi-hop", "open-domain", "adversarial"]
    labels = ["Single-hop", "Temporal", "Multi-hop", "Open-domain", "Adversarial"]
    first = [simple["by_category"][name]["accuracy"] * 100 for name in categories]
    second = [nexus["by_category"][name]["accuracy"] * 100 for name in categories]

    fig, (quality, efficiency) = plt.subplots(
        1, 2, figsize=(12.6, 4.7), gridspec_kw={"width_ratios": [1.23, 1]}, constrained_layout=True
    )
    x = np.arange(len(categories))
    width = 0.38
    quality.bar(x - width / 2, first, width, color="#9a6700", label="SimpleMem")
    quality.bar(x + width / 2, second, width, color="#1769aa", label="MemNexus")
    quality.set_xticks(x, labels, rotation=22, ha="right")
    quality.set_ylim(0, 105)
    quality.set_ylabel("Judge accuracy (%)")
    quality.set_title("Answer quality by LoCoMo category")
    quality.grid(axis="y", alpha=0.2)
    quality.set_axisbelow(True)
    for position, value in zip(x - width / 2, first):
        quality.text(position, value + 1.3, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    for position, value in zip(x + width / 2, second):
        quality.text(position, value + 1.3, f"{value:.0f}", ha="center", va="bottom", fontsize=8)

    measures = [
        ("Answer calls", "provider_attempts"),
        ("Input tokens", "input_tokens"),
        ("Output tokens", "output_tokens"),
        ("Answer cost", "answer_cost_rmb"),
        ("P50 latency", "latency_p50_ms"),
        ("P95 latency", "latency_p95_ms"),
    ]
    normalized = [100 * nexus[key] / simple[key] for _, key in measures]
    y = np.arange(len(measures))
    efficiency.barh(y, [100] * len(y), height=0.35, color="#9a6700", label="SimpleMem = 100%")
    efficiency.barh(y + 0.35, normalized, height=0.35, color="#1769aa", label="MemNexus")
    efficiency.set_yticks(y + 0.175, [label for label, _ in measures])
    efficiency.invert_yaxis()
    efficiency.set_xlim(0, 119)
    efficiency.set_xlabel("Relative to SimpleMem (%) · lower is better")
    efficiency.set_title("Efficiency at 100 paired questions")
    efficiency.grid(axis="x", alpha=0.2)
    efficiency.set_axisbelow(True)
    for position, value in zip(y + 0.35, normalized):
        efficiency.text(value + 1.5, position, f"{value:.0f}%", va="center", fontsize=8)

    fig.suptitle(
        "LoCoMo D1-D2 subset · 100 paired questions · brown: SimpleMem · blue: MemNexus",
        fontsize=13,
    )
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()

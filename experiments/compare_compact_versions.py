"""Compare complete paired runs without changing predictions or judge labels."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_locomo_disjoint import configure_chinese_font
from benchmark_mixed200 import aggregate, digest, read, save


def main(before, after):
    for folder in (before, after):
        if read(folder / "run.json")["status"] != "completed":
            raise ValueError("Both complete runs are required")
    old, new = [read(p / "results.json") for p in (before, after)]
    a = {(r["item_id"], r["system"]): r for r in old["rows"]}
    b = {(r["item_id"], r["system"]): r for r in new["rows"]}
    if a.keys() != b.keys():
        raise ValueError("Cohorts differ")
    for key in a:
        for field in ("question", "gold_answer", "evaluation_query"):
            if a[key][field] != b[key][field]:
                raise ValueError(f"Changed paired input: {key}/{field}")
        if key[1] == "SimpleMem" and a[key]["prediction"] != b[key]["prediction"]:
            raise ValueError("This comparison requires the same immutable baseline predictions")
    stats = [aggregate(run["rows"], run["stages"]) for run in (old, new)]
    changes = []
    for key in a:
        if key[1] != "MemNexus":
            continue
        baseline = b[key[0], "SimpleMem"]
        changes.append({"item_id": key[0], "dataset": b[key]["dataset"],
                        "question": b[key]["question"], "gold": b[key]["gold_answer"],
                        "before": a[key]["prediction"], "after": b[key]["prediction"],
                        "simplemem": baseline["prediction"],
                        "before_label": a[key]["judge_result"], "after_label": b[key]["judge_result"],
                        "baseline_label": baseline["judge_result"],
                        "f1_before": a[key]["f1"], "f1_after": b[key]["f1"],
                        "before_path": a[key].get("reasoning", {}).get("answer_path"),
                        "after_path": b[key].get("reasoning", {}).get("answer_path")})
    save(after / "version_comparison.json", {
        "before": str(before.resolve()), "after": str(after.resolve()),
        "input_hashes": {str(p / "results.json"): digest(p / "results.json") for p in (before, after)},
        "stats_before": stats[0], "stats_after": stats[1], "items": changes,
        "limitations": ["Repeated development cohort, not held-out generalization",
                        "New memories and stochastic model outputs can change; no isolated causal attribution",
                        "Reused baseline predictions, newly blinded judgments may change labels",
                        "Each LongMemEval subtype has only one item; no significance claim"]})
    configure_chinese_font()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    colors = ["#64748b", "#f59e0b", "#0891b2"]
    labels = ["SimpleMem", "MemNexus 修复前", "MemNexus 修复后"]
    panels = [("accuracy", "正确率", 100), ("mean_f1", "Token F1", 100),
              ("input_tokens", "问答输入 token", 1), ("cost_rmb", "问答费用（元，固定费率估算）", 1)]
    for row, dataset in enumerate(("LoCoMo", "LongMemEval-S")):
        values = [stats[1][dataset]["SimpleMem"], stats[0][dataset]["MemNexus"], stats[1][dataset]["MemNexus"]]
        for col, (metric, title, scale) in enumerate(panels):
            ax = axes[row, col]
            bars = ax.bar(np.arange(3), [v[metric] * scale for v in values], color=colors)
            ax.set_title(f"{dataset}（n={values[0]['n']}）\n{title}")
            ax.set_xticks(np.arange(3), ["基线", "修复前", "修复后"])
            ax.set_ylim(bottom=0)
            if scale == 100:
                ax.set_ylim(0, 110)
            else:
                ax.margins(y=.18)
            ax.bar_label(bars, labels=[f"{v[metric] * scale:.2f}" if scale == 100 or metric == "cost_rmb"
                                      else f"{v[metric]:,}" for v in values], padding=3, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], labels,
               loc="outside lower center", ncol=3)
    fig.suptitle("同题配对开发实验：完整原始历史独立建库；低样本结果不代表显著泛化优势", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(after / f"version_comparison.{suffix}", dpi=180)
    plt.close(fig)
    lines = ["# 修复前后逐题对照", "", "保留所有题目和失败；SimpleMem 原始回答复用、统一裁判重新盲评。",
             "这是一轮开发验证，不是独立测试集，不将一次差值归因于某一个模块。", "",
             "|数据集|题目|修复前|修复后|SimpleMem|F1 前→后|", "|---|---|---|---|---|---|"]
    for item in changes:
        lines.append(f"|{item['dataset']}|{item['item_id']}|{item['before_label']}|{item['after_label']}|"
                     f"{item['baseline_label']}|{item['f1_before']:.3f}→{item['f1_after']:.3f}|")
    for item in changes:
        if item["dataset"] == "LongMemEval-S" or item["after_label"] != "CORRECT":
            lines += ["", f"## {item['item_id']}", "", item["question"], "",
                      f"- Gold：{item['gold']}", f"- SimpleMem：{item['simplemem']}",
                      f"- MemNexus 修复前：{item['before']}", f"- MemNexus 修复后：{item['after']}",
                      f"- 路径：{item['before_path']} → {item['after_path']}"]
    (after / "version_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    main(args.before, args.after)

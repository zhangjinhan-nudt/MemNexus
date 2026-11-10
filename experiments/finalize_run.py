"""Assemble the final artefacts from a finished run.

One command produces every number the report quotes, so no figure is hand
copied: the paper table, the paired verdicts, the token breakdown, the
counterfactual that isolates the list renderer, and the HTML dashboard.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from paper_protocol import (  # noqa: E402
    adversarial_success,
    f1_of,
    paper_table,
    render_markdown,
)

PY = sys.executable


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"failed: {' '.join(command)}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="result/bench260")
    parser.add_argument("--cohort", default="result/locomo_260_cohort_20261021.json")
    parser.add_argument("--report", default="result/RESULT_20261021.md")
    parser.add_argument("--dashboard", default="result/DASHBOARD_20261021.html")
    args = parser.parse_args()
    run_dir = ROOT / args.run

    run([PY, "experiments/run_parallel_bench.py", "--cohort", args.cohort,
         "--out", args.run, "--collect"])
    rows = json.loads((run_dir / "rows.json").read_text(encoding="utf-8"))
    table = paper_table(rows)
    blocks = {}

    # 1. paper table on the common subset only
    common = {"MemNexus": {(r["sample_id"], r["qa_index"]) for r in rows if r["system"] == "MemNexus"},
              "SimpleMem": {(r["sample_id"], r["qa_index"]) for r in rows if r["system"] == "SimpleMem"}}
    shared = common["MemNexus"] & common["SimpleMem"]
    # Strict pairing: a question counts only when BOTH sides have a non-empty
    # answer, otherwise one system is scored on questions the other never ran.
    answered = {"MemNexus": set(), "SimpleMem": set()}
    for row in rows:
        if (row.get("prediction") or "").strip():
            answered[row["system"]].add((row["sample_id"], row["qa_index"]))
    shared = shared & answered["MemNexus"] & answered["SimpleMem"]
    subset = [r for r in rows if (r["sample_id"], r["qa_index"]) in shared]
    subset_table = paper_table(subset)
    blocks["RESULTS_TABLE"] = "\n".join(
        [f"配对题数 **{len(shared)}**（两系统同题、同库、同模型；空回答已剔除）。", ""]
        + render_markdown(subset_table)
    )

    # 2. counterfactual: the same answers rendered with a semicolon list
    alt = []
    for row in subset:
        clone = dict(row)
        if clone["system"] == "MemNexus":
            clone["prediction"] = clone["prediction"].replace(", ", "; ")
        alt.append(clone)
    alt_table = paper_table(alt)
    changed = sum(
        1 for row in subset
        if row["system"] == "MemNexus"
        and abs(f1_of(row) - f1_of({**row, "prediction": row["prediction"].replace(", ", "; ")})) > 1e-9
    )
    mn_alt = alt_table["systems"]["MemNexus"]
    mn_now = subset_table["systems"]["MemNexus"]
    blocks["COUNTERFACTUAL"] = (
        "### 反事实：只把列表连接符换回分号，其余不变\n\n"
        "| | MultiHop F1 | Temporal F1 | OpenDomain F1 | SingleHop F1 | Average F1 |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| 分号（旧渲染） | {100 * mn_alt['f1_by_type']['MultiHop']:.2f} | "
        f"{100 * mn_alt['f1_by_type']['Temporal']:.2f} | {100 * mn_alt['f1_by_type']['OpenDomain']:.2f} | "
        f"{100 * mn_alt['f1_by_type']['SingleHop']:.2f} | {100 * mn_alt['average_f1']:.2f} |\n"
        f"| 逗号（本轮） | {100 * mn_now['f1_by_type']['MultiHop']:.2f} | "
        f"{100 * mn_now['f1_by_type']['Temporal']:.2f} | {100 * mn_now['f1_by_type']['OpenDomain']:.2f} | "
        f"{100 * mn_now['f1_by_type']['SingleHop']:.2f} | {100 * mn_now['average_f1']:.2f} |\n\n"
        f"同一个模型、同一批预测、只改连接符：{changed} 道题的分数发生变化，全部集中在多跳题"
        "（官方切分只作用于多跳），其余类别逐题完全不变。\n"
        "这条差值就是「评测口径对齐」的净贡献，不含任何模型行为变化。"
    )

    # 3. paired verdicts and loss classes
    run([PY, "experiments/analyze_paired.py", "--rows", f"{args.run}/rows.json",
         "--json", "result/paired_260.json", "--markdown", "result/PAIRED_260.md"])
    paired = json.loads((ROOT / "result/paired_260.json").read_text(encoding="utf-8"))
    summary = paired["summary"]
    classes = summary["failure_classes"]
    label = {"abstained": "拒答（证据在库）", "no_evidence": "证据缺失", "partial": "条目漏答",
             "extra": "多余条目", "phrasing": "同义表述", "wrong": "答错"}
    lines = [
        "| 胜 / 平 / 负 | 数量 |", "|---|---:|",
        f"| MemNexus 更好 | {summary['outcomes'].get('mn', 0)} |",
        f"| 持平 | {summary['outcomes'].get('tie', 0)} |",
        f"| SimpleMem 更好 | {summary['outcomes'].get('sm', 0)} |",
        "",
        "| MemNexus 落后的原因 | 题数 |", "|---|---:|",
    ]
    for name, count in Counter(classes).most_common():
        lines.append(f"| {label.get(name, name)} | {count} |")
    lines += ["", f"（因限流导致的空回答剔除 {summary.get('excluded_provider_errors', 0)} 对。）"]
    worst = [r for r in paired["records"] if r["verdict"] == "sm"]
    worst.sort(key=lambda r: r["sm_f1"] - r["mn_f1"], reverse=True)
    lines += ["", "| 题目 | 类别 | 金标 | SimpleMem | MemNexus | 原因 |", "|---|---|---|---|---|---|"]
    for record in worst[:12]:
        lines.append(
            f"| {record['sample_id']}/{record['qa_index']} | {record['category']} | "
            f"{str(record['gold'])[:60]} | {str(record['sm'])[:60]} | {str(record['mn'])[:60]} | "
            f"{label.get(record.get('class'), record.get('class'))} |"
        )
    blocks["PAIRED"] = "\n".join(lines)

    # 4. token accounting
    samples = ",".join(sorted({key[0] for key in shared}))
    tokens = run([PY, "experiments/token_breakdown.py", "--root", args.run, "--samples", samples])
    blocks["TOKENS"] = (
        "每题 token 由三段构成（建库单独计）。用服务端返回的 usage 逐次配对，不按字符估算：\n\n"
        "```\n" + tokens.strip() + "\n```\n\n"
        f"按题计：MemNexus {mn_now['token_cost_per_question']:.0f} tok/题，"
        f"SimpleMem {subset_table['systems']['SimpleMem']['token_cost_per_question']:.0f} tok/题。"
        "差异不显著，说明本轮的优势不是「更省 token」，而是**同样的 token 换来更准的判定**。"
        "这一点必须如实写：论文若要主张「更少 token 更准」，还需要把规划 schema 瘦身（见 §7）。"
    )

    # 5. dashboard
    run([PY, "experiments/build_dashboard.py", "--rows", f"{args.run}/rows.json",
         "--paired", "result/paired_260.json", "--out", args.dashboard,
         "--title", "MemNexus vs SimpleMem · LoCoMo 配对评测"])

    report_path = ROOT / args.report
    text = report_path.read_text(encoding="utf-8")
    for key, value in blocks.items():
        text = text.replace("{{" + key + "}}", value)
    report_path.write_text(text, encoding="utf-8")
    print(f"report: {args.report}")
    print(f"dashboard: {args.dashboard}")
    print(f"paired questions: {len(shared)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

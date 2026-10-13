"""Audit the live category-1 rerun against the saved development baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "result/locomo_100_disjoint_v2_complete.json"
COHORT = ROOT / "result/locomo_100_cohort_v2.json"
DEFAULT = ROOT / "result/locomo_multihop_retest_20261013.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_coverage(row: dict, ledger_dir: Path) -> int:
    snapshot = read(ledger_dir / f"{row['sample_id']}_memnexus.memory.json")
    raw = read(ledger_dir / f"{row['sample_id']}_simplemem.memory.json")["raw_turns"] if (
        ledger_dir / f"{row['sample_id']}_simplemem.memory.json"
    ).exists() else None
    if raw is None:
        # The new rerun saves the same turn order as the original paired run.
        raw = read(ROOT / "result/locomo_100_disjoint_v2_complete_ledgers" / f"{row['sample_id']}_simplemem.memory.json")["raw_turns"]
    assert len(snapshot["turns"]) == len(raw)
    mapping = {
        turn["turn_id"]: source["dia_id"]
        for turn, source in zip(snapshot["turns"], raw, strict=True)
    }
    chosen = {
        mapping[event["turn_id"]]
        for hit in row["retrieved_sources"]
        for event in hit["metadata"].get("source_events", [])
        if event["turn_id"] in mapping
    }
    return len(chosen & set(row["gold_evidence"]))


def analyze(report: dict, run_path: Path) -> dict:
    if report["status"] != "completed" or len(report["rows"]) != 13:
        raise ValueError("Requires 13 fully judged live answers")
    cohort = read(COHORT)
    old = read(OLD)
    assert report["selection_sha256"] == cohort["selection_sha256"] == old["selection_sha256"]
    changed = [path for path, sha in report["source_sha256"].items()
               if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha]
    if changed:
        raise ValueError(f"Evaluated source changed: {changed}")
    old_rows = {(r["sample_id"], r["qa_index"], r["system"]): r for r in old["rows"]}
    selected = {(item["sample_id"], item["qa_index"]) for item in cohort["questions"] if item["category_id"] == 1}
    assert len(selected) == 13
    assert {(r["sample_id"], r["qa_index"]) for r in report["rows"]} == selected
    assert len(report["stages"]) == len(report["judge_usage"]) == len({sid for sid, _ in selected})
    ledger_dir = run_path.with_name(run_path.stem + "_ledgers")
    stage_ledgers = [read(path) for path in ledger_dir.glob("*.json") if not path.name.endswith((".memory.json", ".stage.json"))]
    ledger_total = sum(item.get("cost_rmb", 0) + item.get("reserved_rmb", 0) for item in stage_ledgers)
    assert abs(ledger_total - report["run_cost_rmb"]) < 1e-8
    assert all(item.get("reserved_rmb", 0) == 0 for item in stage_ledgers)

    old_dir = OLD.with_name(OLD.stem + "_ledgers")
    details = []
    for row in report["rows"]:
        key = row["sample_id"], row["qa_index"]
        old_mem = old_rows[(*key, "MemNexus")]
        old_simple = old_rows[(*key, "SimpleMem")]
        if row["judge_result"] not in {"CORRECT", "WRONG"}:
            raise ValueError("Missing judge label")
        gold_count = len(set(row["gold_evidence"]))
        before = source_coverage(old_mem, old_dir)
        after = source_coverage(row, ledger_dir)
        details.append({
            "sample_id": key[0], "qa_index": key[1], "question": row["question"],
            "gold_answer": row["gold_answer"],
            "old_MemNexus_prediction": old_mem["prediction"],
            "new_MemNexus_prediction": row["prediction"],
            "SimpleMem_prediction": old_simple["prediction"],
            "old_MemNexus_correct": old_mem["judge_result"] == "CORRECT",
            "new_MemNexus_correct": row["judge_result"] == "CORRECT",
            "SimpleMem_correct": old_simple["judge_result"] == "CORRECT",
            "new_judge_reason": row["judge_reason"],
            "gold_evidence_count": gold_count,
            "old_selected_gold_count": before,
            "new_selected_gold_count": after,
            "old_all_gold": before == gold_count,
            "new_all_gold": after == gold_count,
            "new_grounding": row["grounding"],
            "new_model_calls": row["model_calls"],
            "old_model_calls": old_mem["model_calls"],
            "new_latency_ms": row["latency_ms"],
            "old_latency_ms": old_mem["latency_ms"],
        })
    wins = sum(not item["old_MemNexus_correct"] and item["new_MemNexus_correct"] for item in details)
    losses = sum(item["old_MemNexus_correct"] and not item["new_MemNexus_correct"] for item in details)
    result = {
        "scope": "rerun of previously tested category-1 development questions; no held-out accuracy claim",
        "source": str(run_path),
        "question_count": len(details),
        "old_MemNexus_correct": sum(item["old_MemNexus_correct"] for item in details),
        "new_MemNexus_correct": sum(item["new_MemNexus_correct"] for item in details),
        "old_SimpleMem_correct": sum(item["SimpleMem_correct"] for item in details),
        "old_wrong_to_new_correct": wins,
        "old_correct_to_new_wrong": losses,
        "old_all_gold": sum(item["old_all_gold"] for item in details),
        "new_all_gold": sum(item["new_all_gold"] for item in details),
        "old_wrong_new_all_gold": sum(item["new_all_gold"] for item in details if not item["old_MemNexus_correct"]),
        "qa_provider_attempts_old": sum(item["old_model_calls"]["provider_attempts"] for item in details),
        "qa_provider_attempts_new": sum(item["new_model_calls"]["provider_attempts"] for item in details),
        "qa_input_tokens_old": sum(item["old_model_calls"]["input_tokens"] for item in details),
        "qa_input_tokens_new": sum(item["new_model_calls"]["input_tokens"] for item in details),
        "qa_output_tokens_old": sum(item["old_model_calls"]["output_tokens"] for item in details),
        "qa_output_tokens_new": sum(item["new_model_calls"]["output_tokens"] for item in details),
        "qa_cost_rmb_old": sum(item["old_model_calls"]["cost_rmb"] for item in details),
        "qa_cost_rmb_new": sum(item["new_model_calls"]["cost_rmb"] for item in details),
        "qa_p50_seconds_old": statistics.median(item["old_latency_ms"] for item in details) / 1000,
        "qa_p50_seconds_new": statistics.median(item["new_latency_ms"] for item in details) / 1000,
        "qa_p95_seconds_old": sorted(item["old_latency_ms"] for item in details)[12] / 1000,
        "qa_p95_seconds_new": sorted(item["new_latency_ms"] for item in details)[12] / 1000,
        "build_cost_rmb_new": sum(stage["build_usage"]["cost_rmb"] for stage in report["stages"]),
        "total_run_cost_rmb": report["run_cost_rmb"],
        "details": details,
    }
    return result


def plot(result: dict, base: Path):
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    old, new = result["old_MemNexus_correct"], result["new_MemNexus_correct"]
    baseline = result["old_SimpleMem_correct"]
    bars = axes[0].bar(["MemNexus 旧版", "MemNexus 新版", "SimpleMem 原实验"],
                       [old, new, baseline], color=["#A6B3BF", "#2878B3", "#D1882E"])
    axes[0].bar_label(bars, labels=[f"{value}/13" for value in (old, new, baseline)], padding=4)
    axes[0].set_ylim(0, 14.5)
    axes[0].set_title("类别1：统一裁判正确题数")
    axes[0].tick_params(axis="x", rotation=18)
    bars = axes[1].bar(["旧版最终来源", "新版最终来源"],
                       [result["old_all_gold"], result["new_all_gold"]],
                       color=["#A6B3BF", "#2878B3"])
    axes[1].bar_label(bars, padding=4)
    axes[1].set_ylim(0, 14.5)
    axes[1].set_title("检齐全部 gold turn 的题数")
    axes[1].set_ylabel("题数 / 13")
    values = [result["qa_input_tokens_old"] / 1000, result["qa_input_tokens_new"] / 1000]
    bars = axes[2].bar(["旧版", "新版"], values, color=["#A6B3BF", "#2878B3"])
    axes[2].bar_label(bars, labels=[f"{value:.1f}k" for value in values], padding=4)
    axes[2].set_title("13题问答输入 tokens")
    axes[2].set_ylabel("千 tokens")
    for axis in axes:
        axis.grid(axis="y", alpha=.16)
        axis.set_axisbelow(True)
    fig.suptitle("既有多跳错题的开发集复测", fontsize=14, y=.99)
    fig.text(.5, .015, "同一13题重新建库、生成与裁判；结果受模型波动影响，不是独立保留测试或单组件因果效应。", ha="center", fontsize=9)
    fig.subplots_adjust(left=.06, right=.99, top=.83, bottom=.23, wspace=.35)
    for extension in ("png", "svg"):
        fig.savefig(base.with_suffix("." + extension), dpi=180)
    plt.close(fig)


def markdown(result: dict) -> str:
    lines = [
        "# MemNexus 多跳类别13题真实模型复测", "",
        "这13题来自上一轮 LoCoMo 对比的类别1，题目与旧版逐题相同，已经用于错误分析，属于开发集复测。重新从相同 D1–D5 原始对话独立构建 MemNexus 记忆，并由相同 DeepSeek 裁判评分。", "",
        f"旧版 MemNexus {result['old_MemNexus_correct']}/13，新版 {result['new_MemNexus_correct']}/13，原实验 SimpleMem {result['old_SimpleMem_correct']}/13。旧错→新对 {result['old_wrong_to_new_correct']} 题，旧对→新错 {result['old_correct_to_new_wrong']} 题。", "",
        f"最终来源检齐全部官方 gold turn 的题数 {result['old_all_gold']}→{result['new_all_gold']}。问答输入 tokens {result['qa_input_tokens_old']:,}→{result['qa_input_tokens_new']:,}；provider attempts {result['qa_provider_attempts_old']}→{result['qa_provider_attempts_new']}；问答模型费用估算 ¥{result['qa_cost_rmb_old']:.4f}→¥{result['qa_cost_rmb_new']:.4f}；P50 {result['qa_p50_seconds_old']:.3f}s→{result['qa_p50_seconds_new']:.3f}s，P95 {result['qa_p95_seconds_old']:.3f}s→{result['qa_p95_seconds_new']:.3f}s。", "",
        "## 逐题原始对照", "", "| ID | 问题 / gold | 旧版→新版 | gold 来源覆盖 | 裁判理由 |", "|---|---|---|---|---|",
    ]
    for item in result["details"]:
        label = f"{'✓' if item['old_MemNexus_correct'] else '×'}→{'✓' if item['new_MemNexus_correct'] else '×'}"
        reason = item["new_judge_reason"].replace("|", "\\|").replace("\n", "<br>")
        question = item["question"].replace("|", "\\|")
        gold = str(item["gold_answer"]).replace("|", "\\|")
        lines.append(
            f"| {item['sample_id']}/{item['qa_index']} | {question}<br>{gold} | {label} | "
            f"{item['old_selected_gold_count']}/{item['gold_evidence_count']}→"
            f"{item['new_selected_gold_count']}/{item['gold_evidence_count']} | {reason} |"
        )
    lines += ["", "## 净变化案例", ""]
    for item in result["details"]:
        if item["old_MemNexus_correct"] == item["new_MemNexus_correct"]:
            continue
        title = "修复" if item["new_MemNexus_correct"] else "退步"
        lines += [
            f"### {title}：{item['sample_id']}/{item['qa_index']}", "",
            f"问题：{item['question']}", "",
            f"Gold：{item['gold_answer']}", "",
            f"旧版：{item['old_MemNexus_prediction']}", "",
            f"新版：{item['new_MemNexus_prediction']}", "",
            f"裁判：{item['new_judge_reason']}", "",
        ]
    lines += ["", "## 解释边界", "", "离线来源回放使用旧记忆快照，仅验证选择算法；本次真实模型复测重新建库，抽取和裁判存在随机性。准确率变化不能全部归因于一个检索组件。题目已用于设计，不能报告为独立泛化结果。后续要在新的对话与任务上固定版本验证。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT)
    parser.add_argument("--artifact-stem", default="locomo_multihop_retest")
    args = parser.parse_args()
    path = args.run.resolve()
    result = analyze(read(path), path)
    base = ROOT / "result" / args.artifact_stem
    base.with_name(base.name + "_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base.with_name(base.name.upper() + "_ANALYSIS.md").write_text(markdown(result), encoding="utf-8")
    plot(result, base.with_name(base.name + "_comparison"))
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Write a source-aware error table for a completed disjoint paired run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "result/locomo_100_disjoint_v2_final.json"
ANALYSIS = ROOT / "result/locomo_100_disjoint_v2_analysis.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    args = parser.parse_args()
    report = read(args.run)
    analysis = read(args.analysis)
    if report["status"] != "completed":
        raise ValueError("Run must have completed 100 paired questions")

    simple = {
        (row["sample_id"], row["qa_index"]): row
        for row in report["rows"]
        if row["system"] == "SimpleMem"
    }
    evidence = {
        (row["sample_id"], row["qa_index"]): row
        for row in analysis["evidence_recall"]["rows"]
    }
    cases = []
    for row in report["rows"]:
        if row["system"] != "MemNexus" or row["judge_result"] != "WRONG":
            continue
        key = row["sample_id"], row["qa_index"]
        baseline = simple[key]
        source = evidence.get(key)
        if row["category_id"] == 5:
            stage = "negative_answer_or_judge"
        elif source is None or not source["any_gold"]:
            stage = (
                "gold_source_referenced_in_memory_but_not_selected"
                if source is not None and source["memory_any_gold"]
                else "gold_source_not_referenced_by_memory"
            )
        elif not source["all_gold"]:
            stage = "gold_source_partially_covered"
        else:
            stage = "gold_sources_present_answer_or_inference_failed"
        cases.append(
            {
                "sample_id": key[0],
                "qa_index": key[1],
                "category_id": row["category_id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "MemNexus_prediction": row["prediction"],
                "SimpleMem_prediction": baseline["prediction"],
                "SimpleMem_correct": baseline["judge_result"] == "CORRECT",
                "judge_reason": row.get("judge_reason", ""),
                "triage_stage": stage,
                "gold_evidence": row.get("gold_evidence", []),
                "evidence_hit": source,
                "question_route": row.get("plan", {}).get("query_type")
                if row.get("plan")
                else None,
                "grounding": row.get("grounding", {}),
                "citations": row.get("citations", []),
            }
        )
    assert len(cases) == 100 - analysis["systems"]["MemNexus"]["correct"]
    counts = Counter(case["triage_stage"] for case in cases)
    payload = {
        "run": str(args.run),
        "wrong_count": len(cases),
        "stage_counts": dict(counts),
        "cases": cases,
        "warning": "Triage stages use final gold-turn coverage; they are diagnostic labels, not proven causal attributions.",
    }
    output = ROOT / "result/LOCOMO_100_DISJOINT_V2_ERRORS.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 第二组 LoCoMo 100 题：MemNexus 逐题错误审计",
        "",
        f"裁判标为错误的 MemNexus 答案共有 **{len(cases)}** 道。以下分类只按最终来源是否覆盖官方 evidence 判断问题可能发生的阶段，不能将其直接解释为算法的唯一根因。",
        "",
        "| 可观察阶段 | 题数 | 判断口径 |",
        "|---|---:|---|",
    ]
    descriptions = {
        "gold_source_referenced_in_memory_but_not_selected": "建库引用过 gold turn，最终来源没有；优先检查召回与证据选择。",
        "gold_source_not_referenced_by_memory": "建库记忆没有引用 gold turn；检查抽取和原文索引。",
        "gold_source_partially_covered": "只包含部分 gold turn；关注多跳、集合和因果链覆盖。",
        "gold_sources_present_answer_or_inference_failed": "全部 gold turn 已在最终来源；关注推理、指代、时间、答案格式及 gold 争议。",
        "negative_answer_or_judge": "对抗题的 gold 是未提及；turn 覆盖率不能证明否定答案。",
    }
    for name, count in counts.most_common():
        lines.append(f"| {name} | {count} | {descriptions[name]} |")
    lines += [
        "",
        "## 逐题核对",
        "",
        "每行均来自真实预测、统一裁判和保存的来源。完整来源、裁判原始响应及 token 记录在运行 JSON 和 ledgers 中。",
        "",
        "| ID / 类别 | 问题与 gold | 两套答案和裁判理由 | 证据情况 |",
        "|---|---|---|---|",
    ]
    for case in cases:
        source = case["evidence_hit"]
        coverage = (
            "对抗题：不适用 turn 召回"
            if source is None
            else f"any={source['any_gold']}，all={source['all_gold']}，{case['triage_stage']}"
        )
        lines.append(
            f"| {case['sample_id']}/{case['qa_index']} / {case['category_id']} | "
            f"{cell(case['question'])}<br>Gold: {cell(case['gold_answer'])} | "
            f"SimpleMem: {cell(case['SimpleMem_prediction'])}<br>"
            f"MemNexus: {cell(case['MemNexus_prediction'])}<br>"
            f"裁判: {cell(case['judge_reason'])} | {coverage} |"
        )
    lines += [
        "",
        "## 后续算法实验",
        "",
        "1. 对 gold 来源缺失题，固定问答器，仅改变事件图写入、查询路由和原文召回策略；测 turn Recall@K、输入 token 与准确率。",
        "2. 对 gold 来源已齐仍答错题，固定检索上下文，分别消融时间区间运算、关系 join、类型约束和答案选择；人工复核 gold 争议。",
        "3. 对对抗题，测主体、对象类型、时间三个约束的 false-positive 与 false-negative，不用单一总体分数掩盖两种错误。",
        "4. 将回答预算按问题类型分配：单事实保留低成本；多跳、集合和计数题提高来源覆盖上限，记录 accuracy–token–latency 曲线。",
        "5. 完成问答改进后，再将可追溯事件图用作 ICAPS belief state，测执行成功率、局部重规划成功率和约束违反。",
        "",
    ]
    (ROOT / "result/LOCOMO_100_DISJOINT_V2_ERRORS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {"wrong": len(cases), "stage_counts": dict(counts), "output": str(output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

"""Audit and visualize the disjoint LoCoMo 100-question paired experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_optimized_comparison import (
    audit,
    cluster_ci,
    correct,
    mcnemar,
    metric,
    rows_by_key,
)
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OLD_COHORT = ROOT / "result/locomo_100_cohort.json"
DEFAULT_RUN = ROOT / "result/locomo_100_disjoint_v2_final.json"
SYSTEMS = ("SimpleMem", "MemNexus")
COLORS = {"SimpleMem": "#D1882E", "MemNexus": "#2878B3"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_recall(report: dict, path: Path) -> dict:
    """Map verified MemNexus turn IDs back to official dia_id by ingest order."""
    ledgers = path.with_name(path.stem + "_ledgers")
    turn_ids: dict[str, dict[str, str]] = {}
    memory_references: dict[str, set[str]] = {}
    for stage in report["stages"]:
        if stage["system"] != "MemNexus":
            continue
        sample_id = stage["sample_id"]
        mem = read(ledgers / f"{sample_id}_memnexus.memory.json")
        simple = read(ledgers / f"{sample_id}_simplemem.memory.json")
        assert len(mem["turns"]) == len(simple["raw_turns"]) == stage["raw_turns"]
        turn_ids[sample_id] = {
            turn["turn_id"]: raw["dia_id"]
            for turn, raw in zip(mem["turns"], simple["raw_turns"], strict=True)
        }
        memory_references[sample_id] = {
            turn_ids[sample_id][evidence["turn_id"]]
            for memory in mem["memories"]
            for evidence in memory["evidence"]
            if evidence["turn_id"] in turn_ids[sample_id]
        }

    outcomes = []
    for row in report["rows"]:
        if row["system"] != "MemNexus" or row["category_id"] == 5:
            continue
        official = set(row["gold_evidence"])
        mapped = turn_ids[row["sample_id"]]
        referenced = memory_references[row["sample_id"]]
        final = {
            mapped[record["turn_id"]]
            for hit in row["retrieved_sources"]
            for record in hit["metadata"].get("source_events", [])
            if record["turn_id"] in mapped
        }
        outcomes.append(
            {
                "sample_id": row["sample_id"],
                "qa_index": row["qa_index"],
                "gold_count": len(official),
                "retrieved_count": len(final),
                "any_gold": bool(official & final),
                "all_gold": official <= final,
                "memory_any_gold": bool(official & referenced),
                "memory_all_gold": official <= referenced,
                "judge_correct": bool(correct(row)),
            }
        )
    wrong = [item for item in outcomes if not item["judge_correct"]]
    return {
        "non_adversarial_n": len(outcomes),
        "memory_any_gold_count": sum(item["memory_any_gold"] for item in outcomes),
        "memory_all_gold_count": sum(item["memory_all_gold"] for item in outcomes),
        "any_gold_count": sum(item["any_gold"] for item in outcomes),
        "all_gold_count": sum(item["all_gold"] for item in outcomes),
        "stored_all_but_final_incomplete": sum(
            item["memory_all_gold"] and not item["all_gold"] for item in outcomes
        ),
        "wrong_n": len(wrong),
        "wrong_with_all_gold": sum(item["all_gold"] for item in wrong),
        "wrong_without_any_gold": sum(not item["any_gold"] for item in wrong),
        "wrong_without_any_memory_reference": sum(
            not item["memory_any_gold"] for item in wrong
        ),
        "wrong_with_memory_reference_but_no_final_source": sum(
            item["memory_any_gold"] and not item["any_gold"] for item in wrong
        ),
        "rows": outcomes,
        "mapping": "ingest-order turn_id to official dia_id; final selected source_events only",
    }


def compare(report: dict, path: Path) -> dict:
    checked = audit(report, path)
    previous = read(OLD_COHORT)
    old_keys = {(q["sample_id"], q["qa_index"]) for q in previous["questions"]}
    new_keys = {(q["sample_id"], q["qa_index"]) for q in report["cohort"]["questions"]}
    assert len(old_keys) == len(new_keys) == 100
    assert not old_keys & new_keys
    assert report["cohort"]["max_session"] == report["environment"]["max_session"] == 5
    assert len(report["judge_usage"]) == 10

    indexed = rows_by_key(report)
    clusters: dict[str, list[int]] = defaultdict(list)
    pairs = Counter()
    per_category = {}
    for item in report["cohort"]["questions"]:
        key = item["sample_id"], item["qa_index"]
        simple, mem = (indexed[(*key, system)] for system in SYSTEMS)
        a, b = correct(simple), correct(mem)
        clusters[key[0]].append(b - a)
        pairs[(a, b)] += 1
    for category_id in range(1, 6):
        selected = [
            q for q in report["cohort"]["questions"] if q["category_id"] == category_id
        ]
        per_category[str(category_id)] = {
            "n": len(selected),
            "SimpleMem": sum(
                correct(indexed[(q["sample_id"], q["qa_index"], "SimpleMem")])
                for q in selected
            ),
            "MemNexus": sum(
                correct(indexed[(q["sample_id"], q["qa_index"], "MemNexus")])
                for q in selected
            ),
        }

    metrics = {system: metric(report, system) for system in SYSTEMS}
    mem_rows = [row for row in report["rows"] if row["system"] == "MemNexus"]
    failed_stage_costs = {}
    for stem in (
        "locomo_100_disjoint_v2_final",
        "locomo_100_disjoint_v2_resumed",
        "locomo_100_disjoint_v2_continue",
    ):
        folder = ROOT / "result" / (stem + "_ledgers")
        failed_stage_costs[stem] = sum(
            read(ledger).get("cost_rmb", 0) + read(ledger).get("reserved_rmb", 0)
            for ledger in folder.glob("conv-48_*.json")
            if not ledger.name.endswith((".stage.json", ".memory.json"))
        )
    result = {
        "scope": "LoCoMo 100 disjoint questions; official evidence in D1-D5; both systems rebuilt from same D1-D5 turns",
        "run": str(path),
        "selection_sha256": report["selection_sha256"],
        "previous_selection_sha256": previous["selection_sha256"],
        "overlap_count": 0,
        "systems": metrics,
        "by_category_id": per_category,
        "paired": {
            "both_correct": pairs[(1, 1)],
            "mem_only": pairs[(0, 1)],
            "simple_only": pairs[(1, 0)],
            "both_wrong": pairs[(0, 0)],
            "difference_pp": metrics["MemNexus"]["correct"]
            - metrics["SimpleMem"]["correct"],
            "cluster_95ci_pp": [
                100 * x for x in cluster_ci(clusters, n=10000, seed=20261013)
            ],
            "mcnemar_p_question_independence": mcnemar(pairs[(0, 1)], pairs[(1, 0)]),
        },
        "evidence_recall": evidence_recall(report, path),
        "audit": checked,
        "reasoning_observation": {
            "routed_questions": dict(
                Counter(row["plan"].get("query_type") for row in mem_rows)
            ),
            "answer_model_calls": sum(
                row["runtime_model_calls"]["answer_calls"] for row in mem_rows
            ),
            "direct_gate_answers": sum(
                row["runtime_model_calls"]["answer_calls"] == 0 for row in mem_rows
            ),
            "gate_trace_saved": sum(bool(row.get("reasoning")) for row in mem_rows),
        },
        "cost_accounting": {
            "completed_run_upper_rmb": checked["cost_rmb"],
            "abandoned_stage_upper_rmb": failed_stage_costs,
            "all_attempts_upper_rmb": checked["cost_rmb"]
            + sum(failed_stage_costs.values()),
        },
        "per_conversation": {
            sample_id: {
                "n": len(differences),
                "SimpleMem": metrics["SimpleMem"]["conversations"][sample_id][
                    "correct"
                ],
                "MemNexus": metrics["MemNexus"]["conversations"][sample_id]["correct"],
            }
            for sample_id, differences in sorted(clusters.items())
        },
    }
    return result


def configure_chinese_font():
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font_path)
        ).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FAFBFD",
            "axes.facecolor": "#FAFBFD",
            "svg.fonttype": "none",
        }
    )


def graph(result: dict, base: Path):
    configure_chinese_font()
    s, m = (result["systems"][name] for name in SYSTEMS)
    fig, axes = plt.subplots(2, 3, figsize=(15.3, 9.1))

    ax = axes[0, 0]
    bars = ax.bar(
        SYSTEMS,
        [s["correct"], m["correct"]],
        color=[COLORS[name] for name in SYSTEMS],
        width=0.55,
    )
    ax.bar_label(
        bars, labels=[f"{v}%" for v in (s["correct"], m["correct"])], padding=4
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("裁判正确率 (%)，每组100题")
    ax.set_title("回答正确率 ↑")
    lo, hi = result["paired"]["cluster_95ci_pp"]
    ax.set_xlabel(
        f"MemNexus−SimpleMem：{result['paired']['difference_pp']:+d}百分点；对话重采样95%区间 [{lo:.1f}, {hi:.1f}]"
    )

    ax = axes[0, 1]
    labels = [
        f"类别{k}\n(n={result['by_category_id'][str(k)]['n']})" for k in range(1, 6)
    ]
    x = np.arange(5)
    for offset, system in ((-0.18, "SimpleMem"), (0.18, "MemNexus")):
        values = [
            100
            * result["by_category_id"][str(k)][system]
            / result["by_category_id"][str(k)]["n"]
            for k in range(1, 6)
        ]
        bars = ax.bar(x + offset, values, 0.36, color=COLORS[system], label=system)
        ax.bar_label(bars, labels=[f"{v:.0f}" for v in values], padding=3, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 113)
    ax.set_title("原始类别编号分层")
    ax.set_ylabel("裁判正确率 (%)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 2]
    labels = ["输入", "输出", "建库+问答"]
    a = [
        s["qa_input_tokens"] / 1000,
        s["qa_output_tokens"] / 1000,
        s["main_total_tokens"] / 1000,
    ]
    b = [
        m["qa_input_tokens"] / 1000,
        m["qa_output_tokens"] / 1000,
        m["main_total_tokens"] / 1000,
    ]
    x = np.arange(3)
    for offset, system, values in ((-0.18, "SimpleMem", a), (0.18, "MemNexus", b)):
        bars = ax.bar(x + offset, values, 0.36, color=COLORS[system])
        ax.bar_label(bars, labels=[f"{v:.0f}" for v in values], padding=3, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_title("模型 token 用量 ↓")
    ax.set_ylabel("千 tokens")
    ax.margins(y=0.17)

    ax = axes[1, 0]
    for index, (system, stats) in enumerate((("SimpleMem", s), ("MemNexus", m))):
        ax.bar(
            index, stats["build_cost_rmb"], width=0.55, color=COLORS[system], alpha=0.43
        )
        ax.bar(
            index,
            stats["qa_cost_rmb"],
            width=0.55,
            bottom=stats["build_cost_rmb"],
            color=COLORS[system],
        )
        ax.text(
            index,
            stats["main_cost_rmb"] + 0.1,
            f"¥{stats['main_cost_rmb']:.2f}",
            ha="center",
        )
    ax.set_xticks([0, 1], SYSTEMS)
    ax.set_title("主模型费用估算 ↓")
    ax.set_ylabel("人民币；浅色建库，深色问答")
    ax.margins(y=0.2)

    ax = axes[1, 1]
    measures = [
        ("问答请求", "qa_provider_attempts"),
        ("P50 延迟", "p50_seconds"),
        ("P95 延迟", "p95_seconds"),
    ]
    labels = [item[0] for item in measures]
    values = [
        100 * m[field] / s[field] if s[field] else math.nan for _, field in measures
    ]
    bars = ax.bar(labels, values, color=COLORS["MemNexus"], width=0.55)
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in values], padding=4)
    ax.axhline(100, color="#777", linestyle="--", linewidth=1)
    ax.set_ylim(0, max(120, max(values) * 1.25))
    ax.set_title("相对 SimpleMem 的效率 ↓")
    ax.set_ylabel("SimpleMem = 100%")

    ax = axes[1, 2]
    keys = ("both_correct", "mem_only", "simple_only", "both_wrong")
    labels = ("两者都对", "仅MemNexus对", "仅SimpleMem对", "两者都错")
    values = [result["paired"][key] for key in keys]
    bars = ax.bar(
        range(4),
        values,
        color=["#72937E", COLORS["MemNexus"], COLORS["SimpleMem"], "#AEB5BD"],
    )
    ax.bar_label(bars, padding=3)
    ax.set_xticks(range(4), labels, rotation=15)
    ax.set_title("逐题配对胜负")
    ax.set_ylabel("题数")
    ax.margins(y=0.16)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.17)
        ax.set_axisbelow(True)
    fig.suptitle(
        "LoCoMo 新100题：MemNexus 与 SimpleMem 完整配对比较",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "问题与旧100题零重叠；10段对话D1–D5；两边分别建库。费用为按实际token估算，非账户账单。",
        ha="center",
        color="#555",
        fontsize=9,
    )
    fig.subplots_adjust(
        top=0.90, bottom=0.13, left=0.07, right=0.98, hspace=0.52, wspace=0.36
    )
    for extension in ("png", "svg"):
        fig.savefig(base.with_suffix("." + extension), dpi=170)
    plt.close(fig)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.8, 5.3))
    ids = list(result["per_conversation"])
    x = np.arange(len(ids))
    for system in SYSTEMS:
        vals = [
            100
            * result["per_conversation"][key][system]
            / result["per_conversation"][key]["n"]
            for key in ids
        ]
        ax.plot(
            x, vals, "o-", linewidth=2, markersize=5, color=COLORS[system], label=system
        )
    ax.set_xticks(
        x,
        [f"{key}\n(n={result['per_conversation'][key]['n']})" for key in ids],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_ylim(0, 110)
    ax.set_title("各对话的正确率")
    ax.set_ylabel("正确率 (%)")
    ax.legend(frameon=False)
    recall = result["evidence_recall"]
    labels = ("建库引用齐全", "最终至少一条", "最终全部齐全", "错题已检齐")
    values = (
        recall["memory_all_gold_count"] / recall["non_adversarial_n"] * 100,
        recall["any_gold_count"] / recall["non_adversarial_n"] * 100,
        recall["all_gold_count"] / recall["non_adversarial_n"] * 100,
        recall["wrong_with_all_gold"] / recall["wrong_n"] * 100
        if recall["wrong_n"]
        else 0,
    )
    bars = bx.bar(range(4), values, color=["#6E9995", "#347F9B", "#6BA8C0", "#B9764A"])
    bx.bar_label(bars, labels=[f"{v:.0f}%" for v in values], padding=3)
    bx.set_xticks(range(4), labels, rotation=12)
    bx.set_ylim(0, 110)
    bx.set_title("MemNexus 来源引用到最终上下文")
    bx.set_ylabel("占非对抗题的比例 / 错题比例")
    for axis in (ax, bx):
        axis.grid(axis="y", alpha=0.17)
        axis.set_axisbelow(True)
    fig.suptitle("对话差异与证据覆盖", fontsize=15, y=0.98)
    fig.text(
        0.5,
        0.012,
        "来源覆盖按 MemNexus turn_id 与官方 dia_id 的同序映射；SimpleMem 无原始来源ID，故不绘制其 Recall。",
        ha="center",
        fontsize=9,
        color="#555",
    )
    fig.subplots_adjust(top=0.85, bottom=0.25, left=0.07, right=0.98, wspace=0.24)
    detail = base.with_name(base.name + "_detail")
    for extension in ("png", "svg"):
        fig.savefig(detail.with_suffix("." + extension), dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    wrong_stages = [
        ("建库无 gold 来源引用", recall["wrong_without_any_memory_reference"]),
        (
            "已建库引用，最终未选中",
            recall["wrong_with_memory_reference_but_no_final_source"],
        ),
        (
            "最终只覆盖部分 gold 来源",
            recall["wrong_n"]
            - recall["wrong_without_any_gold"]
            - recall["wrong_with_all_gold"],
        ),
        ("gold 来源已齐，回答仍错", recall["wrong_with_all_gold"]),
        (
            "对抗题错误",
            100 - result["systems"]["MemNexus"]["correct"] - recall["wrong_n"],
        ),
    ]
    positions = np.arange(len(wrong_stages))
    bars = ax.barh(
        positions,
        [count for _, count in wrong_stages],
        color=["#83939C", "#347F9B", "#78A7BA", "#B9764A", "#8B83A8"],
        height=0.58,
    )
    ax.set_yticks(positions, [label for label, _ in wrong_stages])
    ax.invert_yaxis()
    ax.bar_label(bars, padding=6)
    ax.set_xlim(0, max(count for _, count in wrong_stages) + 3)
    ax.set_xlabel("MemNexus 被裁判判错的题数")
    ax.set_title("29道错题按可观察证据阶段拆分", loc="left")
    ax.grid(axis="x", alpha=0.17)
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.02,
        "来源引用和金标 turn 命中不等于语义证明；各组为诊断定位，不是算法根因的因果估计。",
        ha="center",
        color="#555",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.31, right=0.94, top=0.87, bottom=0.17)
    causes = base.with_name(base.name + "_errors")
    for extension in ("png", "svg"):
        fig.savefig(causes.with_suffix("." + extension), dpi=170)
    plt.close(fig)


def report_markdown(result: dict, report: dict) -> str:
    simple, mem = (result["systems"][system] for system in SYSTEMS)
    paired = result["paired"]
    recall = result["evidence_recall"]
    lines = [
        "# LoCoMo 第二组不重复100题配对测试",
        "",
        f"新旧 cohort 的题目键重叠 **{result['overlap_count']}**；新 cohort 哈希 `{result['selection_sha256']}`。实验输入为10段对话的 D1–D5，共 {simple['raw_turns']} 条原始 turn；题目官方 evidence 均落在该范围内，其中 {sum(any(int(ref.split(':')[0][1:]) > 2 for ref in item['evidence']) for item in report['cohort']['questions'])} 题需要 D3–D5 的来源。",
        "",
        f"统一裁判正确率：SimpleMem **{simple['correct']}/100**，MemNexus **{mem['correct']}/100**。MemNexus−SimpleMem = **{paired['difference_pp']:+d} 个百分点**，按对话聚类重采样的95%区间为 **[{paired['cluster_95ci_pp'][0]:.1f}, {paired['cluster_95ci_pp'][1]:.1f}]** 个百分点。",
        "",
        "## 实验核查",
        "",
        f"- 200 条非空回答、200 个裁判标签、20 个非空独立记忆库；provider attempts = {result['audit']['provider_attempts']}。账本中1次 error 是首段裁判阶段进程中断时未返回的请求，已按预留上界入账并补做该题裁判。",
        f"- 所有请求/响应 usage、阶段用量和费用账本已核对。供应商返回模型标识：`{result['audit']['returned_models']}`。",
        "- 两边使用 DeepSeek V4 Flash、Qwen3-Embedding-0.6B、英文回答，依次从同一 D1–D5 原始对话建库；答案缓存关闭。",
        "- 类别5沿用固定 A=Not mentioned/B=干扰答案协议；问题按官方 category_id 分组。该100题因按 gold evidence horizon 选择，不代表官方全集分布。",
        "- 旧 cohort 的 D1–D2 分数不能与本次 D1–D5 分数直接作算法前后因果对照。",
        f"- 已保存的问题路由覆盖100题；100题的回答模型调用均为1次，直接 gate 回答为 {result['reasoning_observation']['direct_gate_answers']} 次。该 runner 没有保存 `AnswerResponse.reasoning` 字段，因此不能把类别5的结果归因于 typed negative gate；下轮需逐题保存 gate trace。",
        "",
        "## 总体指标",
        "",
        "| 指标 | SimpleMem | MemNexus | 相对变化 |",
        "|---|---:|---:|---:|",
    ]
    measures = (
        ("裁判正确题数 /100", "correct", ".0f"),
        ("按 LoCoMo 词面规则重算 F1，非对抗题原样", "locomo_f1_75_raw", ".4f"),
        ("问答输入 tokens", "qa_input_tokens", ",.0f"),
        ("问答输出 tokens", "qa_output_tokens", ",.0f"),
        ("问答模型请求", "qa_provider_attempts", ",.0f"),
        ("建库模型请求", "build_provider_attempts", ",.0f"),
        ("建库＋问答 tokens", "main_total_tokens", ",.0f"),
        ("建库＋问答费用估算（元）", "main_cost_rmb", ".4f"),
        ("每答对一题模型成本（元）", "main_cost_per_correct", ".5f"),
        ("回答延迟 P50（秒）", "p50_seconds", ".3f"),
        ("回答延迟 P95（秒）", "p95_seconds", ".3f"),
        ("记忆条目总数", "memory_count", ",.0f"),
    )
    for label, field, fmt in measures:
        a, b = simple[field], mem[field]
        delta = (
            f"{b - a:+.0f}百分点"
            if field == "correct"
            else f"{(b / a - 1) * 100:+.1f}%"
            if a
            else "—"
        )
        lines.append(f"| {label} | {a:{fmt}} | {b:{fmt}} | {delta} |")
    lines += [
        "",
        "费用用 provider 实际 token 与同一保守单价估算，不是账户扣款账单；本地 embedding 与硬件费用不计入。",
        f"本次完成的配对流程（双系统与裁判）费用估算上界为 ¥{result['cost_accounting']['completed_run_upper_rmb']:.4f}；三次未完成的 conv-48 阶段另计 ¥{sum(result['cost_accounting']['abandoned_stage_upper_rmb'].values()):.4f}，本次所有调用上界约 ¥{result['cost_accounting']['all_attempts_upper_rmb']:.4f}。其中中断或连接失败的未知用量按预留上界计入；供应商实际账单可能更低。",
        "",
        "## 类别与配对结果",
        "",
        "| 官方类别 ID | 题数 | SimpleMem 答对 | MemNexus 答对 |",
        "|---|---:|---:|---:|",
    ]
    for key, row in result["by_category_id"].items():
        lines.append(f"| {key} | {row['n']} | {row['SimpleMem']} | {row['MemNexus']} |")
    lines += [
        "",
        f"两边都对 {paired['both_correct']} 题；仅 MemNexus 对 {paired['mem_only']} 题；仅 SimpleMem 对 {paired['simple_only']} 题；两边都错 {paired['both_wrong']} 题。问题级 McNemar 双侧 p={paired['mcnemar_p_question_independence']:.5f}；该检验假设题目独立，同一对话内的相关性使其偏乐观。",
        "",
        "## 来源覆盖与错因线索",
        "",
        f"MemNexus 在非对抗题 {recall['non_adversarial_n']} 道中，建库记忆的 evidence 引用覆盖全部 gold turn 的有 {recall['memory_all_gold_count']} 道，最终来源全部覆盖仅 {recall['all_gold_count']} 道；其中 {recall['stored_all_but_final_incomplete']} 道在建库到最终选择之间丢失至少一条引用。最终来源至少覆盖一条 gold turn 的有 {recall['any_gold_count']} 道。",
        "",
        f"其非对抗错题 {recall['wrong_n']} 道中，{recall['wrong_without_any_memory_reference']} 道在建库记忆中没有 gold turn 引用，{recall['wrong_with_memory_reference_but_no_final_source']} 道有引用却未进入最终来源，{recall['wrong_with_all_gold']} 道已检齐 gold turn 仍答错。记忆引用覆盖仅证明原始 turn 被索引，不证明抽取内容正确；最终 turn 命中也不等于正确推理。",
        "",
        "SimpleMem 的返回上下文只有压缩文本，没有原始 turn ID，因此无法用同一严格方式计算其 evidence Recall。",
        "",
        "## 逐题不一致案例",
        "",
    ]
    keys = {
        (row["sample_id"], row["qa_index"]): row
        for row in report["rows"]
        if row["system"] == "SimpleMem"
    }
    for system in SYSTEMS:
        if system == "SimpleMem":
            selected = [
                row
                for row in report["rows"]
                if row["system"] == "MemNexus"
                and correct(row)
                and not correct(keys[(row["sample_id"], row["qa_index"])])
            ]
            title = "仅 MemNexus 答对"
        else:
            selected = [
                row
                for row in report["rows"]
                if row["system"] == "MemNexus"
                and not correct(row)
                and correct(keys[(row["sample_id"], row["qa_index"])])
            ]
            title = "仅 SimpleMem 答对"
        lines += [f"### {title}", ""]
        for row in selected:
            baseline = keys[(row["sample_id"], row["qa_index"])]
            lines += [
                f"**{row['sample_id']}/{row['qa_index']}，类别{row['category_id']}：{row['question']}**",
                "",
                f"Gold：{row['gold_answer']}",
                "",
                f"SimpleMem：{baseline['prediction']}",
                "",
                f"MemNexus：{row['prediction']}",
                "",
                f"裁判理由：{row.get('judge_reason', '')}",
                "",
            ]
        if not selected:
            lines += ["无。", ""]
    lines += [
        "## 方法限制与下一步",
        "",
        "这100题与旧题目零重叠，但使用同一10段对话，因而不是对话级独立保留集；MemNexus 的结构和错误分析曾基于旧题设计。LoCoMo 问答成绩只能支持记忆回答结论，ICAPS 的规划贡献仍须用执行、约束违反和重规划成功率检验。",
        "",
        "建议先对错题按‘来源未召回 / 来源已齐但推理错误 / 主体类型时间错配 / 裁判或 gold 争议’作人工复核，再做事件图、问题路由、typed negative gate 的逐组件消融。将决策阈值固定在开发集后，以新的对话或 LongMemEval-S 完整历史验证。",
        "",
        "原始逐题输出、来源、调用、用量、裁判理由和源码哈希见本次 JSON 与同名 `_ledgers` 目录。",
        "逐题错误阶段表见 `result/LOCOMO_100_DISJOINT_V2_ERRORS.md`；原始证据与 gold 争议核查见 `result/LOCOMO_100_DISJOINT_V2_DATA_QUALITY.md`；按本轮瓶颈排序的算法和架构方案见 `result/LOCOMO_100_DISJOINT_V2_RESEARCH_PLAN.md`。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    path = args.run.resolve()
    report = read(path)
    result = compare(report, path)
    result_path = ROOT / "result/locomo_100_disjoint_v2_analysis.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_base = ROOT / "result/locomo_100_disjoint_v2_comparison"
    graph(result, plot_base)
    report_path = ROOT / "result/LOCOMO_100_DISJOINT_V2_ANALYSIS.md"
    report_path.write_text(report_markdown(result, report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "audited",
                "simple_correct": result["systems"]["SimpleMem"]["correct"],
                "mem_correct": result["systems"]["MemNexus"]["correct"],
                "paired": result["paired"],
                "evidence_recall": {
                    key: value
                    for key, value in result["evidence_recall"].items()
                    if key != "rows"
                },
                "audit": result["audit"],
                "report": str(report_path),
                "figures": str(plot_base),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

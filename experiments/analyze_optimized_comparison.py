"""Audit a completed paired run and render Chinese figures from actual records.

No provider calls. Old and new runs stay immutable. Category IDs are authoritative;
legacy English category names in the runner are deliberately not used for grouping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import string
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from nltk.stem import PorterStemmer

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("SimpleMem", "MemNexus")
COLORS = {"SimpleMem": "#CF8730", "MemNexus": "#2477B3", "旧版 MemNexus": "#A7B2BC"}

STEMMER = PorterStemmer()


def official_normalize(text):
    """LoCoMo task_eval/evaluation.py normalization, verified 2026-10-12."""
    text = str(text).replace(",", "").lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the|and)\b", " ", text).split())


def official_f1(prediction, gold):
    p = [STEMMER.stem(word) for word in official_normalize(prediction).split()]
    g = [STEMMER.stem(word) for word in official_normalize(gold).split()]
    same = sum((Counter(p) & Counter(g)).values())
    return 2 * same / (len(p) + len(g)) if same else 0.0


def locomo_score(row, remove_citations=False):
    prediction, gold, category = (
        row["prediction"],
        str(row["gold_answer"]),
        row["category_id"],
    )
    if remove_citations:
        prediction = re.sub(r"【来源\d+】", "", prediction)
    if category == 1:
        return statistics.mean(
            max(official_f1(p.strip(), g.strip()) for p in prediction.split(","))
            for g in gold.split(",")
        )
    if category == 3:
        gold = gold.split(";")[0].strip()
    if category == 5:
        return float(
            "no information available" in prediction.lower()
            or "not mentioned" in prediction.lower()
        )
    return official_f1(prediction, gold)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_by_key(report):
    result = {}
    for row in report["rows"]:
        key = row["sample_id"], row["qa_index"], row["system"]
        assert key not in result, f"Duplicate answer: {key}"
        assert row["prediction"].strip(), f"Empty answer: {key}"
        assert row["judge_result"] in {"CORRECT", "WRONG"}, f"Missing judge: {key}"
        result[key] = row
    assert len(result) == 200
    assert Counter(key[2] for key in result) == {system: 100 for system in SYSTEMS}
    expected = {
        (q["sample_id"], q["qa_index"], s)
        for q in report["cohort"]["questions"]
        for s in SYSTEMS
    }
    assert set(result) == expected
    return result


def correct(row):
    return int(row["judge_result"] == "CORRECT")


def q95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def cluster_ci(groups, n=10000, seed=20261012):
    """Resample whole conversations; preserve every within-conversation pair."""
    rng = random.Random(seed)
    clusters = list(groups.values())
    values = []
    for _ in range(n):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values.append(sum(sum(c) for c in sampled) / sum(len(c) for c in sampled))
    values.sort()
    return [values[int(n * 0.025)], values[int(n * 0.975)]]


def mcnemar(a, b):
    n = a + b
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2**n)


def metric(report, system):
    rows = [r for r in report["rows"] if r["system"] == system]
    stages = [s for s in report["stages"] if s["system"] == system]
    result = {"n": len(rows), "correct": sum(map(correct, rows))}
    for field in (
        "input_tokens",
        "output_tokens",
        "provider_attempts",
        "provider_errors",
        "cost_rmb",
    ):
        result["qa_" + field] = sum(r["model_calls"].get(field, 0) for r in rows)
        result["build_" + field] = sum(s["build_usage"].get(field, 0) for s in stages)
    result["qa_total_tokens"] = result["qa_input_tokens"] + result["qa_output_tokens"]
    result["main_total_tokens"] = (
        result["qa_total_tokens"]
        + result["build_input_tokens"]
        + result["build_output_tokens"]
    )
    result["main_cost_rmb"] = result["qa_cost_rmb"] + result["build_cost_rmb"]
    result["main_cost_per_correct"] = (
        result["main_cost_rmb"] / result["correct"] if result["correct"] else None
    )
    result["p50_seconds"] = statistics.median(r["latency_ms"] / 1000 for r in rows)
    result["p95_seconds"] = q95([r["latency_ms"] / 1000 for r in rows])
    result["memory_count"] = sum(s["memory_count"] for s in stages)
    result["raw_turns"] = sum(s["raw_turns"] for s in stages)
    result["mean_retrieved_sources"] = statistics.mean(
        r["retrieval_count"] for r in rows
    )
    result["mean_simple_token_f1"] = statistics.mean(r["f1"] for r in rows)
    result["mean_simple_bleu1"] = statistics.mean(r["bleu1"] for r in rows)
    result["locomo_f1_75_raw"] = statistics.mean(
        locomo_score(r) for r in rows if r["category_id"] != 5
    )
    result["locomo_f1_75_without_source_markers"] = statistics.mean(
        locomo_score(r, remove_citations=True) for r in rows if r["category_id"] != 5
    )
    result["categories"] = {
        str(k): {"n": len(subset), "correct": sum(map(correct, subset))}
        for k in range(1, 6)
        if (subset := [r for r in rows if r["category_id"] == k])
    }
    result["conversations"] = {
        key: {"n": len(subset), "correct": sum(map(correct, subset))}
        for key in sorted({r["sample_id"] for r in rows})
        if (subset := [r for r in rows if r["sample_id"] == key])
    }
    return result


def audit(report, path):
    assert report["status"] == "completed"
    rows_by_key(report)
    assert len(report["stages"]) == 20
    assert len(report["judge_usage"]) == 10
    assert all(s["memory_count"] > 0 for s in report["stages"])
    assert all(s.get("failed_windows", 0) == 0 for s in report["stages"])
    dataset_path = ROOT / report["cohort"]["dataset"]
    dataset = read(dataset_path)
    keyed = rows_by_key(report)
    for item in report["cohort"]["questions"]:
        qa = dataset[item["sample_index"]]["qa"][item["qa_index"]]
        for system in SYSTEMS:
            row = keyed[(item["sample_id"], item["qa_index"], system)]
            assert row["question"] == qa["question"]
            assert row["category_id"] == qa["category"]
            assert row["gold_answer"] == (
                "Not mentioned in the conversation"
                if qa["category"] == 5
                else qa["answer"]
            )
    assert (
        hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        == report["dataset_sha256"]
    )
    changed = [
        p
        for p, digest in report["source_sha256"].items()
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest
    ]
    assert not changed, f"Evaluated source changed after manifest: {changed}"
    ledger_folder = path.with_name(path.stem + "_ledgers")
    ledger_files = [
        p
        for p in ledger_folder.glob("*.json")
        if not p.name.endswith((".stage.json", ".memory.json"))
    ]
    assert len(ledger_files) == 30
    attempts = errors = input_tokens = output_tokens = 0
    total = 0.0
    returned_models = Counter()
    missing_usage = []
    for ledger_path in ledger_files:
        ledger = read(ledger_path)
        assert ledger.get("reserved_rmb", 0) == 0
        trace = [
            json.loads(s)
            for s in ledger_path.with_suffix(".calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        requests = [t for t in trace if t["event"] == "request"]
        responses = [t for t in trace if t["event"] == "response"]
        assert len(requests) == ledger["provider_attempts"]
        for request in requests:
            assert request["body"]["model"] == "deepseek-v4-flash"
            assert request["body"]["temperature"] == 0
            assert request["body"]["max_tokens"] in {4096, 8192}
        for response in responses:
            body = response["body"]
            returned_models[body.get("model", "missing")] += 1
            if not body.get("usage"):
                missing_usage.append(str(ledger_path))
        trace_input = sum(
            t["body"].get("usage", {}).get("prompt_tokens", 0) for t in responses
        )
        trace_output = sum(
            t["body"].get("usage", {}).get("completion_tokens", 0) for t in responses
        )
        assert trace_input == ledger["input_tokens"]
        assert trace_output == ledger["output_tokens"]
        attempts += ledger["provider_attempts"]
        errors += ledger.get("provider_errors", 0)
        input_tokens += trace_input
        output_tokens += trace_output
        total += ledger["cost_rmb"]
    assert not missing_usage
    accounting = sum(s["total_usage"]["cost_rmb"] for s in report["stages"]) + sum(
        j["cost_rmb"] for j in report["judge_usage"]
    )
    assert math.isclose(accounting, total, abs_tol=1e-8)
    assert math.isclose(total, report["run_cost_rmb"], abs_tol=1e-8)
    for stage in report["stages"]:
        subset = [
            r
            for r in report["rows"]
            if r["system"] == stage["system"] and r["sample_id"] == stage["sample_id"]
        ]
        for field in ("provider_attempts", "input_tokens", "output_tokens", "cost_rmb"):
            partitioned = stage["build_usage"].get(field, 0) + sum(
                r["model_calls"].get(field, 0) for r in subset
            )
            assert math.isclose(
                partitioned, stage["total_usage"].get(field, 0), abs_tol=1e-8
            )
    return {
        "answers": 200,
        "judgments": 200,
        "memory_builds": 20,
        "ledger_files": 30,
        "source_hashes_verified": len(report["source_sha256"]),
        "provider_attempts": attempts,
        "provider_errors": errors,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "returned_models": dict(returned_models),
        "cost_rmb": total,
        "usage_reconciled": True,
    }


def analyze(report, old):
    new_by_key, old_by_key = rows_by_key(report), rows_by_key(old)
    assert report["selection_sha256"] == old["selection_sha256"]
    assert set(new_by_key) == set(old_by_key)
    pairs = defaultdict(list)
    changes = defaultdict(list)
    outcomes = Counter()
    transitions = []
    for item in report["cohort"]["questions"]:
        sample, index = item["sample_id"], item["qa_index"]
        s, m = [new_by_key[(sample, index, system)] for system in SYSTEMS]
        old_m = old_by_key[(sample, index, "MemNexus")]
        pairs[sample].append(correct(m) - correct(s))
        changes[sample].append(correct(m) - correct(old_m))
        outcomes[
            {
                (1, 1): "both_correct",
                (0, 0): "both_wrong",
                (0, 1): "mem_only",
                (1, 0): "simple_only",
            }[(correct(s), correct(m))]
        ] += 1
        transitions.append(
            {
                "sample_id": sample,
                "qa_index": index,
                "category_id": m["category_id"],
                "question": m["question"],
                "gold_answer": m["gold_answer"],
                "SimpleMem": s["prediction"],
                "MemNexus": m["prediction"],
                "old_MemNexus": old_m["prediction"],
                "simple_correct": correct(s),
                "mem_correct": correct(m),
                "old_mem_correct": correct(old_m),
                "judge_reason": m["judge_reason"],
            }
        )
    mem_rows = [r for r in report["rows"] if r["system"] == "MemNexus"]
    return {
        "systems": {s: metric(report, s) for s in SYSTEMS},
        "old_systems": {s: metric(old, s) for s in SYSTEMS},
        "paired_outcomes": dict(outcomes),
        "cluster_95ci_difference": cluster_ci(pairs),
        "cluster_95ci_change_mem": cluster_ci(changes),
        "mcnemar_question_independence_p": mcnemar(
            outcomes["mem_only"], outcomes["simple_only"]
        ),
        "old_wrong_new_correct": sum(
            not t["old_mem_correct"] and t["mem_correct"] for t in transitions
        ),
        "old_correct_new_wrong": sum(
            t["old_mem_correct"] and not t["mem_correct"] for t in transitions
        ),
        "grounding": {
            "questions_with_supplement": sum(
                bool(r["grounding"]["supplemental_rounds"]) for r in mem_rows
            ),
            "questions_with_raw_selected": sum(
                bool(r["grounding"]["raw_sources_selected"]) for r in mem_rows
            ),
            "questions_with_temporal_derivations": sum(
                bool(r["grounding"]["temporal_derivations"]) for r in mem_rows
            ),
            "questions_with_attribution_conflicts": sum(
                bool(r["grounding"]["attribution_conflicts"]) for r in mem_rows
            ),
            "mean_grounding_ms": statistics.mean(
                r["grounding"]["elapsed_ms"] for r in mem_rows
            ),
            "initial_missing_slots": sum(
                len(r["grounding"]["initial_missing_info"]) for r in mem_rows
            ),
            "final_missing_slots": sum(
                len(r["grounding"]["final_missing_info"]) for r in mem_rows
            ),
            "max_supplemental_rounds": max(
                r["grounding"]["supplemental_rounds"] for r in mem_rows
            ),
        },
        "transitions": transitions,
    }


def prepare_plot():
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
            "svg.fonttype": "none",
            "figure.facecolor": "#FAFBFD",
            "axes.facecolor": "#FAFBFD",
        }
    )


def decorate(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)


def barlabels(ax, bars, fmt="{:.0f}"):
    ax.bar_label(
        bars, labels=[fmt.format(b.get_height()) for b in bars], padding=3, fontsize=9
    )


def figures(data, outbase):
    prepare_plot()
    simple, mem = [data["systems"][s] for s in SYSTEMS]
    old_mem = data["old_systems"]["MemNexus"]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.2))
    ax = axes[0, 0]
    bars = ax.bar(
        [0, 1, 2],
        [simple["correct"], mem["correct"], old_mem["correct"]],
        color=[COLORS["SimpleMem"], COLORS["MemNexus"], COLORS["旧版 MemNexus"]],
        width=0.62,
    )
    barlabels(ax, bars, "{:.0f}%")
    ax.set_xticks(
        [0, 1, 2], ["SimpleMem\n本次", "MemNexus\n本次优化版", "MemNexus\n优化前历史值"]
    )
    ax.set_ylim(0, 110)
    decorate(ax, "回答正确率 ↑", "统一裁判正确率（%），每组 n=100")
    low, high = [100 * v for v in data["cluster_95ci_difference"]]
    ax.set_xlabel(
        f"Mem−Simple：{mem['correct'] - simple['correct']:+d}个百分点；对话重采样区间 [{low:.1f}, {high:.1f}]",
        fontsize=8,
    )
    ax = axes[0, 1]
    x = np.arange(2)
    for offset, system in [(-0.18, "SimpleMem"), (0.18, "MemNexus")]:
        stats = data["systems"][system]
        bars = ax.bar(
            x + offset,
            [stats["qa_input_tokens"] / 1000, stats["qa_output_tokens"] / 1000],
            0.36,
            color=COLORS[system],
            label=system,
        )
        barlabels(ax, bars, "{:.1f}")
    ax.set_xticks(x, ["输入 tokens", "输出 tokens"])
    ax.margins(y=0.17)
    decorate(ax, "问答阶段 token 用量 ↓", "千 tokens；含规划、检索反思和回答")
    ax.legend(frameon=False)
    ax = axes[0, 2]
    bars = ax.bar(
        SYSTEMS,
        [simple["qa_provider_attempts"], mem["qa_provider_attempts"]],
        color=[COLORS[s] for s in SYSTEMS],
        width=0.56,
    )
    barlabels(ax, bars)
    ax.margins(y=0.18)
    decorate(ax, "问答阶段模型请求 ↓", "实际 provider attempts / 100 题")
    ax = axes[1, 0]
    for index, system in enumerate(SYSTEMS):
        stats = data["systems"][system]
        ax.bar(index, stats["build_cost_rmb"], 0.55, color=COLORS[system], alpha=0.38)
        ax.bar(
            index,
            stats["qa_cost_rmb"],
            0.55,
            bottom=stats["build_cost_rmb"],
            color=COLORS[system],
        )
        ax.text(
            index,
            stats["main_cost_rmb"] + 0.04,
            f"¥{stats['main_cost_rmb']:.2f}",
            ha="center",
            fontsize=10,
        )
        ax.text(
            index,
            stats["build_cost_rmb"] / 2,
            f"建库 {stats['build_cost_rmb']:.2f}",
            ha="center",
            fontsize=9,
        )
        ax.text(
            index,
            stats["build_cost_rmb"] + stats["qa_cost_rmb"] / 2,
            f"问答 {stats['qa_cost_rmb']:.2f}",
            ha="center",
            color="white",
            fontsize=9,
        )
    ax.set_xticks([0, 1], SYSTEMS)
    ax.margins(y=0.18)
    decorate(ax, "主模型费用估算 ↓", "人民币；建库＋100题问答，不含裁判")
    ax = axes[1, 1]
    for offset, system in [(-0.18, "SimpleMem"), (0.18, "MemNexus")]:
        stats = data["systems"][system]
        bars = ax.bar(
            x + offset,
            [stats["p50_seconds"], stats["p95_seconds"]],
            0.36,
            color=COLORS[system],
            label=system,
        )
        barlabels(ax, bars, "{:.2f}")
    ax.set_xticks(x, ["P50 中位数", "P95"])
    ax.margins(y=0.18)
    decorate(ax, "单题端到端延迟 ↓", "秒；无答案缓存，含规划与检索")
    ax = axes[1, 2]
    outcomes = data["paired_outcomes"]
    bars = ax.bar(
        np.arange(4),
        [
            outcomes.get(k, 0)
            for k in ("both_correct", "mem_only", "simple_only", "both_wrong")
        ],
        color=["#698A76", COLORS["MemNexus"], COLORS["SimpleMem"], "#B3B8C0"],
        width=0.6,
    )
    barlabels(ax, bars)
    ax.set_xticks(
        np.arange(4),
        ["两者都对", "仅MemNexus对", "仅SimpleMem对", "两者都错"],
        rotation=15,
    )
    ax.margins(y=0.18)
    decorate(ax, "逐题配对结果", "题数；同样的100题")
    fig.suptitle(
        "优化后的 MemNexus 与 SimpleMem：真实 LoCoMo 100 题配对复测",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.022,
        "10 段对话，仅 D1–D2；开发集复测，非独立泛化验证。费用按实际 token × 保守单价估算，非账户账单。",
        ha="center",
        fontsize=10,
        color="#535F6C",
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, top=0.90, bottom=0.12, hspace=0.52, wspace=0.33
    )
    for suffix in (".png", ".svg"):
        fig.savefig(outbase.with_suffix(suffix), dpi=170)
    plt.close(fig)

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(15.2, 5.4), gridspec_kw={"width_ratios": [1.13, 1]}
    )
    labels = [
        "类别1\n多跳 n=13",
        "类别2\n时间 n=25",
        "类别3\nn=9",
        "类别4\nn=28",
        "类别5\n对抗 n=25",
    ]
    x = np.arange(5)
    for offset, stats, label in [
        (-0.26, simple, "SimpleMem 本次"),
        (0, mem, "MemNexus 本次"),
        (0.26, old_mem, "MemNexus 优化前"),
    ]:
        values = [
            100
            * stats["categories"][str(k)]["correct"]
            / stats["categories"][str(k)]["n"]
            for k in range(1, 6)
        ]
        color = (
            COLORS["SimpleMem"]
            if offset < 0
            else COLORS["MemNexus"]
            if offset == 0
            else COLORS["旧版 MemNexus"]
        )
        bars = ax.bar(x + offset, values, 0.25, label=label, color=color)
        barlabels(ax, bars, "{:.0f}")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 116)
    decorate(ax, "按原始类别编号分层", "正确率（%）")
    ax.legend(
        frameon=False,
        fontsize=9,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
    )
    ids = list(simple["conversations"])
    for stats, label, color in [
        (simple, "SimpleMem 本次", COLORS["SimpleMem"]),
        (mem, "MemNexus 本次", COLORS["MemNexus"]),
    ]:
        bx.plot(
            np.arange(len(ids)),
            [
                100
                * stats["conversations"][i]["correct"]
                / stats["conversations"][i]["n"]
                for i in ids
            ],
            "o-",
            color=color,
            label=label,
            linewidth=2,
            markersize=5,
        )
    bx.set_xticks(
        np.arange(len(ids)),
        [f"{i}\n(n={simple['conversations'][i]['n']})" for i in ids],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    bx.set_ylim(0, 110)
    decorate(bx, "各对话上的正确率", "正确率（%）；各对话题数不同")
    bx.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "质量变化的分布：不同类别、不同对话的变化并不一致", fontsize=16, y=0.98
    )
    fig.text(
        0.5,
        0.015,
        "旧 runner 的类别名称存在误标，本图只按官方 category_id 分组；类别3/4保留编号。折线连接的是离散对话，不是时间趋势。",
        ha="center",
        fontsize=9,
        color="#535F6C",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.24, wspace=0.23)
    detail = outbase.with_name(outbase.name + "_detail")
    for suffix in (".png", ".svg"):
        fig.savefig(detail.with_suffix(suffix), dpi=170)
    plt.close(fig)


def report_text(data, raw_path, old_path, fail_cost, overview):
    s, m = [data["systems"][x] for x in SYSTEMS]
    old_s, old_m = [data["old_systems"][x] for x in SYSTEMS]
    ci = [100 * x for x in data["cluster_95ci_difference"]]
    diff = m["correct"] - s["correct"]
    direction = "高于" if diff > 0 else "低于" if diff < 0 else "等于"
    significant = ci[0] > 0 or ci[1] < 0
    lines = [
        "# 优化后 MemNexus 与 SimpleMem：100 题配对复测（2026-10-12）",
        "",
        f"本次统一裁判正确率：MemNexus **{m['correct']}/100**，SimpleMem **{s['correct']}/100**，MemNexus {direction} SimpleMem **{abs(diff)} 个百分点**。",
        f"旧版 MemNexus 为 {old_m['correct']}/100，本次净变化 {m['correct'] - old_m['correct']:+d} 题；SimpleMem 历史值 {old_s['correct']}/100，本次也重新建库、生成和评分。",
        "",
        "这是优化所用开发集上的回归复测，不是独立保留测试，不能据此宣称跨数据集泛化或 ICAPS 论文主张已成立。",
        "",
        "## 数据与审计",
        "",
        "- 原始 LoCoMo 文件中预先冻结的100道题，10段对话，每段仅输入D1–D2，共450 turns。两套系统独立从原始对话建库；不共享压缩记忆。两边均使用数据集已有blip_caption作为图片的文本描述，未另调用视觉模型。",
        "- 选择条件使用官方 evidence 是否位于D1–D2，存在已知的 evidence horizon 筛选；不是完整长程LoCoMo，也不代表完整1986题的分布。",
        "- 主模型/裁判配置均为 deepseek-v4-flash；实际供应商返回的 model 标识见下方审计。Qwen3-Embedding-0.6B，1024维，query prompt相同；英文回答、temperature=0、关闭答案缓存、顺序运行。",
        "- SimpleMem保留原有规划与反思检索；MemNexus使用model planner、原文来源恢复、时间区间推导与最多一轮原文补检索。它们不是同等上下文长度或等token预算的消融实验。",
        f"- 20次建库均非空，200条回答与200个裁判标签完整；核对了{data['audit']['source_hashes_verified']}个源码哈希、30份ledger和请求/响应sidecar。",
        f"- 实际请求 {data['audit']['provider_attempts']} 次，provider errors={data['audit']['provider_errors']}；所有token、阶段分账与总账一致，未完成预留均归零。",
        f"- 供应商返回model标识计数：`{json.dumps(data['audit']['returned_models'], ensure_ascii=False)}`。",
        "- 首次尝试因MemNexus提示词预算溢出中止，完整保留失败记录。修复为先预留实际序列化任务/规划/规则，再进行有界证据选择；54项相关回归通过，算法随后冻结。",
        "- 第二次尝试完成conv-26后，SimpleMem在conv-30建库连续三次输出截断。两套provider与裁判加入一致规则：仅明确finish_reason=length时允许一次8192上限恢复，普通请求仍4096；27项provider回归通过。",
        "- 最终执行从第二次尝试保留已完整配对评分的conv-26（其调用未触发截断），只继续其余会话；没有选择分数更高的回答。继承源码哈希和恢复原因保存在resumed_from。未完成conv-30的失败建库成本单独计入本次总开销。",
        "",
        "## 总体指标",
        "",
        "| 指标 | SimpleMem 本次 | MemNexus 优化后 | MemNexus 相对变化 |",
        "|---|---:|---:|---:|",
    ]
    for label, field, fmt in [
        ("裁判正确题数 /100", "correct", ".0f"),
        ("问答输入 tokens", "qa_input_tokens", ",.0f"),
        ("问答输出 tokens", "qa_output_tokens", ",.0f"),
        ("问答总 tokens", "qa_total_tokens", ",.0f"),
        ("问答模型请求", "qa_provider_attempts", ",.0f"),
        ("建库模型请求", "build_provider_attempts", ",.0f"),
        ("建库＋问答总 tokens", "main_total_tokens", ",.0f"),
        ("问答费用估算（元）", "qa_cost_rmb", ".4f"),
        ("建库费用估算（元）", "build_cost_rmb", ".4f"),
        ("建库＋问答费用估算（元）", "main_cost_rmb", ".4f"),
        ("每答对一题的建库＋问答成本（元）", "main_cost_per_correct", ".5f"),
        ("回答延迟P50（秒）", "p50_seconds", ".3f"),
        ("回答延迟P95（秒）", "p95_seconds", ".3f"),
        ("建库记忆条目数", "memory_count", ",.0f"),
        ("平均最终来源数", "mean_retrieved_sources", ".2f"),
    ]:
        change = (
            f"{m[field] - s[field]:+d}个百分点"
            if field == "correct"
            else f"{100 * (m[field] / s[field] - 1):+.1f}%"
        )
        lines.append(f"| {label} | {s[field]:{fmt}} | {m[field]:{fmt}} | {change} |")
    lines.extend(
        [
            "",
            "token和请求数直接来自provider usage，问答阶段包括规划、检索反思与最终生成。费用采用与旧实验一致的保守单价：输入¥2.4/百万、输出¥9.6/百万，不抵扣cache hit或错峰折扣；它不是DeepSeek账户的实际扣款账单，也不包含本地embedding、电力与硬件成本。",
            "",
            f"完整本轮（双系统建库＋问答＋裁判）估算 **¥{data['audit']['cost_rmb']:.6f}**；中止尝试 **¥{fail_cost:.6f}**；本次请求合计 **¥{data['audit']['cost_rmb'] + fail_cost:.6f}**，低于用户提供的¥31余额。历史费用没有被当作本次账户扣款。",
            "",
            "上表是最终成功配对流程的成本，其中包含流程内所有恢复调用；运行协议修复前废弃的部分不混入算法单题均值，但已在中止尝试总额单独报告。因此同时提供了成功流程成本和实际测试开销，不能把失败费用隐藏。",
            "",
            "## 分层结果与统计不确定性",
            "",
            f"补充词面指标：依据官方evaluation.py的归一化、Porter词干化与类别1多答案规则，非对抗75题的原样回答F1为 SimpleMem {s['locomo_f1_75_raw']:.4f}、MemNexus {m['locomo_f1_75_raw']:.4f}。对两边统一删除【来源N】后的格式敏感性检查为 {s['locomo_f1_75_without_source_markers']:.4f}、{m['locomo_f1_75_without_source_markers']:.4f}。原样分数是主词面指标，清理版单独标注，未改动原始回答。",
            "",
            "| 官方类别ID | 题数 | SimpleMem 本次 | MemNexus 本次 | MemNexus 旧版 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for k in range(1, 6):
        cat = str(k)
        lines.append(
            f"| {k} | {m['categories'][cat]['n']} | {s['categories'][cat]['correct']} | {m['categories'][cat]['correct']} | {old_m['categories'][cat]['correct']} |"
        )
    lines.extend(
        [
            "",
            "类别名称校正：旧runner将类别1标为single-hop，但[官方evaluation.py](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py)明确将1作为multi-hop。类别2是时间、5是对抗；3/4在本报告保留原始编号，不沿用旧报告未经核准的英文名称。这个修正只影响展示解释，不改变抽样、题目或评分。",
            "",
            f"逐题配对：MemNexus独对 {data['paired_outcomes'].get('mem_only', 0)} 题，SimpleMem独对 {data['paired_outcomes'].get('simple_only', 0)} 题；两者都对 {data['paired_outcomes'].get('both_correct', 0)} 题，两者都错 {data['paired_outcomes'].get('both_wrong', 0)} 题。",
            f"按10段对话为cluster、重复抽样10,000次的MemNexus−SimpleMem正确率差95%区间为 **[{ci[0]:.1f}, {ci[1]:.1f}] 个百分点**。区间{'未跨越' if significant else '跨越'}0；样本只有10个cluster，结果仅作探索性估计。",
            f"假定题目独立时的McNemar双侧p={data['mcnemar_question_independence_p']:.6g}；实际题目在对话内相关，不能只用这个p值作论文显著性结论。",
            f"MemNexus旧错→新对 {data['old_wrong_new_correct']} 题，旧对→新错 {data['old_correct_new_wrong']} 题。两轮均重新建库和生成，且裁判有随机性，因此差值不是严格的单组件因果效应。",
            "",
            "## 机制观测",
            "",
            f'相对旧版MemNexus，问答总tokens为 {old_m["qa_total_tokens"]:,}→{m["qa_total_tokens"]:,}（{100*(m["qa_total_tokens"]/old_m["qa_total_tokens"]-1):+.1f}%），问答成本估算 ¥{old_m["qa_cost_rmb"]:.4f}→¥{m["qa_cost_rmb"]:.4f}，建库＋问答成本 ¥{old_m["main_cost_rmb"]:.4f}→¥{m["main_cost_rmb"]:.4f}。质量改善与成本变化必须一起阅读；相对SimpleMem节省费用，不等于相对旧版MemNexus所有开销都下降。',
            "",
        ]
    )
    g = data["grounding"]
    lines.extend(
        [
            f"100题中，触发一次补检索 {g['questions_with_supplement']} 题，最终选中原文补充 {g['questions_with_raw_selected']} 题，出现时间推导 {g['questions_with_temporal_derivations']} 题，检测到归属冲突 {g['questions_with_attribution_conflicts']} 题。最大补检索轮数={g['max_supplemental_rounds']}，本地grounding平均耗时 {g['mean_grounding_ms']:.2f} ms。",
            f"词面missing_info槽位总数 {g['initial_missing_slots']}→{g['final_missing_slots']}。这些是检索启发式日志，不是逻辑蕴含证明；有时间推导、槽位减少都不能直接等同回答正确。",
            "",
            "## 可复核案例",
            "",
            "以下按固定题目顺序各取最多3例：优化修复、仍落后于SimpleMem、以及优化退步。完整200条回答与全部新旧转换保存在JSON；不修改裁判标签。",
            "",
        ]
    )
    for title, predicate in [
        ("旧错→新对", lambda t: not t["old_mem_correct"] and t["mem_correct"]),
        (
            "本次SimpleMem对、MemNexus错",
            lambda t: t["simple_correct"] and not t["mem_correct"],
        ),
        ("旧对→新错", lambda t: t["old_mem_correct"] and not t["mem_correct"]),
    ]:
        lines.extend([f"### {title}", ""])
        selected = [t for t in data["transitions"] if predicate(t)][:3]
        if not selected:
            lines.extend(["本轮没有此类题目。", ""])
        for t in selected:
            lines.extend(
                [
                    f"**{t['sample_id']} / qa_index={t['qa_index']}**：{t['question']}",
                    "",
                    f"- Gold：{t['gold_answer']}",
                    f"- SimpleMem：{t['SimpleMem']}",
                    f"- MemNexus优化后：{t['MemNexus']}",
                    f"- MemNexus优化前：{t['old_MemNexus']}",
                    f"- 本次MemNexus裁判理由：{t['judge_reason']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## 解释边界与面向ICAPS的下一步",
            "",
            "**已核实的gold冲突和裁判不一致（不改分）**：conv-26 / qa_index=5，问题是Melanie何时参加慈善跑。官方证据D2:1的记录日期为2023-05-25，原文明确写“I ran a charity race for mental health last Saturday”，对应2023-05-20；但gold写“The sunday before 25 May 2023”。SimpleMem回答“20 May 2023”被判对，理由错误地声称这一天是周日；MemNexus回答“Saturday, May 20, 2023”被判错。两边给出同一天，原始证据也支持该日。这说明gold-only LLM裁判存在标注与一致性问题。主表保留原始评分；此处不单方面给某个系统加分，也未声称已完成全数据人工复核。",
            "",
            "1. 同一模型生成并裁判，且裁判只看gold、问题和预测，不看完整原始对话；评分可能存在偏差。必要人工复核应单独记录，不能为了提升分数覆盖原始标签。",
            "2. 类别5沿用此前A=Not mentioned、B=对抗答案的固定选项协议，存在位置偏差。25%的题为这类题，不能把总体分数等同官方全量LoCoMo指标。",
            "3. runner原始字段f1/bleu1是简单词面诊断，不是官方LoCoMo评分函数。本文另计算官方F1规则下非对抗75题的原样与去引用敏感性指标；词面指标受冗长输出影响，不能与语义裁判准确率混为一谈，更不能称为完整LoCoMo成绩。",
            "4. 这100题参与了优化诊断，因此下一阶段应冻结算法，在未用于开发的LongMemEval-S完整haystack和更长LoCoMo历史上测试，并对时间层、来源恢复和补检索做受控消融。",
            "5. ICAPS是规划与调度会议。QA证据不足以支撑规划贡献；应把记忆误差落实到belief state、行动前置条件与执行监控，比较真实局部修复和全局重规划的plan success、replanning success、约束违反、搜索节点与修复开销。当前测试没有测这些指标。",
            "",
            "## 原始文件",
            "",
            f"- 本次：`{raw_path.relative_to(ROOT)}`",
            f"- 历史：`{old_path.relative_to(ROOT)}`",
            f"- 总览：`{overview.relative_to(ROOT)}.png`；另有SVG与detail图。",
            "- 每个会话的原始请求/响应、构建记忆、逐题输出与ledger位于本次JSON同名的 `_ledgers` 目录。",
            "- `locomo_100_optimized_source_snapshot.zip`：经哈希验证的最终源码、分析脚本和公开实验manifest；不包含.env或带API key的config.py。继承的首段会话使用截断恢复之前的provider，其差异保留在原始结果resumed_from中。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--old", type=Path, default=ROOT / "result/locomo_100_results_v3.json"
    )
    parser.add_argument("--failed", type=Path, action="append")
    args = parser.parse_args()
    path, old_path = args.run.resolve(), args.old.resolve()
    report, old = read(path), read(old_path)
    validated = audit(report, path)
    data = analyze(report, old)
    data["audit"] = validated
    data["source"] = str(path)
    failed_paths = args.failed or [ROOT / "result/locomo_100_optimized_20261012.json"]
    fail_cost = 0.0
    failed_summaries = []
    for failed_path in failed_paths:
        failed = read(failed_path)
        assert failed["status"] == "aborted"
        failed_ledger_dir = failed_path.with_name(failed_path.stem + "_ledgers")
        completed_cost = 0.0
        for item in failed.get("judge_usage", []):
            for suffix in ("_simplemem.json", "_memnexus.json", "_judge.json"):
                ledger_path = failed_ledger_dir / (item["sample_id"] + suffix)
                if ledger_path.exists():
                    ledger = read(ledger_path)
                    completed_cost += ledger.get("cost_rmb", 0.0) + ledger.get("reserved_rmb", 0.0)
        incremental_failure = failed["upper_run_cost_rmb"] - completed_cost
        fail_cost += incremental_failure
        failed_summaries.append({
            "path": str(failed_path), "upper_cost_rmb": failed["upper_run_cost_rmb"],
            "completed_checkpoint_cost_rmb": completed_cost,
            "incremental_incomplete_cost_rmb": incremental_failure,
            "failure": failed.get("failure"),
        })
    data["failed_attempt_cost_rmb"] = fail_cost
    data["failed_attempts"] = failed_summaries
    data["this_request_cost_rmb"] = validated["cost_rmb"] + fail_cost
    snapshot = ROOT / "result/locomo_100_optimized_source_snapshot.zip"
    with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in report["source_sha256"]:
            archive.write(ROOT / name, arcname=name.replace("\\", "/"))
        archive.write(
            Path(__file__), arcname="experiments/analyze_optimized_comparison.py"
        )
        archive.writestr(
            "experiment_manifest.json",
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "dataset_sha256",
                        "selection_sha256",
                        "source_sha256",
                        "cohort",
                        "environment",
                        "budget",
                        "resumed_from",
                    )
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    data["source_snapshot"] = str(snapshot)
    base = ROOT / "result/locomo_100_optimized_comparison"
    figures(data, base)
    (ROOT / "result/locomo_100_optimized_analysis.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "result/LOCOMO_100_OPTIMIZED_20261012_ANALYSIS.md").write_text(
        report_text(data, path, old_path, fail_cost, base), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in data.items() if key != "transitions"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Audit the frozen collection-upgrade comparison and plot actual paired results."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_locomo_disjoint import configure_chinese_font, evidence_recall
from analyze_optimized_comparison import (
    audit,
    cluster_ci,
    correct,
    mcnemar,
    metric,
    rows_by_key,
)
from run_locomo_100 import ledger_total

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("SimpleMem", "MemNexus")
COLORS = {"SimpleMem": "#D18A35", "MemNexus": "#2478AE"}
CATEGORY_LABELS = {1: "多跳", 2: "时间", 3: "开放域", 4: "单跳", 5: "对抗"}


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def completed_subset(raw, path):
    """Audit only complete pairs; never score missing questions as wrong or as zero-cost answers."""
    report = deepcopy(raw)
    completed = {j["sample_id"] for j in raw["judge_usage"]}
    report["requested_question_count"] = raw["cohort"]["question_count"]
    report["cohort"]["questions"] = [q for q in raw["cohort"]["questions"] if q["sample_id"] in completed]
    n = len(report["cohort"]["questions"])
    report["cohort"]["question_count"] = n
    report["cohort"]["conversation_count"] = len(completed)
    report["rows"] = [r for r in raw["rows"] if r["sample_id"] in completed]
    report["stages"] = [s for s in raw["stages"] if s["sample_id"] in completed]
    indexed = {(r["sample_id"], r["qa_index"], r["system"]): r for r in report["rows"]}
    expected = {(q["sample_id"], q["qa_index"], s) for q in report["cohort"]["questions"] for s in SYSTEMS}
    assert len(indexed) == len(report["rows"]) == 2*n and set(indexed) == expected
    assert all(r["prediction"].strip() and r["judge_result"] in {"CORRECT", "WRONG"} for r in indexed.values())
    assert len(report["stages"]) == 2*len(completed) and all(s["memory_count"] > 0 for s in report["stages"])
    dataset_path = ROOT / raw["cohort"]["dataset"]
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == raw["dataset_sha256"]
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    for q in report["cohort"]["questions"]:
        qa = dataset[q["sample_index"]]["qa"][q["qa_index"]]
        for system in SYSTEMS:
            row = indexed[q["sample_id"], q["qa_index"], system]
            assert row["question"] == qa["question"] and row["category_id"] == qa["category"]
            assert row["gold_answer"] == ("Not mentioned in the conversation" if qa["category"] == 5 else qa["answer"])
    for name, digest in raw["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    totals = Counter()
    returned_models = Counter()
    folder = path.with_name(path.stem+"_ledgers")
    for sample in completed:
        for system in ("simplemem", "memnexus", "judge"):
            ledger_path = folder/f"{sample}_{system}.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            events = [json.loads(line) for line in ledger_path.with_suffix(".calls.jsonl").read_text(encoding="utf-8").splitlines()]
            requests = [e for e in events if e["event"] == "request"]
            responses = [e for e in events if e["event"] == "response"]
            assert len(requests) == ledger["provider_attempts"] and ledger.get("reserved_rmb",0) == 0
            assert all(e["body"]["model"] == "deepseek-v4-flash" and e["body"]["temperature"] == 0 for e in requests)
            assert all(e["body"].get("usage") for e in responses)
            assert sum(e["body"]["usage"].get("prompt_tokens",0) for e in responses) == ledger["input_tokens"]
            assert sum(e["body"]["usage"].get("completion_tokens",0) for e in responses) == ledger["output_tokens"]
            returned_models.update(e["body"].get("model") for e in responses)
            for field in ("provider_attempts","provider_errors","input_tokens","output_tokens","cost_rmb"):
                totals[field] += ledger.get(field,0)
    for stage in report["stages"]:
        rows = [r for r in report["rows"] if r["sample_id"] == stage["sample_id"] and r["system"] == stage["system"]]
        for field in ("provider_attempts","input_tokens","output_tokens","cost_rmb"):
            assert math.isclose(stage["build_usage"].get(field,0)+sum(r["model_calls"].get(field,0) for r in rows),
                                stage["total_usage"].get(field,0), abs_tol=1e-8)
    assert math.isclose(totals["cost_rmb"], sum(s["total_usage"]["cost_rmb"] for s in report["stages"])+sum(j["cost_rmb"] for j in report["judge_usage"]), abs_tol=1e-8)
    checked = {**totals, "answers":2*n,"judgments":2*n,"memory_builds":len(report["stages"]),
               "ledger_files":3*len(completed),"source_hashes_verified":len(raw["source_sha256"]),
               "returned_models":dict(returned_models),"usage_reconciled":True,
               "incomplete_stage_upper_rmb":ledger_total(folder)-totals["cost_rmb"]}
    return report, checked, indexed


def analyze(path, aborted_paths=(), allow_partial=False):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "completed":
        if not allow_partial:
            raise ValueError("Run incomplete; --allow-partial is required and plots will state the actual denominator")
        report, checked, indexed = completed_subset(report, path)
    else:
        checked = audit(report, path)
        indexed = rows_by_key(report)
        report["requested_question_count"] = report["cohort"]["question_count"]
    metrics = {system: metric(report, system) for system in SYSTEMS}
    for system in SYSTEMS:
        metrics[system]["build_seconds"] = sum(s["build_ms"] for s in report["stages"] if s["system"] == system) / 1000
        metrics[system]["qa_total_seconds"] = sum(r["latency_ms"] for r in report["rows"] if r["system"] == system) / 1000
        metrics[system]["main_total_seconds"] = metrics[system]["build_seconds"] + metrics[system]["qa_total_seconds"]
    clusters = defaultdict(list)
    pairs = Counter()
    subgroups = {}
    for question in report["cohort"]["questions"]:
        k = question["sample_id"], question["qa_index"]
        a, b = (correct(indexed[(*k, system)]) for system in SYSTEMS)
        pairs[a, b] += 1
        clusters[k[0]].append(b - a)
    for reused in (False, True):
        subset = [q for q in report["cohort"]["questions"] if q["previously_tested"] == reused]
        subgroups["reused" if reused else "new"] = {
            "n": len(subset),
            **{s: sum(correct(indexed[(q["sample_id"], q["qa_index"], s)]) for q in subset) for s in SYSTEMS},
        }
    coverage = evidence_recall(report, path)
    source_map = {(r["sample_id"], r["qa_index"]): r for r in coverage["rows"]}
    errors = []
    architecture = defaultdict(lambda: {"n": 0, "correct": 0, "output_complete": 0})
    table = []
    for q in report["cohort"]["questions"]:
        k = q["sample_id"], q["qa_index"]
        simple, mem = (indexed[(*k, s)] for s in SYSTEMS)
        ir = mem.get("reasoning", {}).get("answer_ir", {})
        mode = ir.get("mode", "ordinary_answer")
        architecture[mode]["n"] += 1
        architecture[mode]["correct"] += correct(mem)
        architecture[mode]["output_complete"] += bool(ir.get("coverage", {}).get("output_complete"))
        record = {"sample_id": k[0], "qa_index": k[1], "category_id": q["category_id"],
                  "previously_tested": q["previously_tested"], "question": mem["question"],
                  "gold": mem["gold_answer"], "SimpleMem": simple["prediction"], "MemNexus": mem["prediction"],
                  "SimpleMem_correct": correct(simple), "MemNexus_correct": correct(mem),
                  "SimpleMem_judge_reason": simple.get("judge_reason"),
                  "MemNexus_judge_reason": mem.get("judge_reason"), "mode": mode,
                  "coverage": ir.get("coverage"), "source_coverage": source_map.get(k)}
        table.append(record)
        if not correct(mem):
            evidence = source_map.get(k)
            if q["category_id"] == 5:
                stage = "对抗题错误"
            elif not evidence["any_gold"]:
                stage = "已入库引用但最终未选中" if evidence["memory_any_gold"] else "建库没有 gold 来源引用"
            elif not evidence["all_gold"]:
                stage = "最终来源只覆盖部分 gold"
            else:
                stage = "gold 来源齐全仍答错"
            errors.append({**record, "triage_stage": stage})
    aborted = []
    for old_path in aborted_paths:
        old = json.loads(old_path.read_text(encoding="utf-8"))
        assert old["status"] == "aborted" and old["selection_sha256"] == report["selection_sha256"]
        folder = old_path.with_name(old_path.stem + "_ledgers")
        abandoned_cost = ledger_total(folder)
        retained_samples = set()
        resumed = report.get("resumed_from", {})
        if resumed and Path(resumed["path"]).resolve() == old_path.resolve():
            retained_samples = set(resumed["completed_samples"])
            retained_cost = sum(json.loads((folder/f"{sample}_{system}.json").read_text(encoding="utf-8"))["cost_rmb"]
                                for sample in resumed["completed_samples"]
                                for system in ("simplemem", "memnexus", "judge"))
            abandoned_cost -= retained_cost
        unknown_cost = 0.0
        for ledger_path in folder.glob("*.json"):
            if ledger_path.name.endswith((".stage.json", ".memory.json")):
                continue
            if any(ledger_path.stem == f"{sample}_{system}" for sample in retained_samples
                   for system in ("simplemem", "memnexus", "judge")):
                continue
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            unknown_cost += ledger.get("unknown_usage_upper_bound_rmb", 0) + ledger.get("reserved_rmb", 0)
        aborted.append({"run": str(old_path), "failure": old["failure"], "cost_rmb": abandoned_cost,
                        "unknown_usage_upper_bound_rmb": unknown_cost,
                        "returned_usage_cost_rmb": abandoned_cost - unknown_cost,
                        "retained_completed_samples": sorted(retained_samples),
                        "completed_judgments": sum(r.get("judge_result") in {"CORRECT", "WRONG"} for r in old["rows"])})
    return report, {
        "run": str(path), "selection_sha256": report["selection_sha256"], "audit": checked,
        "systems": metrics, "subgroups": subgroups, "architecture_modes": dict(architecture),
        "paired": {"both_correct": pairs[1, 1], "mem_only": pairs[0, 1], "simple_only": pairs[1, 0],
                   "both_wrong": pairs[0, 0],
                   "difference_pp": 100*(metrics["MemNexus"]["correct"] - metrics["SimpleMem"]["correct"])/metrics["SimpleMem"]["n"],
                   "cluster_95ci_pp": [100 * v for v in cluster_ci(clusters, seed=20261013)],
                   "mcnemar_question_independence_p": mcnemar(pairs[0, 1], pairs[1, 0])},
        "evidence_recall": coverage, "errors": errors,
        "error_stages": dict(Counter(e["triage_stage"] for e in errors)), "per_question": table,
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aborted_runs": aborted,
        "all_attempts_cost_rmb": checked["cost_rmb"] + checked.get("incomplete_stage_upper_rmb",0) + sum(r["cost_rmb"] for r in aborted),
        "completed_question_count": metrics["SimpleMem"]["n"],
        "requested_question_count": report["requested_question_count"],
    }


def plot(report, result, base):
    configure_chinese_font()
    stats = result["systems"]
    n, target = result["completed_question_count"], result["requested_question_count"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    ax = axes[0, 0]
    bars = ax.bar(SYSTEMS, [stats[s]["correct"] for s in SYSTEMS], color=[COLORS[s] for s in SYSTEMS], width=.55)
    ax.bar_label(bars, labels=[f"{stats[s]['correct']}/{n}" for s in SYSTEMS], padding=4)
    ax.set_ylim(0, n*1.12)
    ax.set_title("统一裁判正确题数 ↑")
    lo, hi = result["paired"]["cluster_95ci_pp"]
    ax.set_xlabel(f"差值 {result['paired']['difference_pp']:+.1f} 个百分点\n对话聚类95%区间 [{lo:.1f}, {hi:.1f}]")
    ax = axes[0, 1]
    x = np.arange(5)
    for offset, s in [(-.18, "SimpleMem"), (.18, "MemNexus")]:
        categories = stats[s]["categories"]
        values = [100 * categories[str(k)]["correct"] / categories[str(k)]["n"] for k in range(1, 6)]
        bars = ax.bar(x + offset, values, .36, color=COLORS[s], label=s)
        ax.bar_label(bars, labels=[f"{categories[str(k)]['correct']}/{categories[str(k)]['n']}" for k in range(1, 6)],
                     padding=3, fontsize=8)
    ax.set_xticks(x, [f"{k}·{CATEGORY_LABELS[k]}" for k in range(1, 6)])
    ax.set_ylim(0, 117)
    ax.set_title("按原始类别 ID 分层的正确率 (%)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 2]
    labels = ["建库输入", "建库输出", "问答输入", "问答输出"]
    fields = ["build_input_tokens", "build_output_tokens", "qa_input_tokens", "qa_output_tokens"]
    for offset, s in [(-.18, "SimpleMem"), (.18, "MemNexus")]:
        values = [stats[s][field] / 1000 for field in fields]
        bars = ax.bar(np.arange(4) + offset, values, .36, color=COLORS[s])
        ax.bar_label(bars, labels=[f"{v:.0f}" for v in values], fontsize=8, padding=3)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("千 tokens")
    ax.set_title("真实模型 token 用量 ↓")
    ax.margins(y=.18)
    ax = axes[1, 0]
    for i, s in enumerate(SYSTEMS):
        ax.bar(i, stats[s]["build_cost_rmb"], color=COLORS[s], alpha=.4, width=.55)
        ax.bar(i, stats[s]["qa_cost_rmb"], bottom=stats[s]["build_cost_rmb"], color=COLORS[s], width=.55)
        ax.text(i, stats[s]["main_cost_rmb"], f"¥{stats[s]['main_cost_rmb']:.2f}", ha="center", va="bottom")
    ax.set_xticks([0, 1], SYSTEMS)
    ax.set_title("建库＋问答费用估算 ↓")
    ax.set_ylabel("人民币；浅色建库，深色问答")
    ax.margins(y=.2)
    ax = axes[1, 1]
    for s in SYSTEMS:
        latencies = sorted(r["latency_ms"] / 1000 for r in report["rows"] if r["system"] == s)
        ax.step(latencies, np.arange(1, n+1)*100/n, where="post", color=COLORS[s], label=s)
    ax.set_title("回答延迟分布：越靠左越快")
    ax.set_ylabel("累计题目比例 (%)")
    ax.set_xlabel("秒；包括检索、模型调用与本地处理")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 2]
    fields = ["both_correct", "mem_only", "simple_only", "both_wrong"]
    values = [result["paired"][f] for f in fields]
    bars = ax.bar(range(4), values, color=["#7A9985", COLORS["MemNexus"], COLORS["SimpleMem"], "#ADB7C0"])
    ax.bar_label(bars, padding=3)
    ax.set_xticks(range(4), ["两者都对", "仅MemNexus对", "仅SimpleMem对", "两者都错"], rotation=15)
    ax.set_title("同题配对结果")
    ax.set_ylabel("题数")
    ax.margins(y=.18)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
    fig.suptitle(f"MemNexus 架构升级后：LoCoMo 已完成 {n}/{target} 题配对", fontsize=17, y=.98)
    note = f"纳入{result['subgroups']['new']['n']}道未测题、{result['subgroups']['reused']['n']}道复用题；两边独立建库，关闭答案缓存。"
    if n < target:
        note += f" 余下{target-n}题因API余额不足未完成，未计入分母。"
    fig.text(.5, .014, note,
             ha="center", fontsize=10, color="#555")
    fig.subplots_adjust(top=.90, bottom=.13, left=.065, right=.98, hspace=.6, wspace=.34)
    for suffix in (".png", ".svg"):
        fig.savefig(base.with_suffix(suffix), dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    ids = sorted(stats["SimpleMem"]["conversations"])
    for s in SYSTEMS:
        vals = [100 * stats[s]["conversations"][i]["correct"] / stats[s]["conversations"][i]["n"] for i in ids]
        axes[0].plot(range(len(ids)), vals, "o-", color=COLORS[s], label=s, linewidth=2)
    axes[0].set_xticks(range(len(ids)),
                      [f"{i}\n(n={stats['SimpleMem']['conversations'][i]['n']})" for i in ids],
                      rotation=40, ha="right", fontsize=8)
    axes[0].set_ylim(0, 110)
    axes[0].set_title("各对话正确率 (%)")
    axes[0].legend(frameon=False, fontsize=8)
    names = ["建库没有 gold 来源引用", "已入库引用但最终未选中", "最终来源只覆盖部分 gold", "gold 来源齐全仍答错", "对抗题错误"]
    values = [result["error_stages"].get(n, 0) for n in names]
    bars = axes[1].barh(range(5), values, color="#578FA8")
    axes[1].bar_label(bars, padding=4)
    axes[1].set_yticks(range(5), names, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(values, default=0) + 3)
    axes[1].set_title("MemNexus 错题：可观察证据阶段")
    axes[1].set_xlabel("题数；不等同于已证实的因果归因")
    for offset, s in [(-.18, "SimpleMem"), (.18, "MemNexus")]:
        vals = [100 * result["subgroups"][g][s] / result["subgroups"][g]["n"] for g in ("new", "reused")]
        bars = axes[2].bar(np.arange(2) + offset, vals, .36, color=COLORS[s], label=s)
        axes[2].bar_label(bars, labels=[f"{v:.1f}%" for v in vals], padding=3)
    axes[2].set_xticks([0, 1], [f"未测题 (n={result['subgroups']['new']['n']})", f"复用题 (n={result['subgroups']['reused']['n']})"])
    axes[2].set_ylim(0, 112)
    axes[2].set_title("题目重叠分层")
    for ax in axes:
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("对话差异、证据线索与复用题分层", fontsize=16)
    fig.text(.5, .015, "新题仍来自已参与开发的对话；来源覆盖仅按 turn ID 判断，不能证明语义正确。", ha="center", fontsize=10)
    fig.subplots_adjust(top=.83, bottom=.22, left=.055, right=.98, wspace=.60)
    detail = base.with_name(base.name + "_detail")
    for suffix in (".png", ".svg"):
        fig.savefig(detail.with_suffix(suffix), dpi=170)
    plt.close(fig)


def write_report(report, result, base):
    s, m = (result["systems"][name] for name in SYSTEMS)
    n, target = result["completed_question_count"], result["requested_question_count"]
    non_adversarial = sum(q["category_id"] != 5 for q in report["cohort"]["questions"])
    metrics = [(f"裁判正确题数 /{n}", "correct", ".0f"),
               (f"非对抗{non_adversarial}题 LoCoMo 词面 F1（原样）", "locomo_f1_75_raw", ".4f"),
               (f"非对抗{non_adversarial}题 F1（仅去引用标记）", "locomo_f1_75_without_source_markers", ".4f"),
               ("BLEU-1（沿用旧 runner 口径）", "mean_simple_bleu1", ".4f"),
               ("问答输入 tokens", "qa_input_tokens", ",.0f"), ("问答输出 tokens", "qa_output_tokens", ",.0f"),
               ("问答真实请求", "qa_provider_attempts", ".0f"), ("建库真实请求", "build_provider_attempts", ".0f"),
               ("建库＋问答 tokens", "main_total_tokens", ",.0f"), ("建库＋问答费用（元）", "main_cost_rmb", ".4f"),
               ("每答对一题成本（元，含建库）", "main_cost_per_correct", ".5f"),
               ("建库总耗时（秒）", "build_seconds", ".1f"),
               ("建库＋问答总耗时（秒）", "main_total_seconds", ".1f"),
               ("回答 P50（秒）", "p50_seconds", ".3f"), ("回答 P95（秒）", "p95_seconds", ".3f")]
    lo, hi = result["paired"]["cluster_95ci_pp"]
    lines = [f"# MemNexus 架构升级后 LoCoMo 对比：已完成 {n}/{target} 题", "",
             f"统一裁判：SimpleMem {s['correct']}/{n}，MemNexus {m['correct']}/{n}。差值 {result['paired']['difference_pp']:+.2f} 个百分点；按{report['cohort']['conversation_count']}段对话聚类重采样的95%区间 [{lo:.1f}, {hi:.1f}]。", "",
             "## 固定协议与审计", "",
             f"计划固定100题、79道此前两组未测题及21道复用题，类别ID配额13/25/9/28/25，D1–D5共1104条原始turn。实际纳入{n}道已完整配对题，两边分别处理相同{s['raw_turns']}条原始turn；其中{result['subgroups']['new']['n']}道未测题、{result['subgroups']['reused']['n']}道复用题。新题仍来自已参与开发的对话，不能称对话级独立泛化。", "",
             "两边主模型与裁判 deepseek-v4-flash，temperature=0，英文输出；相同Qwen3-Embedding-0.6B、1024维，query prompt与document编码预检匹配。SimpleMem窗口20/stride5，保留原检索与反思策略；MemNexus使用模型planner及新集合路径；答案缓存关闭，单路顺序运行。", "",
             "类别5沿用A=未提及/B=干扰选项协议；裁判不看到系统名称，不编辑预测答案。正式主要结果为统一裁判结果，词面指标原样另列。新增集合IR和完整性状态逐题保存，不能把output_complete视为答案正确。", "",
             f"审计确认{result['audit']['answers']}条非空回答、{result['audit']['judgments']}个有效裁判标签、{result['audit']['memory_builds']}个非空独立记忆库，费用与真实usage逐笔核对。", "",
             "## 主要指标", "", "| 指标 | SimpleMem | MemNexus |", "|---|---:|---:|"]
    for title, field, fmt in metrics:
        a = format(s[field], fmt) if s[field] is not None else "N/A"
        b = format(m[field], fmt) if m[field] is not None else "N/A"
        lines.append(f"| {title} | {a} | {b} |")
    lines += ["", f"本次完成运行（包括裁判）费用估算 ¥{result['audit']['cost_rmb']:.4f}。采用相同的保守单价2.4/9.6元每百万输入/输出token，不是账户账单；不含本地embedding和硬件成本。", "",
              f"账本包含中断尝试的费用上界 ¥{result['all_attempts_cost_rmb']:.4f}；其中失败请求预留费用不等于实际扣款。主表和图只纳入同一修复版本的{n}道完整配对题，不用缺失题或空答案凑足100题。", "",
              "## 中断记录", ""]
    if n < target:
        lines += [f"当前运行在 conv-50 遇到 DeepSeek HTTP 402 Insufficient Balance，剩余{target-n}题尚未完成。程序已停止；该段没有形成可用于评分的完整记忆库，未纳入质量指标。保留前{n}题，充值后可断点续跑。", "",
                  f"本次未完成阶段账本上界为 ¥{result['audit'].get('incomplete_stage_upper_rmb',0):.4f}，包含没有 usage 的402失败请求预留上界；它不是已确认费用。", ""]
    for old in result["aborted_runs"]:
        retained = old["retained_completed_samples"]
        disposition = (f"其中{len(retained)}段已完成对话的回答和账本已纳入本次续跑；此处仅加计被放弃阶段，未重复计费。"
                       if retained else "该次回答未纳入主结果。")
        lines.append(f"- `{old['run']}`：{old['failure']}；额外费用上界 ¥{old['cost_rmb']:.4f}，其中返回usage对应估算 ¥{old['returned_usage_cost_rmb']:.4f}、未知usage预留上界 ¥{old['unknown_usage_upper_bound_rmb']:.4f}。保留 {old['completed_judgments']} 条裁判记录。{disposition}")
    lines += ["", "首次运行发现校验器错误：模型正确拒绝其他主体的候选时，校验器仍要求被拒绝候选属于查询主体。修复仅将主体/数量的查询约束限定于accepted候选；被拒绝候选仍校验原文来源。新增回归后主项目282项测试通过，随后固定相同100题、两套系统全部重新建库与评分，未只重跑错题。", "",
              "## 类别和配对结果", "", "| 类别ID | 题数 | SimpleMem对 | MemNexus对 |", "|---|---:|---:|---:|"]
    for k in range(1,6):
        a,b=s["categories"][str(k)],m["categories"][str(k)]
        lines.append(f"| {k}（{CATEGORY_LABELS[k]}） | {a['n']} | {a['correct']} | {b['correct']} |")
    lines += ["", f"配对：两者都对{result['paired']['both_correct']}，仅MemNexus对{result['paired']['mem_only']}，仅SimpleMem对{result['paired']['simple_only']}，两者都错{result['paired']['both_wrong']}。题目独立假设下McNemar p={result['paired']['mcnemar_question_independence_p']:.5f}；主解释使用对话聚类区间。", "",
              "## 新题与复用题", "", "| 分组 | 题数 | SimpleMem对 | MemNexus对 |", "|---|---:|---:|---:|"]
    for name in ("new","reused"):
        group=result["subgroups"][name]
        lines.append(f"| {name} | {group['n']} | {group['SimpleMem']} | {group['MemNexus']} |")
    lines += ["", "## 新架构路径观察", "", "| 路径 | 题数 | 答对 | output_complete |", "|---|---:|---:|---:|"]
    for mode, group in result["architecture_modes"].items():
        lines.append(f"| {mode} | {group['n']} | {group['correct']} | {group['output_complete']} |")
    lines += ["", "路径组的题型不同，不能直接用组间正确率证明某路径因果上更好。结构化输出保全成员，也可能保全错误接受或未正确归一的成员。", "",
              "本地直接输出但被判错的案例：", ""]
    native_errors = [r for r in result["errors"] if r["mode"] == "indexed"]
    for row in native_errors:
        lines.append(f"- {row['sample_id']}/{row['qa_index']}：{row['question']}；输出：{row['MemNexus']}；Gold：{row['gold']}。裁判：{row['MemNexus_judge_reason']}")
    if not native_errors:
        lines.append("本轮未观察到此类错误；不能据此推断未测问题也不会出错。")
    lines += ["",
              "## 错误线索", ""]
    for stage, count in result["error_stages"].items():
        lines.append(f"- {stage}：{count}题。")
    lines += ["", "SimpleMem返回的是压缩文本，缺少同一粒度的原始turn ID，因此不能虚构其Evidence Recall来对比。MemNexus来源覆盖也只证明turn进入上下文，不能证明具体答案片段、主体或事件匹配。", "",
              "## 解释边界", "",
              "这是单次、固定证据范围的开发对比，不是完整LoCoMo论文主结果；按gold evidence horizon选题带来选择偏差。历史版本的100题样本不同，不能把旧分数与本轮分数之差当作单组件提升。温度0仍可能有模型随机性，延迟还受网络、供应商负载与顺序运行影响。是否支持显著优势应由本轮配对区间判断。", "",
              "## 逐题结果", "", "| ID / 类别 | 问题与gold | SimpleMem | MemNexus | 判定与裁判理由 |", "|---|---|---|---|---|"]
    for r in result["per_question"]:
        lines.append(f"| {r['sample_id']}/{r['qa_index']} / {r['category_id']} | {cell(r['question'])}<br>Gold: {cell(r['gold'])} | {cell(r['SimpleMem'])} | {cell(r['MemNexus'])} | SimpleMem={r['SimpleMem_correct']}: {cell(r['SimpleMem_judge_reason'])}<br>MemNexus={r['MemNexus_correct']}: {cell(r['MemNexus_judge_reason'])} |")
    base.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "result/locomo_100_collection_20261013.json")
    parser.add_argument("--aborted-run", type=Path, action="append", default=[])
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report, result = analyze(args.run, args.aborted_run, args.allow_partial)
    suffix = f"_PARTIAL_{result['completed_question_count']}" if result["completed_question_count"] < result["requested_question_count"] else ""
    base = ROOT / ("result/LOCOMO_100_COLLECTION_20261013_ANALYSIS"+suffix)
    base.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(report, result, ROOT / ("result/locomo_100_collection_20261013_comparison"+suffix.lower()))
    write_report(report, result, base)
    print(json.dumps({"systems": result["systems"], "paired": result["paired"], "audit": result["audit"]}, ensure_ascii=False))

"""Offline analysis of completed frozen runs. Never calls a model or alters predictions."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_locomo_disjoint import configure_chinese_font

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(folder, old_folder):
    state = read(folder / "run.json")
    if state["status"] != "completed":
        raise ValueError("A partial run cannot be reported as a completed comparison")
    summary = read(folder / "summary.json")
    old = read(old_folder / "summary.json")
    rows = read(folder / "results.json")["rows"]
    old_rows = read(old_folder / "results.json")["rows"]
    current = summary["stats"]["LoCoMo"]
    past = old["stats"]["LoCoMo"]
    previous = {(r["item_id"], r["system"]): r for r in old_rows}
    if set(previous) != {(r["item_id"], r["system"]) for r in rows}:
        raise ValueError("Question identities differ")
    phase = defaultdict(lambda: Counter())
    for path in sorted((folder / "attempts").glob("locomo-*/usage.calls.jsonl")):
        system = "MemNexus" if "memnexus" in path.parent.name else "SimpleMem"
        requests = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["event"] == "request":
                requests[event["attempt"]] = event["body"]
            elif event["event"] == "response":
                request = requests.get(event["attempt"], {})
                text = "\n".join(str(m.get("content", "")) for m in request.get("messages", []))
                if system != "MemNexus":
                    continue
                label = ("extract" if "短期记忆的增量生成" in text else
                         "plan" if "用户意图感知查询规划" in text else
                         "bind" if "Fill each mandatory output" in text else
                         "synthesize" if "Return one concise answer table" in text else "other")
                if "previous_response" in text:
                    label += "_repair"
                usage = event["body"].get("usage", {})
                phase[label]["calls"] += 1
                phase[label]["input_tokens"] += usage.get("prompt_tokens", 0)
                phase[label]["output_tokens"] += usage.get("completion_tokens", 0)
                phase[label]["prompt_chars"] += len(text)

    diagnostic = []
    transitions = Counter()
    for row in rows:
        if row["system"] != "MemNexus":
            continue
        reasoning = row.get("reasoning", {})
        execution = reasoning.get("execution") or {}
        before = previous[row["item_id"], "MemNexus"]
        transitions[before["judge_result"] + "->" + row["judge_result"]] += 1
        diagnostic.append({"id": row["item_id"], "category": row["category"],
            "question": row["question"], "gold": row["gold_answer"], "prediction": row["prediction"],
            "judge": row["judge_result"], "judge_reason": row.get("judge_reason"),
            "previous_judge": before["judge_result"], "path": reasoning.get("answer_path"),
            "error": reasoning.get("semantic_error"), "gaps": execution.get("gaps", []),
            "rejected": execution.get("rejected", []), "followup": reasoning.get("premise_retrieval"),
            "provider_budget": reasoning.get("provider_request_budget"),
            "old_prediction": before["prediction"]})
    metrics = [
        ("正确率", lambda s: 100*s["accuracy"], "%"),
        ("Token F1（100题）", lambda s: s["mean_f1"], ""),
        ("LoCoMo风格F1（75道非对抗题）", lambda s: s["locomo_style_nonadversarial_f1"], ""),
        ("问答输入tokens", lambda s: s["input_tokens"], ""),
        ("问答输出tokens", lambda s: s["output_tokens"], ""),
        ("问答总tokens", lambda s: s["input_tokens"]+s["output_tokens"], ""),
        ("建库输入tokens", lambda s: s["build_input_tokens"], ""),
        ("建库输出tokens", lambda s: s["build_output_tokens"], ""),
        ("全流程总tokens", lambda s: s["input_tokens"]+s["output_tokens"]+s["build_input_tokens"]+s["build_output_tokens"], ""),
        ("建库费用", lambda s: s["build_cost_rmb"], "元"),
        ("问答费用", lambda s: s["cost_rmb"], "元"),
        ("建库+问答费用", lambda s: s["cost_rmb"]+s["build_cost_rmb"], "元"),
        ("问答P50", lambda s: s["p50_latency_ms"]/1000, "秒"),
        ("问答P95", lambda s: s["p95_latency_ms"]/1000, "秒"),
    ]
    labels = ["上一完整轮MemNexus", "本轮SimpleMem", "本轮MemNexus"]
    values = [past["MemNexus"], current["SimpleMem"], current["MemNexus"]]
    lines = ["# 时间执行与调用额度修复：LoCoMo-100复测", "",
             "本轮100题、10段对话D1–D5、20个独立记忆库。主模型/裁判deepseek-v4-flash，缓存关闭；LongMemEval未运行。",
             "原始结果和裁判标签保持不变。相同题组重复使用属于开发回归，不能称独立泛化。", "",
             "| 指标 | 上一完整轮MemNexus | 本轮SimpleMem | 本轮MemNexus |", "|---|---:|---:|---:|"]
    for label, fn, unit in metrics:
        rendered = [f"{fn(v):,.4f}{unit}" for v in values]
        lines.append("| " + " | ".join([label, *rendered]) + " |")
    paired = summary["paired"]["LoCoMo"]
    lines += ["", "## 配对统计", "", json.dumps(paired, ensure_ascii=False, indent=2), "",
              "置信区间按整段对话聚类；只有10个聚类，精度有限。McNemar基于题目独立假设，仅作补充。",
              "多项指标和逐题比较均属探索性分析，未作多重比较校正。", "",
              "## 分题型", "", "| 类别 | SimpleMem正确/总数 | MemNexus正确/总数 | SimpleMem F1 | MemNexus F1 |", "|---|---:|---:|---:|---:|"]
    for category, a in current["SimpleMem"]["by_category"].items():
        b = current["MemNexus"]["by_category"][category]
        lines.append(f"| {category} | {a['correct']}/{a['n']} | {b['correct']}/{b['n']} | {a['mean_f1']:.4f} | {b['mean_f1']:.4f} |")
    lines += ["", "## MemNexus真实调用分解", "", "|阶段|请求数|输入tokens|输出tokens|估算费用|", "|---|---:|---:|---:|---:|"]
    for label, data in phase.items():
        cost = (data['input_tokens']*2.4+data['output_tokens']*9.6)/1e6
        lines.append(f"|{label}|{data['calls']}|{data['input_tokens']}|{data['output_tokens']}|{cost:.4f}|")
    lines += ["", "## 版本间变化", "", json.dumps(transitions, ensure_ascii=False),
              "同时重新建库、调用模型和评分，不能将版本差异全部归因于时间修复一个因素。", "",
              "## 审计", "", json.dumps(read(folder/'audit.json'), ensure_ascii=False, indent=2),
              "费用是配置单价乘真实usage，未扣供应商缓存折扣；不是账户余额或实扣账单。", "",
              "## 逐题失败与变化", ""]
    for d in diagnostic:
        if d['judge'] != 'CORRECT' or d['judge'] != d['previous_judge']:
            lines += [f"### {d['id']} ({d['category']})", "", d['question'],
                      f"Gold: {d['gold']}", f"之前: {d['old_prediction']}", f"现在: {d['prediction']}",
                      f"裁判: {d['judge']} / {d['judge_reason']}",
                      f"路径: {d['path']}；缺口: {json.dumps(d['gaps'], ensure_ascii=False)}", ""]
    out = {"phase_usage": dict(phase), "transitions": dict(transitions), "diagnostics": diagnostic}
    (folder/'repair_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    (folder/'repair_analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    configure_chinese_font()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, index in zip(axes.flat, (0, 1, 5, 10, 11, 12)):
        label, fn, unit = metrics[index]
        ys = [fn(v) for v in values]
        bars = ax.bar(labels, ys, color=['#9aa5b1', '#d18a35', '#2478ae'])
        ax.bar_label(bars, labels=[f"{y:,.3f}" for y in ys], padding=4, fontsize=9)
        ax.set_ylim(0, max(ys)*1.25 or 1); ax.set_title(label+' '+unit)
        ax.tick_params(axis='x', labelrotation=12); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle("同一LoCoMo-100开发题组：修复前后与SimpleMem对照", fontsize=16)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('png','svg'):
        fig.savefig(folder/f'repair_comparison.{ext}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--old', type=Path, default=ROOT/'result/benchmark_locomo100_valid_20261016')
    args = parser.parse_args()
    analyze(args.folder, args.old)

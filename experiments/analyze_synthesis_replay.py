"""Audit and plot the completed fixed-memory replay, without rewriting predictions."""

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_locomo_disjoint import configure_chinese_font
from analyze_optimized_comparison import cluster_ci, mcnemar

ROOT = Path(__file__).resolve().parents[1]
LABELS = {1: "多跳", 2: "时间", 3: "开放域", 4: "单跳", 5: "对抗"}


def analyze(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "completed", "Do not present partial data as a complete comparison"
    source = Path(report["source"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"]
    old = json.loads(source.read_text(encoding="utf-8"))
    for p, digest in report["code_sha256"].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest, p
    for sid, digest in report["snapshot_sha256"].items():
        p = source.with_name(source.stem+"_ledgers") / f"{sid}_memnexus.memory.json"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    indexed = {(r["system"], r["sample_id"], r["qa_index"]): r for r in report["rows"]}
    assert len(indexed) == len(report["rows"]) == 200
    original = {(r["system"], r["sample_id"], r["qa_index"]): r for r in old["rows"]}
    assert set(indexed) == set(original)
    for key, row in indexed.items():
        assert row["prediction"].strip() and row["judge_result"] in {"CORRECT", "WRONG"}
        assert row["question"] == original[key]["question"] and row["gold_answer"] == original[key]["gold_answer"]
        if key[0] == "SimpleMem":
            assert row["prediction"] == original[key]["prediction"]
    continuation = report.get("resumed_from")
    if continuation:
        previous = json.loads(Path(continuation["path"]).read_text(encoding="utf-8"))
        sys.path.insert(0,str(ROOT/"MemNexus/memory-system/src"))
        from agent_memory.models import Hit
        from agent_memory.synthesis import AnswerTable, answer_type, render_table, validate_table
        retained = 0
        for row in previous["rows"]:
            key = row["system"],row["sample_id"],row["qa_index"]
            assert indexed[key]["prediction"] == row["prediction"]
            if row["system"] != "MemNexus":
                continue
            if "answer_table" in row.get("reasoning", {}):
                table = AnswerTable.model_validate(row["reasoning"]["answer_table"])
                validate_table(table,[Hit.model_validate(h) for h in row["retrieved_sources"]],
                               row["plan"]["required_info"] or [row["evaluation_query"]],answer_type(row["evaluation_query"]))
                assert render_table(table) == row["prediction"]
            elif row.get("reasoning", {}).get("answer_path") not in {"semantic_execution", "source_bound_synthesis"}:
                raise AssertionError("Retained row lacks an auditable answer receipt")
            retained += 1
        continuation = {"retained_answers_revalidated":retained,"policy":continuation["policy"]}
    audit = Counter()
    for p in path.with_name(path.stem+"_ledgers").glob("*.json"):
        ledger = json.loads(p.read_text(encoding="utf-8"))
        trace = [json.loads(s) for s in p.with_suffix(".calls.jsonl").read_text(encoding="utf-8").splitlines()]
        requests = [e for e in trace if e["event"] == "request"]
        responses = [e for e in trace if e["event"] == "response"]
        assert len(requests) == ledger["provider_attempts"] and ledger.get("reserved_rmb", 0) == 0
        assert all(e["body"]["model"] == "deepseek-v4-flash" and e["body"]["temperature"] == 0 for e in requests)
        assert all(e["body"].get("usage") for e in responses)
        for field, usage_key in [("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")]:
            assert sum(e["body"]["usage"].get(usage_key, 0) for e in responses) == ledger[field]
        for field in ("provider_attempts", "provider_errors", "input_tokens", "output_tokens", "cost_rmb"):
            audit[field] += ledger.get(field, 0)
        audit["ledgers"] += 1
    assert audit["ledgers"] == 20
    assert math.isclose(audit["cost_rmb"],sum(x["cost_rmb"] for x in report["qa_usage"]+report["judge_usage"]),abs_tol=1e-8)
    stats = {}
    for system in ("SimpleMem", "MemNexus"):
        rows = [r for r in report["rows"] if r["system"] == system]
        stages = [s for s in old["stages"] if s["system"] == system]
        build = {f: sum(s["build_usage"].get(f,0) for s in stages) for f in ("input_tokens","output_tokens","cost_rmb","provider_attempts")}
        answer_usage = {f: sum(r["model_calls"].get(f,0) for r in rows) for f in build}
        qa = ({f:sum(u.get(f,0) for u in report["qa_usage"]) for f in build}
              if system == "MemNexus" else answer_usage)
        stats[system] = {"correct":sum(r["judge_result"]=="CORRECT" for r in rows), "n":len(rows),
                         "categories":{c:{"n":sum(r["category_id"]==c for r in rows),
                                          "correct":sum(r["category_id"]==c and r["judge_result"]=="CORRECT" for r in rows)} for c in range(1,6)},
                         "qa":qa,"successful_answer_usage":answer_usage,
                         "interrupted_attempt_usage":{f:qa[f]-answer_usage[f] for f in build},
                         "build":build,"main_cost_rmb":qa["cost_rmb"]+build["cost_rmb"],
                         "qa_tokens":qa["input_tokens"]+qa["output_tokens"],
                         "total_tokens":qa["input_tokens"]+qa["output_tokens"]+build["input_tokens"]+build["output_tokens"],
                         "p50_seconds":statistics.median(r["latency_ms"]/1000 for r in rows),
                         "p95_seconds":float(np.percentile([r["latency_ms"]/1000 for r in rows],95))}
    assert stats["MemNexus"]["main_cost_rmb"] + report["prior_development_spend_rmb"] <= stats["SimpleMem"]["main_cost_rmb"]
    clusters = defaultdict(list)
    pairs = Counter()
    transitions = Counter()
    for key,row in indexed.items():
        if key[0] != "MemNexus":continue
        baseline=indexed[("SimpleMem",*key[1:])]
        a,b=baseline["judge_result"]=="CORRECT", row["judge_result"]=="CORRECT"
        pairs[a,b]+=1
        clusters[key[1]].append(int(b)-int(a))
        transitions[original[key]["judge_result"],row["judge_result"]]+=1
    result={"stats":stats,"audit":dict(audit),"paired":{"both_correct":pairs[True,True],"mem_only":pairs[False,True],
            "simple_only":pairs[True,False],"both_wrong":pairs[False,False],
            "difference_pp":stats["MemNexus"]["correct"]-stats["SimpleMem"]["correct"],
            "cluster_95ci_pp":[100*x for x in cluster_ci(clusters, seed=20261013)],
            "mcnemar_p":mcnemar(pairs[False,True],pairs[True,False])},
            "simplemem_rejudge_changed":sum(r["judge_result"]!=r["original_judge_result"] for r in report["rows"] if r["system"]=="SimpleMem"),
            "old_to_new_memnexus":{f"{a}->{b}":v for (a,b),v in transitions.items()},
            "prior_development_spend_rmb":report["prior_development_spend_rmb"],"limitations":report["limitations"],
            "continuation_audit":continuation}
    base=path.with_name(path.stem+"_analysis")
    base.with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    configure_chinese_font()
    systems=list(stats);colors=["#D18A35","#2478AE"]
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for ax,field,title in [(axes[0,0],"correct","统一重新盲评：正确题数 /100 ↑"),
                           (axes[0,2],"qa_tokens","问答阶段真实 tokens ↓"),
                           (axes[1,0],"main_cost_rmb","原建库＋本版本问答：费用估算 ↓"),
                           (axes[1,1],"p50_seconds","回答 P50（秒）↓")]:
        vals=[stats[s][field] for s in systems]
        bars=ax.bar(systems,vals,color=colors,width=.55)
        ax.bar_label(bars,labels=[f"{v:,.2f}" if field in {"main_cost_rmb","p50_seconds"} else f"{v:,}" for v in vals],padding=4)
        ax.set_title(title);ax.margins(y=.2)
    axes[0,0].set_ylim(0,112)
    for j,s in enumerate(systems):
        cats=stats[s]["categories"]
        vals=[100*cats[c]["correct"]/cats[c]["n"] for c in range(1,6)]
        bars=axes[0,1].bar(np.arange(5)+(j-.5)*.36,vals,.36,label=s,color=colors[j])
        axes[0,1].bar_label(bars,labels=[f"{cats[c]['correct']}/{cats[c]['n']}" for c in range(1,6)],fontsize=8,padding=3)
    axes[0,1].set_xticks(range(5),list(LABELS.values()));axes[0,1].set_ylim(0,115)
    axes[0,1].set_title("分题型正确率 (%)");axes[0,1].legend(fontsize=8)
    names=["both_correct","mem_only","simple_only","both_wrong"]
    bars=axes[1,2].bar(range(4),[result['paired'][k] for k in names],color=["#7A9985",colors[1],colors[0],"#ADB7C0"])
    axes[1,2].bar_label(bars,padding=3);axes[1,2].margins(y=.2)
    axes[1,2].set_xticks(range(4),["都对","仅MemNexus对","仅SimpleMem对","都错"],rotation=12)
    axes[1,2].set_title("同题配对")
    for ax in axes.flat:ax.grid(axis="y",alpha=.16);ax.set_axisbelow(True)
    lo,hi=result["paired"]["cluster_95ci_pp"]
    fig.suptitle(f"统一答案表＋话轮证据组：固定记忆库100题开发复测\n正确率差 {result['paired']['difference_pp']:+.0f} 个百分点；对话聚类95%区间 [{lo:.1f}, {hi:.1f}]",fontsize=15)
    fig.text(.5,.02,"SimpleMem答案复用，双方重新盲评；未重新建库；不是独立泛化测试。费用非实际账单，延迟来自不同时段。",ha="center",fontsize=10)
    fig.subplots_adjust(top=.85,bottom=.12,hspace=.4,wspace=.3)
    for suffix in (".png",".svg"):fig.savefig(base.with_suffix(suffix),dpi=170)
    plt.close(fig)
    lines=["# 固定记忆库100题复测", "", "这是开发集上的检索和回答端比较；原始记忆保持不变，SimpleMem原答案复用，双方重新盲评。不是新建库全流程实验，也不是独立泛化测试。", "",
           "| 指标 | SimpleMem | MemNexus |","|---|---:|---:|"]
    for title,k in [("正确题数","correct"),("问答tokens","qa_tokens"),("含建库tokens","total_tokens"),("含建库费用估算","main_cost_rmb"),("P50秒","p50_seconds"),("P95秒","p95_seconds")]:
        lines.append(f"| {title} | {stats['SimpleMem'][k]} | {stats['MemNexus'][k]} |")
    lines += ["",f"配对统计：{json.dumps(result['paired'],ensure_ascii=False)}。10个对话簇的区间有不稳定性；没有足够证据时不能称显著优势。", "",
              f"续跑审计：{continuation}。前84题后修复反向年份推导；保留所有已完成回答，按最终验证器重新校验且渲染一致，不按正确/错误筛选保留。这是开发中断续跑，非单版本从头运行。", "",
              f"SimpleMem重新评分改变{result['simplemem_rejudge_changed']}题标签；旧版MemNexus到新版本的标签变化为{result['old_to_new_memnexus']}，包含裁判波动因素。", "",
              f"本次实际新增主模型和裁判费用估算¥{audit['cost_rmb']:.4f}，此前中断开发尝试¥{report['prior_development_spend_rmb']:.4f}。系统成本比较另计原始MemNexus建库费用；未将复用记忆当免费建库。", "",
              "审计：200条非空回答及评分，原题与gold一致、SimpleMem答案未改动，源码和10份记忆快照哈希核验，20份新账本请求与usage对账。", "",
              "## 逐题对照", "", "| ID/类别 | 问题 | Gold | SimpleMem / 评分 | MemNexus / 评分 | MemNexus裁判理由 |", "|---|---|---|---|---|---|"]
    def cell(x):return str(x).replace("|","\\|").replace("\n","<br>")
    for row in report["rows"]:
        if row["system"]!="MemNexus":continue
        b=indexed[("SimpleMem",row["sample_id"],row["qa_index"])]
        lines.append("| "+" | ".join(map(cell,[f"{row['sample_id']}/{row['qa_index']} · {LABELS[row['category_id']]}",row['question'],row['gold_answer'],f"{b['prediction']} / {b['judge_result']}",f"{row['prediction']} / {row['judge_result']}",row.get('judge_reason')]))+" |")
    base.with_suffix(".md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path",type=Path)
    analyze(parser.parse_args().path)

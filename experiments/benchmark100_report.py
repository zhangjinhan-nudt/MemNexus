"""Audit and render a portable report for any completed benchmark100 run."""

from __future__ import annotations

import hashlib
import html
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_locomo_disjoint import configure_chinese_font
from analyze_optimized_comparison import cluster_ci, locomo_score, mcnemar
from benchmark100 import FIELDS, SYSTEMS, digest, parse_judgment, read, save

LABELS={1:"多跳",2:"时间",3:"开放域",4:"单跳",5:"对抗"}
COLORS=("#D18A35","#2478AE")


def equal(a,b):
    return math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-10)


def ledger_audit(path):
    ledger=read(path)
    events=[json.loads(s) for s in path.with_suffix(".calls.jsonl").read_text(encoding="utf-8").splitlines()]
    req=[e for e in events if e["event"]=="request"]
    responses=[e for e in events if e["event"]=="response"]
    unaccounted=ledger["provider_attempts"]-len(req)
    if unaccounted:
        # Historical recovery is a separate, hash-bound audit receipt. Never
        # invent a provider event or silently forgive a gap from a log keyword.
        console=path.parent/"console.log"
        text=console.read_text(encoding="utf-8",errors="replace") if console.exists() else ""
        receipt_path=path.with_suffix(".recovery.json")
        receipt=read(receipt_path) if receipt_path.exists() else {}
        ids=[e.get("attempt") for e in req]
        response_ids=[e.get("attempt") for e in responses]
        missing=set(range(1,ledger["provider_attempts"]+1))-set(ids)
        files={"usage.json":path,"usage.calls.jsonl":path.with_suffix(".calls.jsonl"),"console.log":console}
        valid_hashes=(set(receipt.get("sha256",{}))==set(files)
                      and all(p.exists() and digest(p)==receipt["sha256"][name] for name,p in files.items()))
        failures=[line for line in text.splitlines() if "Failed to generate targeted queries:" in line
                  and "WinError 5" in line and "usage.json.tmp" in line
                  and str(path).replace("\\", "\\\\") in line]
        if (unaccounted == 1 and valid_hashes and len(failures)==receipt.get("log_occurrences", 1)
                and receipt.get("kind")=="pre_request_ledger_write_failure"
                and missing==set(receipt.get("missing_attempt_ids",[]))
                and len(ids)==len(set(ids)) and sorted(ids)==sorted(response_ids)
                and set(ids)<=set(range(1,ledger["provider_attempts"]+1))
                and not ledger.get("reserved_rmb",0)):
            ledger["audit_recovery"]={**receipt,"receipt":str(receipt_path),
                                      "verified_request_count":len(req)}
            unaccounted=0
    if unaccounted or ledger.get("reserved_rmb",0):
        raise ValueError(f"账本请求未结算：{path}")
    if not all(e["body"]["model"]=="deepseek-v4-flash" and e["body"]["temperature"]==0 for e in req):
        raise ValueError("调用的模型/温度不符合协议。")
    if any(not e["body"].get("usage") for e in responses):raise ValueError("已响应调用缺少usage。")
    for name,field in (("input_tokens","prompt_tokens"),("output_tokens","completion_tokens")):
        if sum(e["body"]["usage"].get(field,0) for e in responses)!=ledger[name]:
            raise ValueError(f"token账本不一致：{path}")
    returned=(ledger["input_tokens"]*2.4+ledger["output_tokens"]*9.6)/1e6
    if not equal(returned+ledger.get("unknown_usage_upper_bound_rmb",0),ledger["cost_rmb"]):
        raise ValueError(f"费用账本不一致：{path}")
    return ledger,responses


def audit(folder):
    report=read(folder/"run.json")
    if report.get("invalidated"):
        raise ValueError("运行中代码变化，实验版本不一致；请新建测试。")
    cohort=report["cohort"]
    expected={(q["sample_id"],q["qa_index"],s) for q in cohort["questions"] for s in SYSTEMS}
    samples={q["sample_id"] for q in cohort["questions"]}
    if len(expected)!=200 or len(report["judgments"])!=200 or len(report["stages"])!=2*len(samples):
        raise ValueError("未完成100题配对及评分，不能生成完整报告。")
    with zipfile.ZipFile(folder/"source.zip") as archive:
        for name,sha in report["signature"]["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=sha:raise ValueError("实验代码快照哈希不一致。")
    dataset_path=Path(report["dataset_path"])
    if digest(dataset_path)!=cohort["dataset_sha256"]:raise ValueError("数据集哈希不一致。")
    dataset=read(dataset_path)
    items={(q["sample_id"],q["qa_index"]):q for q in cohort["questions"]}
    rows,stages,successful_ledgers=[],[],set()
    recoveries=[]
    for checkpoint in report["stages"].values():
        path=folder/checkpoint
        data=read(path)
        stage=data["stage"]
        if stage["memory_count"]<=0 or stage.get("failed_windows",0):raise ValueError("发现空记忆库或失败窗口。")
        memory=read(path.parent/"usage.memory.json")
        if len(memory["memories"])!=stage["memory_count"]:raise ValueError("记忆快照与阶段记录不一致。")
        raw=memory.get("turns",memory.get("raw_turns",[]))
        if len(raw)!=stage["raw_turns"]:raise ValueError("原始话轮计数不一致。")
        ledger_path=path.parent/"usage.json"
        ledger,_=ledger_audit(ledger_path)
        if ledger.get("audit_recovery"):
            recoveries.append(ledger["audit_recovery"])
        successful_ledgers.add(ledger_path)
        for f in FIELDS:
            if not equal(ledger.get(f,0),stage["total_usage"].get(f,0)):
                raise ValueError("阶段累计usage与账本不一致。")
            partition=stage["build_usage"].get(f,0)+sum(r["model_calls"].get(f,0) for r in data["rows"])
            if not equal(partition,ledger.get(f,0)):raise ValueError("建库/问答用量分解不一致。")
        stages.append(stage)
        rows.extend(data["rows"])
    indexed={(r["sample_id"],r["qa_index"],r["system"]):r for r in rows}
    if len(rows)!=200 or set(indexed)!=expected:raise ValueError("回答重复或题目缺失。")
    judge_cost=0.0
    judge_repairs=0
    for k,row in indexed.items():
        if not row["prediction"].strip():raise ValueError("存在空答案。")
        item=items[k[:2]]
        qa=dataset[item["sample_index"]]["qa"][item["qa_index"]]
        gold="Not mentioned in the conversation" if qa["category"]==5 else qa["answer"]
        if row["question"]!=qa["question"] or row["gold_answer"]!=gold or row["category_id"]!=qa["category"]:
            raise ValueError("问题或gold不匹配。")
        unit=f"{row['sample_id']}_{row['qa_index']}_{row['system'].lower()}"
        path=folder/report["judgments"][unit]
        judgment=read(path)
        if judgment["prediction_sha256"]!=hashlib.sha256(row["prediction"].encode()).hexdigest():
            raise ValueError("裁判预测哈希不匹配。")
        verdict=parse_judgment(judgment["judge_raw_response"])
        if verdict["label"]!=judgment["judge_result"]:raise ValueError("裁判标签不匹配。")
        ledger_path=path.parent/"usage.json"
        ledger,responses=ledger_audit(ledger_path)
        if not any(e["body"]["choices"][0]["message"]["content"]==judgment["judge_raw_response"] for e in responses):
            raise ValueError("裁判结果不在真实响应中。")
        events=[json.loads(s) for s in ledger_path.with_suffix(".calls.jsonl").read_text(encoding="utf-8").splitlines()]
        requests=[e for e in events if e["event"]=="request"]
        if not requests:
            raise ValueError("裁判账本没有请求。")
        expected_payload={k:row[k] for k in ("question","gold_answer","prediction")}
        for request in requests:
            # Repairs append a message but retain the same original question.
            payload=json.loads(request["body"]["messages"][1]["content"].split("\n",1)[1])
            if payload!=expected_payload:
                raise ValueError("裁判请求不是该题的原始问题、gold和预测。")
        if not all(equal(ledger.get(f,0),judgment["usage"].get(f,0)) for f in FIELDS):
            raise ValueError("裁判摘要用量与账本不一致。")
        successful_ledgers.add(ledger_path)
        judge_cost+=ledger["cost_rmb"]
        judge_repairs+=len(judgment["format_repairs"])
        row.update(judge_result=verdict["label"],judge_reason=verdict["reason"])
    all_cost=0.0
    unknown=0.0
    unsuccessful=[]
    for path in (folder/"attempts").glob("*/usage.json"):
        ledger=read(path)
        cost=ledger.get("cost_rmb",0)+ledger.get("reserved_rmb",0)
        all_cost+=cost
        unknown+=ledger.get("unknown_usage_upper_bound_rmb",0)+ledger.get("reserved_rmb",0)
        if path not in successful_ledgers:
            unsuccessful.append({"ledger":path.relative_to(folder).as_posix(),"cost_upper_rmb":cost})
    effective=sum(s["total_usage"]["cost_rmb"] for s in stages)+judge_cost
    if not equal(effective+sum(x["cost_upper_rmb"] for x in unsuccessful),all_cost):raise ValueError("失败尝试费用未对齐。")
    if all_cost>report["budget_rmb"]+1e-8:raise ValueError("总用量超过配置预算，需人工核查。")
    return report,rows,stages,{"answers":200,"judgments":200,"independent_memory_builds":len(stages),
        "successful_ledger_files":len(successful_ledgers),"judge_format_repairs":judge_repairs,
        "judge_cost_rmb":judge_cost,"effective_completed_cost_rmb":effective,
        "all_attempts_cost_upper_rmb":all_cost,"unknown_usage_upper_rmb":unknown,"abandoned_attempts":unsuccessful,
        "ledger_recoveries":recoveries,"protocol_deviations":bool(recoveries),
        "source_snapshot_verified":True,"dataset_verified":True,"usage_reconciled":True}


def summarize(rows,stages):
    stats={}
    for system in SYSTEMS:
        subset=[r for r in rows if r["system"]==system]
        selected=[s for s in stages if s["system"]==system]
        build={f:sum(s["build_usage"].get(f,0) for s in selected) for f in FIELDS}
        qa={f:sum(r["model_calls"].get(f,0) for r in subset) for f in FIELDS}
        correct=sum(r["judge_result"]=="CORRECT" for r in subset)
        stats[system]={"n":len(subset),"correct":correct,"accuracy":correct/len(subset),"build":build,"qa":qa,
            "main_cost_rmb":build["cost_rmb"]+qa["cost_rmb"],
            "main_tokens":sum(build[k]+qa[k] for k in ("input_tokens","output_tokens")),
            "qa_tokens":qa["input_tokens"]+qa["output_tokens"],
            "cost_per_correct":(build["cost_rmb"]+qa["cost_rmb"])/correct if correct else None,
            "p50_seconds":statistics.median(r["latency_ms"]/1000 for r in subset),
            "p95_seconds":float(np.percentile([r["latency_ms"]/1000 for r in subset],95)),
            "build_seconds":sum(s["build_ms"] for s in selected)/1000,
            "qa_seconds":sum(r["latency_ms"] for r in subset)/1000,
            "nonadversarial_locomo_f1":statistics.mean(locomo_score(r) for r in subset if r["category_id"]!=5),
            "simple_bleu1":statistics.mean(r["bleu1"] for r in subset),
            "categories":{c:{"n":sum(r["category_id"]==c for r in subset),
                "correct":sum(r["category_id"]==c and r["judge_result"]=="CORRECT" for r in subset)} for c in LABELS}}
    idx={(r["sample_id"],r["qa_index"],r["system"]):r for r in rows}
    clusters=defaultdict(list)
    pairs=Counter()
    for row in rows:
        if row["system"]!="MemNexus":continue
        other=idx[row["sample_id"],row["qa_index"],"SimpleMem"]
        a,b=int(other["judge_result"]=="CORRECT"),int(row["judge_result"]=="CORRECT")
        pairs[a,b]+=1
        clusters[row["sample_id"]].append(b-a)
    paired={"both_correct":pairs[1,1],"mem_only":pairs[0,1],"simple_only":pairs[1,0],"both_wrong":pairs[0,0],
            "difference_pp":100*(stats["MemNexus"]["accuracy"]-stats["SimpleMem"]["accuracy"]),
            "cluster_95ci_pp":[100*x for x in cluster_ci(clusters,seed=20261014)],
            "mcnemar_question_independence_p":mcnemar(pairs[0,1],pairs[1,0])}
    return stats,paired


def figures(folder,stats,paired,rows):
    configure_chinese_font()
    fig,axes=plt.subplots(2,3,figsize=(16,9.4))
    for ax,field,title in ((axes[0,0],"correct","统一盲评：正确题数 /100 ↑"),
                           (axes[0,2],"qa_tokens","问答输入＋输出 tokens ↓"),
                           (axes[1,0],"main_cost_rmb","建库＋问答费用估算（元）↓")):
        vals=[stats[s][field] for s in SYSTEMS]
        bars=ax.bar(SYSTEMS,vals,color=COLORS,width=.55)
        ax.bar_label(bars,labels=[f"{v:.2f}" if field=="main_cost_rmb" else f"{v:,}" for v in vals],padding=4)
        ax.set_title(title);ax.margins(y=.22)
    axes[0,0].set_ylim(0,112)
    for i,s in enumerate(SYSTEMS):
        categories=stats[s]["categories"]
        vals=[100*categories[c]["correct"]/categories[c]["n"] if categories[c]["n"] else 0 for c in LABELS]
        bars=axes[0,1].bar(np.arange(5)+(i-.5)*.36,vals,.36,color=COLORS[i],label=s)
        axes[0,1].bar_label(bars,labels=[f"{categories[c]['correct']}/{categories[c]['n']}" for c in LABELS],fontsize=8,padding=3)
    axes[0,1].set_xticks(range(5),list(LABELS.values()));axes[0,1].set_ylim(0,116)
    axes[0,1].set_title("分题型正确率 (%)");axes[0,1].legend(frameon=False,fontsize=8)
    for i,s in enumerate(SYSTEMS):
        latency=sorted(r["latency_ms"]/1000 for r in rows if r["system"]==s)
        axes[1,1].step(latency,np.arange(1,len(latency)+1),where="post",color=COLORS[i],label=s)
    axes[1,1].set_title("回答延迟累计分布：靠左更快");axes[1,1].set_xlabel("秒")
    axes[1,1].set_ylabel("累计题目比例 (%)");axes[1,1].legend(frameon=False,fontsize=8)
    fields=("both_correct","mem_only","simple_only","both_wrong")
    bars=axes[1,2].bar(range(4),[paired[k] for k in fields],color=["#7A9985",COLORS[1],COLORS[0],"#ADB7C0"])
    axes[1,2].bar_label(bars,padding=4);axes[1,2].margins(y=.2)
    axes[1,2].set_xticks(range(4),["两者都对","仅MemNexus对","仅SimpleMem对","两者都错"],rotation=12)
    axes[1,2].set_title("同题配对")
    for ax in axes.flat:ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    lo,hi=paired["cluster_95ci_pp"]
    fig.suptitle(f"LoCoMo 100题：两套系统独立建库与回答\n正确率差（MemNexus − SimpleMem）{paired['difference_pp']:+.1f}个百分点；聚类95%区间 [{lo:.1f}, {hi:.1f}]",fontsize=15)
    fig.text(.5,.018,"固定开发题组，关闭答案缓存；费用估算非账单；失败阶段额外费用见报告，不并入成功系统指标。",ha="center",fontsize=10)
    fig.subplots_adjust(top=.85,bottom=.14,hspace=.45,wspace=.32,left=.065,right=.98)
    for suffix in ("png","svg"):fig.savefig(folder/f"comparison.{suffix}",dpi=170)
    plt.close(fig)


def generate(folder):
    folder=Path(folder)
    if folder.is_file():folder=folder.parent
    report,rows,stages,checked=audit(folder)
    stats,paired=summarize(rows,stages)
    save(folder/"results.json",{"rows":rows,"stages":stages,"cohort":report["cohort"]})
    summary={"stats":stats,"paired":paired,"audit":checked,"settings":report["signature"]["settings"]}
    save(folder/"summary.json",summary)
    figures(folder,stats,paired,rows)
    lo,hi=paired["cluster_95ci_pp"]
    conclusion=("观察到MemNexus正确率领先，聚类区间高于0。仍需独立泛化验证。" if lo>0 else
                "观察到SimpleMem正确率领先，聚类区间低于0。" if hi<0 else
                "聚类置信区间跨过0，不能据此声称显著优势或两者等效。")
    labels=[("正确题数","correct",".0f"),("非对抗题LoCoMo词面F1","nonadversarial_locomo_f1",".4f"),
            ("简化BLEU-1（历史runner口径）","simple_bleu1",".4f"),("问答tokens","qa_tokens",",.0f"),
            ("含建库tokens","main_tokens",",.0f"),("建库＋问答费用（元）","main_cost_rmb",".4f"),
            ("每正确答案费用（元）","cost_per_correct",".4f"),("回答P50秒","p50_seconds",".3f"),
            ("回答P95秒","p95_seconds",".3f"),("建库总秒数","build_seconds",".1f"),("问答总秒数","qa_seconds",".1f")]
    lines=["# LoCoMo 100题对比报告","",conclusion,"",
           f"SimpleMem {stats['SimpleMem']['correct']}/100；MemNexus {stats['MemNexus']['correct']}/100。差值{paired['difference_pp']:+.1f}个百分点；对话聚类95%区间[{lo:.2f}, {hi:.2f}]。", "",
           "| 指标 | SimpleMem | MemNexus |","|---|---:|---:|"]
    html_rows=[]
    for title,field,fmt in labels:
        values=[format(stats[s][field],fmt) if stats[s][field] is not None else "N/A" for s in SYSTEMS]
        lines.append(f"| {title} | {values[0]} | {values[1]} |")
        html_rows.append(f"<tr><td>{html.escape(title)}</td><td>{values[0]}</td><td>{values[1]}</td></tr>")
    notes=[
        f"实际配置：semantic_execution={report['signature']['settings']['semantic_execution']}；模型和裁判deepseek-v4-flash，温度0；相同Qwen query/document编码、1024维。",
        f"本轮全部尝试费用上界¥{checked['all_attempts_cost_upper_rmb']:.4f}；成功阶段含裁判¥{checked['effective_completed_cost_rmb']:.4f}；其中裁判¥{checked['judge_cost_rmb']:.4f}。",
        f"失败/废弃阶段额外上界¥{sum(x['cost_upper_rmb'] for x in checked['abandoned_attempts']):.4f}；未知usage预留上界¥{checked['unknown_usage_upper_rmb']:.4f}，预留不代表实际扣款。",
        "费用统一按输入¥2.4/百万、输出¥9.6/百万估算，未含本地硬件费用。",
        f"100题为固定开发题组，原始对话范围D1–D{report['cohort']['max_session']}，不是完整LoCoMo。两套系统本轮分别从原始对话构建记忆；不是历史预测复用。",
        "题组来自已参与开发的对话，不能称独立泛化；对抗题沿用A=未提及/B=干扰选项，SimpleMem对抗题关闭reflection。",
        "答案保持原样；官方风格词面F1与语义裁判分开，简化BLEU-1不是完整官方评测复现。相同模型作答案和裁判仍可能存在偏差。",
        f"McNemar p={paired['mcnemar_question_independence_p']:.5f}基于题目独立假设，仅作补充；聚类数少，置信区间也可能不稳定。",
        f"审计通过：{checked['answers']}条非空唯一答案、{checked['judgments']}条有效评分、{checked['independent_memory_builds']}个独立记忆库，源码/数据哈希和调用账本核对。"
    ]
    for recovery in checked.get("ledger_recoveries",[]):
        notes.append("历史账本恢复（原文件未改）："+recovery["explanation"]+
                     " 此运行包含检索回退异常，图表呈现实际运行结果，不作为无异常的性能优越性证据。")
    lines += ["","## 费用、协议与解释边界","",*notes,"","![对比图](comparison.png)","",
              "## 逐题答案","","| ID/类别 | 问题 / Gold | SimpleMem | MemNexus | 裁判理由 |","|---|---|---|---|---|"]
    def cell(x):return str(x).replace("|","\\|").replace("\n","<br>")
    idx={(r["sample_id"],r["qa_index"],r["system"]):r for r in rows}
    details=[]
    for q in report["cohort"]["questions"]:
        a,b=(idx[q["sample_id"],q["qa_index"],s] for s in SYSTEMS)
        ident=f"{q['sample_id']}/{q['qa_index']} · {LABELS[q['category_id']]}"
        lines.append("| "+" | ".join(map(cell,[ident,a["question"]+" / Gold: "+str(a["gold_answer"]),
             a["prediction"]+" / "+a["judge_result"],b["prediction"]+" / "+b["judge_result"],
             "SimpleMem: "+a["judge_reason"]+"; MemNexus: "+b["judge_reason"]]))+" |")
        details.append(f"<details><summary>{html.escape(ident)} — SimpleMem {a['judge_result']} / MemNexus {b['judge_result']}</summary>"
                       f"<p>{html.escape(a['question'])}</p><p>Gold: {html.escape(str(a['gold_answer']))}</p>"+
                       "".join(f"<h4>{s}</h4><pre>{html.escape(r['prediction'])}</pre><p>{html.escape(r['judge_reason'])}</p>" for s,r in zip(SYSTEMS,(a,b),strict=True))+"</details>")
    (folder/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    document="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>100题对比报告</title>
<style>body{font:16px/1.7 'Microsoft YaHei',sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#24313c;background:#f7fafc}table{border-collapse:collapse;width:100%;background:white}td,th{border-bottom:1px solid #ddd;padding:8px 16px;text-align:left}img{width:100%;height:auto}details{background:white;padding:12px;margin:10px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}h1{font-size:28px}.notes{font-size:14px;color:#536271}</style>
<h1>LoCoMo 100题完整对比</h1>"""+f"<p>{html.escape(conclusion)}</p>"+\
        "<table><tr><th>指标</th><th>SimpleMem</th><th>MemNexus</th></tr>"+"".join(html_rows)+"</table>"+\
        '<img src="comparison.png" alt="正确率、token、费用和延迟对比">'+\
        '<div class="notes">'+"".join(f"<p>{html.escape(note)}</p>" for note in notes)+"</div>"+\
        "<h2>逐题答案与裁判理由</h2>"+"".join(details)+"</html>"
    (folder/"report.html").write_text(document,encoding="utf-8")
    report["status"]="completed"
    report["audit"]=checked
    save(folder/"run.json",report)
    print(f"报告与图表已生成：{folder/'report.html'}",flush=True)
    return summary

"""Read-only audit of the semantic execution development replay and Chinese figures.

Reconcile duplicate cumulative usage against provider traces, without modifying
the original answers, judge labels, or run files. This is not a fresh-build run.
"""

import hashlib
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_locomo_disjoint import configure_chinese_font
from analyze_optimized_comparison import cluster_ci, mcnemar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus/memory-system/src"))
from agent_memory.models import AnswerContract, Hit
from agent_memory.synthesis import BoundAnswer, check_and_execute

RUN = ROOT / "result/locomo_semantic_replay_20261013_v4.json"
PREVIOUS = ROOT / "result/locomo_synthesis_replay_20261013_v4.json"
LABELS = {1:"多跳", 2:"时间", 3:"开放域", 4:"单跳", 5:"对抗"}
FIELDS = ("provider_attempts", "provider_errors", "input_tokens", "output_tokens", "cost_rmb")
COLORS = ("#D18A35", "#2478AE")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def key(row):
    return row["system"], row["sample_id"], row["qa_index"]


def correct(row):
    return int(row["judge_result"] == "CORRECT")


def usage_cost(body):
    u = body["usage"]
    return (2.4*u["prompt_tokens"] + 9.6*u["completion_tokens"]) / 1e6


def main():
    report, previous = read(RUN), read(PREVIOUS)
    source = Path(report["source"])
    original = read(source)
    assert report["status"] == "completed"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"]
    indexed = {key(r):r for r in report["rows"]}
    old = {key(r):r for r in original["rows"]}
    prev = {key(r):r for r in previous["rows"]}
    assert len(indexed) == len(report["rows"]) == 200
    assert set(indexed) == set(old) == set(prev)
    archive = RUN.with_suffix(".source.zip")
    with zipfile.ZipFile(archive) as z:
        for name, digest in report["code_sha256"].items():
            assert hashlib.sha256(z.read(name.replace("\\", "/"))).hexdigest() == digest
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    before_resume = read(Path(report["resumed_from"]["path"]))
    changed = [p for p,h in before_resume["code_sha256"].items()
               if p.startswith("MemNexus") and report["code_sha256"][p] != h]
    assert not changed, "Application code must be identical across this continuation"
    for row in before_resume["rows"]:
        assert indexed[key(row)]["prediction"] == row["prediction"]
    for sid,digest in report["snapshot_sha256"].items():
        p = source.with_name(source.stem+"_ledgers") / f"{sid}_memnexus.memory.json"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest

    duplicates, qa, judges = [], {}, {}
    for field, dest in (("qa_usage",qa), ("judge_usage",judges)):
        for u in report[field]:
            sid = u["sample_id"]
            if sid in dest:
                assert dest[sid] == u, "Non-identical cumulative entries cannot be silently deduplicated"
                duplicates.append({"field":field,"sample_id":sid,"cost_rmb":u["cost_rmb"]})
            dest[sid] = u
    assert len(qa) == len(judges) == 10

    audit = Counter()
    returned_models = Counter()
    completed_ids = set()
    judge_by_prompt = {}
    for path in RUN.with_name(RUN.stem+"_ledgers").glob("*.json"):
        ledger = read(path)
        events = [json.loads(s) for s in path.with_suffix(".calls.jsonl").read_text(encoding="utf-8").splitlines()]
        requests = [e for e in events if e["event"] == "request"]
        responses = [e for e in events if e["event"] == "response"]
        assert len(requests) == ledger["provider_attempts"] and not ledger.get("reserved_rmb",0)
        assert len(responses) == len(requests) and not ledger.get("provider_errors",0)
        assert all(e["body"]["model"] == "deepseek-v4-flash" and e["body"]["temperature"] == 0 for e in requests)
        for name,field in (("input_tokens","prompt_tokens"),("output_tokens","completion_tokens")):
            assert sum(e["body"]["usage"][field] for e in responses) == ledger[name]
        assert math.isclose(sum(usage_cost(e["body"]) for e in responses), ledger["cost_rmb"], abs_tol=1e-8)
        sid, system = path.stem.rsplit("_",1)
        recorded = qa[sid] if system=="memnexus" else judges[sid]
        assert all(math.isclose(ledger.get(f,0), recorded.get(f,0), abs_tol=1e-8) for f in FIELDS)
        returned_models.update(e["body"]["model"] for e in responses)
        completed_ids.update(e["body"]["id"] for e in responses)
        if system == "judge":
            for req, resp in zip(requests,responses,strict=True):
                payload = json.loads(req["body"]["messages"][-1]["content"].split("\n",1)[1])
                raw = resp["body"]["choices"][0]["message"]["content"]
                judge_by_prompt.setdefault((payload["question"],str(payload["gold_answer"]),payload["prediction"]),[]).append(raw)
        for f in FIELDS:
            audit[f] += ledger.get(f,0)
        audit["ledger_files"] += 1
    assert audit["ledger_files"] == 20

    contradictions = []
    for k,row in indexed.items():
        assert row["prediction"].strip() and row["judge_result"] in {"CORRECT","WRONG"}
        assert row["question"] == old[k]["question"] and row["gold_answer"] == old[k]["gold_answer"]
        raw = row["judge_raw_response"]
        assert raw in judge_by_prompt[row["question"],str(row["gold_answer"]),row["prediction"]]
        verdict = json.loads(raw)
        assert verdict["label"] == row["judge_result"]
        if verdict["label"] == "CORRECT" and re.search(r"should be WRONG",verdict["reason"],re.I):
            contradictions.append({"key":k,"reason":verdict["reason"]})
        if row["system"] == "SimpleMem":
            assert row["prediction"] == old[k]["prediction"]
        else:
            reasoning = row["reasoning"]
            assert reasoning["answer_path"] == "semantic_execution"
            outcome = check_and_execute(AnswerContract.model_validate(reasoning["answer_contract"]),
                                        BoundAnswer.model_validate(reasoning["bound_answer"]),
                                        [Hit.model_validate(h) for h in row["retrieved_sources"]])
            assert outcome.model_dump(mode="json") == reasoning["execution"]
            assert outcome.render(choice_question="Choose exactly one" in row["evaluation_query"]) == row["prediction"]
    audit["answers"] = audit["judgments"] = 200
    audit["executions_revalidated"] = 100

    systems = {}
    for system in ("SimpleMem","MemNexus"):
        rows = [r for r in report["rows"] if r["system"]==system]
        stages = [s for s in original["stages"] if s["system"]==system]
        build = {f:sum(s["build_usage"].get(f,0) for s in stages) for f in FIELDS}
        query = {f:sum(r["model_calls"].get(f,0) for r in rows) for f in FIELDS}
        if system == "MemNexus":
            for f in FIELDS:
                assert math.isclose(query[f],sum(u.get(f,0) for u in qa.values()),abs_tol=1e-8)
        systems[system] = {"correct":sum(map(correct,rows)),"n":len(rows),"build":build,"qa":query,
                           "main_cost_rmb":build["cost_rmb"]+query["cost_rmb"],
                           "qa_tokens":query["input_tokens"]+query["output_tokens"],
                           "total_tokens":build["input_tokens"]+build["output_tokens"]+query["input_tokens"]+query["output_tokens"],
                           "p50_seconds":statistics.median(r["latency_ms"]/1000 for r in rows),
                           "p95_seconds":float(np.percentile([r["latency_ms"]/1000 for r in rows],95)),
                           "categories":{c:{"n":sum(r["category_id"]==c for r in rows),
                                            "correct":sum(r["category_id"]==c and correct(r) for r in rows)} for c in LABELS}}
    pairs = Counter()
    clusters = defaultdict(list)
    transitions = Counter()
    traces = Counter()
    question_table = []
    for row in report["rows"]:
        if row["system"]!="MemNexus":continue
        b = indexed["SimpleMem",row["sample_id"],row["qa_index"]]
        prior = prev[key(row)]
        pairs[correct(b),correct(row)] += 1
        clusters[row["sample_id"]].append(correct(row)-correct(b))
        transitions[correct(prior),correct(row)] += 1
        reason = row["reasoning"]
        traces[reason["premise_retrieval"]["stop_reason"]] += 1
        if reason["execution"]["gaps"]:traces["unresolved_final"] += 1
        if row["prediction"] in {"Insufficient evidence to determine the answer.","Not mentioned in the conversation"}:
            traces["abstention"] += 1
            if row["category_id"]!=5:traces["nonadversarial_abstention"] += 1
        question_table.append({"sample_id":row["sample_id"],"qa_index":row["qa_index"],"category_id":row["category_id"],
                               "question":row["question"],"gold":row["gold_answer"],"SimpleMem":b["prediction"],
                               "MemNexus":row["prediction"],"previous_MemNexus":prior["prediction"],
                               "simple_correct":correct(b),"mem_correct":correct(row),"previous_correct":correct(prior),
                               "judge_reason":row["judge_reason"],"gaps":reason["execution"]["gaps"],
                               "rejected":reason["execution"]["rejected"],"retrieval":reason["premise_retrieval"]})

    # De-duplicate copied response IDs across interrupted/continued semantic runs.
    unique = {}
    for folder in (ROOT/"result").glob("locomo_semantic_replay_20261013*_ledgers"):
        for path in folder.glob("*.calls.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                event=json.loads(line)
                if event["event"]=="response":
                    body=event["body"]
                    if body["id"] in unique:
                        assert unique[body["id"]]["usage"] == body["usage"]
                    unique[body["id"]]=body
    archived_extra=sum(usage_cost(body) for id_,body in unique.items() if id_ not in completed_ids)
    smoke=sum(row["usage"]["cost_rmb"] for p in (ROOT/"result").glob("semantic_smoke_20261013*.json") for row in read(p)["rows"])
    result={"run":str(RUN),"systems":systems,"audit":dict(audit),"returned_models":dict(returned_models),
            "usage_deduplication":duplicates,"qa_summary_original_rmb":sum(x["cost_rmb"] for x in report["qa_usage"]),
            "qa_reconciled_rmb":sum(x["cost_rmb"] for x in qa.values()),
            "paired":{"both_correct":pairs[1,1],"mem_only":pairs[0,1],"simple_only":pairs[1,0],"both_wrong":pairs[0,0],
                      "difference_pp":systems["MemNexus"]["correct"]-systems["SimpleMem"]["correct"],
                      "cluster_95ci_pp":[100*x for x in cluster_ci(clusters,seed=20261014)],
                      "mcnemar_p":mcnemar(pairs[0,1],pairs[1,0])},
            "previous_transition":{f"{a}->{b}":n for (a,b),n in transitions.items()},
            "path_counts":dict(traces),"judge_contradictions":contradictions,"per_question":question_table,
            "accounting":{"completed_replay_including_judge_rmb":audit["cost_rmb"],
                          "archived_additional_attempts_rmb":archived_extra,"smoke_estimate_rmb":smoke,
                          "total_new_semantic_experiments_estimate_rmb":audit["cost_rmb"]+archived_extra+smoke},
            "application_changed_across_resume":False,"retained_mem_answers":94,
            "limitations":["Fixed-memory development cohort: 79 new question keys and 21 reused, from previously studied conversations",
                           "SimpleMem predictions/timings are reused; both systems rejudged; no fresh ingestion",
                           "Dataset-specific dog/pup expansion present in evaluated runtime; cannot claim general method validation",
                           "One judge CORRECT label contradicts its own WRONG rationale; raw scores retained",
                           "Earlier tests, repeat scoring and retry costs are separate from per-system workload cost"]}
    base=ROOT/"result/LOCOMO_SEMANTIC_100_20261014"
    base.with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    plot(result,report,previous,base)
    write_report(result,base)
    print(json.dumps({k:v for k,v in result.items() if k not in {"per_question","judge_contradictions"}},ensure_ascii=False))


def plot(result,report,previous,base):
    configure_chinese_font()
    stats=result["systems"];names=list(stats)
    fig,axes=plt.subplots(2,3,figsize=(16,9.4))
    for ax,field,title in [(axes[0,0],"correct","正确题数 /100 ↑"),
                           (axes[0,2],"qa_tokens","问答 tokens ↓"),
                           (axes[1,0],"main_cost_rmb","原建库＋本轮问答费用估算（元）↓")]:
        vals=[stats[s][field] for s in names]
        bars=ax.bar(names,vals,color=COLORS,width=.55)
        labels=[f"{v:.2f}" if field=="main_cost_rmb" else f"{v:,}" for v in vals]
        ax.bar_label(bars,labels=labels,padding=4);ax.set_title(title);ax.margins(y=.20)
    axes[0,0].set_ylim(0,112)
    for j,s in enumerate(names):
        cats=stats[s]["categories"]
        vals=[100*cats[c]["correct"]/cats[c]["n"] for c in LABELS]
        bars=axes[0,1].bar(np.arange(5)+(j-.5)*.36,vals,.36,color=COLORS[j],label=s)
        axes[0,1].bar_label(bars,labels=[f"{cats[c]['correct']}/{cats[c]['n']}" for c in LABELS],fontsize=9,padding=3)
    axes[0,1].set_xticks(range(5),list(LABELS.values()));axes[0,1].set_ylim(0,116)
    axes[0,1].set_title("分题型正确率 (%)");axes[0,1].legend(frameon=False,fontsize=8)
    for s,color in zip(names,COLORS,strict=True):
        lat=sorted(r["latency_ms"]/1000 for r in report["rows"] if r["system"]==s)
        axes[1,1].step(lat,np.arange(1,101),where="post",color=color,label=s)
    axes[1,1].set_title("单题延迟累计分布：靠左更快")
    axes[1,1].set_xlabel("秒（两次运行时段不同）");axes[1,1].set_ylabel("累计题目比例 (%)")
    axes[1,1].legend(frameon=False,fontsize=8)
    keys=("both_correct","mem_only","simple_only","both_wrong")
    bars=axes[1,2].bar(range(4),[result["paired"][k] for k in keys],color=["#7A9985",COLORS[1],COLORS[0],"#ADB7C0"])
    axes[1,2].bar_label(bars,padding=4);axes[1,2].margins(y=.2)
    axes[1,2].set_xticks(range(4),["两者都对","仅MemNexus对","仅SimpleMem对","两者都错"],rotation=12)
    axes[1,2].set_title("同题配对结果")
    for ax in axes.flat:ax.grid(axis="y",alpha=.16);ax.set_axisbelow(True)
    lo,hi=result["paired"]["cluster_95ci_pp"]
    fig.suptitle(f"统一语义执行版本：LoCoMo 100题开发对比\nMemNexus − SimpleMem：{result['paired']['difference_pp']:+.0f}个百分点；聚类95%区间 [{lo:.1f}, {hi:.1f}]",fontsize=16)
    fig.text(.5,.02,"两套记忆分别来自原始建库快照；SimpleMem答案复用、双方重新评分。全部100题完成，费用不含历史开发尝试。",ha="center",fontsize=10)
    fig.subplots_adjust(top=.85,bottom=.14,hspace=.42,wspace=.30,left=.065,right=.98)
    for ext in (".png",".svg"):fig.savefig(base.with_suffix(ext),dpi=170)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    xs=np.arange(5)
    for offset,system,color in [(-.23,"previous","#86A5B5"),(0,"MemNexus",COLORS[1]),(.23,"SimpleMem",COLORS[0])]:
        if system=="previous":
            vals=[sum(r["system"]=="MemNexus" and r["category_id"]==c and correct(r) for r in previous["rows"]) for c in LABELS]
            label="上一版 MemNexus"
        else:
            vals=[stats[system]["categories"][c]["correct"] for c in LABELS];label=system
        bars=axes[0].bar(xs+offset,vals,.23,color=color,label=label)
        axes[0].bar_label(bars,fontsize=8,padding=3)
    axes[0].set_xticks(xs,list(LABELS.values()));axes[0].set_title("各类别正确题数：历史版本对照")
    axes[0].set_ylim(0,32);axes[0].legend(fontsize=8,frameon=False)
    transition=result["previous_transition"]
    bars=axes[1].bar(range(4),[transition.get(k,0) for k in ("1->1","0->1","1->0","0->0")],
                     color=["#7A9985","#2478AE","#C65A53","#ADB7C0"])
    axes[1].bar_label(bars,padding=3);axes[1].margins(y=.2)
    axes[1].set_xticks(range(4),["保持正确","旧错→新对","旧对→新错","仍错误"],rotation=15)
    axes[1].set_title("相对上一版78/100：改进与退步")
    groups={False:[],True:[]}
    for r in result["per_question"]:
        groups[r["retrieval"]["stop_reason"]=="one_round_completed"].append(r)
    bars=axes[2].bar(range(2),[100*sum(r["mem_correct"] for r in groups[g])/len(groups[g]) for g in (False,True)],color=["#86A5B5","#2478AE"])
    axes[2].bar_label(bars,labels=[f"{sum(r['mem_correct'] for r in groups[g])}/{len(groups[g])}" for g in (False,True)],padding=4)
    axes[2].set_xticks(range(2),["无需补检索","执行一次补检索"]);axes[2].set_ylim(0,112)
    axes[2].set_title("补检索路径的观察性结果 (%)")
    for ax in axes:ax.grid(axis="y",alpha=.16);ax.set_axisbelow(True)
    fig.suptitle("新版本没有通过准确率不退步的验收",fontsize=16)
    fig.text(.5,.02,"历史对照包含重新生成和裁判波动；路径组难度不同，组间差异不代表补检索的因果效果。",ha="center",fontsize=10)
    fig.subplots_adjust(top=.82,bottom=.21,wspace=.25,left=.055,right=.98)
    for ext in (".png",".svg"):fig.savefig(base.with_name(base.name+"_detail").with_suffix(ext),dpi=170)
    plt.close(fig)


def write_report(r,base):
    s,m=r["systems"]["SimpleMem"],r["systems"]["MemNexus"]
    lines=["# 完整100题：统一语义执行版本与SimpleMem", "",
           "分析日期：2026-10-14。原始文件为locomo_semantic_replay_20261013_v4.json。", "",
           f"原始统一裁判结果：SimpleMem {s['correct']}/100，MemNexus {m['correct']}/100。相对上一版MemNexus 78/100退步；本版本未满足准确率不降低、更优于SimpleMem的要求。", "",
           "## 主要指标", "", "| 指标 | SimpleMem | MemNexus |", "|---|---:|---:|"]
    for title,k,fmt in [("正确题数","correct",",.0f"),("问答tokens","qa_tokens",",.0f"),("含建库tokens","total_tokens",",.0f"),
                        ("建库＋问答费用估算（元）","main_cost_rmb",".4f"),("P50秒","p50_seconds",".3f"),("P95秒","p95_seconds",".3f")]:
        lines.append(f"| {title} | {s[k]:{fmt}} | {m[k]:{fmt}} |")
    lines += ["", "费用采用输入¥2.4/百万、输出¥9.6/百万的保守统一价格，不是账单；建库成本来自原始独立建库；本地embedding和硬件成本未计入。延迟来自不同时段，不能当严格并行性能试验。", "",
              "| 题型 | 题数 | SimpleMem对 | MemNexus对 |", "|---|---:|---:|---:|"]
    for c in LABELS:
        lines.append(f"| {LABELS[c]} | {s['categories'][c]['n']} | {s['categories'][c]['correct']} | {m['categories'][c]['correct']} |")
    lines += ["",f"配对统计：{r['paired']}。置信区间采用10个对话簇bootstrap，样本簇少；题目独立假设下McNemar仅作补充。", "",
              "## 审计与费用纠正", "",
              f"200条唯一且非空回答、200条有效标签、20份新账本、100份执行记录全部复验。原始SimpleMem预测未修改；两套系统各自独立的记忆快照哈希核对通过。provider返回模型标识为{r['returned_models']}。", "",
              f"续跑把conv-49累计QA账本摘要添加两次（多计¥{r['usage_deduplication'][0]['cost_rmb']:.7f}），但没有重复生成该段回答。本分析先验证重复条目逐字段相同，再与原始provider请求/response逐笔对账，实际QA费用为¥{r['qa_reconciled_rmb']:.7f}。原始JSON保留不改。", "",
              "续跑前后应用源码哈希一致，保留前94题的预测；先前的裁判JSON格式错误及runner重复行故障只影响评分/记录层。resumed_from.policy沿用的旧年份推导说明不适用于本轮，本次据实际源码哈希判断。", "",
              f"有效本轮新调用（主模型＋裁判）：¥{r['accounting']['completed_replay_including_judge_rmb']:.4f}。此前语义版本中止及多次续跑的额外调用按response ID去重后：¥{r['accounting']['archived_additional_attempts_rmb']:.4f}。八轮开发smoke保存用量估算：¥{r['accounting']['smoke_estimate_rmb']:.4f}。本阶段累计新增实验估算：¥{r['accounting']['total_new_semantic_experiments_estimate_rmb']:.4f}，不含更早建库和历史实验。smoke仅有保存usage估算，审计强度低于逐次调用账本。", "",
              "## 退步与机制观察", "",
              f"相对上一版78题正确的同题输出，标签变化为{r['previous_transition']}。路径统计：{r['path_counts']}。这包含随机生成/裁判波动，尚不是消融因果结果。", "",
              "本轮的典型退步包括：武术漏kickboxing、狗名漏Ned、购买时间拒答、母亲礼物多主体题拒答、店铺评价和信心提升遗漏。列表/count结构不能补救遗漏候选；前提/主体约束仍可能使可回答问题被拒绝。仅记录gap或完成一次补检索不能证明缺口解决。", "",
              "补检索中的固定字符串 `names dog puppy adopted owner` 出现在被测runtime中，对所有问题追加，是开发时加入的领域特定规则。该实现违反普适性目标，因此本轮只能作为该实现的诊断结果，不能证明所提通用算法有效。下一版必须移除并以契约/缺口生成查询，再冻结复测。此处没有修改被测代码后宣称同一结果。", "",
              "## 裁判异常与解释边界", "",
              "conv-49/26的JSON label为CORRECT，但reason明确写应为WRONG。主表保留原自动评分70题，不人工改标签；若单独按这条理由改判，MemNexus为69/100。这是敏感性说明，不是新的正式评分。conv-49/27将smart watch与fitness tracker判为不等价，也需统一人工复核。", "",
              "题组为79道此前两组未测题、21道复用题，来自已经用于开发的10段对话；范围为D1–D5共1104原始turn。SimpleMem答案与时间复用，本轮只重新生成MemNexus并重新评分双方。不能称为完整LoCoMo、独立泛化或两套系统都从头运行。", "",
              "目前默认MEMORY_SEMANTIC_EXECUTION仍为false。保留原78题版本作为参考，当前新版本不应因费用较低而提升为质量改进版本。下一步应先修复通用查询构造、语义约束过严和增量绑定，再做等预算消融与未参与开发的数据验证。", "",
              "## 逐题结果", "", "| ID/类别 | 问题 / Gold | SimpleMem | 新MemNexus | 上一版MemNexus | 本轮判定与理由 |", "|---|---|---|---|---|---|"]
    def cell(x):return str(x).replace("|","\\|").replace("\n","<br>")
    for q in r["per_question"]:
        lines.append("| "+" | ".join(map(cell,[f"{q['sample_id']}/{q['qa_index']} / {LABELS[q['category_id']]}",
            q["question"]+" / Gold: "+str(q["gold"]),q["SimpleMem"],q["MemNexus"],q["previous_MemNexus"],
            f"SimpleMem={q['simple_correct']}, MemNexus={q['mem_correct']}: {q['judge_reason']}"]))+" |")
    base.with_suffix(".md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()

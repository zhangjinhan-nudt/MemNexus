"""Post-run analysis only: read preserved results, never call providers or change scores."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_locomo_disjoint import configure_chinese_font
from benchmark_mixed200 import aggregate, read, save

ROOT = Path(__file__).resolve().parents[1]


def audit_quote_spans(folder):
    receipts=[]
    for stage_path in (folder/'attempts').glob('*memnexus*/stage_result.json'):
        snapshot=read(stage_path.parent/'usage.memory.json')
        sources={(t['turn_id'],i):e['content'] for t in snapshot['turns'] for i,e in enumerate(t['events'])}
        for line in (stage_path.parent/'usage.calls.jsonl').read_text(encoding='utf-8').splitlines():
            event=json.loads(line)
            if event['event']!='evidence_alignment':
                continue
            for correction in event['corrections']:
                text=sources.get((correction['turn_id'],correction['event_index']))
                literal=correction['resolved_quote']
                valid=text is not None and text[correction['start']:correction['end']]==literal
                unique=valid and text.find(literal)==text.rfind(literal)
                receipts.append({'stage':stage_path.parent.name,**correction,
                                 'exact_source_span':valid,'unique_including_overlaps':unique})
    result={'corrections':receipts,'n':len(receipts),
            'all_exact_unique':all(r['exact_source_span'] and r['unique_including_overlaps'] for r in receipts),
            'note':'Local alignment attempts, not new model calls or proof of semantic correctness.'}
    save(folder/'quote_alignment_audit.json',result)
    return result


def phase_usage(folder):
    result = defaultdict(Counter)
    manifest=read(folder/'manifest.json')
    files=[]
    if manifest.get('source_runs'):
        for dataset,root in manifest['source_runs'].items():
            files.extend(p for p in sorted((Path(root)/'attempts').glob('*/stage_result.json'))
                         if read(p)['stage']['dataset']==dataset)
    else:
        files=sorted((folder/'attempts').glob('*/stage_result.json'))
    seen=set()
    for file in files:
        data = read(file)
        system = data['stage']['system']
        dataset = data['stage']['dataset']
        identity=(dataset,data['stage']['unit_id'],system)
        if identity in seen:
            continue
        seen.add(identity)
        requests = {}
        for line in (file.parent/'usage.calls.jsonl').read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            if event['event'] == 'request':
                requests[event['attempt']] = event['body']
            elif event['event'] == 'response':
                prompt = '\n'.join(str(m.get('content','')) for m in requests.get(event['attempt'],{}).get('messages',[]))
                if system != 'MemNexus':
                    continue
                phase = ('extract' if prompt.startswith('你负责短期记忆的增量生成') else
                         'plan' if prompt.startswith(('Plan retrieval for this factual question','你负责 SimpleMem')) else
                         'bind' if prompt.startswith('Fill each mandatory output') else
                         'synthesize' if prompt.startswith('Answer the original question using the numbered evidence') else 'other')
                if '"previous_bindings":' in prompt:
                    phase += '_incremental'
                if '\n上次输出没有通过 JSON 格式、Schema 或证据约束校验。' in prompt:
                    phase += '_repair'
                usage = event['body'].get('usage',{})
                stats = result[dataset+'/'+phase]
                stats['calls'] += 1
                stats['input_tokens'] += usage.get('prompt_tokens',0)
                stats['output_tokens'] += usage.get('completion_tokens',0)
                stats['prompt_chars'] += len(prompt)
    return {k:dict(v) for k,v in result.items()}


def main(folder):
    if read(folder/'run.json')['status'] != 'completed':
        raise ValueError('Refuse to report a partial run as completed')
    data = read(folder/'results.json')
    rows, stages = data['rows'], data['stages']
    stats = aggregate(rows,stages)
    old = read(ROOT/'result/benchmark_locomo100_timefix_v2_20261016/results.json')['rows']
    ids = {r['item_id'] for r in rows if r['dataset']=='LoCoMo'}
    matched = [r for r in old if r['item_id'] in ids]
    old_stats = aggregate(matched)['LoCoMo']
    phases = phase_usage(folder)
    quote_audit = audit_quote_spans(folder)
    pairs = {(r['item_id'],r['system']):r for r in rows}
    errors=[]
    for row in rows:
        if row['system']!='MemNexus':
            continue
        reasoning = row.get('reasoning',{})
        errors.append({'item_id':row['item_id'], 'dataset':row['dataset'],'category':row['category'],
                       'question':row['question'],'gold':row['gold_answer'], 'prediction':row['prediction'],
                       'judge':row['judge_result'],'judge_reason':row.get('judge_reason'),
                       'simplemem_prediction':pairs[row['item_id'],'SimpleMem']['prediction'],
                       'simplemem_judge':pairs[row['item_id'],'SimpleMem']['judge_result'],
                       'path':reasoning.get('answer_path'), 'semantic_error':reasoning.get('semantic_error'),
                       'execution':reasoning.get('execution'), 'answer_table':reasoning.get('answer_table'),
                       'followup':reasoning.get('premise_retrieval'),
                       'request_budget':reasoning.get('provider_request_budget')})
    save(folder/'analysis_details.json', {'phases':phases,'quote_alignment_audit':quote_audit,'all_memnexus_cases':errors,
                                        'old_same_locomo_subset':old_stats})
    lines=['# 紧凑阅读层：实际实验与逐阶段分析','', '## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: validate',
           '- Origin Date: 2026-10-17','- Verification Status: ANALYZED','- Version Label: compact-reader-v2','',
           '本报告保留原始分数；accuracy来自统一盲化自定义裁判，非LongMemEval官方rubric。两数据集按子类均衡抽取，不能当官方总体分数。',
           'LoCoMo是D1–D5开发题组的固定子集；LongMemEval用每题完整haystack。详见manifest及audit。','',
           '本轮全部MemNexus重新建库。部分已完成的同配置SimpleMem测量按原签名和账本哈希复用，未完成的基线新执行；两边统一重新评分。复用来源见manifest.baseline_reuse。','',
           '|数据集|指标|SimpleMem|MemNexus|Mem/Simple变化|','|---|---|---:|---:|---:|']
    metrics=[('正确率',lambda s:s['accuracy']),('Token F1',lambda s:s['mean_f1']),
             ('问答输入token',lambda s:s['input_tokens']),('问答输出token',lambda s:s['output_tokens']),
             ('问答费用¥',lambda s:s['cost_rmb']),('建库输入token',lambda s:s['build_input_tokens']),
             ('建库输出token',lambda s:s['build_output_tokens']),('建库费用¥',lambda s:s['build_cost_rmb']),
             ('全流程输入token',lambda s:s['build_input_tokens']+s['input_tokens']),
             ('全流程输出token',lambda s:s['build_output_tokens']+s['output_tokens']),
             ('建库+问答费用¥',lambda s:s['build_cost_rmb']+s['cost_rmb']),
             ('问答P50秒',lambda s:s['p50_latency_ms']/1000),('问答P95秒',lambda s:s['p95_latency_ms']/1000)]
    for dataset in ('LoCoMo','LongMemEval-S'):
        for name, fn in metrics:
            a,b = [fn(stats[dataset][s]) for s in ('SimpleMem','MemNexus')]
            delta = f'{b/a-1:+.1%}' if a else 'N/A'
            lines.append(f'|{dataset}|{name}|{a:,.4f}|{b:,.4f}|{delta}|')
    lines += ['','## 同一LoCoMo子集的历史参考','',
              '旧轮与新轮均重新建库/生成/评分，不是只切换一个变量的消融。历史题组重复使用不可声称独立泛化。',
              '|版本|系统|正确数|Token F1|QA输入|QA输出|QA费用|','|---|---|---:|---:|---:|---:|---:|']
    for version, group in [('旧轮相同20题',old_stats),('新冻结轮',stats['LoCoMo'])]:
        for system,s in group.items():
            lines.append(f"|{version}|{system}|{s['correct']}/{s['n']}|{s['mean_f1']:.4f}|{s['input_tokens']}|{s['output_tokens']}|{s['cost_rmb']:.4f}|")
    lines += ['','## MemNexus请求级分解','',
              '按保存的真实请求内容识别阶段，包含修复；长度重试仍计真实次数。细项与usage账本分别保存。',
              '|数据集/阶段|已响应请求数|输入token|输出token|','|---|---:|---:|---:|']
    for name,p in sorted(phases.items()):
        lines.append(f"|{name}|{p['calls']}|{p['input_tokens']}|{p['output_tokens']}|")
    lines += ['',f"记录原文字面片段重新锚定 {quote_audit['n']} 次；全部为同事件唯一精确片段（包括重叠检查）：{quote_audit['all_exact_unique']}。",
              '这些是本地定位操作，不是额外LLM请求，也不自动证明事实语义正确；仍受原有主体、作用域和证据校验约束。']
    lines += ['','## 新轮错误与执行症状','',
              '以下为自动汇集的真实输出与轨迹，不把症状自动当成根因。来源是否充分需要回看原始文本，不能只按gold词重叠判定。']
    for e in errors:
        if e['judge']=='CORRECT':
            continue
        lines += ['',f"### {e['item_id']} / {e['category']}",'',e['question'],f"Gold：{e['gold']}",
                  f"MemNexus：{e['prediction']}；裁判：{e['judge_reason']}",
                  f"SimpleMem（{e['simplemem_judge']}）：{e['simplemem_prediction']}",
                  f"路径：{e['path']}；错误：{e['semantic_error']}",
                  '执行：'+json.dumps(e['execution'],ensure_ascii=False),
                  '补取/重读：'+json.dumps(e['followup'],ensure_ascii=False)]
    lines += ['','## 统计误用检查（11/11）','',
              '1. Simpson：分数据集、类别报告，合计不代替各层结果。',
              '2. 生态谬误：群体均值不推出每题或每个用户均改善。',
              '3. 选择偏差：仅ID哈希抽样，但LoCoMo来自反复使用的开发题组，且类别均衡非总体分布。',
              '4. Collider：不按是否答对或是否补检索筛选主分析；错误子组仅作诊断。',
              '5. 基率：公开每类题数，不把均衡抽样准确率当自然分布准确率。',
              '6. 均值回归：不凭旧错误题重新变对证明算法；报告全部冻结题和基线。',
              '7. 幸存者：要求52条答案/评分和28个独立记忆库的完整测量；失败尝试单列。未将截断或证据校验失败阶段伪装成错误答案或完整结果。',
              '8. 多重比较：多指标为探索性描述，不筛选显著项或声称显著全面胜出。',
              '9. 分析路径：冻结代码/配置/题目哈希；日期修正前的中止试跑另记，未形成配对分数。',
              '10. 因果：组合改动和随机API输出不能证明某个机制单独造成提升；固定证据呈现重放只支持格式效应。',
              '11. 反向因果：困难题会诱发额外调用，调用多与答错相关不等于多调用致错。','',
              '小样本、单次运行、同族主模型/裁判、LoCoMo开发集重复使用均限制结论。LME每类仅1题，0%/100%不构成可靠子类能力估计。',
              '时延来自关闭推理缓存的串行配置，不代表各系统的最大并发吞吐率。',
              '费用是实际usage按固定输入/输出单价计算，不含本地算力、供应商缓存优惠或未返回usage的请求费用。',
              '旧版中止与失败尝试保留在原目录。v4对全部MemNexus重新测试，仅复用代码、配置、原始历史均校验一致的SimpleMem完整测量；不选择最佳答案。',
              '有效实验成本包括复用基线的原测量成本；本轮新增API费用与之前已付款费用分开，见audit.json。LongMemEval恢复上限32768、LoCoMo8192。','',
              '![质量与问答成本](compact_details.png)']
    (folder/'detailed_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    configure_chinese_font()
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    selected=[metrics[i] for i in (0,1,2,3,4,7,10,11)]
    x=np.arange(2)
    for ax,(name,fn) in zip(axes.flat,selected,strict=True):
        for index,system in enumerate(('SimpleMem','MemNexus')):
            values=[fn(stats[d][system]) for d in ('LoCoMo','LongMemEval-S')]
            bars=ax.bar(x+(index-.5)*.36,values,.36,label=system,color=('#2878ad','#d68535')[index])
            ax.bar_label(bars,labels=[f'{v:,.3f}' if abs(v)<10 else f'{v:,.1f}' for v in values],fontsize=8,padding=3)
        ax.set_title(name);ax.set_xticks(x,['LoCoMo n=20','LongMemEval n=6'])
        ax.set_ylim(0,max([fn(stats[d][s]) for d in ('LoCoMo','LongMemEval-S') for s in ('SimpleMem','MemNexus')])*1.25 or 1)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('冻结小样本：真实质量与分阶段成本（开发验证，不作显著性宣称）',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ('png','svg'):
        fig.savefig(folder/f'compact_details.{suffix}',dpi=170)
    plt.close(fig)
    print(json.dumps({'stats':stats,'phases':phases},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder.resolve())

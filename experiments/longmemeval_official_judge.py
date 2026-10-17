"""Supplementary official-template evaluation with the user's DeepSeek judge.

Keeps the frozen unified judge scores unchanged. Downloads the pinned evaluator
as DATA, extracts its literal prompt templates with AST, and never executes it.
"""
import argparse
import ast
import base64
import hashlib
import json
import os
import random
from pathlib import Path

import httpx

from benchmark_mixed200 import LME, SYSTEMS, read, save, digest, retained_spend

SOURCE_URL='https://raw.githubusercontent.com/xiaowu0162/LongMemEval/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py'
SOURCE_SHA='ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251'


def templates(folder):
    path=folder/'source.json'
    if path.exists():
        source=read(path)['source_text']
    else:
        try:
            response=httpx.get(SOURCE_URL,timeout=30,follow_redirects=True)
            response.raise_for_status()
            source=response.text
        except httpx.TransportError:
            # Same pinned object through GitHub's content API; hash check below
            # forbids changing to a different evaluator on a network failure.
            url='https://api.github.com/repos/xiaowu0162/LongMemEval/contents/src/evaluation/evaluate_qa.py?ref=9e0b455f4ef0e2ab8f2e582289761153549043fc'
            response=httpx.get(url,timeout=30,follow_redirects=True)
            response.raise_for_status()
            source=base64.b64decode(response.json()['content']).decode('utf-8')
    if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Official evaluator source hash mismatch')
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='get_anscheck_prompt')
    values=[]
    def visit(node):
        if (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in node.targets)
                and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)):
            values.append(node.value.value)
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(function)
    assert len(values)==5 and 'off-by-one' in values[1] and 'personalized' in values[3] and 'unanswerable' in values[4]
    save(path,{'source_url':SOURCE_URL,'sha256':SOURCE_SHA,'source_text':source,
               'method':'Literal AST extraction only; no downloaded code executed'})
    return values


def main(args):
    root=args.run.resolve()
    folder=root/'official_lme_supplement'
    folder.mkdir(parents=True,exist_ok=True)
    prompts=templates(folder)
    manifest=read(root/'manifest.json')
    ids=[q for u in manifest['units'] if u['dataset']=='LongMemEval-S' for q in u['item_ids']]
    protocol={'kind':'supplementary_official_prompt_with_replaced_judge_model',
              'source_sha256':SOURCE_SHA,'source_url':SOURCE_URL,'items':ids,
              'judge_model':'deepseek-v4-flash','temperature':0,'max_tokens':10,
              'label_rule':'yes in response.lower(), as in the official evaluator',
              'main_scores':'Preserved unchanged; this supplement is reported even if less favorable',
              'difference_from_upstream':'User-requested DeepSeek judge instead of upstream model_zoo; no official leaderboard claim'}
    if (folder/'protocol.json').exists() and read(folder/'protocol.json')!=protocol:
        raise ValueError('Supplementary judge protocol changed')
    save(folder/'protocol.json',protocol)
    if args.prepare_only:
        print(f'Prepared official-template supplement for {len(ids)} questions; no model calls')
        return
    if read(root/'run.json')['status']!='completed':
        raise ValueError('Wait for the complete frozen benchmark; no partial supplementary score')
    rows=[r for r in read(root/'results.json')['rows'] if r['dataset']=='LongMemEval-S']
    if {(r['item_id'],r['system']) for r in rows}!={(i,s) for i in ids for s in SYSTEMS}:
        raise ValueError('Supplement is not the whole paired LME cohort')
    refs={r['question_id']:r for r in read(LME)}
    from benchmark100 import configure, new_attempt
    configure({'variant':'semantic'})
    from simplemem.core.utils.llm_client import LLMClient
    import config
    os.environ['EXPERIMENT_TRUNCATION_MAX_TOKENS']='0'
    os.environ['EXPERIMENT_COMPONENT_BUDGET_RMB']='10000'
    random.Random(20261017).shuffle(rows)
    records=[]
    for row in rows:
        qid=row['item_id'].removeprefix('longmemeval-')
        ref=refs[qid]
        if row['question']!=ref['question'] or str(row['gold_answer'])!=str(ref['answer']):
            raise ValueError('Official reference alignment mismatch')
        kind=ref['question_type']
        index=4 if '_abs' in qid else {'temporal-reasoning':1,'knowledge-update':2,'single-session-preference':3}.get(kind,0)
        prompt=prompts[index].format(ref['question'],ref['answer'],row['prediction'])
        signature=hashlib.sha256(prompt.encode()).hexdigest()
        key='judge_'+qid+'_'+row['system'].lower()
        existing=sorted((folder/'attempts').glob(key+'__*/judgment.json'))
        if existing:
            result_path=existing[0]
            record=read(result_path)
            if record['prompt_sha256']!=signature:
                raise ValueError('Prediction/prompt changed during supplementary resume')
        else:
            attempt=new_attempt(folder,key)
            result_path=attempt/'judgment.json'
            os.environ['EXPERIMENT_USAGE_LEDGER_PATH']=str(attempt/'usage.json')
            judge=LLMClient(api_key=config.JUDGE_API_KEY or config.OPENAI_API_KEY,model=config.JUDGE_MODEL,
                            base_url=config.JUDGE_BASE_URL,enable_thinking=False,use_streaming=False)
            try:
                response=judge.chat_completion([{'role':'user','content':prompt}],temperature=0,max_retries=1,_output_limit=10)
                normalized=response.strip().lower()
                record={'item_id':row['item_id'],'question_type':kind,'system':row['system'],
                        'prediction':row['prediction'],'prompt_sha256':signature,'raw_response':response,
                        'correct':'yes' in normalized,'strict_yes_no':normalized.rstrip('.!') in {'yes','no'},
                        'unified_judge_result':row['judge_result'],'usage':dict(judge.usage)}
                save(result_path,record)
            finally:
                judge.client.close()
        records.append(record)
    from benchmark100_report import ledger_audit
    for path in (folder/'attempts').glob('*/usage.json'):
        ledger_audit(path)
    stats={s:{'n':sum(r['system']==s for r in records),
              'correct':sum(r['system']==s and r['correct'] for r in records),
              'changed_from_unified':sum(r['system']==s and r['correct']!=(r['unified_judge_result']=='CORRECT') for r in records)}
           for s in SYSTEMS}
    out={'protocol':protocol,'original_results_sha256':digest(root/'results.json'),'stats':stats,'rows':records,
         'cost_rmb':sum(r['usage']['cost_rmb'] for r in records),
         'all_attempt_cost_rmb':retained_spend(folder)}
    save(folder/'results.json',out)
    lines=['# LongMemEval官方提示模板补充评分','',
           '使用官方分任务提示模板，裁判按用户要求仍为deepseek-v4-flash；不等同原仓库GPT-4o的官方排行榜分数。',
           '原统一裁判结果保持不变，两个口径均报告。官方时间题容许off-by-one，偏好题不要求覆盖rubric所有点，知识更新题容许同时包含旧信息。',
           '因此分数变化不代表系统输出改变，也不能掩盖无关建议或其它人工核查发现的问题。','',
           '|系统|题数|官方模板判对|相对统一裁判标签变化|','|---|---:|---:|---:|']
    for s,v in stats.items():
        lines.append(f"|{s}|{v['n']}|{v['correct']}|{v['changed_from_unified']}|")
    lines += ['',f'来源：{SOURCE_URL}',f'源文件SHA256：{SOURCE_SHA}',
              f"本次补充评分实际usage估算费用：¥{out['cost_rmb']:.6f}",
              f"非严格yes/no输出：{sum(not r['strict_yes_no'] for r in records)}条（原文全部保留）。",'']
    for r in records:
        lines.append(f"- {r['item_id']} / {r['system']}: unified={r['unified_judge_result']}; official-template={r['correct']}; raw={r['raw_response']!r}")
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    main(parser.parse_args())

"""Export completed dataset strata with explicit frozen-run provenance.

No provider calls, no changed predictions, no best-of-retry selection. A parent
run may have failed AFTER one dataset completed; that completed stratum can be
reported independently after per-stage and per-judge ledger validation.
"""
import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

from benchmark100_report import ledger_audit
from benchmark_mixed200 import SYSTEMS, aggregate, paired, read, retained_spend, save, token_f1, write_report


def load_dataset(folder, dataset):
    manifest, state = read(folder/'manifest.json'), read(folder/'run.json')
    data = read(folder/('results.json' if (folder/'results.json').exists() else 'progress.json'))
    rows = [r for r in data['rows'] if r['dataset']==dataset]
    stages = [s for s in data['stages'] if s['dataset']==dataset]
    units = [u for u in manifest['units'] if u['dataset']==dataset]
    expected = {(item,s) for u in units for item in u['item_ids'] for s in SYSTEMS}
    observed = [(r['item_id'],r['system']) for r in rows]
    if not expected or len(observed)!=len(set(observed)) or set(observed)!=expected:
        raise ValueError(f'{dataset}: incomplete paired rows')
    if len(stages)!=len(units)*2:
        raise ValueError(f'{dataset}: incomplete builds')
    with zipfile.ZipFile(folder/'source.zip') as archive:
        for name,sha in manifest['source_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=sha:
                raise ValueError('Frozen source archive does not match manifest')
    cost=0.0
    fields=('input_tokens','output_tokens','cost_rmb','provider_attempts')
    units_by_id={u['unit_id']:u for u in units}
    for stage in stages:
        key=stage['unit_id']+'_'+stage['system'].lower()
        path=sorted((folder/'attempts').glob(f'{key}__*/stage_result.json'))[0]
        saved=read(path)
        unit=units_by_id[stage['unit_id']]
        if saved['signature']!=state['signature'] or saved['raw_sha256']!=unit['raw_sha256']:
            raise ValueError('Stage signature/input mismatch')
        if saved['stage']!=stage or stage['raw_turns']!=unit['raw_messages'] or stage.get('failed_windows',0):
            raise ValueError('Stage ingestion mismatch/failure')
        snapshot=read(path.parent/'usage.memory.json')
        if len(snapshot['memories'])!=stage['memory_count'] or stage['memory_count']<=0:
            raise ValueError('Memory snapshot mismatch/empty')
        ledger,_=ledger_audit(path.parent/'usage.json')
        selected=[r for r in rows if r['unit_id']==stage['unit_id'] and r['system']==stage['system']]
        original={r['item_id']:r for r in saved['rows']}
        for row in selected:
            for field in ('prediction','question','gold_answer','model_calls'):
                if row[field]!=original[row['item_id']][field]:
                    raise ValueError('Result differs from committed stage')
        for field in fields:
            total=stage['build_usage'].get(field,0)+sum(r['model_calls'].get(field,0) for r in selected)
            if not math.isclose(total,ledger.get(field,0),abs_tol=1e-7):
                raise ValueError(f'Usage mismatch: {field}')
        cost+=ledger['cost_rmb']
    for row in rows:
        if not row['prediction'].strip() or row['judge_result'] not in {'CORRECT','WRONG'}:
            raise ValueError('Empty/unjudged prediction')
        key='judge_'+row['item_id']+'_'+row['system'].lower()
        path=sorted((folder/'attempts').glob(f'{key}__*/judgment.json'))[0]
        judgment=read(path)
        if judgment['prediction_sha256']!=hashlib.sha256(row['prediction'].encode()).hexdigest():
            raise ValueError('Judge prediction hash mismatch')
        if judgment['judge_result']!=row['judge_result'] or judgment['judge_reason']!=row['judge_reason']:
            raise ValueError('Judge label/reason changed')
        if not math.isclose(row['f1'],token_f1(row['prediction'],row['gold_answer']),abs_tol=1e-12):
            raise ValueError('F1 changed')
        ledger,_=ledger_audit(path.parent/'usage.json')
        cost+=ledger['cost_rmb']
    audit={'source_run':str(folder),'source_run_status':state['status'],
           'paired_answers':len(rows),'independent_builds':len(stages),
           'included_cost_with_judge_rmb':cost,
           'excluded_other_or_failed_cost_in_source_run_rmb':retained_spend(folder)-cost}
    return rows,stages,manifest,audit


def main(args):
    runs={'LoCoMo':args.locomo_run.resolve()}
    if args.lme_run:
        runs['LongMemEval-S']=args.lme_run.resolve()
    rows,stages,manifests,audits=[],[],{},{}
    for dataset,folder in runs.items():
        rr,ss,mm,aa=load_dataset(folder,dataset)
        rows.extend(rr);stages.extend(ss);manifests[dataset]=mm;audits[dataset]=aa
    if len(manifests)==2:
        a,b=[manifests[d]['source_sha256'] for d in runs]
        changed={name for name in set(a)|set(b) if a.get(name)!=b.get(name)}
        if changed-{'experiments/benchmark_mixed200.py'}:
            raise ValueError(f'Production code differs across datasets: {changed}')
        old=manifests['LoCoMo'];new=manifests['LongMemEval-S']
        for dataset in runs:
            if [u for u in old['units'] if u['dataset']==dataset]!=[u for u in new['units'] if u['dataset']==dataset] and dataset=='LongMemEval-S':
                raise ValueError('LongMemEval IDs/history changed')
        if old['settings']!=new['settings']:
            raise ValueError('System settings differ')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest={'kind':'completed_dataset_exports','source_runs':{k:str(v) for k,v in runs.items()},
              'question_count':len(rows)//2,'dataset_protocols':manifests,
              'note':'LoCoMo v2; LongMemEval v3 recovery cap 32768 (initial 8192). Same production code. Separate dataset results, not one unchanged global protocol.'}
    save(out/'manifest.json',manifest)
    save(out/'audit.json',audits)
    write_report(out,manifest,rows,stages,aggregate(rows,stages),paired(rows))
    # The exported dataset completion does not rewrite the interrupted parent.
    save(out/'run.json',{'status':'completed','kind':'audited_dataset_export','source_runs':manifest['source_runs']})
    lines=['# 数据集级结果来源审计','',
           '本目录是已完成数据集的只读导出，不是一场重新执行的单一协议实验。原目录、预测、gold和裁判均未修改。',
           'LoCoMo来自v2冻结运行；若含LongMemEval，则来自两系统恢复上限统一提高至32768后的v3运行。',
           'v2在第二个LongMemEval抽取被16384上限截断后停止；其旧LongMemEval结果和失败阶段不进入新版主统计。',
           '每个已纳入阶段验证源码压缩包哈希、消息数、记忆快照、真实调用账本、预测及裁判哈希；不选最佳重试。','',
           json.dumps(audits,ensure_ascii=False,indent=2)]
    (out/'PROVENANCE.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(audits,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--locomo-run',type=Path,required=True)
    parser.add_argument('--lme-run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())

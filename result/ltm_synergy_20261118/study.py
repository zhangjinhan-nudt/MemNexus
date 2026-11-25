"""Budget admission and isolated processes around the existing benchmark functions.

No new memory/answer implementation. All artifacts are immutable per component;
failed units count toward cost and the registered evaluation denominator.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'SimpleMem')]
CAPS = {'MemNexus': 8.0, 'SimpleMem': 14.0, 'flat': .6, 'before': .6, 'no_promotion':8.0, 'judge': .2,
        'locomo_current':1.0,'locomo_before':1.0,'locomo_semantic':1.0,'final_reader':.4,
        'coverage_reader':.6,'locomo_coverage':1.0}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    # Windows scanners/readers can briefly hold an otherwise writable output.
    # Retry only the atomic filesystem commit, never the paid model operation.
    for attempt in range(6):
        try:
            tmp.replace(p)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.1 * 2**attempt)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def configure():
    from dotenv import dotenv_values
    private = dotenv_values(ROOT / 'MemNexus/memory-system/.env')
    for k, v in private.items():
        if v is not None and (k.startswith('MEMORY_MODEL_') or k.startswith('MEMORY_EMBEDDING_')):
            os.environ[k] = v
    os.environ.update(MEMORY_MODEL_NAME='deepseek-v4.1-flash', MEMORY_MODEL_DISABLE_THINKING='true',
                      MEMORY_INPUT_PRICE_RMB_PER_MILLION_TOKENS='2.4',
                      MEMORY_OUTPUT_PRICE_RMB_PER_MILLION_TOKENS='9.6',
                      MEMORY_WINDOW_TOKEN_LIMIT='24000', MEMORY_MAX_TURN_CHARS='200000',
                      MEMORY_STATE_CLAMP='true', MEMORY_CONTEXT_TURNS='15',
                      MEMORY_ALLOW_ASSISTANT_EVIDENCE='true', MEMORY_PLANNER_MODE='model',
                      MEMORY_PROMOTION_ENABLED='true', EXPERIMENT_MEMNEXUS_MAX_TOKENS='32768',
                      EXPERIMENT_TRUNCATION_MAX_TOKENS='32768',
                      NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    import config
    config.OPENAI_API_KEY = config.JUDGE_API_KEY = private['MEMORY_MODEL_API_KEY']
    config.OPENAI_BASE_URL = config.JUDGE_BASE_URL = private['MEMORY_MODEL_BASE_URL']
    config.LLM_MODEL = config.JUDGE_MODEL = 'deepseek-v4.1-flash'
    config.TEMPERATURE, config.MAX_TOKENS = 0.0, 32768
    config.INPUT_PRICE_RMB_PER_MILLION_TOKENS = 2.4
    config.OUTPUT_PRICE_RMB_PER_MILLION_TOKENS = 9.6
    config.ENABLE_THINKING = config.USE_STREAMING = False
    config.WINDOW_SIZE, config.OVERLAP_SIZE = 20, 15
    return config


def prepare():
    if (HERE / 'protocol.json').exists():
        raise RuntimeError('Registered protocol already exists')
    all_q = read(ROOT / 'result/longmemeval_1000_cohort_20261023.json')
    groups = {}
    for q in all_q['questions']:
        if q['setting'] == 'S':
            groups.setdefault(q['category'], []).append(q)
    rng = random.Random(20261118)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    selected = []
    while len(selected) < 40:
        for key in sorted(groups):
            if groups[key] and len(selected) < 40:
                selected.append(groups[key].pop())
    cohort = {**{k: all_q[k] for k in ('dataset', 'dataset_sha256')},
              'questions': selected, 'target_n': 40,
              'selection': 'seed 20261118; category-balanced round robin; IDs/types only; not an untouched publication holdout',
              'stopping': 'fixed order prefix admitted only on conservative budget; no quality stopping'}
    save(HERE / 'cohort.json', cohort)
    source = ROOT / 'MemNexus/memory-system/src/agent_memory'
    shutil.copytree(source, HERE / 'source_candidate/agent_memory', ignore=shutil.ignore_patterns('__pycache__'))
    files = list(source.rglob('*.py')) + list((ROOT / 'SimpleMem/simplemem').rglob('*.py'))
    files += [ROOT / 'SimpleMem/main.py', ROOT / 'SimpleMem/config.py',
              ROOT / 'experiments/run_locomo_pilot.py', ROOT / 'experiments/run_parallel_bench.py', Path(__file__)]
    save(HERE / 'protocol.json', {
        'registered_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'budget_rmb': 500,
        'cohort_sha256': sha(HERE / 'cohort.json'), 'target_pairs': 40,
        'first_two': 'technical pilot within exploratory cohort; review before admitting remaining unchanged cohort',
        'primary': 'paired accuracy with SimpleMem Appendix A.4 prompt, SAME changed DeepSeek judge',
        'limitations': ['gpt-4.1-mini absent from configured provider catalog; NOT exact paper judge replication',
                       'DeepSeek backbone substitution; not original paper numerical reproduction',
                       'development exposure exists; not a publication holdout',
                       'category-balanced subset; report macro and micro separately',
                       'normalized conservative cost, not actual invoice or assumed night discount'],
        'baseline': {'window':20,'stride':5,'overlap':15,'paper_ambiguity':'Table7 stride5 conflicts with parenthetical 25% overlap; use explicit stride5',
                     'embedding':'Qwen3-Embedding-0.6B 1024 fp32 eager; same local encoder both systems',
                     'planning':'native enabled','reflection':'native enabled','temperature':0},
        'model':'deepseek-v4.1-flash','judge':'deepseek-v4.1-flash', 'prices_per_million':{'input':2.4,'output':9.6},
        'generation':{'max_tokens':4096,'bounded_length_recovery':32768,'unit_retries':0},
        'inputs':'full S haystack both systems; original chronology/role/timestamp; no question/gold in construction',
        'caps':CAPS,'reserved_analysis_rmb':30,
        'ablation':'first 12 completed MN snapshots: same facts, scopes, validity, raw history; flatten tiers to short; indexed sweep retained; QA-only marginal cost',
        'before_replay':'first 12 MN snapshots with entry-version reader; isolates reading changes, not write changes',
        'literature':[
          {'url':'https://arxiv.org/html/2510.18866v3','use':'incremental memory organization; do not hide consolidation costs'},
          {'url':'https://arxiv.org/html/2410.10813v2','use':'fact indexing plus original evidence, not destructive fact-only replacement'},
          {'url':'https://aclanthology.org/2026.findings-acl.1091.pdf','use':'test complementary detailed evidence and abstraction; no assumed additive gain'},
          {'url':'https://arxiv.org/html/2601.02553','use':'baseline native pipeline and A.4 judge prompt; disclose substitutions'}],
        'code':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    print('Prepared 40 S questions, source freeze, protocol; zero paid calls', flush=True)


def component_path(arm, sid):
    return HERE / 'components' / arm / sid


def cost(path):
    ledger = path / 'usage.json'
    if ledger.exists():
        u = read(ledger)
        return float(u.get('cost_rmb', 0)) + float(u.get('reserved_rmb', 0))
    return 0.0


def infrastructure_failure(path):
    """Stop admission after transport failures; never reinterpret answer quality."""
    completion = path / 'completion.json'
    if not completion.exists() or read(completion).get('status') != 'failed':
        return False
    kinds = {'ConnectError', 'APIConnectionError', 'APITimeoutError', 'TimeoutError'}
    if read(completion).get('error_type') in kinds:
        return True
    # Native retries can exhaust the component budget after a transport outage.
    if read(completion).get('error_type') == 'BudgetExceededError':
        log = path / 'usage.calls.jsonl'
        if log.exists():
            events = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
            terminal = [e for e in events if e.get('event') in ('response', 'error')]
            return bool(terminal and terminal[-1].get('error_type') in kinds)
    return False


def reconcile():
    admissions = read(HERE / 'admissions.json') if (HERE / 'admissions.json').exists() else []
    spent, reserved = 0.0, 0.0
    for a in admissions:
        p = component_path(a['arm'], a['sample_id'])
        if (p / 'completion.json').exists():
            spent += cost(p)
        else:
            reserved += a['cap']
    budget = read(HERE / 'budget.json')
    budget.update(spent_rmb=round(spent,8), reserved_rmb=round(reserved,8),
                  conservative_total_rmb=round(spent+reserved,8))
    save(HERE / 'budget.json',budget)
    if spent + reserved > 500 + 1e-6:
        raise RuntimeError('Study budget invariant failed')
    return admissions, spent, reserved


async def worker(arm, sid):
    configure()
    if arm=='no_promotion':
        os.environ['MEMORY_PROMOTION_ENABLED']='false'
    p = component_path(arm,sid)
    if (p / 'started.json').exists():
        raise RuntimeError('Refuse component rerun/reset')
    save(p / 'started.json',{'time':time.time(),'cap':CAPS[arm]})
    locomo=arm.startswith('locomo_')
    if arm=='locomo_coverage':
        import config
        config.MAX_TOKENS=4096
        os.environ['EXPERIMENT_MEMNEXUS_MAX_TOKENS']='4096'
    cohort = read(ROOT/'result/research_iteration_20261117/locomo41_full_cohort.json' if locomo else HERE/'cohort.json')
    q = next(q for q in cohort['questions'] if q['sample_id']==sid)
    variant = ('source_relation_coverage' if arm in ('coverage_reader','locomo_coverage') else
               'source_candidate_v3' if arm=='locomo_semantic' else
               'source_before' if arm in ('before','locomo_before') else
               'source_candidate' if q.get('phase')=='pilot' else
               'source_relation_coverage' if q.get('phase')=='main' else
               read(HERE/'accepted_source.json')['variant'] if (HERE/'accepted_source.json').exists()
               else 'source_candidate_v2')
    sys.path.insert(0,str(HERE / variant))
    import agent_memory
    assert Path(agent_memory.__file__).resolve().is_relative_to(HERE / variant)
    if arm in ('flat','no_promotion'):
        from agent_memory.retrieval import Retriever
        original_recall=Retriever._recall
        async def flat_recall(self,query,plan,session,memories,limit,now,warnings):
            # A flat index gets the WHOLE existing candidate budget. Leaving
            # 2/3 reserved for its empty LT tier would weaken the control.
            if memories and all(m.tier=='short' for m in memories):
                depth=max(3,min(20,max(plan.depth,len(plan.required_info))))
                limit=min(self.settings.max_candidates,(10 if plan.recall_policy=='recall_first' else 6)*depth)
            return await original_recall(self,query,plan,session,memories,limit,now,warnings)
        Retriever._recall=flat_recall
    from run_parallel_bench import read_selected_samples, run_unit
    sample = read_selected_samples(ROOT / cohort['dataset'],{q['sample_index']})[q['sample_index']]
    max_session = max(int(k.split('_')[1]) for k in sample['conversation'] if k.startswith('session_') and not k.endswith('_date_time'))
    sample['expected_raw_turns'] = sum(bool(t['text'].strip()) for v in sample['conversation'].values()
                                       if isinstance(v,list) for t in v)
    save(p/'configuration.json',{'arm':arm,'max_session':max_session,'raw_turns':sample['expected_raw_turns'],
                                'model':'deepseek-v4.1-flash','window_chars':24000,'max_output_tokens':int(os.environ['EXPERIMENT_MEMNEXUS_MAX_TOKENS']),
                                'promotion':arm!='no_promotion','source_variant':variant,
                                'source_hashes':{f.name:sha(f) for f in (HERE/variant/'agent_memory').glob('*.py')},
                                'cohort_sha256':sha(HERE/'cohort.json')})
    items=[x for x in cohort['questions'] if x['sample_id']==sid] if locomo else [q]
    qa = [{**sample['qa'][x['qa_index']],'qa_index':x['qa_index']} for x in items]
    try:
        if arm == 'judge':
            import config
            from run_locomo_pilot import judge_rows
            config.MAX_TOKENS = 512
            os.environ['EXPERIMENT_TRUNCATION_MAX_TOKENS']='0'
            rows=[]
            for source_arm in ('MemNexus','SimpleMem','flat','before','no_promotion','final_reader','coverage_reader'):
                f=component_path(source_arm,sid)/'stage_result.json'
                if f.exists():
                    rows.extend({**r,'arm':source_arm} for r in read(f)['rows'])
            # Blind model prompt contains question/gold/answer only, no system name.
            usage = judge_rows(rows,CAPS[arm],usage_ledger=p/'usage.json')
            save(p/'rows.json',rows)
            save(p/'stage_result.json',{'stage':{'total_usage':usage},'rows':rows})
        else:
            snapshots=HERE/'snapshots'
            if locomo:
                snapshots=ROOT/'result/research_iteration_20261117/locomo41_full_mn_w24000_max32768_snapshots'
            if arm=='no_promotion':
                snapshots=HERE/'snapshots_no_promotion'
            if arm=='flat':
                from agent_memory import snapshot
                from agent_memory.models import Scope
                from agent_memory.store import SQLiteStore
                snapshots=HERE/'snapshots_flat'
                db=p/'flat.sqlite3'
                snapshot.thaw(HERE/'snapshots',sid,db)
                st=SQLiteStore(db)
                scope=Scope(tenant_id='locomo-pilot',owner_id=sid)
                original=st.memories(scope)
                for m in original:
                    if m.tier=='long':
                        st._save_memory(scope,m.model_copy(update={'tier':'short'}))
                st._db.commit()
                snapshot.freeze(st,snapshots,sid,metadata={'ablation':'same facts flattened to short; no raw deletion','original_long':sum(m.tier=='long' for m in original)})
                st.close()
            result=await run_unit('SimpleMem' if arm=='SimpleMem' else 'MemNexus',sample,qa,items,p,CAPS[arm],max_session,
                                  snapshots if arm!='SimpleMem' else None,require_snapshots=locomo or arm in ('before','flat','final_reader','coverage_reader'))
            save(p/'stage_result.json',result)
        save(p/'completion.json',{'status':'complete','time':time.time()})
    except BaseException as e:
        if isinstance(e,(KeyboardInterrupt,SystemExit)): raise
        save(p/'completion.json',{'status':'failed','error_type':type(e).__name__,'error':str(e),'time':time.time()})
        raise


async def supervise(mode, limit):
    cohort=read(HERE/'cohort.json')
    selected=[q for q in cohort['questions'] if mode=='judge' or q.get('phase')=='main'][:limit]
    if mode=='coverage_reader':selected=[q for q in cohort['questions'] if q.get('phase')=='engineering'][:limit]
    if mode in ('locomo','locomo_semantic','locomo_coverage'):selected=[{'sample_id':'conv-41'}]
    batch_size=limit if mode=='main' and limit>2 else 2
    offset=0
    while offset<len(selected):
        next_offset=min(len(selected),offset+batch_size)
        jobs=[]
        for q in selected[offset:offset+batch_size]:
            arms=('MemNexus','SimpleMem') if mode=='main' else ('flat','before') if mode=='ablate' else ('no_promotion',) if mode=='write_ablate' else ('judge',)
            if mode=='locomo':arms=('locomo_current','locomo_before')
            if mode=='locomo_semantic':arms=('locomo_semantic',)
            if mode=='locomo_coverage':arms=('locomo_coverage',)
            if mode=='final_reader':arms=('final_reader',)
            if mode=='coverage_reader':arms=('coverage_reader',)
            for arm in arms:
                if any((component_path(arm,q['sample_id'])/name).exists() for name in ('started.json','completion.json')):continue
                if mode=='judge':
                    if not any((component_path(a,q['sample_id'])/'stage_result.json').exists()
                               for a in ('MemNexus','SimpleMem')):continue
                elif mode not in ('main','locomo','locomo_semantic','locomo_coverage') and not (component_path('MemNexus',q['sample_id'])/'stage_result.json').exists():continue
                jobs.append((arm,q['sample_id']))
        if not jobs:
            offset=next_offset
            continue
        admissions,spent,reserved=reconcile()
        admitted={(x['arm'],x['sample_id']) for x in admissions}
        # A restarted scheduler can inherit reserved jobs that have not started.
        # Reuse their reservation; never append a second ledger admission.
        extra_cap=lambda:sum(CAPS[a] for a,s in jobs if (a,s) not in admitted)
        cap=extra_cap()
        ceiling=470 if mode=='main' else 500
        if mode=='main':
            # Admit only whole pairs, in frozen order, when a final full batch
            # cannot fit. This is budget stopping, never outcome selection.
            while jobs and spent+reserved+extra_cap()>ceiling:
                last_sid=jobs[-1][1]
                next_offset=min(next_offset,next(i for i,q in enumerate(selected) if q['sample_id']==last_sid))
                jobs=[job for job in jobs if job[1]!=last_sid]
            cap=extra_cap()
        if not jobs or spent+reserved+cap>ceiling:
            save(HERE/f'{mode}_budget_stop.json',{'offset':offset,'spent':spent,'reserved':reserved,'next_cap':cap})
            break
        for arm,sid in jobs:
            if (arm,sid) not in admitted:
                admissions.append({'arm':arm,'sample_id':sid,'cap':CAPS[arm],'time':time.time()})
        save(HERE/'admissions.json',admissions)
        reconcile()
        launch_lock=asyncio.Lock()
        outage=asyncio.Event()
        async def run(arm,sid,delay):
            await asyncio.sleep(delay)
            p=component_path(arm,sid);p.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONHASHSEED='0')
            with (p/'process.log').open('w',encoding='utf-8') as f:
                async with launch_lock:
                    while True:
                        if outage.is_set():
                            save(p/'completion.json',{'status':'failed','error_type':'InfrastructureCircuitOpen',
                                'error':'Cancelled before provider request after another worker transport failure',
                                'cancelled_before_start':True,'time':time.time()})
                            return
                        count=0
                        for process in psutil.process_iter(['cmdline']):
                            try:
                                cmd=process.info['cmdline'] or []
                                count+=int('worker' in cmd and any(str(v).endswith('study.py') for v in cmd))
                            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                        if count<8:break
                        await asyncio.sleep(2)
                    proc=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__)),'worker','--arm',arm,'--sid',sid,
                                                             stdout=f,stderr=f,env=env)
                rc=await proc.wait()
                if infrastructure_failure(p):
                    outage.set()
                    save(HERE/'infrastructure_stop.json',{'arm':arm,'sample_id':sid,'time':time.time(),
                        'policy':'Let active workers settle; cancel unstarted admissions; stop subsequent stages. No answer-dependent stopping.'})
            print(json.dumps({'arm':arm,'sample':sid,'exit':rc,'cost':cost(p)}),flush=True)
        await asyncio.gather(*(run(a,s,2*i) for i,(a,s) in enumerate(jobs)))
        reconcile()
        if outage.is_set():
            raise RuntimeError('Infrastructure outage: queued requests cancelled; active workers settled. See infrastructure_stop.json.')
        # Revisit the first budget-trimmed ID after actual costs settle. A for
        # loop advancing a full batch here would silently skip cohort members.
        offset=next_offset
    print(json.dumps(read(HERE/'budget.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','worker','main','ablate','write_ablate','judge','locomo','locomo_semantic','locomo_coverage','coverage_reader','final_reader','complete'])
    parser.add_argument('--arm',choices=list(CAPS));parser.add_argument('--sid')
    parser.add_argument('--limit',type=int,default=40)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='worker':asyncio.run(worker(args.arm,args.sid))
    elif args.mode=='complete':
        async def complete():
            await supervise('main',40)
            await supervise('ablate',12)
            await supervise('write_ablate',6)
            await supervise('judge',len(read(HERE/'cohort.json')['questions']))
        asyncio.run(complete())
        from analyze import main as analyze
        analyze()
    else:asyncio.run(supervise(args.mode,args.limit))

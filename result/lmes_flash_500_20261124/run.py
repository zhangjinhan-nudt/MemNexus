"""Budgeted study orchestrator; calls existing native methods, never alters production."""
import asyncio, json, os, sys, time, shutil, hashlib, random, subprocess, importlib.util
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'SimpleMem')]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('accounting',ROOT/'result/assistant_validation_20261123/runner.py');old.LIMIT=500
read,write,sha=old.read,old.write,old.sha
CAPS={'mn_reuse':1.5,'sm_reuse':1.5,'mn_fresh':10.,'sm_fresh':15.,'judge':.2}
def configure():
 import lme_paired_protocol as lp
 sys.path.insert(0,str(lp.CONFIG));import study
 original=study.configure
 def patched():
  c=original();c.LLM_MODEL=c.JUDGE_MODEL='deepseek-flash';os.environ['MEMORY_MODEL_NAME']='deepseek-flash';return c
 study.configure=patched;lp.MODEL='deepseek-flash'
 return lp.runtime_config()
def balance():
 b=old.balance(HERE);write(HERE/'budget.json',b);return b
def verify():
 p=read(HERE/'protocol.json')
 for name,digest in p['files'].items():assert sha(ROOT/name)==digest,'Frozen code changed: '+name
 return p

def prepare():
 from run_parallel_bench import read_selected_samples
 from run_locomo_pilot import raw_turns
 from lme_paired_protocol import inspect_inputs,effective_method
 config=configure();base=read(ROOT/'result/assistant_validation_20261123/cohort.json');allq=read(ROOT/'result/longmemeval_1000_cohort_20261023.json');inventory=read(HERE/'sm_inventory.json')
 existing={q['sample_id']:q for q in base['questions']};questions=[]
 for q in allq['questions']:
  if q['setting']!='S':continue
  q=dict(q);q['memory_mode']='reuse' if q['sample_id'] in existing else 'fresh'
  if q['memory_mode']=='reuse':q['mn_snapshot']=existing[q['sample_id']]['snapshot_dir']
  questions.append(q)
 groups=defaultdict(list)
 for q in questions:groups[(q['memory_mode'],q['category'])].append(q)
 rng=random.Random(20261124)
 for g in groups.values():rng.shuffle(g)
 ordered=[]
 for mode in ['reuse','fresh']:
  keys=sorted(k for k in groups if k[0]==mode)
  while any(groups[k] for k in keys):
   for k in keys:
    if groups[k]:ordered.append(groups[k].pop())
 dataset=ROOT/base['dataset'];samples=read_selected_samples(dataset,{q['sample_index'] for q in ordered});info=inspect_inputs(ordered,samples)
 for q in ordered:
  if q['memory_mode']!='reuse':continue
  sid=q['sample_id'];expected=raw_turns(samples[q['sample_index']],info[sid]['max_session']);valid=[]
  for filename in inventory[sid]:
   mem=read(filename);saved=mem['raw_turns'];without=[{k:v for k,v in x.items() if k!='source_session'} for x in expected]
   if saved==expected or saved==without:valid.append(filename)
  assert valid,'No full-history SM memory for '+sid
  q['sm_memory']=sorted(valid)[0];q['sm_memory_sha256']=sha(q['sm_memory'])
  q['mn_snapshot_sha256']=sha(Path(q['mn_snapshot'])/(sid+'.sqlite3'))
 files=list((ROOT/'MemNexus/memory-system/src/agent_memory').rglob('*.py'))+list((ROOT/'SimpleMem/simplemem/core').rglob('*.py'))
 files += [ROOT/'SimpleMem/config.py',ROOT/'SimpleMem/main.py',Path(__file__),ROOT/'experiments/run_locomo_pilot.py',ROOT/'experiments/run_parallel_bench.py',ROOT/'experiments/paper_protocol.py']
 p={'limit':500,'analysis_reserve':20,'caps':CAPS,'model':'deepseek-flash','judge':'DeepSeek substitute, unchanged SimpleMem Appendix A.4, shuffled anonymous answers','dataset':str(dataset),'dataset_sha256':sha(dataset),'questions':ordered,'info':info,'effective_method':effective_method(config),'files':{str(f.relative_to(ROOT)):sha(f) for f in files},'checkpoint_rule':read(HERE/'budget_plan.json')['stop_rule'],'scope':'reuse and fresh reported separately; reused stores were built by historical models, all QA regenerated with deepseek-flash','seed':20261124}
 assert not (HERE/'protocol.json').exists();write(HERE/'protocol.json',p)
 folder=HERE/'components/probe/readiness';write(folder/'admission.json',{'cap_rmb':.02});write(folder/'started.json',{});shutil.copy2(HERE/'probe_usage.json',folder/'usage.json');shutil.copy2(HERE/'probe_usage.calls.jsonl',folder/'usage.calls.jsonl');write(folder/'completion.json',{})
 balance();print(json.dumps({'prepared':len(ordered),'reuse':sum(q['memory_mode']=='reuse' for q in ordered)}))

def sm_reuse(q,sample,folder,cap):
 from run_locomo_pilot import RemoteQwenEmbedding,evaluation_query,usage_delta,token_f1,bleu1,is_forced_abstain
 from main import SimpleMemSystem
 saved=read(q['sm_memory']);assert sha(q['sm_memory'])==q['sm_memory_sha256']
 index=folder/'index';shutil.copytree(Path(q['sm_memory']).parent/'usage.index/lancedb',index)
 os.environ.update(EXPERIMENT_COMPONENT_BUDGET_RMB=str(cap),EXPERIMENT_USAGE_LEDGER_PATH=str(folder/'usage.json'))
 system=SimpleMemSystem(db_path=str(index),clear_db=False,embedding_model=RemoteQwenEmbedding(),enable_parallel_processing=False,max_parallel_workers=1,enable_parallel_retrieval=False,max_retrieval_workers=1)
 entries=system.get_all_memories();assert {e.entry_id:e.lossless_restatement for e in entries}=={e['entry_id']:e['lossless_restatement'] for e in saved['memories']}
 qa=sample['qa'][0];query,gold=evaluation_query(qa);before=dict(system.llm_client.usage);start=time.perf_counter();contexts=system.hybrid_retriever.retrieve(query);rms=(time.perf_counter()-start)*1000;start=time.perf_counter();answer=system.answer_generator.generate_answer(query,contexts);gms=(time.perf_counter()-start)*1000
 row={'system':'SimpleMem','sample_id':q['sample_id'],'qa_index':0,'question':q['question'],'evaluation_query':query,'gold_answer':gold,'category':q['category'],'category_id':0,'prediction':answer,'retrieved_sources':[e.model_dump(mode='json') for e in contexts],'model_calls':usage_delta(system.llm_client.usage,before),'latency_ms':rms+gms,'retrieval_ms':rms,'generation_ms':gms,'f1':token_f1(answer,gold),'bleu1':bleu1(answer,gold),'judge_result':None,'memory_mode':'reuse'}
 assert answer and answer!='Failed to generate answer'
 return {'stage':{'total_usage':dict(system.llm_client.usage),'historical_build_usage':saved['build_usage'],'build_usage':{},'memory_base_replayed':True},'rows':[row]}

async def worker(phase,sid):
 p=verify();c=configure();q=next(q for q in p['questions'] if q['sample_id']==sid);folder=HERE/'components'/phase/sid;cap=read(folder/'admission.json')['cap_rmb'];assert not (folder/'started.json').exists();write(folder/'started.json',{'pid':os.getpid()})
 try:
  from run_parallel_bench import run_unit,read_selected_samples,cohort_questions
  from run_locomo_pilot import judge_rows,validate_completed_qa_history
  if phase=='judge':
   rows=[]
   for system in ['mn','sm']:rows.extend(read(HERE/'components'/(system+'_'+q['memory_mode'])/sid/'stage_result.json')['rows'])
   random.Random(p['seed']+int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)).shuffle(rows)
   c.MAX_TOKENS=512;os.environ['EXPERIMENT_TRUNCATION_MAX_TOKENS']='0';usage=judge_rows(rows,cap,usage_ledger=folder/'usage.json');result={'stage':{'total_usage':usage},'rows':rows}
  else:
   sample=read_selected_samples(Path(p['dataset']),{q['sample_index']})[q['sample_index']];sample['expected_raw_turns']=p['info'][sid]['raw_turns']
   if phase=='sm_reuse':result=sm_reuse(q,sample,folder,cap)
   else:
    system='MemNexus' if phase.startswith('mn') else 'SimpleMem';snapshot=Path(q['mn_snapshot']) if phase=='mn_reuse' else folder/'snapshots'
    if phase=='mn_reuse':assert sha(snapshot/(sid+'.sqlite3'))==q['mn_snapshot_sha256']
    result=await run_unit(system,sample,cohort_questions(sample,[q]),[q],folder,cap,p['info'][sid]['max_session'],snapshot,phase=='mn_reuse')
    if system=='MemNexus':validate_completed_qa_history(result,'events')
   for r in result['rows']:r['memory_mode']=q['memory_mode']
  write(folder/'stage_result.json',result);verify();write(folder/'completion.json',{'status':'complete'});write(folder/'ended.json',{'status':'complete'})
 except BaseException as e:write(folder/'ended.json',{'status':'failed','error':str(e)});raise
 finally:balance()

def checkpoint(label):
 import math
 rows=[]
 for f in (HERE/'components/judge').glob('*/stage_result.json'):rows.extend(read(f)['rows'])
 pairs=defaultdict(dict)
 for r in rows:pairs[r['sample_id']][r['system']]=r
 pairs={k:v for k,v in pairs.items() if len(v)==2};wins=losses=0
 for pair in pairs.values():
  m=pair['MemNexus']['judge_result']=='CORRECT';s=pair['SimpleMem']['judge_result']=='CORRECT';wins+=m and not s;losses+=s and not m
 n=len(pairs);d=wins+losses;pv=sum(math.comb(d,k) for k in range(losses,d+1))/2**d if d else 1
 stop=n>=24 and ((losses-wins)/n>=.10 or (losses>wins and pv<.05))
 result={'label':label,'pairs':n,'wins':wins,'losses':losses,'delta_accuracy':(wins-losses)/n if n else None,'exploratory_one_sided_p':pv,'stop_for_quality':stop,'budget':balance()}
 write(HERE/'checkpoints'/(label+'.json'),result);write(HERE/'results.json',{'rows':rows,'checkpoint':result});print(json.dumps(result),flush=True);return stop

def run():
 from lme_paired_protocol import coordinator_lock,encoder_check
 configure();encoder_check()
 with coordinator_lock(HERE/'coordinator.lock'):
  p=verify();status=HERE/'status.json';assert not status.exists();write(status,{'status':'running','pid':os.getpid()});mark=100;done=0
  for q in p['questions']:
   sid=q['sample_id'];phases=['mn_'+q['memory_mode'],'sm_'+q['memory_mode'],'judge'];required=sum(CAPS[x] for x in phases)
   if balance()['available_rmb']<required+20:checkpoint('budget_final');write(status,{'status':'budget_complete','pairs':done});return
   for phase in phases:
    folder=HERE/'components'/phase/sid;assert not folder.exists();write(folder/'admission.json',{'cap_rmb':CAPS[phase],'protocol_sha256':sha(HERE/'protocol.json')});balance()
    with (folder/'stdout.log').open('w',encoding='utf8') as out,(folder/'stderr.log').open('w',encoding='utf8') as err:
     process=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__)),'worker',phase,sid],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if process.returncode:checkpoint('technical_stop');write(status,{'status':'technical_stop','phase':phase,'sample_id':sid});return
   done+=1;spent=balance()['settled_rmb']
   if done in [12,24,62] or spent>=mark:
    if checkpoint('pairs_'+str(done)):
     write(status,{'status':'quality_stop','pairs':done});return
    if spent>=mark:mark=(int(spent//100)+1)*100
   print(json.dumps({'completed_pairs':done,'cost_rmb':spent,'sample_id':sid}),flush=True)
  checkpoint('all_complete');write(status,{'status':'complete','pairs':done})
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='run':run()
 else:asyncio.run(worker(sys.argv[2],sys.argv[3]))

import json,html,statistics,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
TYPES=['temporal-reasoning','multi-session','knowledge-update','single-session-user','single-session-assistant','single-session-preference']
LABELS=['Temporal','Multi-session','Knowledge update','Single-session user','Single-session assistant','Single-session preference']
SYSTEMS=['MemNexus','SimpleMem'];COLORS=['#6ba3c8','#dd9b75']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def mean(xs):return float(np.mean(xs)) if xs else None
pairs={}
for p in (H/'components/judge').glob('*/stage_result.json'):
 rows=read(p)['rows']
 if len(rows)==2 and all(r.get('judge_result') in ['CORRECT','WRONG'] for r in rows):pairs[p.parent.name]={r['system']:r for r in rows}
if not pairs:sys.exit('No completed pairs')
rows=[r for p in pairs.values() for r in p.values()]
def summarize(rs):
 out={}
 for s in SYSTEMS:
  selected=[r for r in rs if r['system']==s];types={}
  for t in TYPES:
   group=[r for r in selected if r['category']==t]
   types[t]={'n':len(group),'accuracy':mean([100*(r['judge_result']=='CORRECT') for r in group]),'f1':mean([r['f1']*100 for r in group]),'bleu1':mean([r['bleu1']*100 for r in group])}
  out[s]={'n':len(selected),'types':types,'micro_accuracy':mean([100*(r['judge_result']=='CORRECT') for r in selected]),'macro_accuracy':mean([v['accuracy'] for v in types.values()]) if all(v['n'] for v in types.values()) else None,'f1':mean([r['f1']*100 for r in selected]),'bleu1':mean([r['bleu1']*100 for r in selected]),'qa_input':sum(r['model_calls'].get('input_tokens',0) for r in selected),'qa_output':sum(r['model_calls'].get('output_tokens',0) for r in selected),'qa_cost':sum(r['model_calls'].get('cost_rmb',0) for r in selected),'attempts':sum(r['model_calls'].get('provider_attempts',0) for r in selected),'latency_median_s':float(np.median([r['latency_ms']/1000 for r in selected])) if selected else None,'latency_p95_s':float(np.percentile([r['latency_ms']/1000 for r in selected],95)) if selected else None}
 return out
summary={'matched_pairs':len(pairs),'all':summarize(rows),'status':read(H/'status.json'),'budget':read(H/'budget.json'),'analysis_population':'All complete matched pairs pooled; no construction-cohort stratification'}
summary['answerable']=summarize([r for r in rows if '_abs_' not in r['sample_id']])
summary['unanswerable']=summarize([r for r in rows if '_abs_' in r['sample_id']])
for scope in ['all']:
 for s in SYSTEMS:
  entry=summary[scope][s]
  for metric in ['f1','bleu1']:
   entry['macro_'+metric]=mean([v[metric] for v in entry['types'].values()]) if all(v['n'] for v in entry['types'].values()) else None
summary['measured_build']={}
for s,prefix in zip(SYSTEMS,['mn','sm']):
 stages=[read(H/'components'/(prefix+'_fresh')/sid/'stage_result.json')['stage'] for sid,p in pairs.items() if p[s].get('memory_mode')=='fresh']
 summary['measured_build'][s]={'n':len(stages),'input_tokens':sum(x.get('build_usage',{}).get('input_tokens',0) for x in stages),'output_tokens':sum(x.get('build_usage',{}).get('output_tokens',0) for x in stages),'normalized_rmb':sum(x.get('build_usage',{}).get('cost_rmb',0) for x in stages),'total_build_seconds':sum(x.get('build_ms',0)/1000 for x in stages),'provider_attempts':sum(x.get('build_usage',{}).get('provider_attempts',0) for x in stages)}
if (H/'billing_audit.json').exists():summary['provider_billing_estimate']=read(H/'billing_audit.json')
rng=np.random.default_rng(20261124);d=np.array([int(p['MemNexus']['judge_result']=='CORRECT')-int(p['SimpleMem']['judge_result']=='CORRECT') for p in pairs.values()]);draw=rng.choice(d,(10000,len(d)),replace=True).mean(1)*100
summary['paired_micro_delta_pp']=float(d.mean()*100);summary['paired_micro_bootstrap_95ci_pp']=np.percentile(draw,[2.5,97.5]).tolist()
summary['stratified_paired_macro']={}
for scope in ['all']:
 groups=[[int(p['MemNexus']['judge_result']=='CORRECT')-int(p['SimpleMem']['judge_result']=='CORRECT') for p in pairs.values() if p['MemNexus']['category']==t and (scope=='all' or p['MemNexus'].get('memory_mode')==scope)] for t in TYPES]
 if not all(groups):continue
 rng_scope=np.random.default_rng(20261124)
 draws=np.mean([rng_scope.choice(np.asarray(g),(10000,len(g)),replace=True).mean(axis=1) for g in groups],axis=0)*100
 summary['stratified_paired_macro'][scope]={'delta_pp':float(np.mean([np.mean(g) for g in groups])*100),'bootstrap_95ci_pp':np.percentile(draws,[2.5,97.5]).tolist(),'n_by_type':[len(g) for g in groups],'method':'Paired resampling within each task type; six type means weighted equally; exploratory fixed-seed10000 draws.'}
(H/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False})

# One population for all headline quality and inference comparisons.
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
x=np.arange(3)
for i,s in enumerate(SYSTEMS):
 vals=[summary['all'][s][k] for k in ['macro_accuracy','macro_f1','macro_bleu1']]
 bars=axes[0].bar(x+(i-.5)*.36,vals,.36,color=COLORS[i],label=s)
 axes[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
axes[0].set_xticks(x,['Accuracy','Token F1','BLEU-1'])
axes[0].set_ylim(0,105);axes[0].set_ylabel('Macro score (%)')
axes[0].set_title('Answer quality | equal weight across six tasks')
keys=['qa_input','qa_output','qa_cost','attempts'];x=np.arange(4)
for i,s in enumerate(SYSTEMS):
 vals=[100*summary['all'][s][k]/summary['all']['SimpleMem'][k] for k in keys]
 bars=axes[1].bar(x+(i-.5)*.36,vals,.36,color=COLORS[i],label=s)
 axes[1].bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
axes[1].set_xticks(x,['Input tokens','Output tokens','QA cost','QA calls'])
axes[1].set_ylim(0,120);axes[1].set_ylabel('Resource index (SimpleMem = 100)')
axes[1].set_title('Inference resources | lower is better')
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle(f'LongMemEval-S | {len(pairs)} matched pairs',fontsize=18,fontweight='bold',y=.99)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
fig.text(.025,.025,'All complete pairs pooled. Same DeepSeek generation and substitute judge. QA cost uses a common token price.',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.84))
fig.savefig(H/'final_overview.png',dpi=200);fig.savefig(H/'final_overview.pdf');plt.close(fig)
def quality_plot(scope,stem):
 fig,axes=plt.subplots(1,3,figsize=(17,5));x=np.arange(6)
 for ax,metric,title in zip(axes,['accuracy','f1','bleu1'],['Accuracy (primary)','Token F1 (supplementary)','BLEU-1 (supplementary)']):
  for i,s in enumerate(SYSTEMS):
   values=[summary[scope][s]['types'][t][metric] for t in TYPES];ax.bar(x+(i-.5)*.36,[v if v is not None else 0 for v in values],.36,label=s,color=COLORS[i],edgecolor='#555555',linewidth=.4)
  ax.set_xticks(x,LABELS,rotation=35,ha='right',fontsize=9);ax.set_ylim(0,105);ax.set_ylabel('Score (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False);fig.suptitle(f"LongMemEval-S: {summary[scope]['MemNexus']['n']} matched pairs | {scope} | deepseek-flash",fontweight='bold');fig.text(.01,.01,'Same DeepSeek judge substitutes for paper GPT-4.1-mini. All complete matched pairs pooled; task types retained.',fontsize=9);fig.tight_layout(rect=(0,.045,1,.85));fig.savefig(H/(stem+'.png'),dpi=180);fig.savefig(H/(stem+'.pdf'));plt.close(fig)
quality_plot('all','quality')
def resource_plot(scope,stem):
 fig,axes=plt.subplots(1,4,figsize=(14,3.8))
 for ax,key,title in zip(axes,['qa_input','qa_output','qa_cost','latency_median_s'],['QA input tokens / question','QA output tokens / question','Normalized QA RMB / question','Median QA latency (s)']):
  vals=[summary[scope][s][key]/summary[scope]['MemNexus']['n'] if key!='latency_median_s' else summary[scope][s][key] for s in SYSTEMS];ax.bar(SYSTEMS,vals,color=COLORS);ax.set_title(title,fontsize=11)
  for i,v in enumerate(vals):ax.text(i,v,f'{v:,.3f}' if v<10 else f'{v:,.1f}',ha='center',va='bottom')
  ax.margins(y=.2)
 fig.suptitle(f'Measured inference resources | {scope} | all attempts included',fontweight='bold');fig.text(.01,.01,'Cost normalization: input 2.4 / output 9.6 RMB per million tokens; actual account billing is reported separately.',fontsize=8);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(H/(stem+'.png'),dpi=180);fig.savefig(H/(stem+'.pdf'));plt.close(fig)
resource_plot('all','resources')
scopes=['all']
fig,axes=plt.subplots(1,len(scopes),figsize=(6*len(scopes),4.5),squeeze=False)
for ax,scope in zip(axes[0],scopes):
 for s,color in zip(SYSTEMS,COLORS):
  v=summary[scope][s];y=v['macro_accuracy']
  if y is None:continue
  x=(v['qa_input']+v['qa_output'])/v['n'];ax.scatter(x,y,s=130,c=color,edgecolor='#555555',zorder=3);ax.annotate(s,(x,y),xytext=(6,8 if s=='MemNexus' else -16),textcoords='offset points',fontsize=10)
 ax.set_xscale('log');ax.set_ylim(0,105);ax.set_xlabel('All QA tokens / question (log scale)');ax.set_ylabel('Macro accuracy (%)');ax.set_title('All matched pairs'+f' | n={summary[scope]["MemNexus"]["n"]}');ax.grid(alpha=.2);ax.margins(x=.55)
fig.suptitle('Answer quality and inference token cost',fontweight='bold');fig.tight_layout();fig.savefig(H/'efficiency.png',dpi=180);fig.savefig(H/'efficiency.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(16,5.8));ax.axis('off');cells=[];values=[]
for metric,label in [('accuracy','Accuracy'),('f1','Token F1'),('bleu1','BLEU-1')]:
 for s in SYSTEMS:
  vals=[summary['all'][s]['types'][t][metric] for t in TYPES]+[summary['all'][s]['macro_'+metric]]
  values.append(vals);cells.append([label,s]+['—' if v is None else f'{v:.2f}' for v in vals])
tab=ax.table(cellText=cells,colLabels=['Metric','Method','Temporal','Multi-\nsession','Knowledge\nupdate','Single-session\nuser','Single-session\nassistant','Single-session\npreference','Macro\naverage'],cellLoc='center',colWidths=[.11,.12,.10,.10,.11,.12,.12,.12,.10],bbox=[0,.13,1,.79]);tab.auto_set_font_size(False);tab.set_fontsize(11)
for (i,j),cell in tab.get_celld().items():
 cell.set_edgecolor('#dddddd');cell.set_linewidth(.45)
 if i==0:cell.set_text_props(fontweight='bold');cell.set_facecolor('white')
 elif i%2==0:cell.set_facecolor('#eeeeee')
for pair_start in [0,2,4]:
 for col in range(7):
  vv=[values[pair_start+k][col] for k in [0,1]]
  if all(v is not None for v in vv):
   for k in [0,1]:
    if vv[k]>=max(vv)-1e-9:tab[pair_start+k+1,col+2].set_text_props(fontweight='bold')
ax.set_title(f'LongMemEval-S across {len(pairs)} matched pairs',loc='left',fontsize=18,fontweight='bold',pad=12)
counts=', '.join(str(summary['all']['MemNexus']['types'][t]['n']) for t in TYPES)
fig.text(.025,.04,f'Questions per type (left to right): {counts}. Macro averages give equal weight to six types.\nAccuracy is primary; F1/BLEU-1 are supplementary. DeepSeek judge substitutes for GPT-4.1-mini.',fontsize=10)
fig.tight_layout(rect=(0,.06,1,1));fig.savefig(H/'table.png',dpi=180);fig.savefig(H/'table.pdf');plt.close(fig)
if summary['measured_build']['MemNexus']['n']:
 fig,axes=plt.subplots(1,4,figsize=(15,4))
 for ax,key,title in zip(axes,['tokens','normalized_rmb','provider_attempts','total_build_seconds'],['Build tokens / question','Normalized build RMB / question','Build calls / question','Build minutes / question']):
  vals=[]
  for s in SYSTEMS:
   b=summary['measured_build'][s];value=(b['input_tokens']+b['output_tokens']) if key=='tokens' else b[key];vals.append(value/b['n']/(60 if key=='total_build_seconds' else 1))
  ax.bar(SYSTEMS,vals,color=COLORS);ax.set_title(title,fontsize=10);ax.margins(y=.2)
  for i,v in enumerate(vals):ax.text(i,v,f'{v:,.2f}',ha='center',va='bottom',fontsize=9)
 fig.suptitle('Recorded construction costs | 44 measured pairs',fontweight='bold');fig.text(.01,.01,'All measured build calls included. Time is observed under the recorded scheduling conditions; costs use the common normalization.',fontsize=8);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(H/'build_resources.png',dpi=180);fig.savefig(H/'build_resources.pdf');plt.close(fig)
def esc(x):return html.escape(str(x))
def table(scope):
 ss=summary[scope];head='<tr><th>Method</th>'+''.join('<th>'+x+'</th>' for x in LABELS)+ '<th>Macro average</th><th>Micro average</th></tr>'
 body=''
 for s in SYSTEMS:
  vals=[ss[s]['types'][t]['accuracy'] for t in TYPES]+[ss[s]['macro_accuracy'],ss[s]['micro_accuracy']]
  body+='<tr><th>'+s+'</th>'+''.join('<td>'+('—' if v is None else f'{v:.2f}')+'</td>' for v in vals)+'</tr>'
 return '<table>'+head+body+'</table>'
audit={r['sample_id']:r for r in read(H/'manual_error_audit.json')} if (H/'manual_error_audit.json').exists() else {}
wrong=[];details=''
for sid,p in pairs.items():
 m,s=p['MemNexus'],p['SimpleMem'];row={'id':sid,'category':m['category'],'question':m['question'],'gold':m['gold_answer'],'mn':m['prediction'],'sm':s['prediction'],'mn_judge':m['judge_result'],'sm_judge':s['judge_result'],'cause_status':'pending manual evidence audit'}
 if m['judge_result']=='WRONG':
  if sid in audit:row['cause_status']=audit[sid]
  wrong.append(row)
 details+='<details><summary>'+esc(sid)+' | '+esc(m['category'])+' | MN '+m['judge_result']+' / SM '+s['judge_result']+'</summary><p>'+esc(m['question'])+'</p><p><b>Gold:</b> '+esc(m['gold_answer'])+'</p><p><b>MN:</b> '+esc(m['prediction'])+'</p><p><b>SM:</b> '+esc(s['prediction'])+'</p>'
 if sid in audit:details+='<p><b>Verified failure stage:</b> '+esc(audit[sid]['stage'])+'</p><p>'+esc(audit[sid]['root_cause'])+'</p><p>'+esc(audit[sid]['evidence'])+'</p>'
 details+='</details>'
(H/'error_queue.json').write_text(json.dumps(wrong,ensure_ascii=False,indent=2),encoding='utf8')
page='<!doctype html><meta charset="utf-8"><title>LME-S matched comparison</title><style>body{max-width:1250px;margin:35px auto;padding:0 20px;color:#222;font:17px Georgia,"Microsoft YaHei",serif}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0;border-top:2px solid;border-bottom:2px solid}th,td{padding:10px;text-align:center}tr:first-child{border-bottom:1px solid}tr:nth-child(3){background:#f1f1f1}img{width:100%}details{border-top:1px solid #ddd;padding:10px}p{line-height:1.6}</style>'
state_label={'user_stopped':'已按用户要求停止，以下为最终完整配对结果','running':'测评进行中','handoff_complete':'批次切换中','quality_stop':'质量检查触发暂停','technical_stop':'技术检查暂停','budget_complete':'本阶段预算检查点','complete':'本阶段全部完成','account_budget_complete':'已到账户余额保护边界，测评结束'}.get(summary['status']['status'],summary['status']['status'])
page+=f'<h1>LongMemEval-S：{len(pairs)} 道配对结果</h1><p>状态：{esc(state_label)}。结果随完整配对更新，未完成的单侧答案不进入对比。</p><p>生成模型 deepseek-flash；评审沿用 SimpleMem Appendix A.4，DeepSeek 替代 GPT-4.1-mini。准确率为主指标；F1、BLEU-1为补充。QA token 覆盖全部规划、反思、生成与修复调用，不等同于论文仅检索上下文的 Token Cost。评审调用费用纳入实验总支出，单独记账，不计入任一系统的 QA 成本。</p>'
m,sm=summary['all']['MemNexus'],summary['all']['SimpleMem']
page+='<img src="final_overview.png" alt="Final paired quality and cost overview">'
page+='<h2>当前结果如何解读</h2><p>全部配对中，MN / SM 的宏平均准确率为 '+f'{m["macro_accuracy"]:.2f}% / {sm["macro_accuracy"]:.2f}%'+'，按题目等权的准确率为 '+f'{m["micro_accuracy"]:.2f}% / {sm["micro_accuracy"]:.2f}%'+'。MN 的 QA 输入 token 比 SM '+f'{(m["qa_input"]/sm["qa_input"]-1)*100:+.1f}%'+'，输出 token '+f'{(m["qa_output"]/sm["qa_output"]-1)*100:+.1f}%'+'，统一价格下 QA 费用 '+f'{(m["qa_cost"]/sm["qa_cost"]-1)*100:+.1f}%'+'。</p>'
page+='<p>目前最明确的贡献证据是 QA 资源节省及可追溯的原文引用、算术执行；是否质量更高，要看统一样本的配对差异与分题型结果。差异区间包含零时，不据此声称显著优于或已经证明非劣。</p>'
page+='<p>统一106题中，MN答对91题，SM答对92题；84题共同答对，7题仅MN答对，8题仅SM答对，7题共同答错。宏平均 Token F1 为53.03 / 52.21，BLEU-1为46.91 / 44.44。当前结果支持“接近的准确率、更好的词汇重合指标、更低的QA资源消耗”。</p>'
page+='<p>按题型看，MN在历史助手回答（18/18 对17/18）、跨会话（14/19 对13/19）与偏好（12/15 对11/15）上领先；知识更新（16/19 对18/19）是最大的净失分来源，时间推理（15/17 对16/17）和用户事实（16/18 对17/18）各少答对一题。后续优先完善新旧状态选择、时间语义和候选证据进入答案端的过程。</p>'
page+='<h2>逐题审计反映的主要缺口</h2><p>① 历史意图或习惯保留为会话短期记忆后，原文搜索虽有机会找到，仍可能被主检索和最终排序弱化；② 正确前提已经可见时，推断仍可能过度保守，或把条件不满足的背景数值放到主答案；③ 日期窗口、状态截止时点和请求单位没有始终一致；④ 个性化依据有时只出现在内部推导，未进入最终建议。修复方向是现有索引、排序和单一答案表的语义契约，不是新增检索分支或题型专用生成器。</p>'
if (H/'trace_audit.json').exists():
 tr=read(H/'trace_audit.json')
 page+='<h2>长期记忆实际参与情况</h2><p>已配对样本中，'+str(tr['long_visible_questions'])+'/'+str(tr['n'])+' 道答案端看到长期记忆，'+str(tr['long_cited_questions'])+'/'+str(tr['n'])+' 道引用了长期记忆。这证明长期记忆已经参与检索和回答；引用次数不能替代关闭长期记忆的消融实验，也不能独立证明准确率提升。未晋升但仍有用的用户意图与更新事实，需要通过现有原文回读通道保持可访问。</p>'
if (H/'final_accounting.json').exists():
 fa=read(H/'final_accounting.json')
 page+='<h2>停止与费用结算</h2><p>按用户要求停止所有实验进程，统一统计106道完整配对。另4道仅MN完成、SM建库未完成，不纳入任何配对质量或资源均值。全部调用回执按服务商时段与缓存价格估算 '+f'{fa["receipt_estimated_actual_rmb"]:.2f}'+' 元，其中未完成配对工作约 '+f'{fa["unmatched_work_estimate_rmb"]:.2f}'+' 元。停止后读取的账户余额为 '+f'{fa["account_balance_rmb"]:.2f}'+' 元。另有1条停止时在途请求未取得回执，其保守折算上限为 '+f'{fa["unresolved_normalized_upper_rmb"]:.3f}'+' 元；不视作免费，也不计入已核实的token汇总。</p><p>所有生产代码全程冻结，48个源文件哈希通过校验。仅修改了实验报告与审计工具；没有根据本轮错题修改答案、评分或生产逻辑。</p>'
page+='<h2>统一106道：分题型准确率（%）</h2>'+table('all')+'<img src="table.png"><h2>统一质量与资源对比</h2><img src="quality.png"><img src="resources.png">'
page+='<img src="efficiency.png"><p>每题 QA 调用次数（含修复）：'+', '.join(s+f' {summary["all"][s]["attempts"]/len(pairs):.2f}' for s in SYSTEMS)+'。</p>'
page+='<h2>可回答与不可回答题</h2><p>按数据集题目 ID 的 abs 标记分组，不以系统是否拒答分组。</p>'+table('answerable')+table('unanswerable')
page+='<p>各题型配对数：'+', '.join(f'{label}: {summary["all"]["MemNexus"]["types"][t]["n"]}' for label,t in zip(LABELS,TYPES))+ '。宏平均对六类等权，微平均对题目等权。</p>'
page+='<p>资源图费用统一按输入 2.4、输出 9.6 元/百万 token 归一化，便于比较，不代表账户扣款。实际价格估算另外依据缓存命中与时段计价。</p>'
page+='<p>延迟统一汇总全部106道；测试期间存在不同并行负载，因此这里展示实测分布，不将其解释为隔离环境的加速比。</p><table><tr><th>Method</th><th>n</th><th>Median QA seconds</th><th>P95 QA seconds</th></tr>'
for system in SYSTEMS:
 v=summary['all'][system]
 page+=f'<tr><td>{system}</td><td>{v["n"]}</td><td>{v["latency_median_s"]:.3f}</td><td>{v["latency_p95_s"]:.3f}</td></tr>'
page+='</table>'
page+='<h2>建库成本记录（有完整测量的44道；不外推到106道）</h2><table><tr><th>Method</th><th>n</th><th>Input tokens / question</th><th>Output tokens / question</th><th>Normalized RMB / question</th><th>Calls / question</th></tr>'
for s in SYSTEMS:
 b=summary['measured_build'][s];n=b['n']
 page+='<tr><td>'+s+'</td><td>'+str(n)+'</td>'+''.join('<td>'+ (f'{b[k]/n:,.3f}' if n else '—')+'</td>' for k in ['input_tokens','output_tokens','normalized_rmb','provider_attempts'])+'</tr>'
page+='</table>'
if summary['measured_build']['MemNexus']['n']:page+='<img src="build_resources.png">'
if 'provider_billing_estimate' in summary:page+='<p>截至计价审计时，有 token 回执调用按官网价格估算合计 '+f'{summary["provider_billing_estimate"]["estimated_receipted_rmb"]:.4f}'+' 元；未含缺少回执的未知扣费。归一化预算账本另计。</p>'
page+='<h2>判对样本中仍需改善的证据链</h2><p>以下问题来自原始证据和答案轨迹核查；保留正式盲评结果，不改写准确率。判对并不表示检索和输出完整无缺。</p>'
for name,label in [('fresh_temporal_audit.json','时间区间与状态时点混用'),('fresh_preference_audit.json','相关用户前提被上下文预算排除'),('fresh_unit_audit.json','确定性计算覆盖了请求单位'),('fresh_update_audit.json','新旧状态均可见但主答案优先级不清')]:
 if (H/name).exists():
  a=read(H/name)
  if a['sample_id'] in pairs:
   page+='<details><summary>'+esc(a['sample_id'])+' | '+label+'</summary><pre style="white-space:pre-wrap;font-size:14px">'+esc(json.dumps(a,ensure_ascii=False,indent=2))+'</pre></details>'
if (H/'error_analysis_zh.json').exists():
 page+='<h2>15道 MN 错题的中文诊断</h2><p>以下是本轮冻结版本的轨迹分析，尚未据此修改生产代码。其中原文时间或指代不明确的题目，保留证据歧义，不为匹配参考答案强行补事实。</p><table><tr><th>题目</th><th>环节</th><th>观察到的问题</th><th>后续方向</th></tr>'
 for a in read(H/'error_analysis_zh.json'):page+='<tr><td>'+esc(a['topic'])+'<br><small>'+esc(a['id'])+'</small></td><td>'+esc(a['stage'])+'</td><td>'+esc(a['cause'])+'</td><td>'+esc(a['next_step'])+'</td></tr>'
 page+='</table>'
page+='<h2>配对差异区间</h2><table><tr><th>Population</th><th>Macro accuracy delta (MN−SM), pp</th><th>Stratified paired bootstrap95% interval, pp</th></tr>'
for scope,v in summary['stratified_paired_macro'].items():page+='<tr><td>'+scope+'</td><td>'+f'{v["delta_pp"]:.2f}'+'</td><td>'+esc([round(x,2) for x in v['bootstrap_95ci_pp']])+'</td></tr>'
page+='</table><p>上述宏平均区间按题型分层配对重采样。下述区间为按题目等权的微平均差异，两者不混用；均为探索性描述。</p>'
page+='<p>均值差 '+f'{summary["paired_micro_delta_pp"]:.2f} pp；配对 bootstrap 95% 区间 '+esc(summary['paired_micro_bootstrap_95ci_pp'])+'。检查点为开发阶段的探索性停止判断，不作为反复检验后的正式显著性结论。</p><h2>逐题答案</h2>'+details
(H/'report.html').write_text(page,encoding='utf8');print(json.dumps({'pairs':len(pairs),'wrong_mn':len(wrong),'report':str(H/'report.html')}))

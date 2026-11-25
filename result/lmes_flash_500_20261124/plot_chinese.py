import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
S=json.loads((H/'summary.json').read_text(encoding='utf8'))
D=json.loads((H.parent/'evidence_chain_20261124/v3_summary.json').read_text(encoding='utf8'))
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
types=['temporal-reasoning','multi-session','knowledge-update','single-session-user','single-session-assistant','single-session-preference']
labels=['时间推理','跨会话','知识更新','用户事实','助手历史回答','用户偏好']
colors=['#669FC5','#DB9974']
def save(fig,name):
 fig.savefig(H/(name+'.png'),dpi=180,facecolor='white');fig.savefig(H/(name+'.pdf'));plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(18,6))
for ax,key,title in zip(axes,['accuracy','f1','bleu1'],['准确率（主要指标）','词元 F1（补充指标）','BLEU-1（补充指标）']):
 x=np.arange(6)
 for i,s in enumerate(['MemNexus','SimpleMem']):
  vals=[S['all'][s]['types'][t][key] for t in types]
  b=ax.bar(x+(i-.5)*.38,vals,.38,color=colors[i],label=s)
  ax.bar_label(b,fmt='%.1f',fontsize=9,padding=3)
 ax.set_xticks(x,[f'{l}\n（{S["all"]["MemNexus"]["types"][t]["n"]}题）' for l,t in zip(labels,types)],rotation=30,ha='right')
 ax.set_ylim(0,113);ax.set_yticks(range(0,101,20));ax.set_ylabel('得分（%）');ax.set_title(title,pad=15);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('LongMemEval-S：优化后 MN 与 SM 分题型对比｜统一106道完整配对',fontsize=19,y=.98)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.90),ncol=2,frameon=False)
fig.text(.03,.025,'生成与替代盲评均使用同一 DeepSeek 模型。按题型统计，不按建库批次拆分；本图反映当前表现，不直接估计优化收益。',fontsize=11)
fig.tight_layout(rect=(0,.09,1,.83));save(fig,'task_comparison_zh')
q=D['all19']['questions'];fig,axes=plt.subplots(1,3,figsize=(18,6))
audit={}
for ax,key,title in zip(axes,['accuracy','f1','bleu1'],['准确率变化','词元 F1 变化','BLEU-1 变化']):
 x=np.arange(6)
 for i,(arm,label) in enumerate([('before','MN 优化前'),('after','MN 优化后（V3）')]):
  vals=[]
  for t in types:
   rows=[r['arms'][arm] for r in q if r['category']==t]
   vals.append(np.mean([100*(r['judge_result']=='CORRECT') if key=='accuracy' else 100*r[key] for r in rows]) if rows else np.nan)
  b=ax.bar(x+(i-.5)*.38,vals,.38,color=colors[i],label=label);ax.bar_label(b,fmt='%.1f',fontsize=9,padding=3)
  audit[arm+'_'+key]=vals
 ax.set_xticks(x,[f'{l}\n（{sum(r["category"]==t for r in q)}题）' for l,t in zip(labels,types)],rotation=30,ha='right')
 ax.set_ylim(0,113);ax.set_yticks(range(0,101,20));ax.set_ylabel('得分（%）');ax.set_title(title,pad=15);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('上次优化的直接变化：同19道开发题、优化前后答案统一重评',fontsize=19,y=.98)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.90),ncol=2,frameon=False)
fig.text(.03,.025,'这是已暴露开发题的回归验证，不是106题的前后对照。各类型样本较少；柱高变化仅描述这组题的实测结果。',fontsize=11)
fig.tight_layout(rect=(0,.09,1,.83));save(fig,'optimization_change_zh')
(H/'optimization_change_data.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(audit,ensure_ascii=False))

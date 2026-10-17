"""Offline audit of all prior 100 QA rows plus current prompt serialization.
Gold is used here ONLY for retrospective audit, never passed to the reader.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'MemNexus/memory-system/src'))

from agent_memory.models import AnswerContract, Hit, QueryPlan
from agent_memory.providers import CallableModel

OLD = ROOT/'result/benchmark_locomo100_timefix_v2_20261016'
OUT = ROOT/'result/compact_reader_audit_20261017'

CAUSES = {
 ('conv-26',11): ('已有两段支持但未完成指代连接', 'home country与Sweden均在来源，回答停留在泛称。应保持原文顺序并要求输出具体指代对象。'),
 ('conv-26',161): ('标签与原文冲突', 'Caroline原文明说碗提醒她art and self-expression；不能为了对抗标签抹去事实。保留原标签，列人工复核。'),
 ('conv-30',44): ('检索证据不足', '来源中没有Gina评价舞者的关键句；Gold证据D1:26本身是Jon，正确Gina句在D1:25/27。不能把此题仅归因于reader拒答。'),
 ('conv-41',8): ('推断问题被按明确属性处理', '孩子拥有很多物品是弱间接证据，不等于明确财务状态。允许有依据的模态推断，但不能制造收入。'),
 ('conv-42',4): ('参考答案推断边界争议', '没有皮毛不保证过敏患者安全。不能为评分硬编码安全宠物或删除医学证据约束。'),
 ('conv-42',11): ('集合证据覆盖与关系语义', '缺dairy；原文只说不能吃乳制品，不明确是过敏。须区分未召回与诊断泛化，不把饮食限制自动改写为过敏。'),
 ('conv-42',82): ('粒度重复与参考列表范围', '列出具体冰淇淋又重复泛称ice cream；另有原文支持的蛋糕却不在gold。应去重复概念，保持额外事实的证据而不修改gold。'),
 ('conv-43',86): ('主体前提冲突及错误事件替代', '原文Skype说话者是Tim，题问John；本轮回答篮球照片属于另一个事件。应保持事件限定，必要时解释主体不符而非转移事实。'),
 ('conv-48',17): ('已有来源但跨句指代未消解', 'Seraphim和I bought it a year ago均已到位；缺少顺序化源句联结，非日历算术问题。'),
 ('conv-48',23): ('错误路由与推断证据覆盖', 'times正则误命中sometimes，走count。游戏偏好句未进初次上下文；因果解释本身也是推断而非明确陈述。'),
 ('conv-48',28): ('答案类型投影错误', '已有Bogota但问country，返回城市。通用要求推断所问属性且保留推断依据，禁止地名专用补丁。'),
 ('conv-48',78): ('证据映射/输入时间范围争议', 'gold证据D2:1是父亲去世，与Phuket无关；不得从gold引入不存在的地点。先复核数据而非调reader迎合。'),
 ('conv-49',24): ('主体与Gold证据不一致', 'gold恋爱句说话者是Evan，问题问Sam。本轮误答其他Sam事件；需要拒绝错误事件替代，不能把Evan经历转给Sam。'),
 ('conv-49',26): ('分散前提遗漏', 'family在当前来源，tracker代词和hikes分散在相邻后续turn。需要来源顺序、局部补全和完整动机投影。'),
 ('conv-49',97): ('已有完整来源但原因输出不全', '同一来源同时含象征意义和motivates through tough times，只抽出前半句；需保留所问因果结果。'),
 ('conv-50',7): ('图像线索到人数的弱推断', 'caption只描述一群人站车前，未明确雇员数；不能把集体照片当员工规模证明。Gold=Yes需单列推断边界。'),
}


class Captured(Exception):
    pass


class Probe(CallableModel):
    async def _chat_structured(self, prompt):
        self.captured = prompt
        raise Captured()


async def main():
    OUT.mkdir(exist_ok=True)
    rows=json.loads((OLD/'results.json').read_text(encoding='utf-8'))['rows']
    raw={s['sample_id']:s for s in json.loads((ROOT/'benchmark-data/official/locomo/locomo10.json').read_text(encoding='utf-8'))}
    index={(r['sample_id'],r['qa_index'],r['system']):r for r in rows}
    audit=[]; costs=[]
    for r in rows:
        if r['system']!='MemNexus':continue
        s=raw[r['sample_id']]; q=s['qa'][r['qa_index']]
        turns={t['dia_id']:t for v in s['conversation'].values() if isinstance(v,list) for t in v}
        sources=[Hit.model_validate(h) for h in r['retrieved_sources']]
        texts=[e['text'] for h in sources for e in h.metadata.get('source_events',[])]
        evidence=[]
        for eid in q.get('evidence',[]):
            t=turns.get(eid,{})
            text=t.get('text','')
            evidence.append({'id':eid,'speaker':t.get('speaker'),'text':text,
                             'full_utterance_in_context':bool(text) and any(text in x for x in texts)})
        cause,action=CAUSES.get((r['sample_id'],r['qa_index']),('本轮正确，保留原路径语义','不做逐题特化调整'))
        audit.append({'id':r['item_id'],'category':r['category'],'question':r['question'],'gold':r['gold_answer'],
                      'prediction':r['prediction'],'result':r['judge_result'],
                      'f1':r['f1'], 'baseline_f1':index[r['sample_id'],r['qa_index'],'SimpleMem']['f1'],
                      'baseline_result':index[r['sample_id'],r['qa_index'],'SimpleMem']['judge_result'],
                      'baseline_prediction':index[r['sample_id'],r['qa_index'],'SimpleMem']['prediction'],
                      'cause':cause,'action':action,'gold_evidence_audit':evidence})
        probe=Probe(lambda _: '')
        plan=QueryPlan.model_validate(r['plan'])
        try:
            if r.get('reasoning',{}).get('answer_path')=='semantic_execution':
                contract=AnswerContract.model_validate(r['reasoning']['answer_contract'])
                await probe.bind_answer(r['evaluation_query'],contract,sources)
            else:
                await probe.synthesize(r['evaluation_query'],plan,sources)
        except Captured:
            pass
        old_chars=r.get('reasoning',{}).get('provider_request_budget',{}).get('prompt_chars',[])
        # Final stored sources can include reread expansion, so this is NOT a
        # controlled before/after comparison of the first reader request.
        old_chars=old_chars[1] if len(old_chars)>1 else None
        costs.append({'id':r['item_id'],'old_initial_reader_chars':old_chars,'new_reader_chars':len(probe.captured),
                      'measurement':'new prompt on FINAL stored sources; old size may be INITIAL subset; not comparable savings or provider tokens'})
    out={'status':'ANALYZED','audit_rows':audit,'prompt_probe':costs,'wrong_count':sum(r['result']=='WRONG' for r in audit)}
    (OUT/'audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 100题逐项审计：压缩阅读路径优化前','', '## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: validate',
           '- Origin Date: 2026-10-17','- Verification Status: ANALYZED','- Version Label: compact_reader_audit_v1','',
           '本文件逐项审计原100题；原始标签不修改。错误归因是来源/轨迹支持的诊断，不是因果消融或新benchmark分数。',
           'full_utterance_in_context仅检查完整原句是否出现；False不能证明相关部分完全缺失。', '',
           '|题型|题数|MemNexus正确|SimpleMem正确|','|---|---:|---:|---:|']
    for cat in sorted({a['category'] for a in audit}):
        a=[a for a in audit if a['category']==cat]
        lines.append(f"|{cat}|{len(a)}|{sum(x['result']=='CORRECT' for x in a)}|{sum(x['baseline_result']=='CORRECT' for x in a)}|")
    lines += ['', '## 各类别 F1 与答案长度', '',
              '|类别|SimpleMem F1|MemNexus F1|SimpleMem平均词数|MemNexus平均词数|Gold平均词数|',
              '|---|---:|---:|---:|---:|---:|']
    words = lambda text: len(re.findall(r'\w+',str(text)))
    for cat in sorted({a['category'] for a in audit}):
        a=[a for a in audit if a['category']==cat]
        lines.append(f"|{cat}|{sum(x['baseline_f1'] for x in a)/len(a):.4f}|{sum(x['f1'] for x in a)/len(a):.4f}|"
                     f"{sum(words(x['baseline_prediction']) for x in a)/len(a):.1f}|{sum(words(x['prediction']) for x in a)/len(a):.1f}|"
                     f"{sum(words(x['gold']) for x in a)/len(a):.1f}|")
    lines += ['', '单跳F1本轮已高于SimpleMem，不应称所有类别F1退步。时间F1受日期词面格式影响很大；',
              '多跳答案较长，包含过宽集合或解释；开放域还涉及推断边界、漏召回和参考答案争议。长度本身不是错误证明。']
    for a in audit:
        if a['result']=='CORRECT':continue
        lines += ['',f"## {a['id']} / {a['category']}",'',a['question'],f"Gold：{a['gold']}",
                  f"MemNexus：{a['prediction']}",f"SimpleMem（{a['baseline_result']}）：{a['baseline_prediction']}",
                  f"诊断：{a['cause']}。{a['action']}",*['原始证据 '+str(e) for e in a['gold_evidence_audit']]]
    lines += ['', '## 判对但F1低于基线超过0.15的题目', '',
              '这些例子用于区分语义正确、完整性和词面匹配；不改写历史预测或gold。']
    for a in audit:
        if a['result']=='CORRECT' and a['baseline_f1']-a['f1']>.15:
            lines += ['',f"### {a['id']} / {a['category']}",a['question'],f"Gold：{a['gold']}",
                      f"MemNexus F1={a['f1']:.3f}：{a['prediction']}",
                      f"SimpleMem F1={a['baseline_f1']:.3f}：{a['baseline_prediction']}"]
    comparable=[r for r in costs if r['old_initial_reader_chars']]
    lines += ['', '## 无付费序列化检查','',f"成功生成100题当前阅读提示。只有{len(comparable)}题记录旧请求字符数，且保存的最终来源可能多于初次来源；不据此宣称节省。实际token和质量由独立冻结实验测量。",'']
    (OUT/'audit.md').write_text('\n'.join(lines),encoding='utf-8')
    print('audited',len(audit),'wrong',out['wrong_count'],'comparable_reader_prompts',len(comparable))
    print('chars_before_after',sum(r['old_initial_reader_chars'] for r in comparable),sum(r['new_reader_chars'] for r in comparable))


if __name__=='__main__':asyncio.run(main())

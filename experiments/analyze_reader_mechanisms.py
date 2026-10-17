"""Offline fixed-evidence probes. Never modifies benchmark rows or calls an LLM."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'MemNexus/memory-system/src'))
from agent_memory.grounding import source_payload
from agent_memory.models import Hit
from agent_memory.prompt_codec import evidence_catalog
from agent_memory.synthesis import AnswerTable, answer_type, render_table, validate_table
from benchmark_mixed200 import token_f1


def main():
    folder = ROOT / 'result/compact_reader_audit_20261017'
    rows = json.loads((ROOT / 'result/benchmark_locomo100_timefix_v2_20261016/results.json').read_text(encoding='utf-8'))['rows']
    stats = defaultdict(lambda: defaultdict(float))
    probes = []
    for row in rows:
        if row['system'] != 'MemNexus':
            continue
        sources = [Hit.model_validate(h) for h in row['retrieved_sources']]
        kind = answer_type(row['evaluation_query'])
        old = {'sources': [{'source_id': i, **source_payload(h, include_time=kind in {'date','duration'})}
                           for i,h in enumerate(sources,1)]}
        new = evidence_catalog(sources, include_time=kind in {'date','duration'})
        size = lambda x: len(json.dumps(x, ensure_ascii=False, separators=(',',':')))
        probe = {'id': row['item_id'], 'category': row['category'],
                 'same_sources_old_chars': size(old), 'same_sources_catalog_chars': size(new)}
        # Only already-committed, source-bound tables; no model-generated update.
        table_data = row.get('reasoning',{}).get('answer_table')
        if table_data and kind == 'date':
            table = AnswerTable.model_validate(table_data)
            try:
                validate_table(table, sources, [s.target for s in table.slots], 'date')
                rendered = render_table(table)
                probe.update(old_prediction=row['prediction'], presentation_replay=rendered,
                             old_f1=row['f1'], presentation_f1=token_f1(rendered,row['gold_answer']))
            except ValueError as error:
                probe['replay_error'] = str(error)
        probes.append(probe)
        s = stats[row['category']]
        s['n'] += 1
        s['old_chars'] += size(old)
        s['new_chars'] += size(new)
    result = {'status':'ANALYZED', 'protocol':'Same final evidence; compression bytes only. Time replay uses stored answer table, no new answers or judgments.',
              'categories':dict(stats), 'rows':probes}
    (folder/'mechanism_probes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    replay = [p for p in probes if 'presentation_f1' in p]
    lines=['# 固定证据的无API机制检查','',
           '这是历史轨迹上的受控序列化/呈现检查，不是新模型正确率。原实验预测与裁判均不修改。', '',
           '|类别|题数|旧证据紧凑JSON字符|事件目录字符|变化|','|---|---:|---:|---:|---:|']
    for category, s in stats.items():
        lines.append(f"|{category}|{int(s['n'])}|{int(s['old_chars'])}|{int(s['new_chars'])}|{s['new_chars']/s['old_chars']-1:+.1%}|")
    lines += ['',f'可重放的日期答案表：{len(replay)}题。只改变呈现/保留原证据的本地日期执行，未请求新答案，也未重新判定语义正确率。',
              f"这些题原token F1均值={sum(p['old_f1'] for p in replay)/len(replay):.4f}；呈现后={sum(p['presentation_f1'] for p in replay)/len(replay):.4f}。",'']
    for p in replay:
        if p['old_prediction'] != p['presentation_replay']:
            lines.append(f"- {p['id']}: `{p['old_prediction']}` → `{p['presentation_replay']}`；F1 {p['old_f1']:.3f} → {p['presentation_f1']:.3f}")
    lines += ['','字数不是provider tokenizer的token数；最终是否节省以及是否损害质量，需要冻结真实对比验证。']
    (folder/'mechanism_probes.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'evidence_chars_before':sum(p['same_sources_old_chars'] for p in probes),
                      'evidence_chars_after':sum(p['same_sources_catalog_chars'] for p in probes),
                      'date_replays':len(replay), 'replay_old_f1':sum(p['old_f1'] for p in replay)/len(replay),
                      'replay_new_f1':sum(p['presentation_f1'] for p in replay)/len(replay)},indent=2))


if __name__ == '__main__':
    main()

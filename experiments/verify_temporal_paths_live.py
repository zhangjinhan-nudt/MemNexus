"""Small paid path verification on saved evidence; NOT a benchmark or accuracy estimate."""
import asyncio
import json
import os
from pathlib import Path

from agent_memory.models import Hit, QueryPlan
from agent_memory.providers import answer_call_budget, build_model
from agent_memory.settings import Settings
from agent_memory.synthesis import render_table

ROOT = Path(__file__).resolve().parents[1]


async def main():
    folder = ROOT / 'result/temporal_path_verification_20261016'
    folder.mkdir(exist_ok=True)
    fixtures = [
        ('year', ROOT/'result/benchmark_locomo100_optimized_20261016/attempts/locomo-conv-43_memnexus__001/usage.calls.jsonl'),
        ('duration', ROOT/'result/benchmark_locomo100_timefix_20261016/attempts/locomo-conv-43_memnexus__001/usage.calls.jsonl'),
    ]
    for name, path in fixtures:
        output = folder/f'{name}.json'
        if output.exists():
            raise FileExistsError(output)
        events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        requests = [e for e in events if e['event']=='request']
        prompt = requests[-1]['body']['messages'][0]['content']
        payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"question":'):])
        for source in payload['sources']:
            for event in source['source_events']:
                event.setdefault('time_derivations', [])
        sources = [Hit(content=s['claim'], score=1, source_file='saved_evidence', chunk_id=str(s['source_id']),
                       metadata={'source_events': s['source_events']}) for s in payload['sources']]
        os.environ['EXPERIMENT_USAGE_LEDGER_PATH'] = str(folder/f'{name}_usage.json')
        os.environ['EXPERIMENT_COMPONENT_BUDGET_RMB'] = '10'
        os.environ['MEMORY_INPUT_PRICE_RMB_PER_MILLION_TOKENS'] = '2.4'
        os.environ['MEMORY_OUTPUT_PRICE_RMB_PER_MILLION_TOKENS'] = '9.6'
        model = build_model(Settings(model_timeout=180))
        try:
            with answer_call_budget(True, max_calls=2) as budget:
                budget['active'] = True
                table = await model.synthesize(payload['question'], QueryPlan(required_info=payload['targets']), sources)
                result = {'scope': 'saved-evidence path verification, not accuracy', 'question': payload['question'],
                          'table': table.model_dump(mode='json'), 'prediction': render_table(table),
                          'usage': model.usage, 'answer_budget': budget}
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                print(name, result['prediction'], 'calls', budget['calls'], flush=True)
        finally:
            await model.aclose()


if __name__ == '__main__':
    asyncio.run(main())

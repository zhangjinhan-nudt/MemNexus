"""Offline replay of the actually failed extraction, not a QA benchmark score."""
import hashlib
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'MemNexus/memory-system/src'))
from agent_memory.models import ExtractionResult,ExtractionWindow
from agent_memory.providers import _anchor_literal_quotes,_validate_evidence,_parse_response

folder=ROOT/'result/benchmark_compact_lme6_v3_20261017/attempts/longmemeval-7161e7e2_memnexus__001'
parser=argparse.ArgumentParser()
parser.add_argument('--ledger',type=Path,default=folder/'usage.calls.jsonl')
parser.add_argument('--output',type=Path,default=ROOT/'result/compact_reader_audit_20261017/span_replay.json')
args=parser.parse_args()
path=args.ledger
events=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
req=next(r for r in reversed(events) if r['event']=='request')
response=next(r for r in reversed(events) if r['event']=='response')
data,_=json.JSONDecoder().raw_decode(req['body']['messages'][0]['content'].split('输入数据：')[1].lstrip())
window=ExtractionWindow.model_validate(data)
transport=[]
result=_parse_response(response['body']['choices'][0]['message']['content'],ExtractionResult,trace=transport.append)
values=lambda r:[(c.value,[m.value for m in c.semantics.members]) for c in r.candidates]
before=values(result)
receipts=_anchor_literal_quotes(window,result)
os.environ['MEMORY_ALLOW_ASSISTANT_EVIDENCE']='true'
_validate_evidence(window,result)
assert values(result)==before
out={'kind':'offline_source_alignment_replay_not_benchmark','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
     'original_failed_attempt':response['attempt'],'candidate_count':len(result.candidates),
     'member_count':sum(len(c.semantics.members) for c in result.candidates),
     'values_unchanged':True,'strict_validation_passed':True,'receipts':receipts,'transport_receipts':transport}
target=args.output
target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))

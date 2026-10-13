"""Offline transfer checks of raw-evidence fallback, never QA-accuracy claims.

LoCoMo questions exclude the previously tuned 100-question cohort. LongMemEval-S
uses 50 hash-selected complete haystacks. Gold labels are consulted only after
retrieval; question categories and has_answer are never retrieval inputs.
"""

import hashlib
import json
import statistics
import sys
import argparse
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus/memory-system/src"))

from agent_memory.grounding import collect_raw_records, ground_evidence, record_key
from agent_memory.models import Event, QueryPlan, Turn
from agent_memory.retrieval import _cosine
from locomo_100_protocol import available_ids, evidence_ids


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measure(query, turns, unit_by_turn, gold):
    semantic = None
    if DENSE_CLIENT is not None:
        plan = QueryPlan(required_info=[query])
        records = collect_raw_records(query, plan, turns, "eval", set())
        def encode(texts, role):
            missing = list(dict.fromkeys(text for text in texts if (role, text) not in VECTOR_CACHE))
            for start in range(0, len(missing), 32):
                batch = missing[start:start + 32]
                response = DENSE_CLIENT.post("http://127.0.0.1:8099/v1/embeddings", json={
                    "model": "Qwen/Qwen3-Embedding-0.6B", "input": batch, "input_type": role})
                response.raise_for_status()
                vectors = sorted(response.json()["data"], key=lambda row: row["index"])
                for text, vector in zip(batch, vectors, strict=True):
                    VECTOR_CACHE[role, text] = vector["embedding"]
            return [VECTOR_CACHE[role, text] for text in texts]
        query_vector = encode([query], "query")[0]
        document_vectors = encode([text for _, text in records], "document")
        semantic = {record_key(record): _cosine(query_vector, vector) for (record, _), vector in zip(records, document_vectors)}
    result = ground_evidence(query, QueryPlan(required_info=[query]), [],
                             session_id="eval", current_turns=turns, resolved_turns={},
                             blocked_events=set(), raw_limit=3, max_sources=3, budget=5000,
                             semantic_scores=semantic)
    selected = [unit_by_turn[e["turn_id"]] for hit in result.sources for e in hit.metadata["source_events"]]
    matched = set(selected) & set(gold)
    return {
        "gold_units": sorted(set(gold)), "retrieved_units": selected,
        "evidence_recall_at_3": len(matched) / len(set(gold)) if gold else None,
        "all_evidence_found": bool(gold) and matched == set(gold),
        "elapsed_ms": result.trace.elapsed_ms,
        "context_characters": result.trace.context_tokens_estimate,
        "rounds": result.trace.supplemental_rounds,
    }


def locomo():
    dataset = ROOT / "benchmark-data/official/locomo/locomo10.json"
    data = read(dataset)
    used = {(item["sample_id"], item["qa_index"]) for item in read(ROOT / "result/locomo_100_cohort.json")["questions"]}
    rows = []
    for sample in data:
        turns, mapping = [], {}
        for session_index in (1, 2):
            key = f"session_{session_index}"
            anchor = datetime.strptime(sample["conversation"][key + "_date_time"], "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC)
            for original in sample["conversation"][key]:
                identifier = original["dia_id"]
                text = original.get("text", "")
                if original.get("blip_caption"):
                    text += " [Image: " + original["blip_caption"] + "]"
                turns.append(Turn(turn_id=identifier, session_id="eval", request_id=identifier, sequence=len(turns) + 1,
                                  events=[Event(role="user", speaker=original["speaker"], content=text, occurred_at=anchor)]))
                mapping[identifier] = identifier
        valid = available_ids(sample)
        for index, qa in enumerate(sample["qa"]):
            refs = evidence_ids(qa)
            if (sample["sample_id"], index) in used or not refs or not set(refs) <= valid:
                continue
            rows.append({"sample_id": sample["sample_id"], "qa_index": index,
                         **measure(qa["question"], turns, mapping, refs)})
    return rows


def longmemeval():
    data = read(ROOT / "benchmark-data/official/LongMemEval/longmemeval_s_cleaned.json")
    data.sort(key=lambda row: hashlib.sha256(("grounding-v1:" + row["question_id"]).encode()).hexdigest())
    rows = []
    for item in data[:50]:
        turns, mapping = [], {}
        for si, session in enumerate(item["haystack_sessions"]):
            anchor = datetime.strptime(item["haystack_dates"][si], "%Y/%m/%d (%a) %H:%M").replace(tzinfo=UTC)
            for ti, original in enumerate(session):
                text = original.get("content", "")
                if not text.strip():
                    continue
                for offset in range(0, len(text), 16000):
                    identifier = f"{si}:{ti}:{offset}"
                    turns.append(Turn(turn_id=identifier, session_id="eval", request_id=identifier, sequence=len(turns) + 1,
                                      events=[Event(role=original["role"], speaker=original["role"],
                                                    content=text[offset:offset + 16000], occurred_at=anchor)]))
                    mapping[identifier] = item["haystack_session_ids"][si]
        rows.append({"question_id": item["question_id"], "question_type": item["question_type"],
                     "raw_turns": len(turns),
                     **measure(item["question"], turns, mapping, item["answer_session_ids"])})
        print(json.dumps({"dataset": "LongMemEval-S", "completed": len(rows)}), flush=True)
    return rows


def summary(rows):
    scored = [row for row in rows if row["evidence_recall_at_3"] is not None]
    return {"questions": len(rows), "with_gold_evidence": len(scored),
            "mean_evidence_recall_at_3": statistics.mean(row["evidence_recall_at_3"] for row in scored),
            "all_evidence_found": sum(row["all_evidence_found"] for row in scored),
            "p50_local_ms": statistics.median(row["elapsed_ms"] for row in rows),
            "max_context_characters": max(row["context_characters"] for row in rows)}


DENSE_CLIENT = None
VECTOR_CACHE = {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-locomo", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / ("result/grounding_hybrid_validation.json" if args.dense_locomo else "result/grounding_offline_validation.json")
    if output.exists():
        raise FileExistsError("Specify a new --output to preserve existing validation evidence")
    if args.dense_locomo:
        import httpx
        DENSE_CLIENT = httpx.Client(timeout=180)
    try:
        lo = locomo()
    finally:
        if DENSE_CLIENT is not None:
            DENSE_CLIENT.close()
            DENSE_CLIENT = None
    lm = longmemeval()
    report = {"kind": "offline raw fallback component validation; not answer accuracy",
              "paid_api_calls": 0, "cost_rmb": 0,
              "locomo_raw_retrieval": "BM25+Qwen3 RRF" if args.dense_locomo else "BM25",
              "longmemeval_raw_retrieval": "BM25",
              "source_sha256": {name: hashlib.sha256((ROOT / "MemNexus/memory-system/src/agent_memory" / name).read_bytes()).hexdigest()
                                for name in ("grounding.py", "temporal.py", "evidence_policy.py")},
              "locomo_outside_development_100": {"summary": summary(lo), "rows": lo},
              "longmemeval_s_hash_selected_50": {"summary": summary(lm), "rows": lm},
              "limitations": ["No compressed-memory input in this component test", "Local component experiment, no generated answers",
                              "LoCoMo uses turn-level evidence, LongMemEval uses session-level evidence",
                              "LoCoMo D1-D2 horizon is evidence-selected, not a full-history result"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"locomo": summary(lo), "longmemeval_s": summary(lm)}, ensure_ascii=False))

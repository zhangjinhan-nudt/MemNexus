"""Offline source-coverage replay for category 1; no gold enters retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus/memory-system/src"))

from agent_memory.grounding import ground_evidence
from agent_memory.models import Hit, Memory, QueryPlan, Turn
from agent_memory.reasoning import (
    apply_route,
    classify_question,
    indexed_memory_hits,
)

SOURCE = ROOT / "result/locomo_100_disjoint_v2_complete.json"
OUTPUT = ROOT / "result/locomo_100_multihop_source_replay.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--budget", type=int, default=7000)
    parser.add_argument("--raw-limit", type=int, default=6)
    args = parser.parse_args()
    source_path = args.source.resolve()
    report = read(source_path)
    folder = source_path.with_name(source_path.stem + "_ledgers")
    rows = []
    for row in report["rows"]:
        if row["system"] != "MemNexus" or row["category_id"] != 1:
            continue
        sample_id = row["sample_id"]
        snapshot = read(folder / f"{sample_id}_memnexus.memory.json")
        simple_path = folder / f"{sample_id}_simplemem.memory.json"
        if not simple_path.exists():
            simple_path = ROOT / "result/locomo_100_disjoint_v2_complete_ledgers" / f"{sample_id}_simplemem.memory.json"
        raw = read(simple_path)["raw_turns"]
        turns = [Turn.model_validate(turn) for turn in snapshot["turns"]]
        memories = [Memory.model_validate(memory) for memory in snapshot["memories"]]
        assert len(turns) == len(raw)
        by_id = {turn.turn_id: turn for turn in turns}
        official_id = {
            turn.turn_id: source["dia_id"]
            for turn, source in zip(turns, raw, strict=True)
        }
        known = [
            event.speaker for turn in turns for event in turn.events if event.speaker
        ]
        route = classify_question(row["evaluation_query"], known)
        plan = apply_route(QueryPlan.model_validate(row["plan"]), route)
        original = [Hit.model_validate(hit) for hit in row["retrieved_sources"]]
        eligible = [memory for memory in memories if memory.status == "active"]
        extra = indexed_memory_hits(row["evaluation_query"], route, eligible, original)
        recovered = ground_evidence(
            row["evaluation_query"],
            plan,
            [*original, *extra],
            session_id=turns[0].session_id,
            current_turns=turns,
            resolved_turns=by_id,
            blocked_events=set(),
            max_sources=20,
            raw_limit=args.raw_limit if route.recall_policy == "recall_first" else 3,
            budget=args.budget if route.recall_policy == "recall_first" else 5000,
            candidate_limit=120,
        )

        def source_ids(hits, mapping=official_id):
            return {
                mapping[event["turn_id"]]
                for hit in hits
                for event in hit.metadata.get("source_events", [])
                if event["turn_id"] in mapping
            }

        gold = set(row["gold_evidence"])
        before = source_ids(original)
        indexed_gold = {
            official_id[evidence["turn_id"]]
            for hit in extra
            for evidence in hit.metadata.get("evidence", [])
            if evidence["turn_id"] in official_id
        }
        after = source_ids(recovered.sources)
        rows.append(
            {
                "sample_id": sample_id,
                "qa_index": row["qa_index"],
                "question": row["question"],
                "old_judge_result": row["judge_result"],
                "route": route.kind,
                "old_sources": len(original),
                "indexed_candidates": len(extra),
                "new_sources": len(recovered.sources),
                "indexed_selected": recovered.trace.indexed_sources_selected,
                "raw_selected": recovered.trace.raw_sources_selected,
                "old_any_gold": bool(before & gold),
                "old_all_gold": gold <= before,
                "new_any_gold": bool(after & gold),
                "new_all_gold": gold <= after,
                "old_gold_count": len(before & gold),
                "new_gold_count": len(after & gold),
                "indexed_gold_count": len(indexed_gold & gold),
                "candidate_gold_count": len((before | indexed_gold) & gold),
                "gold_count": len(gold),
                "context_characters": recovered.trace.context_tokens_estimate,
                "selected": [
                    {
                        "subject": hit.metadata.get("subject"),
                        "predicate": hit.metadata.get("predicate"),
                        "origin": hit.metadata.get("origin"),
                        "content": hit.content[:180],
                    }
                    for hit in recovered.sources
                ],
                "indexed": [
                    {
                        "subject": hit.metadata.get("subject"),
                        "predicate": hit.metadata.get("predicate"),
                        "score": hit.score,
                        "content": hit.content[:120],
                        "gold_references": sorted(
                            gold
                            & {
                                official_id[evidence["turn_id"]]
                                for evidence in hit.metadata.get("evidence", [])
                                if evidence["turn_id"] in official_id
                            }
                        ),
                    }
                    for hit in extra
                ],
            }
        )
    result = {
        "source": str(source_path),
        "scope": "13 category-1 development questions from saved memory snapshots",
        "note": "Gold IDs are compared only after retrieval. This replay measures source coverage, not answer accuracy.",
        "budget_characters": args.budget,
        "raw_limit": args.raw_limit,
        "old_all_gold": sum(item["old_all_gold"] for item in rows),
        "new_all_gold": sum(item["new_all_gold"] for item in rows),
        "old_wrong_old_all_gold": sum(
            item["old_all_gold"] for item in rows if item["old_judge_result"] == "WRONG"
        ),
        "old_wrong_new_all_gold": sum(
            item["new_all_gold"] for item in rows if item["old_judge_result"] == "WRONG"
        ),
        "rows": rows,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

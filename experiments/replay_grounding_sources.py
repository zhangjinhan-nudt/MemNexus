"""Offline development-set replay of recorded compressed hits plus raw turns.

Original turn IDs in old runs were random and their temporary databases were
deleted. Reconstruct a reference only when its quote matches exactly one raw
event. Ambiguous/unmatched references remain unresolved. No answer is generated.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus/memory-system/src"))
from agent_memory.grounding import ground_evidence
from agent_memory.models import Event, Hit, QueryPlan, Turn


def main():
    official = json.loads((ROOT / "benchmark-data/official/locomo/locomo10.json").read_text(encoding="utf-8"))
    original = json.loads((ROOT / "result/locomo_100_results_v3.json").read_text(encoding="utf-8"))
    turns_by_sample = {}
    for sample in official:
        turns = []
        for index in (1, 2):
            session = f"session_{index}"
            anchor = datetime.strptime(sample["conversation"][session + "_date_time"], "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC)
            for item in sample["conversation"][session]:
                text = item.get("text", "")
                if item.get("blip_caption"):
                    text = f"[Image: {item['blip_caption']}] " + text
                turns.append(Turn(turn_id=item["dia_id"], request_id=item["dia_id"], session_id="eval", sequence=len(turns) + 1,
                                  events=[Event(role="user", speaker=item["speaker"], content=f"{item['speaker']}: {text}", occurred_at=anchor)]))
        turns_by_sample[sample["sample_id"]] = turns
    rows = []
    for recorded in original["rows"]:
        if recorded["system"] != "MemNexus":
            continue
        turns = turns_by_sample[recorded["sample_id"]]
        resolved = {turn.turn_id: turn for turn in turns}
        hits = [Hit.model_validate(h) for h in recorded["retrieved_sources"]]
        before = set()
        unresolved = 0
        for hit in hits:
            for e in hit.metadata.get("evidence", []):
                matches = [turn for turn in turns if e["quote"] in turn.events[0].content]
                if len(matches) == 1:
                    e["turn_id"] = matches[0].turn_id
                    e["event_index"] = 0
                    before.add(matches[0].turn_id)
                else:
                    unresolved += 1
        result = ground_evidence(recorded["question"], QueryPlan.model_validate(recorded["plan"]), hits,
                                 session_id="eval", current_turns=turns, resolved_turns=resolved,
                                 blocked_events=set(), max_sources=20, raw_limit=3, budget=5000)
        after = {e["turn_id"] for hit in result.sources for e in hit.metadata["source_events"]}
        gold = set(recorded["gold_evidence"])
        rows.append({"sample_id": recorded["sample_id"], "qa_index": recorded["qa_index"],
                     "before_any": bool(gold & before), "after_any": bool(gold & after),
                     "before_all": gold <= before, "after_all": gold <= after,
                     "unresolved_quote_mappings": unresolved,
                     "trace": result.trace.model_dump(mode="json")})
    report = {"kind": "development-set evidence reconstruction; no new answer accuracy",
              "rows": rows, "questions": len(rows),
              **{key: sum(row[key] for row in rows) for key in ("before_any", "after_any", "before_all", "after_all")},
              "limitations": ["No old raw database: unique quote reconstruction only",
                              "Not an independent held-out cohort", "No semantic fallback", "No new model answers"]}
    (ROOT / "result/grounding_source_replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

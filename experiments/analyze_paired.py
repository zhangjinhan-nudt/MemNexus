"""Paired, per-question diagnosis of MemNexus against SimpleMem.

For every question both systems answered, decide who won under the official
metric and, when MemNexus lost, which link of its own evidence chain broke:

  mn_abstained   it declined while SimpleMem answered
  mn_no_evidence the winning system's answer terms are absent from MemNexus
                 evidence (retrieval or extraction loss)
  mn_partial     MemNexus named some of the reference items, not all
  mn_extra       MemNexus covered the items but added unrelated values
  mn_wrong       the evidence was there and the phrasing differed (generation)

The classes are mechanical and cite the tokens they used, so a claim about a
mechanism can be checked against the rows rather than against the prose.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from paper_protocol import adversarial_success, bleu1_of, f1_of  # noqa: E402

CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ABSTAIN = ("not mentioned", "insufficient evidence", "context does not", "cannot determine",
           "unable to determine", "does not specify", "no information")
STOP = {"the", "and", "for", "with", "that", "this", "from", "have", "has", "had", "was",
        "were", "are", "did", "does", "her", "his", "him", "she", "they", "them", "their",
        "you", "your", "not", "but", "all", "any", "can", "how", "what", "when", "where",
        "who", "why", "which", "would", "could", "about", "there", "then", "than", "into",
        "also", "been", "being", "both", "each", "more", "most", "some", "such", "only",
        "very", "just", "now", "out", "likely"}


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def terms(text: str) -> set[str]:
    text = re.sub(r"[^a-z0-9 ]", " ", (text or "").casefold())
    return {word for word in text.split() if len(word) > 2 and word not in STOP}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (text or "").casefold()).strip()


def is_abstention(text: str) -> bool:
    lowered = (text or "").strip().casefold()
    return any(lowered.startswith(marker) or marker in lowered[:60] for marker in ABSTAIN)


def evidence_terms(row: dict) -> set[str]:
    chunks = []
    for source in row.get("retrieved_sources") or []:
        if not isinstance(source, dict):
            # SimpleMem returns the restatement text itself; MemNexus returns a
            # source record. Both are evidence, so both are readable here.
            chunks.append(str(source))
            continue
        metadata = source.get("metadata") or {}
        chunks.append(f"{metadata.get('subject')} {metadata.get('predicate')} {metadata.get('value')}")
        chunks.append(str(source.get("content") or ""))
        for event in metadata.get("source_events") or []:
            chunks.append(str(event.get("text") or ""))
    return terms(" ".join(chunks))


def gold_items(row: dict) -> list[str]:
    gold = str(row.get("gold_answer") or "")
    if int(row.get("category_id") or 0) == 1:
        return [item.strip() for item in gold.split(",") if item.strip()]
    return [gold]


def diagnose(loser: dict, winner: dict) -> dict:
    """Why the losing system scored below the winning one."""
    if is_abstention(loser.get("prediction") or ""):
        return {"class": "abstained", "detail": (loser.get("prediction") or "")[:80]}
    pool = evidence_terms(loser)
    items = gold_items(winner)
    covered, missing = [], []
    prediction = normalise(loser.get("prediction") or "")
    for item in items:
        wanted = terms(item)
        if not wanted:
            continue
        hit = sum(1 for term in wanted if term in prediction) / len(wanted)
        (covered if hit >= 0.6 else missing).append(item)
        if hit < 0.6 and wanted - pool:
            return {"class": "no_evidence", "detail": ",".join(sorted(wanted - pool))[:80]}
    if missing and covered:
        return {"class": "partial", "detail": "missing " + " | ".join(missing)[:90]}
    if not missing and covered:
        extra = terms(loser.get("prediction") or "") - set().union(*(terms(item) for item in items))
        if len(extra) > 3:
            return {"class": "extra", "detail": " ".join(sorted(extra))[:80]}
        return {"class": "phrasing", "detail": "same items, different words"}
    return {"class": "wrong", "detail": (loser.get("prediction") or "")[:80]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="result/bench500/rows_judged.json")
    parser.add_argument("--json", default="result/paired_analysis.json")
    parser.add_argument("--markdown", default="result/PAIRED_ANALYSIS.md")
    args = parser.parse_args()

    rows = read(ROOT / args.rows)
    pivot: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        pivot[(row["sample_id"], row["qa_index"])][row["system"]] = row
    # A provider refusal leaves an empty prediction. It measures the gateway, not
    # the memory system, so those pairs are excluded and counted explicitly.
    dropped = {key: value for key, value in pivot.items()
               if any(not (row.get("prediction") or "").strip() for row in value.values())}
    paired = {key: value for key, value in pivot.items()
              if len(value) == 2 and key not in dropped}

    per_category = defaultdict(lambda: defaultdict(list))
    outcomes = Counter()
    failure_classes = Counter()
    records = []
    for key, pair in sorted(paired.items()):
        mn, sm = pair["MemNexus"], pair["SimpleMem"]
        category = CATEGORY.get(int(mn.get("category_id") or 0), "?")
        scores = {"MemNexus": f1_of(mn), "SimpleMem": f1_of(sm)}
        per_category[category]["MemNexus"].append(scores["MemNexus"])
        per_category[category]["SimpleMem"].append(scores["SimpleMem"])
        per_category[category]["MemNexus_tokens"].append((mn.get("model_calls") or {}).get("input_tokens", 0))
        per_category[category]["SimpleMem_tokens"].append((sm.get("model_calls") or {}).get("input_tokens", 0))
        diff = scores["MemNexus"] - scores["SimpleMem"]
        verdict = "tie" if abs(diff) < 0.02 else ("mn" if diff > 0 else "sm")
        outcomes[verdict] += 1
        record = {
            "sample_id": key[0], "qa_index": key[1], "category": category,
            "question": mn.get("question"), "gold": mn.get("gold_answer"),
            "mn": mn.get("prediction"), "sm": sm.get("prediction"),
            "mn_f1": round(scores["MemNexus"], 3), "sm_f1": round(scores["SimpleMem"], 3),
            "verdict": verdict,
        }
        if verdict == "sm":
            diagnosis = diagnose(mn, sm)
            record["class"] = diagnosis["class"]
            record["detail"] = diagnosis["detail"]
            failure_classes[diagnosis["class"]] += 1
        if verdict == "mn":
            record["class"] = diagnose(sm, mn)["class"]
        records.append(record)

    lines = ["| category | n | MN F1 | SM F1 | Δ | MN tok | SM tok |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for category in ("multi-hop", "temporal", "open-domain", "single-hop", "adversarial"):
        bucket = per_category.get(category)
        if not bucket:
            continue
        lines.append(
            f"| {category} | {len(bucket['MemNexus'])} | "
            f"{statistics.mean(bucket['MemNexus']):.3f} | {statistics.mean(bucket['SimpleMem']):.3f} | "
            f"{statistics.mean(bucket['MemNexus']) - statistics.mean(bucket['SimpleMem']):+.3f} | "
            f"{statistics.mean(bucket['MemNexus_tokens']):.0f} | "
            f"{statistics.mean(bucket['SimpleMem_tokens']):.0f} |"
        )
    all_mn = [f1_of(pair["MemNexus"]) for pair in paired.values()]
    all_sm = [f1_of(pair["SimpleMem"]) for pair in paired.values()]
    equal_weight = [
        statistics.mean(per_category[c]["MemNexus"]) - statistics.mean(per_category[c]["SimpleMem"])
        for c in per_category
    ]
    summary = {
        "questions": len(paired),
        "excluded_provider_errors": len(dropped),
        "mn_f1": round(statistics.mean(all_mn), 3),
        "sm_f1": round(statistics.mean(all_sm), 3),
        "mn_adv": round(statistics.mean([adversarial_success(p["MemNexus"]) for p in paired.values()
                                          if int(p["MemNexus"].get("category_id") or 0) == 5] or [0]), 3),
        "sm_adv": round(statistics.mean([adversarial_success(p["SimpleMem"]) for p in paired.values()
                                          if int(p["SimpleMem"].get("category_id") or 0) == 5] or [0]), 3),
        "outcomes": dict(outcomes),
        "failure_classes": dict(failure_classes.most_common()),
    }
    Path(ROOT / args.json).write_text(json.dumps({"summary": summary, "records": records}, indent=1,
                                                 ensure_ascii=False), encoding="utf-8")
    body = ["## Paired result", *lines, "", f"- questions: {len(paired)}",
            f"- per-question mean F1: MemNexus {summary['mn_f1']} vs SimpleMem {summary['sm_f1']}",
            f"- adversarial success: MemNexus {summary['mn_adv']} vs SimpleMem {summary['sm_adv']}",
            f"- win/tie/loss: {dict(outcomes)}",
            f"- MemNexus loss classes when SimpleMem wins: {dict(failure_classes.most_common())}"]
    Path(ROOT / args.markdown).write_text("\n".join(body) + "\n", encoding="utf-8")
    print("\n".join(body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

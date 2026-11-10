"""Item-level coverage audit: store -> evidence -> answer.

For multi-value questions the interesting failure is not "the answer was
wrong" but "which item was dropped, and at which stage". Each reference item is
tracked through three stages, so a dropped item is attributed to exactly one:

    store    some memory in the frozen base mentions the item
    evidence the sources handed to the answer model mention the item
    answer   the rendered prediction mentions the item

An item absent from the store is an extraction loss; present in the store but
absent from the evidence is a retrieval loss; present in the evidence but
absent from the answer is a generation loss.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

STOP = {"the", "and", "for", "with", "that", "this", "from", "have", "has", "had", "was",
        "were", "are", "did", "does", "her", "his", "him", "she", "they", "them", "their",
        "you", "your", "not", "but", "all", "any", "can", "how", "what", "when", "where",
        "who", "why", "which", "would", "could", "about", "there", "then", "than", "into",
        "also", "been", "being", "both", "each", "more", "most", "some", "such", "only",
        "very", "just", "now", "out", "likely", "since", "ones", "main", "causes"}


def terms(text: str) -> set[str]:
    text = re.sub(r"[^a-z0-9 ]", " ", (str(text) or "").casefold())
    return {word for word in text.split() if len(word) > 2 and word not in STOP}


def body_text(body: dict) -> str:
    parts = [body.get("subject"), body.get("predicate"), body.get("value"), body.get("content")]
    for event in body.get("evidence") or []:
        parts.append(event.get("quote"))
    return " ".join(str(part) for part in parts if part)


def store_text(path: Path) -> str:
    connection = sqlite3.connect(path)
    rows = [body_text(json.loads(row[0])) for row in connection.execute("select body from memories")]
    connection.close()
    return " ".join(rows)


def evidence_text(row: dict) -> str:
    chunks = []
    for source in row.get("retrieved_sources") or []:
        metadata = source.get("metadata") or {}
        chunks.append(f"{metadata.get('subject')} {metadata.get('predicate')} {metadata.get('value')}")
        chunks.append(str(source.get("content") or ""))
        for event in metadata.get("source_events") or []:
            chunks.append(str(event.get("text") or ""))
    return " ".join(chunks)


def gold_items(row: dict) -> list[str]:
    gold = str(row.get("gold_answer") or "")
    if int(row.get("category_id") or 0) == 1:
        return [item.strip() for item in gold.split(",") if item.strip()]
    return [item.strip() for item in re.split(r"[;,]", gold) if len(item.strip()) > 2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    run = Path(args.run)
    stores: dict[str, str] = {}
    records = []
    stages = defaultdict(Counter)
    rows_file = run / "rows.json"
    if rows_file.exists() and not list(run.glob("conv-*.json")):
        grouped = defaultdict(list)
        for row in json.loads(rows_file.read_text(encoding="utf-8")):
            if row.get("system") == "MemNexus":
                grouped[row["sample_id"]].append(row)
        workdir = run / ".audit"
        workdir.mkdir(exist_ok=True)
        for sample_id, items in grouped.items():
            path = workdir / f"{sample_id}.json"
            path.write_text(json.dumps({"stage": {"sample_id": sample_id}, "rows": items,
                                        "questions": []}, ensure_ascii=False), encoding="utf-8")
    else:
        workdir = run
    for path in sorted(workdir.glob("conv-*.json")):
        if "_usage" in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload.get("questions") or []
        for index, row in enumerate(payload["rows"]):
            reference = questions[index] if index < len(questions) else {}
            sample_id = row.get("sample_id") or reference.get("sample_id")
            category = int(row.get("category_id") or reference.get("category_id") or 0)
            if category == 5:
                continue
            if sample_id not in stores:
                store_path = Path(args.store) / f"{sample_id}.sqlite3"
                stores[sample_id] = store_text(store_path).casefold() if store_path.exists() else ""
            store = stores[sample_id]
            evidence = evidence_text(row).casefold()
            answer = " ".join((row.get("prediction") or "").casefold().split())
            items = gold_items(row)
            missing = {"store": [], "evidence": [], "answer": []}
            for item in items:
                wanted = terms(item)
                if not wanted:
                    continue
                if not any(term in store for term in wanted):
                    # no single content word survives: treat as absent
                    if sum(term in store for term in wanted) == 0:
                        missing["store"].append(item)
                        continue
                if sum(term in evidence for term in wanted) / len(wanted) < 0.5:
                    missing["evidence"].append(item)
                    continue
                if sum(term in answer for term in wanted) / len(wanted) < 0.5:
                    missing["answer"].append(item)
            if not items:
                continue
            worst = ("store" if missing["store"] else
                     "evidence" if missing["evidence"] else
                     "answer" if missing["answer"] else "complete")
            stages[category][worst] += 1
            if worst != "complete":
                records.append({
                    "sample_id": sample_id, "qa_index": row.get("qa_index", reference.get("qa_index")),
                    "category_id": category, "question": row.get("question"),
                    "gold": row.get("gold_answer"), "prediction": row.get("prediction"),
                    "lost_at": worst,
                    "missing_store": missing["store"], "missing_evidence": missing["evidence"],
                    "missing_answer": missing["answer"],
                })

    label = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
    print(f"{'category':12s}{'n':>4s}{'complete':>10s}{'answer':>8s}{'evidence':>10s}{'store':>7s}")
    for category in sorted(stages):
        counter = stages[category]
        total = sum(counter.values())
        print(f"{label[category]:12s}{total:4d}{counter['complete']:10d}{counter['answer']:8d}"
              f"{counter['evidence']:10d}{counter['store']:7d}")
    for record in records:
        print(f"{record['sample_id']}/{record['qa_index']} {label[record['category_id']]:11s} "
              f"lost_at={record['lost_at']:9s} missing={record['missing_' + record['lost_at']][:3]}")
    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

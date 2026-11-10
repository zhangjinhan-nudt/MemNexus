"""Classify every wrong answer by where the evidence chain broke.

Three mutually exclusive failure stages, measured against the frozen memory
store rather than against the retriever's own report:

  stored      gold terms appear in some memory of the same conversation
  retrieved   gold terms appear in the evidence handed to the answer model
  answered    the answer text covers the gold terms

A question whose gold terms are in the store but never retrieved is a ranking
failure. One whose gold terms never made it into any memory is an extraction
failure. One that retrieved but still missed is a generation/gate failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

STOP = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had",
    "was", "were", "are", "did", "does", "her", "his", "him", "she", "they",
    "them", "their", "you", "your", "not", "but", "all", "any", "can", "how",
    "what", "when", "where", "who", "why", "which", "would", "could", "about",
    "there", "then", "than", "into", "also", "been", "being", "both", "each",
    "more", "most", "some", "such", "only", "very", "just", "now", "out",
}


def terms(text: str) -> set[str]:
    text = re.sub(r"[^a-z0-9 ]", " ", (text or "").casefold())
    return {word for word in text.split() if len(word) > 2 and word not in STOP}


def memory_text(body: dict) -> str:
    parts = [body.get("subject"), body.get("predicate"), body.get("value"), body.get("content")]
    for event in body.get("evidence") or []:
        parts.append(event.get("quote"))
    return " ".join(str(part) for part in parts if part)


def load_store(path: Path) -> list[str]:
    connection = sqlite3.connect(path)
    rows = connection.execute("select body from memories").fetchall()
    connection.close()
    return [memory_text(json.loads(row[0])) for row in rows]


def load_rows(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.glob("conv-*.json")):
        if "_usage" in path.name or path.name == "questions.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload.get("questions") or []
        for index, row in enumerate(payload["rows"]):
            reference = questions[index] if index < len(questions) else {}
            row.setdefault("qa_index", reference.get("qa_index"))
            row.setdefault("sample_id", reference.get("sample_id", payload["stage"]["sample_id"]))
            rows.append(row)
    return rows


def evidence_text(row: dict) -> str:
    chunks = []
    for source in row.get("retrieved_sources") or []:
        metadata = source.get("metadata") or {}
        chunks.append(f"{metadata.get('subject')} {metadata.get('predicate')} {metadata.get('value')}")
        chunks.append(str(source.get("content") or ""))
        for event in metadata.get("source_events") or []:
            chunks.append(str(event.get("text") or event.get("quote") or ""))
    return " ".join(chunks)


def classify(row: dict, store: list[str]) -> dict:
    gold = terms(row.get("gold_answer") or "")
    if not gold:
        return {"stage": "no-gold-terms", "missing": []}
    stored_text = "\n".join(store).casefold()
    missing_store = {term for term in gold if term not in stored_text}
    retrieved = terms(evidence_text(row))
    missing_retrieved = {term for term in gold if term not in retrieved}
    answered = terms(row.get("prediction") or "")
    covered = gold & answered
    if len(missing_store) > len(gold) * 0.5:
        stage = "extraction"
    elif missing_retrieved:
        stage = "retrieval"
    elif len(covered) / len(gold) < 0.6:
        stage = "generation"
    else:
        stage = "ok"
    return {
        "stage": stage,
        "gold_terms": len(gold),
        "missing_store": sorted(missing_store),
        "missing_retrieved": sorted(missing_retrieved),
        "covered": len(covered) / len(gold),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--store", required=True, help="directory holding <conv>.sqlite3")
    parser.add_argument("--json")
    parser.add_argument("--markdown")
    args = parser.parse_args()

    run = Path(args.run)
    stores: dict[str, list[str]] = {}
    rows = load_rows(run)
    for row in rows:
        if row["sample_id"] not in stores:
            path = Path(args.store) / f"{row['sample_id']}.sqlite3"
            stores[row["sample_id"]] = load_store(path) if path.exists() else []
    for row in rows:
        row["audit"] = classify(row, stores[row["sample_id"]])

    by_category = defaultdict(Counter)
    for row in rows:
        by_category[row.get("category") or str(row.get("category_id"))][row["audit"]["stage"]] += 1

    payload = {"run": str(run), "rows": rows, "summary": {k: dict(v) for k, v in by_category.items()}}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = ["| category | total | extraction | retrieval | generation | ok |", "|---|---:|---:|---:|---:|---:|"]
    for category in sorted(by_category):
        counter = by_category[category]
        lines.append(
            f"| {category} | {sum(counter.values())} | {counter['extraction']} | "
            f"{counter['retrieval']} | {counter['generation']} | {counter['ok']} |"
        )
    table = "\n".join(lines)
    print(table)
    for category in sorted(by_category):
        print("\n" + "=" * 30, category)
        for row in rows:
            if (row.get("category") or str(row.get("category_id"))) != category:
                continue
            if row["audit"]["stage"] == "ok":
                continue
            print(
                f"{row['sample_id']}/{row['qa_index']} {row['audit']['stage']:11s} "
                f"{row['audit']['covered']:.2f} miss={row['audit']['missing_retrieved'][:4]} "
                f"pred={(row.get('prediction') or '')[:45]!r}"
            )
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

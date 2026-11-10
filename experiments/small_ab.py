"""Small paired A/B on a fixed question subset (answer layer only).

Both phases replay the same frozen memory base, so the only difference is the
answer-layer switch set. This run exists to check mechanism activation and
per-question movement; it is not a paper-table reading and uses no judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"

PHASES = {
    "baseline": {
        "MEMORY_SUBJECT_RECALL": "false",
        "MEMORY_BOUNDED_INFERENCE": "false",
        "MEMORY_ANSWER_NORMALIZE": "false",
    },
    "optimized": {
        "MEMORY_SUBJECT_RECALL": "true",
        "MEMORY_BOUNDED_INFERENCE": "false",
        "MEMORY_ANSWER_NORMALIZE": "true",
    },
    # The subject sweep measured negative (unrelated facts about the same person
    # displaced relevant evidence), so this cell isolates surface alignment.
    "normalize_only": {
        "MEMORY_SUBJECT_RECALL": "false",
        "MEMORY_BOUNDED_INFERENCE": "false",
        "MEMORY_ANSWER_NORMALIZE": "true",
    },
    # Single variable on top of normalize_only: a labelled inference is allowed
    # only when the direct pass abstained and the queried subject is in evidence.
    "inference": {
        "MEMORY_SUBJECT_RECALL": "false",
        "MEMORY_BOUNDED_INFERENCE": "true",
        "MEMORY_ANSWER_NORMALIZE": "true",
    },
    # No planning call at all: the deterministic planner drives retrieval. This
    # is a token experiment first and an accuracy experiment second.
    "ruleplan": {
        "MEMORY_SUBJECT_RECALL": "false",
        "MEMORY_BOUNDED_INFERENCE": "false",
        "MEMORY_ANSWER_NORMALIZE": "true",
        "MEMORY_PLANNER_MODE": "rule",
    },
}

# Hypernym/hyponym recall, open-domain abstention, adversarial protection, and
# the temporal cases whose failure was purely a surface form.
SUBSET = {
    "multi-hop": None,          # every multi-hop question in the 100-question run
    "open-domain": None,        # every open-domain question
    "adversarial": None,        # every adversarial question (protection check)
    "temporal": {("conv-30", 7), ("conv-48", 12), ("conv-48", 21),
                 ("conv-48", 27), ("conv-48", 29)},
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def select_questions(source_rows):
    """Pick the subset from a completed run's rows, preserving official order."""
    chosen = []
    for row in source_rows:
        category = row["category"]
        wanted = SUBSET.get(category)
        key = (row["sample_id"], row["qa_index"])
        if wanted is None:
            if category in SUBSET:
                chosen.append({"sample_id": row["sample_id"], "qa_index": row["qa_index"],
                               "category": category, "question": row["question"],
                               "gold_answer": row["gold_answer"], "category_id": row["category_id"]})
        elif key in wanted:
            chosen.append({"sample_id": row["sample_id"], "qa_index": row["qa_index"],
                           "category": category, "question": row["question"],
                           "gold_answer": row["gold_answer"], "category_id": row["category_id"]})
    return chosen


def source_rows(folder):
    rows = []
    for path in sorted((Path(folder) / "attempts").glob("conv-*_memnexus__*/stage_result.json")):
        rows += read(path)["rows"]
    return rows


async def run_phase(args):
    rows_in = source_rows(args.source_run)
    questions = select_questions(rows_in)
    if args.limit:
        questions = questions[: args.limit]
    dataset = {sample["sample_id"]: sample for sample in read(DATASET)}

    for name, value in PHASES[args.phase].items():
        os.environ[name] = value
    os.environ["MEMORY_INPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "2.4"
    os.environ["MEMORY_OUTPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "9.6"
    os.environ["EXPERIMENT_TRUNCATION_MAX_TOKENS"] = "8192"

    from run_locomo_pilot import CATEGORY_NAMES, run_memnexus

    # run_locomo_pilot carries an older label map; the official LoCoMo ids are
    # 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial.
    CATEGORY_NAMES.update({
        1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial",
    })

    out = Path(args.out) / args.phase
    out.mkdir(parents=True, exist_ok=True)
    save(out / "questions.json", questions)

    by_sample = {}
    for item in questions:
        by_sample.setdefault(item["sample_id"], []).append(item)

    total_cost = 0.0
    started = time.perf_counter()
    for sample_id, items in by_sample.items():
        stage_path = out / f"{sample_id}.json"
        if stage_path.exists():
            print(f"[skip] {sample_id} already done")
            continue
        sample = dataset[sample_id]
        qa = [sample["qa"][item["qa_index"]] for item in items]
        budget = max(0.05, args.budget_per_sample)
        stage, rows = await run_memnexus(
            sample, qa, budget,
            usage_ledger=out / f"{sample_id}_usage.json",
            max_session=args.max_session,
            snapshot_dir=Path(args.snapshots),
            resilient=True,
            answer_timeout=300.0,
        )
        # run_memnexus returns rows in question order; the benchmark harness adds
        # qa_index later, but this script needs it for pairing.
        if len(rows) != len(items):
            raise RuntimeError(f"{sample_id}: {len(rows)} rows for {len(items)} questions")
        for item, row in zip(items, rows, strict=True):
            row["qa_index"] = item["qa_index"]
            row["sample_id"] = item["sample_id"]
        save(stage_path, {"stage": stage, "rows": rows, "questions": items})
        cost = stage.get("total_usage", {}).get("cost_rmb", 0.0)
        total_cost += cost
        print(f"[done] {sample_id}: {len(rows)} questions, ¥{cost:.4f}, "
              f"replayed={stage.get('memory_base_replayed')}")
    print(f"[phase {args.phase}] {len(by_sample)} conversations, ¥{total_cost:.4f}, "
          f"{time.perf_counter() - started:.0f}s")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--source-run", default=str(
        ROOT / "result" / "benchmark100" / "20261020_161439_654687"))
    parser.add_argument("--out", default=str(ROOT / "result" / "small_ab_20261020"))
    parser.add_argument("--snapshots", default=str(
        ROOT / "result" / "small_ab_20261020" / "snapshots"))
    parser.add_argument("--budget-per-sample", type=float, default=3.0)
    parser.add_argument("--max-session", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    await run_phase(args)


if __name__ == "__main__":
    asyncio.run(main())

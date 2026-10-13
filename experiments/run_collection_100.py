"""Frozen 100-question comparison after the collection architecture upgrade.

Keep the previous D1-D5 horizon and category-ID quotas. Prefer previously untested
question keys, and disclose necessary reuse rather than changing the quota after scoring.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np

from locomo_100_protocol import DATASET, QUOTAS, ROOT, estimate_cost
from run_locomo_100 import run as run_paired
from run_locomo_pilot import CATEGORY_NAMES, RemoteQwenEmbedding

COHORT = ROOT / "result/locomo_100_cohort_collection_20261013.json"
OUTPUT = ROOT / "result/locomo_100_collection_20261013.json"
SEED = 20261013


def freeze():
    previous_paths = [ROOT / "result/locomo_100_cohort.json", ROOT / "result/locomo_100_cohort_v2.json"]
    previous = [json.loads(p.read_text(encoding="utf-8")) for p in previous_paths]
    seen = {(q["sample_id"], q["qa_index"]) for c in previous for q in c["questions"]}
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    candidates = defaultdict(list)
    import re

    for sample_index, sample in enumerate(data):
        ids = {t["dia_id"] for name, turns in sample["conversation"].items()
               if name.startswith("session_") and isinstance(turns, list) and int(name.split("_")[1]) <= 5
               for t in turns}
        for qa_index, qa in enumerate(sample["qa"]):
            refs = [f"D{int(a)}:{int(b)}" for e in qa.get("evidence", [])
                    for a, b in re.findall(r"D(\d+):(\d+)", str(e))]
            if not refs or not set(refs) <= ids:
                continue
            if qa["category"] == 5 and not qa.get("adversarial_answer"):
                continue
            if qa["category"] != 5 and qa.get("answer") is None:
                continue
            candidates[qa["category"]].append({
                "sample_index": sample_index, "sample_id": sample["sample_id"], "qa_index": qa_index,
                "category_id": qa["category"], "category": f"category_{qa['category']}",
                "question": qa["question"], "evidence": refs,
                "previously_tested": (sample["sample_id"], qa_index) in seen,
            })
    rng = random.Random(SEED)
    selected = []
    for category, quota in QUOTAS.items():
        chosen = []
        for reused in (False, True):
            buckets = defaultdict(list)
            for item in candidates[category]:
                if item["previously_tested"] == reused:
                    buckets[item["sample_index"]].append(item)
            for bucket in buckets.values():
                rng.shuffle(bucket)
            while len(chosen) < quota and any(buckets.values()):
                for sample_index in sorted(buckets):
                    if buckets[sample_index] and len(chosen) < quota:
                        chosen.append(buckets[sample_index].pop())
        if len(chosen) != quota:
            raise ValueError(f"Insufficient valid category-{category} questions")
        selected.extend(chosen)
    selected.sort(key=lambda q: (q["sample_index"], q["qa_index"]))
    keys = [(q["sample_id"], q["qa_index"]) for q in selected]
    assert len(keys) == len(set(keys)) == 100
    cohort = {
        "dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
        "protocol": "D1-D5 evidence horizon; same category-ID quotas; seeded conversation round-robin; unseen keys first",
        "seed": SEED, "max_session": 5, "question_count": 100,
        "conversation_count": len({q["sample_id"] for q in selected}),
        "categories": dict(Counter(q["category"] for q in selected)),
        "new_questions": sum(not q["previously_tested"] for q in selected),
        "reused_questions": sum(q["previously_tested"] for q in selected),
        "prior_selections": [c["selection_sha256"] for c in previous],
        "interpretation": "development comparison; same previously studied conversations; not conversation-held-out",
        "questions": selected,
    }
    if COHORT.exists():
        assert json.loads(COHORT.read_text(encoding="utf-8")) == cohort, "Frozen selection changed"
    else:
        COHORT.write_text(json.dumps(cohort, ensure_ascii=False, indent=2), encoding="utf-8")
    return cohort


async def preflight():
    sys.path.insert(0, str(ROOT / "MemNexus/memory-system/src"))
    from agent_memory.embeddings import build_embedder
    from agent_memory.settings import Settings

    settings = Settings()
    sys.path.insert(0, str(ROOT / "SimpleMem"))
    import config

    checks = {
        "model": config.LLM_MODEL == os.getenv("MEMORY_MODEL_NAME") == "deepseek-v4-flash",
        "judge": config.JUDGE_MODEL == "deepseek-v4-flash",
        "temperature": config.TEMPERATURE == 0,
        "window_stride": config.WINDOW_SIZE == 20 and config.OVERLAP_SIZE == 15,
        "retrieval_max": config.SEMANTIC_TOP_K == 20,
        "collection_path_enabled": settings.collection_queries_enabled,
        "query_prompt": os.getenv("MEMORY_EMBEDDING_ENCODING") == "query_prompt",
        "main_key_present": bool(os.getenv("MEMORY_MODEL_API_KEY")),
        "baseline_key_present": bool(config.OPENAI_API_KEY),
        "judge_key_present": bool(config.JUDGE_API_KEY or config.OPENAI_API_KEY),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Configuration preflight failed: {checks}")
    health = httpx.get("http://127.0.0.1:8099/health", timeout=60).json()
    assert health["dimension"] == 1024 and health["model"] == "Qwen/Qwen3-Embedding-0.6B"
    baseline = RemoteQwenEmbedding()
    nexus = build_embedder(settings)
    if nexus is None:
        raise RuntimeError("Missing embedding adapter")
    try:
        query = ["Which activities did both people mention?"]
        document = ["A source statement used only to verify encoding parity."]
        simple_q, simple_d = baseline.encode_query(query), baseline.encode_documents(document)
        mem_q, mem_d = await nexus.encode_query(query), await nexus.encode_documents(document)
        assert np.allclose(simple_q, mem_q, atol=1e-6) and np.allclose(simple_d, mem_d, atol=1e-6)
    finally:
        await nexus.aclose()
    return {"checks": checks, "embedding_health": health, "query_and_document_vectors_match": True}


async def main(args):
    output = args.output.resolve()
    cohort = freeze()
    checked = await preflight()
    manifest = {"cohort": str(COHORT), "preflight": checked, "estimate": estimate_cost(cohort, max_session=5),
                "new_questions": cohort["new_questions"], "reused_questions": cohort["reused_questions"],
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return 0
    if output.exists():
        raise FileExistsError(output)
    output.with_suffix(".preflight.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Correct grouping relies on official IDs; do not perpetuate legacy English mislabels.
    CATEGORY_NAMES.update({i: f"category_{i}" for i in range(1, 6)})
    code_files = [*(ROOT / "MemNexus/memory-system/src/agent_memory").glob("*.py"),
                  *(ROOT / "SimpleMem/simplemem/core").rglob("*.py"),
                  ROOT / "SimpleMem/main.py", *Path(__file__).parent.glob("*.py")]
    with zipfile.ZipFile(output.with_suffix(".source.zip"), "x", zipfile.ZIP_DEFLATED) as archive:
        for path in code_files:
            archive.write(path, path.relative_to(ROOT))
    return await run_paired(SimpleNamespace(cohort=COHORT, output=output, max_session=5,
                                            cumulative_cap=100.0, prior_reserved=0.0, resume_from=args.resume_from))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path)
    raise SystemExit(asyncio.run(main(parser.parse_args())))

"""End-to-end connectivity probe for the configured MemNexus model endpoint.

This is not a benchmark. It answers one question: can the configured model
carry a complete MemNexus request chain (ingest -> extract -> plan -> answer)
without a repair storm, an empty completion, or a schema failure?

Writes only under result/connectivity_<timestamp>/.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus" / "memory-system" / "src"))

TURNS = [
    ("Caroline", "I went to the gym this morning and did a full leg workout."),
    ("Melanie", "That sounds intense. I spent the morning painting a bowl in pottery class."),
    ("Caroline", "I did not know you took pottery. When did you start?"),
    ("Melanie", "I started last November, so about five months ago now."),
    ("Caroline", "My sister Dana is visiting me in June, and she wants to try pottery too."),
    ("Melanie", "You should bring her to my class on the 12th. The teacher is very patient."),
    ("Caroline", "Deal. I also adopted a puppy last week, a beagle named Comet."),
    ("Melanie", "A beagle. You are going to be very busy."),
    ("Caroline", "Comet already chewed two of my running shoes."),
    ("Melanie", "That is why I have a cat. Much less destruction."),
]

QUESTIONS = [
    ("single-hop", "What kind of pet did Caroline adopt?"),
    ("temporal", "When did Melanie start taking pottery classes?"),
    ("multi-hop", "What is the name of Caroline's sister who wants to try pottery?"),
    ("open-domain", "Why does Melanie prefer having a cat?"),
]


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _turn_time(index: int) -> datetime:
    """Deterministic turn timestamps: 2023-01-01 plus one day per turn."""
    from datetime import timedelta
    return datetime(2023, 1, 1, 9, 0, tzinfo=timezone.utc) + timedelta(days=index)


async def main() -> int:
    from agent_memory.models import Event, Scope, TurnInput
    from agent_memory.runtime import MemoryRuntime
    from agent_memory.settings import Settings

    out = ROOT / "result" / f"connectivity_{stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    # Full request/response trace lets us see the exact body the endpoint got.
    os.environ["EXPERIMENT_USAGE_LEDGER_PATH"] = str(out / "usage.json")
    os.environ["EXPERIMENT_COMPONENT_BUDGET_RMB"] = os.getenv("EXPERIMENT_COMPONENT_BUDGET_RMB", "1.0")
    workdir = Path(tempfile.mkdtemp(prefix="memnexus-connectivity-"))
    print(f"workdir: {workdir}")
    print(f"output : {out}")

    runtime = MemoryRuntime.from_settings(
        Settings(
            db_path=workdir / "memory.sqlite3",
            pending_turns=5,
            context_turns=15,
            model_timeout=180,
            model_max_tokens=4096,
        )
    )
    print(f"model  : {runtime.model.model}")
    print(f"endpoint: {runtime.model.endpoint}")
    print(f"provider: {runtime.model.provider}")
    print()

    scope = Scope(tenant_id="connectivity", owner_id="probe")
    session = runtime.store.create_session(scope, "connectivity probe")

    started = time.perf_counter()
    try:
        batches = 0
        for index, (speaker, text) in enumerate(TURNS, 1):
            result = await runtime.append_turn(
                scope,
                session.session_id,
                TurnInput(
                    request_id=f"probe-{index}",
                    events=[Event(role="user", speaker=speaker,
                                  content=f"{speaker}: {text}",
                                  occurred_at=_turn_time(index))],
                ),
            )
            if result["extraction_due"]:
                extracted = await runtime.extract(scope, session.session_id)
                if extracted["status"] != "idle":
                    batches += 1
        while True:
            extracted = await runtime.extract(scope, session.session_id)
            if extracted["status"] == "idle":
                break
            batches += 1
        build_ms = (time.perf_counter() - started) * 1000
        memories = runtime.store.memories(scope)
        print(f"[build] batches={batches} memories={len(memories)} {build_ms:.0f}ms")
        if memories:
            sample = memories[0]
            print(f"[build] sample memory: {str(sample.content)[:110]!r}")
        print()

        rows = []
        for category, question in QUESTIONS:
            question_started = time.perf_counter()
            record = {"category": category, "question": question}
            try:
                response = await runtime.answer(scope, session.session_id, question, timeout=180)
                record["ok"] = True
                record["answer"] = response.generated_text
                record["citations"] = len(response.citations or [])
                record["sources"] = len(getattr(response, "sources", []) or [])
                record["retrieval_mode"] = getattr(response, "retrieval_mode", "")
            except Exception as error:  # noqa: BLE001 - probe reports failures verbatim
                record["ok"] = False
                record["error"] = f"{type(error).__name__}: {error}"
            record["latency_ms"] = (time.perf_counter() - question_started) * 1000
            rows.append(record)
            status = "OK " if record["ok"] else "FAIL"
            text = record.get("answer") or record.get("error") or ""
            print(f"[{status}] {category:12s} {record['latency_ms']:6.0f}ms  {str(text)[:100]}")
        print()

        usage = dict(runtime.model.usage)
        telemetry = list(runtime.model.telemetry)
        print("== usage ==")
        for key, value in usage.items():
            print(f"  {key}: {value}")
        print()
        print("== telemetry ==")
        print(f"  calls={len(telemetry)}")
        if telemetry:
            latencies = sorted(t["latency_ms"] for t in telemetry)
            print(f"  latency p50={latencies[len(latencies)//2]:.0f}ms max={latencies[-1]:.0f}ms")
            print(f"  status={Counter(t['status'] for t in telemetry)}")
            failures = [t for t in telemetry if t["status"] != "completed"]
            for failure in failures[:5]:
                print(f"  failure: {failure}")
        print()

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": runtime.model.model,
            "endpoint": runtime.model.endpoint,
            "provider": runtime.model.provider,
            "build": {"batches": batches, "memories": len(memories), "latency_ms": build_ms},
            "questions": rows,
            "usage": usage,
            "telemetry": telemetry,
        }
        (out / "connectivity.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        ok = sum(1 for row in rows if row["ok"])
        print(f"== 结论: {ok}/{len(rows)} 题成功, {len(telemetry)} 次模型调用, "
              f"¥{usage.get('cost_rmb', 0):.4f} ==")
        return 0 if ok == len(rows) else 1
    finally:
        await runtime.close()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    os.environ.setdefault("MEMORY_EXPERIMENT_DROP_QUESTION_CANDIDATES", "1")
    raise SystemExit(asyncio.run(main()))

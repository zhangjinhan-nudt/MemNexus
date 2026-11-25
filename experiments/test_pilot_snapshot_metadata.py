"""Exercise runner provenance with real SQLite freeze/thaw and no model calls."""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "MemNexus/memory-system/src")]

import run_locomo_pilot as pilot  # noqa: E402
from agent_memory import snapshot  # noqa: E402
from agent_memory.models import AnswerResponse, Evidence, Memory  # noqa: E402
from agent_memory.store import SQLiteStore  # noqa: E402


def offline_answer(text):
    return AnswerResponse(
        generated_text=text, sources=[], citations=[], retrieval_count=0, elapsed_ms=0,
        acceleration={"enabled": False, "cache_hit": False, "cache_status": "disabled",
                      "retrieval_ms": 0, "generation_ms": 0, "total_ms": 0,
                      "context_tokens_before": 0, "context_tokens_after": 0,
                      "avoided_model_calls": 0},
    )


class OfflineRuntime:
    """Keep ingestion and snapshots real; replace only model-backed extraction."""

    def __init__(self, settings):
        self.settings = settings
        self.store = SQLiteStore(settings.db_path)
        self.model = SimpleNamespace(
            usage={"provider_attempts": 0, "input_tokens": 0, "output_tokens": 0,
                   "cost_rmb": 0.0},
            rejected_question_candidates=0,
            validation_rejections={},
        )
        self.retriever = SimpleNamespace(embedder=object())
        self.append_calls = 0
        self.extract_calls = 0
        self.pending = False

    async def append_turn(self, scope, session_id, data):
        self.append_calls += 1
        self.store.append_turn(scope, session_id, data)
        self.pending = True
        return {"extraction_due": True}

    async def extract(self, scope, session_id):
        self.extract_calls += 1
        if not self.pending:
            return {"status": "idle"}
        self.pending = False
        with self.store.transaction():
            self.store._save_memory(
                scope,
                Memory(memory_id="test-memory", session_id=session_id,
                       scope_id=session_id, subject="Alex", predicate="likes",
                       value="hiking", content="Alex likes hiking.",
                       evidence=[Evidence(
                           turn_id=self.store.turns(scope, session_id)[0].turn_id,
                           event_index=0, quote="I like hiking.",
                       )]),
            )
        # Synthetic usage lets the runner's real build accounting be checked.
        self.model.usage.update(provider_attempts=1, input_tokens=10,
                                output_tokens=5, cost_rmb=0.01)
        return {"status": "extracted"}

    async def close(self):
        self.store.close()


class PilotSnapshotMetadataTests(unittest.TestCase):
    sample = {
        "sample_id": "snapshot-metadata-test",
        "conversation": {
            "session_1_date_time": "2026-01-01",
            "session_1": [{"dia_id": "D1:1", "speaker": "Alex",
                           "text": "I like hiking."}],
        },
    }

    def test_fresh_freeze_then_actual_thaw_distinguishes_build_spend(self):
        instances = []

        def make_runtime(settings):
            runtime = OfflineRuntime(settings)
            instances.append(runtime)
            return runtime

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ), patch(
            "agent_memory.runtime.MemoryRuntime.from_settings", side_effect=make_runtime
        ), patch.object(snapshot, "freeze", wraps=snapshot.freeze) as freeze, patch.object(
            snapshot, "thaw", wraps=snapshot.thaw
        ) as thaw:
            base = Path(folder)
            snapshots = base / "snapshots"
            fresh, fresh_rows = asyncio.run(pilot.run_memnexus(
                self.sample, [], 1.0, usage_ledger=base / "fresh.json",
                snapshot_dir=snapshots,
            ))
            freeze.assert_called_once()
            thaw.assert_not_called()
            self.assertEqual(instances[0].append_calls, 1)
            self.assertEqual(instances[0].extract_calls, 2)
            self.assertFalse(fresh["memory_base_replayed"])
            self.assertEqual(fresh["build_usage"]["cost_rmb"], 0.01)
            self.assertEqual(fresh["build_usage"]["provider_attempts"], 1)
            self.assertEqual(fresh["memory_count"], 1)
            self.assertEqual(fresh["extraction_batches"], 1)
            database, manifest_path = snapshot.snapshot_paths(
                snapshots, self.sample["sample_id"]
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(fresh["memory_snapshot_sha256"], snapshot.sha256_of(database))
            self.assertEqual(manifest["build_usage"], fresh["build_usage"])

            replay, replay_rows = asyncio.run(pilot.run_memnexus(
                self.sample, [], 1.0, usage_ledger=base / "replay.json",
                snapshot_dir=snapshots, require_snapshot=True,
            ))
            thaw.assert_called_once()
            freeze.assert_called_once()
            self.assertEqual(instances[1].append_calls, 0)
            self.assertEqual(instances[1].extract_calls, 0)
            self.assertTrue(replay["memory_base_replayed"])
            self.assertEqual(replay["memory_snapshot_sha256"], fresh["memory_snapshot_sha256"])
            self.assertEqual(replay["memory_count"], fresh["memory_count"])
            self.assertEqual(replay["extraction_batches"], fresh["extraction_batches"])
            self.assertEqual(replay["build_ms"], 0.0)
            self.assertTrue(all(value == 0 for value in replay["build_usage"].values()))
            self.assertEqual(fresh["total_usage"], fresh["build_usage"])
            self.assertEqual(replay["total_usage"], replay["build_usage"])
            self.assertEqual(fresh_rows, replay_rows)
            for name, stage in (("fresh", fresh), ("replay", replay)):
                saved = json.loads((base / f"{name}.stage.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, stage)

    def test_fresh_build_without_snapshot_has_no_replay_or_snapshot_hash(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ), patch(
            "agent_memory.runtime.MemoryRuntime.from_settings", side_effect=OfflineRuntime
        ), patch.object(snapshot, "freeze", wraps=snapshot.freeze) as freeze, patch.object(
            snapshot, "thaw", wraps=snapshot.thaw
        ) as thaw:
            stage, _ = asyncio.run(pilot.run_memnexus(
                self.sample, [], 1.0, usage_ledger=Path(folder) / "usage.json",
            ))
            self.assertFalse(stage["memory_base_replayed"])
            self.assertIsNone(stage["memory_snapshot_sha256"])
            self.assertEqual(stage["build_usage"]["cost_rmb"], 0.01)
            freeze.assert_not_called()
            thaw.assert_not_called()

    def test_lme_qa_grants_its_history_and_keeps_legacy_answers_separate(self):
        sample = {
            **self.sample,
            "dataset": "LongMemEval-S",
            "conversation": {
                **self.sample["conversation"],
                "session_2_date_time": "2026-01-02",
                "session_2": [{"dia_id": "D2:1", "speaker": "assistant", "text": "Try hiking."}],
            },
        }
        question = {"qa_index": 0, "dataset": "LongMemEval-S", "category": 1,
                    "question": "What was suggested?", "answer": "Hiking", "question_date": "2026-01-03"}
        capture = AsyncMock(side_effect=RuntimeError("offline capture"))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ), patch(
            "agent_memory.runtime.MemoryRuntime.from_settings", side_effect=OfflineRuntime
        ), patch.object(OfflineRuntime, "answer", capture, create=True):
            base = Path(folder)
            snapshots = base / "snapshots"
            ledger = base / "qa.json"
            stage, _ = asyncio.run(pilot.run_memnexus(
                sample, [], 1.0, usage_ledger=base / "build.json", snapshot_dir=snapshots,
            ))
            _, manifest_path = snapshot.snapshot_paths(snapshots, sample["sample_id"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            grants = sorted(set(manifest["history_session_ids"]) - {stage["qa_session_id"]})
            self.assertEqual(len(grants), 1)
            self.assertEqual(stage["authorized_source_sessions"], grants)
            self.assertEqual(stage["source_read_scope"], "events")
            with self.assertRaisesRegex(RuntimeError, "offline capture"):
                asyncio.run(pilot.run_memnexus(sample, [question], 1.0, usage_ledger=ledger,
                                              snapshot_dir=snapshots, require_snapshot=True))
            self.assertEqual(capture.await_args.kwargs["source_session_ids"], tuple(grants))
            self.assertEqual(capture.await_args.kwargs["source_read_scope"], "events")
            query, gold = pilot.evaluation_query(question)
            prior = {"sample_id": sample["sample_id"], "qa_index": 0, "question": question["question"],
                     "evaluation_query": query, "gold_answer": gold, "prediction": "Hiking"}
            ledger.with_suffix(".rows.jsonl").write_text(json.dumps(prior) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different QA history scope"):
                asyncio.run(pilot.run_memnexus(sample, [question], 1.0, usage_ledger=ledger,
                                              snapshot_dir=snapshots, resume=True))
            prior["reasoning"] = {"authorized_source_sessions": grants}
            for old_scope in (None, "quotes"):
                if old_scope is not None:
                    prior["reasoning"]["source_read_scope"] = old_scope
                checkpoint = ledger.with_suffix(".rows.jsonl")
                checkpoint.write_text(json.dumps(prior) + "\n", encoding="utf-8")
                unchanged = checkpoint.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "different QA history scope"):
                    asyncio.run(pilot.run_memnexus(sample, [question], 1.0, usage_ledger=ledger,
                                                  snapshot_dir=snapshots, resume=True))
                self.assertEqual(checkpoint.read_bytes(), unchanged)
            capture.assert_awaited_once()

            # Record an actual new-format answer from the mock, then verify
            # that the exact events+grant combination is safely reusable.
            checkpoint.write_text("", encoding="utf-8")
            capture.side_effect = None
            capture.return_value = offline_answer("Hiking")
            _, first = asyncio.run(pilot.run_memnexus(sample, [question], 1.0, usage_ledger=ledger,
                                                     snapshot_dir=snapshots, resume=True))
            self.assertEqual(first[0]["source_read_scope"], "events")
            self.assertEqual(first[0]["authorized_source_sessions"], grants)
            self.assertEqual(first[0]["reasoning"]["source_read_scope"], "events")
            capture.reset_mock()
            _, rows = asyncio.run(pilot.run_memnexus(sample, [question], 1.0, usage_ledger=ledger,
                                                   snapshot_dir=snapshots, resume=True))
            self.assertEqual(rows, first)
            capture.assert_not_awaited()

    def test_locomo_keeps_default_quote_scope_and_legacy_row_reuse(self):
        question = {"qa_index": 0, "category": 1, "question": "What does Alex like?", "answer": "hiking"}
        capture = AsyncMock(return_value=offline_answer("hiking"))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ), patch(
            "agent_memory.runtime.MemoryRuntime.from_settings", side_effect=OfflineRuntime
        ), patch.object(OfflineRuntime, "answer", capture, create=True):
            base = Path(folder)
            stage, rows = asyncio.run(pilot.run_memnexus(self.sample, [question], 1.0,
                usage_ledger=base / "qa.json", snapshot_dir=base / "snapshots"))
            self.assertNotIn("source_read_scope", capture.await_args.kwargs)
            self.assertEqual(capture.await_args.kwargs["source_session_ids"], ())
            self.assertEqual(stage["source_read_scope"], "quotes")
            self.assertEqual(rows[0]["source_read_scope"], "quotes")
            legacy = dict(rows[0])
            legacy.pop("source_read_scope")
            legacy.pop("authorized_source_sessions")
            legacy["reasoning"] = {}
            (base / "qa.rows.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
            capture.reset_mock()
            _, reused = asyncio.run(pilot.run_memnexus(self.sample, [question], 1.0,
                usage_ledger=base / "qa.json", snapshot_dir=base / "snapshots", resume=True))
            self.assertEqual(reused, [legacy])
            capture.assert_not_awaited()

    @staticmethod
    def completed_lme_unit():
        access = {"source_read_scope": "events", "authorized_source_sessions": ["past"]}
        return {"stage": {**access, "history_session_ids": ["past", "current"],
                          "qa_session_id": "current", "internal_session_count": 2},
                "rows": [{**access, "sample_id": "lme-test", "qa_index": 0,
                          "system": "MemNexus", "category": "multi-session", "reasoning": dict(access)}]}

    def test_completed_gate_checks_grant_and_scope_in_stage_and_each_row(self):
        pilot.validate_completed_qa_history(self.completed_lme_unit(), "events")
        for mutation in ("old_stage", "old_row", "wrong_grant", "contradictory_row", "missing_history"):
            result = self.completed_lme_unit()
            if mutation == "old_stage":
                result["stage"].pop("source_read_scope")
            elif mutation == "old_row":
                result["rows"][0].pop("source_read_scope")
                result["rows"][0]["reasoning"].pop("source_read_scope")
            elif mutation == "wrong_grant":
                result["rows"][0]["authorized_source_sessions"] = ["elsewhere"]
            elif mutation == "contradictory_row":
                result["rows"][0]["reasoning"]["source_read_scope"] = "quotes"
            else:
                result["stage"].pop("history_session_ids")
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                pilot.validate_completed_qa_history(result, "events")

    def test_outer_runner_rejects_old_scope_before_dispatch_or_collection(self):
        import run_parallel_bench as parallel
        sample = {"sample_id": "lme-test", "dataset": "LongMemEval-S",
                  "qa": [{"question": "Q", "answer": "A", "category": 1}]}
        cohort = {"dataset": "unused", "questions": [{"sample_index": 0, "sample_id": "lme-test",
                    "qa_index": 0, "category_id": 0}]}
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            attempt = base / "attempts/lme-test_memnexus"
            attempt.mkdir(parents=True)
            old = self.completed_lme_unit()
            old["stage"]["source_read_scope"] = "quotes"
            path = attempt / "stage_result.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            old_bytes = path.read_bytes()
            cohort_path = base / "cohort.json"
            cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
            args = SimpleNamespace(cohort=cohort_path, out=base, snapshots=None, limit_conversations=0,
                                   only_sample=None, concurrency=1, only_system="SimpleMem", require_snapshots=False)
            dispatch = AsyncMock()
            with patch.object(parallel, "read_selected_samples", return_value={0: sample}), patch.object(
                parallel, "run_unit", dispatch
            ), self.assertRaisesRegex(RuntimeError, "different QA history scope"):
                asyncio.run(parallel.execute(args))
            dispatch.assert_not_awaited()
            with self.assertRaisesRegex(RuntimeError, "different QA history scope"):
                parallel.collect_run_rows(base, [sample], [0], {0: cohort["questions"]}, ("MemNexus",))
            self.assertEqual(path.read_bytes(), old_bytes)
            fresh = self.completed_lme_unit()
            path.write_text(json.dumps(fresh), encoding="utf-8")
            rows, _ = parallel.collect_run_rows(base, [sample], [0], {0: cohort["questions"]}, ("MemNexus",))
            self.assertEqual(rows, fresh["rows"])
            # Explicit historical collection remains a read/aggregate operation,
            # not a declaration that these rows satisfy the new run protocol.
            path.write_bytes(old_bytes)
            self.assertEqual(parallel.collect(SimpleNamespace(out=base))["rows"], 1)


if __name__ == "__main__":
    unittest.main()

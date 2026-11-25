"""Practical lifecycle cases: aliases, corrections, drift, recovery and provenance."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from agent_memory.long_term import LongTermMemory
from agent_memory.maintenance import correction_relation, retention, summary_groups
from agent_memory.models import (
    Candidate,
    Event,
    Evidence,
    EvolutionInput,
    QueryPlan,
    Scope,
    TurnInput,
    utcnow,
)
from agent_memory.retrieval import Retriever
from agent_memory.store import SQLiteStore


class MaintenanceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(Path(self.temp.name) / "memory.sqlite")
        self.scope = Scope(tenant_id="t", owner_id="alice")
        self.session = self.store.create_session(self.scope, "别名测试", None)
        self.service = LongTermMemory(self.store)
        self.counter = 0

    async def asyncTearDown(self):
        self.store.close()
        self.temp.cleanup()

    def memory(self, value="飞猪", quote=None, **changes):
        self.counter += 1
        quote = quote or f"zfc 的昵称是{value}。"
        turn = self.store.append_turn(
            self.scope,
            self.session.session_id,
            TurnInput(
                request_id=f"turn-{self.counter}",
                events=[Event(role="user", content=quote)],
            ),
        )
        item = Candidate(
            subject="zfc",
            predicate="昵称",
            value=value,
            content=quote,
            durable=True,
            scope_type="user",
            evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote=quote)],
            # These fixtures model one current nickname; unknown/many relations may coexist.
            semantics={"cardinality": "one"},
            **changes,
        )
        return self.store.add_candidates(self.scope, self.session.session_id, [item])[0]

    async def test_durable_statement_promotes_and_wording_duplicate_merges_evidence(self):
        first = self.memory()
        report = await self.service.process_extraction(self.scope, [first])
        self.assertEqual(report["applied"][0]["action"], "promote")
        second = self.memory(quote="请记住，zfc 的昵称是飞猪。")
        report = await self.service.process_extraction(self.scope, [second])
        self.assertEqual(report["applied"][0]["action"], "merge")
        current = self.store.get_memory(self.scope, first.memory_id)
        self.assertEqual(len(current.evidence), 2)
        self.assertEqual(self.store.get_memory(self.scope, second.memory_id).status, "superseded")

    async def test_unexplained_conflicting_value_stays_short_for_review(self):
        first = self.memory()
        await self.service.process_extraction(self.scope, [first])
        canonical = self.store.get_memory(self.scope, first.memory_id).model_dump()
        second = self.memory("大象")
        result = await self.service.process_extraction(self.scope, [second])
        self.assertEqual(result["review"][0]["reason"], "conflicting_values_require_review")
        self.assertNotIn("activate", result["review"][0]["suggested_actions"])
        self.assertEqual(result["applied"], [])
        retained = self.store.get_memory(self.scope, second.memory_id)
        self.assertEqual((retained.status, retained.tier), ("active", "short"))
        self.assertEqual(retained.model_dump(), second.model_dump())
        self.assertEqual(self.store.get_memory(self.scope, first.memory_id).model_dump(), canonical)

    async def test_distinct_trips_without_event_ids_keep_both_original_sources(self):
        memories = []
        for index, (city, day) in enumerate((("Lisbon", "2024-03-02"), ("Rome", "2024-04-10"))):
            quote = f"Nora visited {city} on {day}."
            turn = self.store.append_turn(
                self.scope, self.session.session_id,
                TurnInput(request_id=f"trip-{index}", events=[Event(role="user", content=quote)]),
            )
            candidate = Candidate(
                subject="Nora", predicate="visited_destination", value=city, content=quote,
                kind="fact", durable=True, scope_type="user", semantics={"cardinality": "one"},
                evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote=quote)],
            )
            memory = self.store.add_candidates(self.scope, self.session.session_id, [candidate])[0]
            self.assertIsNone(memory.semantics.event_key)
            self.assertIsNone(memory.semantics.event_identity)
            memories.append(memory)
            result = await self.service.process_extraction(self.scope, [memory])
            if index == 0:
                canonical = self.store.get_memory(self.scope, memory.memory_id).model_dump()
        self.assertEqual(result["applied"], [])
        self.assertEqual(result["review"][0]["target_ids"], [memories[0].memory_id])
        self.assertEqual(self.store.get_memory(self.scope, memories[0].memory_id).model_dump(), canonical)
        retained = [self.store.get_memory(self.scope, memory.memory_id) for memory in memories]
        self.assertEqual([(memory.tier, memory.status) for memory in retained],
                         [("long", "active"), ("short", "active")])
        for memory, original in zip(retained, memories):
            self.assertEqual(memory.evidence, original.evidence)
            self.assertTrue(Retriever._eligible(
                memory, self.session, QueryPlan(temporal_mode="history"), utcnow()
            ))

    async def test_actual_change_preserves_old_validity_and_new_version(self):
        first = self.memory(valid_from=utcnow() - timedelta(days=1))
        await self.service.process_extraction(self.scope, [first])
        second = self.memory("大象", "zfc 的昵称从飞猪改为大象。")
        report = await self.service.process_extraction(self.scope, [second])
        self.assertEqual(report["applied"][0]["action"], "supersede")
        current = self.store.get_memory(self.scope, first.memory_id)
        self.assertEqual((current.value, current.version), ("大象", 2))
        versions = [m for m in self.store.memories(self.scope) if m.memory_id == first.memory_id]
        self.assertEqual(versions[0].valid_to, current.valid_from)

    async def test_correction_retracts_erroneous_assertion_instead_of_pretending_historical_truth(self):
        first = self.memory()
        await self.service.process_extraction(self.scope, [first])
        second = self.memory("大象", "更正：zfc 不是飞猪而是大象，之前说错了。")
        report = await self.service.process_extraction(self.scope, [second])
        self.assertEqual(report["applied"][0]["action"], "correct")
        old = next(
            m for m in self.store.memories(self.scope) if m.memory_id == first.memory_id and m.version == 1
        )
        self.assertEqual(old.status, "retracted")

    async def test_question_or_hypothesis_never_authorizes_correction(self):
        first = self.memory()
        second = first.model_copy(update={"value": "大象"})
        for quote in ("zfc 从飞猪改为大象了吗？", "假如 zfc 从飞猪改为大象。"):
            self.assertIsNone(correction_relation(second, first, [quote]))

    async def test_expired_event_remains_available_to_cross_session_history(self):
        now = utcnow()
        first = self.memory(
            kind="event", valid_from=now - timedelta(days=3), valid_to=now - timedelta(days=1)
        )
        await self.service.process_extraction(self.scope, [first])
        result = await self.service.maintain(self.scope)
        current = self.store.get_memory(self.scope, first.memory_id)
        self.assertEqual(current.status, "active")
        self.assertEqual(current.tier, "long")
        self.assertEqual(current.evidence, first.evidence)
        self.assertEqual(result["applied"], [])
        self.assertTrue(result["retention"][0]["expired"])
        self.assertEqual(result["retention"][0]["recommendation"], "keep")
        other_session = self.store.create_session(self.scope)
        self.assertFalse(Retriever._eligible(current, other_session, QueryPlan(), now))
        self.assertTrue(Retriever._eligible(
            current, other_session, QueryPlan(temporal_mode="history"), now
        ))
        self.assertTrue(Retriever._eligible(
            current, other_session, QueryPlan(as_of=now - timedelta(days=2)), now
        ))
        self.assertEqual(summary_groups([current], now), [])

    async def test_explicit_archive_and_retract_still_exclude_historical_event(self):
        now = utcnow()
        first = self.memory(kind="event", valid_to=now - timedelta(days=1))
        await self.service.process_extraction(self.scope, [first])
        archived = self.store.evolve(
            self.scope, first.memory_id, EvolutionInput(action="archive", expected_version=first.version)
        )
        await self.service.maintain(self.scope)
        self.assertEqual(self.store.get_memory(self.scope, first.memory_id).status, "archived")
        other_session = self.store.create_session(self.scope)
        for plan in (QueryPlan(), QueryPlan(temporal_mode="history"),
                     QueryPlan(as_of=now - timedelta(days=2))):
            self.assertFalse(Retriever._eligible(archived, other_session, plan, now))
        restored = self.store.evolve(
            self.scope,
            archived.memory_id,
            EvolutionInput(
                action="restore",
                expected_version=archived.version,
            ),
        )
        self.assertEqual(restored.status, "active")
        self.assertEqual(restored.evidence, first.evidence)
        self.assertEqual(summary_groups([restored], utcnow()), [])
        retracted = self.store.evolve(
            self.scope, restored.memory_id,
            EvolutionInput(action="retract", expected_version=restored.version),
        )
        for plan in (QueryPlan(), QueryPlan(temporal_mode="history"),
                     QueryPlan(as_of=now - timedelta(days=2))):
            self.assertFalse(Retriever._eligible(retracted, other_session, plan, now))

    async def test_old_event_only_gets_archive_recommendation(self):
        memory = self.memory(kind="event", valid_to=utcnow() - timedelta(days=1))
        memory = memory.model_copy(update={"recorded_at": utcnow() - timedelta(days=900)})
        decision = retention(memory, utcnow())
        self.assertEqual(decision["recommendation"], "review_archive")
        self.assertTrue(decision["expired"])
        self.assertLess(decision["strength"], 0.25)

    async def test_old_stable_fact_only_gets_archive_recommendation(self):
        memory = self.memory().model_copy(update={"recorded_at": utcnow() - timedelta(days=900)})
        decision = retention(memory, utcnow())
        self.assertEqual(decision["recommendation"], "review_archive")
        self.assertLess(decision["strength"], 0.25)

    async def test_summary_is_derived_and_every_item_cites_exact_version(self):
        memory = self.memory()
        await self.service.process_extraction(self.scope, [memory])
        report = await self.service.maintain(self.scope)
        group = report["groups"][0]
        self.assertFalse(group["usable_as_extraction_evidence"])
        self.assertEqual(group["summary_items"][0]["memory_ref"], f"{memory.memory_id}:1")
        self.assertIn("飞猪", group["summary"])
        self.assertEqual(self.store.groups(self.scope)[0]["summary"], group["summary"])

    async def test_model_summary_failure_keeps_extracts_and_extractive_projection(self):
        class BrokenModel:
            async def summarize(self, memories):
                raise RuntimeError("offline")

        memory = self.memory()
        await self.service.process_extraction(self.scope, [memory])
        self.service.model = BrokenModel()
        report = await self.service.maintain(self.scope)
        self.assertTrue(report["warnings"])
        self.assertTrue(report["groups"][0]["summary"])
        self.assertEqual(self.store.get_memory(self.scope, memory.memory_id).status, "active")

    async def test_model_cannot_publish_foreign_summary_sources(self):
        class ForeignModel:
            async def summarize(self, memories):
                return {"summary": "伪造", "source_refs": ["other-owner:1"]}

        memory = self.memory()
        await self.service.process_extraction(self.scope, [memory])
        self.service.model = ForeignModel()
        report = await self.service.maintain(self.scope)
        self.assertTrue(report["warnings"])
        self.assertNotIn("synthesis", self.store.groups(self.scope)[0])

    async def test_summaries_exist_at_every_level_and_cache_only_unchanged_sources(self):
        class SummaryModel:
            calls = 0

            async def summarize(self, memories):
                self.calls += 1
                return {
                    "summary": "zfc 的昵称为飞猪。",
                    "source_refs": [f"{memories[0].memory_id}:{memories[0].version}"],
                }

        memory = self.memory()
        await self.service.process_extraction(self.scope, [memory])
        model = SummaryModel()
        self.service.model = model
        first = await self.service.maintain(self.scope)
        self.assertEqual({g["level"] for g in first["groups"]}, {1, 2, 3})
        self.assertEqual(first["semantic_synthesis"], "model_assisted_with_sources")
        self.assertEqual(model.calls, 3)
        await self.service.maintain(self.scope)
        self.assertEqual(model.calls, 3)
        self.store.evolve(self.scope, memory.memory_id, EvolutionInput(action="retract", expected_version=1))
        self.assertEqual(self.store.groups(self.scope), [])

    async def test_unused_model_relation_advice_is_not_requested_for_conflict(self):
        class RelationModel:
            async def relate(self, source, targets):
                raise AssertionError("Unused relation proposal must not incur a model call")

        first = self.memory()
        await self.service.process_extraction(self.scope, [first])
        self.service.model = RelationModel()
        second = self.memory("大象")
        result = await self.service.process_extraction(self.scope, [second])
        self.assertNotIn("model_proposal", result["review"][0])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(self.store.get_memory(self.scope, first.memory_id).value, "飞猪")
        retained = self.store.get_memory(self.scope, second.memory_id)
        self.assertEqual((retained.status, retained.tier), ("active", "short"))
        self.assertEqual(retained.evidence, second.evidence)

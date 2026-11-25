"""Public search top-k does not truncate the answer's original-source catalog."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from agent_memory.models import Candidate, Event, Evidence, Scope, TurnInput
from agent_memory.prompt_codec import SOURCE_CATALOG_OVERHEAD, compact_json, synthesis_source_cost
from agent_memory.providers import CallableModel
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings


QUESTION = "List all sample items that Mira placed in the cabinet."
OWNER = Scope(tenant_id="fixture", owner_id="mira")


@pytest.mark.parametrize("item_count,context_budget", [(26, 22000), (125, 30000)])
def test_short_original_units_use_available_budget_while_search_retains_top_k(tmp_path, item_count, context_budget):
    calls, payloads = [], []

    async def chat(prompt):
        calls.append(prompt)
        if prompt.startswith("Plan memory retrieval"):
            return json.dumps({"route": "both", "semantic_queries": [QUESTION],
                "keywords": ["Mira", "sample", "cabinet"], "subject": "Mira", "predicate": "placed",
                "answer_contract": {"outputs": [{
                    "output_id": "items", "question_span": QUESTION, "operator": "set",
                    "subjects": ["Mira"], "relation": "placed", "count_unit": "value",
                }]}})
        assert prompt.startswith("Answer the original question")
        payload = json.JSONDecoder().raw_decode(prompt[prompt.index('{"question":'):])[0]
        payloads.append(payload)
        first = payload["sources"][0]
        quote = first["source_events"][0]["text"]
        # A frozen reader response tests the real runtime transport, not model
        # quality or whether a model will always enumerate every item.
        return json.dumps({"slots": [{"target": QUESTION,
            "derivation": "Read the supplied cabinet records.", "values": [{
                "value": quote, "supports": [{"source_id": first["source_id"], "quote": quote}],
            }]}]})

    async def run():
        settings = Settings(db_path=tmp_path / "reading.db", planner_mode="model", context_turns=0,
            prompt_token_limit=context_budget, complex_prompt_token_limit=context_budget,
            max_candidates=max(120, item_count), maintenance_interval=0, acceleration_enabled=False,
            grounding_semantic_enabled=False)
        runtime = MemoryRuntime(settings, model=CallableModel(chat))
        try:
            session = runtime.store.create_session(OWNER)
            originals = {}
            for index in range(item_count):
                text = f"Mira placed sample item {index:03d} in the cabinet."
                turn = runtime.store.append_turn(OWNER, session.session_id, TurnInput(
                    request_id=f"item-{index}", events=[Event(role="user", speaker="Mira", content=text,
                        occurred_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=index))],
                ))
                originals[turn.turn_id] = text
                runtime.store.add_candidates(OWNER, session.session_id, [Candidate(
                    content=text, subject="Mira", predicate="placed", value=f"sample item {index:03d}",
                    evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote=text)],
                )])

            for requested_k in (3, 7):
                found = await runtime.search(OWNER, session.session_id, QUESTION, top_k=requested_k)
                assert len(found.results) == requested_k
                assert len(found.candidates) > len(found.results)

            result = await runtime.answer(OWNER, session.session_id, QUESTION, top_k=3,
                                          prepare=False, accelerate=False, source_read_scope="events")
            assert len(payloads) == 1  # No extra reader/repair stage.
            payload = payloads[0]
            assert 20 < len(payload["sources"]) <= 120
            if item_count == 26:
                assert len(payload["sources"]) == item_count
            assert len(result.sources) == len(payload["sources"])
            assert [s["source_id"] for s in payload["sources"]] == list(range(1, len(result.sources) + 1))
            wire_cost = len(compact_json({"sources": payload["sources"]}))
            accounted = SOURCE_CATALOG_OVERHEAD + sum(synthesis_source_cost(hit, i, needs_time=False)
                for i, hit in enumerate(result.sources, 1))
            assert wire_cost == accounted
            assert result.grounding.context_serialized_chars + SOURCE_CATALOG_OVERHEAD == wire_cost
            assert result.dynamic_k is not None
            assert wire_cost <= result.dynamic_k.token_limit + SOURCE_CATALOG_OVERHEAD
            delivered = set()
            for hit in result.sources:
                for event in hit.metadata["source_events"]:
                    assert originals[event["turn_id"]][event["start"]:event["end"]] == event["text"]
                    delivered.add(event["turn_id"])
            if item_count == 26:
                assert delivered == set(originals)
            assert not any("ModelOutputError" in warning or "schema_error" in warning
                           for warning in result.warnings)
        finally:
            await runtime.close()

    asyncio.run(run())

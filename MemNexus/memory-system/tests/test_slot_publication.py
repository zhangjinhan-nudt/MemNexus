"""The existing slot role has one coverage, rendering and citation contract."""

import asyncio

import pytest
from test_answer_synthesis import hit, turn

from agent_memory.models import Event, QueryPlan, Scope, TurnInput
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.synthesis import AnswerTable, answer_values, render_table, validate_table


def value(text, quote, source_id=1, **extra):
    return {"value": text, "supports": [{"source_id": source_id, "quote": quote}], **extra}


def slot(target, values, **extra):
    return {"target": target, "derivation": "Use the quoted source for this target.", "values": values, **extra}


def verify(slots, sources):
    table = AnswerTable.model_validate({"slots": slots})
    validate_table(table, sources, [item["target"] for item in slots], "text")
    return table


def test_context_slot_accepts_unavailable_without_publishing_its_values():
    quote = "I lead four engineers as Senior Software Engineer."
    table = verify([slot("manager team size", [value("four engineers", quote)], role="context",
                         coverage="unavailable", missing="The requested role is not established")],
                   [hit(turn(1, quote))])
    assert answer_values(table.slots[0]) == []
    assert len(table.slots[0].values) == 1
    assert table.slots[0].coverage == "unavailable"
    assert render_table(table) == "Not mentioned in the conversation"


def test_context_slot_still_requires_valid_source_quotes():
    with pytest.raises(ValueError, match="exact quote"):
        verify([slot("background", [value("four engineers", "I lead four engineers.")],
                     role="context", coverage="unavailable")], [hit(turn(1, "I lead two engineers."))])


def test_context_user_assertion_cannot_supply_another_slots_recommendation():
    quote = "I own a bike."
    with pytest.raises(ValueError, match="quoted user assertion"):
        verify([
            slot("background", [value("bike", quote)], role="context", coverage="unavailable"),
            slot("advice", [{"value": "Visit the music venue", "basis": "recommendation"}]),
        ], [hit(turn(1, quote))])


def test_context_missing_does_not_change_an_independent_answer_slot():
    quote = "I planted five tomatoes."
    table = verify([
        slot("basil", [value("basil", quote, unresolved_premise="Basil quantity is unknown")],
             role="context", coverage="unavailable", missing="Basil quantity is unknown",
             missing_kind="necessary_premise"),
        slot("tomatoes", [value("five tomatoes", quote)]),
    ], [hit(turn(1, quote))])
    assert table.slots[1].coverage == "complete"
    assert table.slots[1].missing == ""
    assert render_table(table) == "five tomatoes"


def test_context_with_partial_missing_cannot_publish_gap_next_to_independent_answer():
    tomato, basil = "I planted five tomatoes.", "I grew basil last year."
    table = verify([
        slot("tomatoes", [value("five tomatoes", tomato)]),
        slot("background", [value("basil", basil, source_id=2)], role="context",
             coverage="partial", missing="Basil quantity is unknown", missing_kind="requested_part"),
    ], [hit(turn(1, tomato)), hit(turn(2, basil))])
    assert render_table(table, question="How many tomatoes and basil plants did I grow?") == "five tomatoes"


def test_original_default_answer_format_and_personal_basis_still_work():
    quote = "I own a power bank."
    table = verify([slot("advice", [value("Carry your power bank", quote, basis="recommendation"),
                                    {"value": "Enable battery saver", "basis": "recommendation"}])],
                   [hit(turn(1, quote))])
    assert table.slots[0].role == "answer"
    assert answer_values(table.slots[0]) == table.slots[0].values
    assert render_table(table) == "Carry your power bank, Enable battery saver"
    assert "role" not in table.slots[0].values[0].model_dump()


@pytest.mark.parametrize("mixed", [False, True])
def test_runtime_only_publishes_citations_from_answer_slots(tmp_path, mixed):
    related = "I lead four engineers as Senior Software Engineer."
    direct = "Our team meets every Monday."
    expected_ids = []

    class Model:
        async def plan(self, query, context):
            return QueryPlan(route="short", keywords=["engineers", "team"], depth=3)

        async def synthesize(self, question, plan, sources, **kwargs):
            def source_id(text):
                return next(i for i, source in enumerate(sources, 1)
                            if any(text in event["text"] for event in source.metadata["source_events"]))

            slots = [slot("manager team size", [value("four engineers", related, source_id(related))],
                          role="context", coverage="unavailable", missing="Manager role is not established")]
            if mixed:
                direct_id = source_id(direct)
                expected_ids.append(direct_id)
                slots.append(slot("meeting day", [value("Monday", direct, direct_id)]))
            return verify(slots, sources)

    async def run():
        runtime = MemoryRuntime(Settings(db_path=tmp_path / "publication.db", maintenance_interval=0), model=Model())
        scope = Scope(tenant_id="test", owner_id="user")
        try:
            session = runtime.store.create_session(scope)
            for index, text in enumerate((related, direct) if mixed else (related,)):
                await runtime.append_turn(scope, session.session_id,
                                          TurnInput(request_id=str(index), events=[Event(role="user", content=text)]))
            result = await runtime.answer(scope, session.session_id,
                                          "Which day does the team meet and how many engineers do I lead as Manager?",
                                          prepare=False, accelerate=False)
            assert result.generated_text == ("Monday" if mixed else "Not mentioned in the conversation")
            assert result.citations == expected_ids
            context = result.reasoning["answer_table"]["slots"][0]
            assert context["role"] == "context" and context["coverage"] == "unavailable"
            assert context["values"]  # Retain the source-bound audit even when it is not published.
        finally:
            await runtime.close()

    asyncio.run(run())

from datetime import UTC, datetime, timedelta

import pytest

from agent_memory.grounding import _source_event, close_evidence_bundles
from agent_memory.models import CollectionQuery, Event, Hit, QueryPlan, Turn
from agent_memory.reasoning import apply_route, classify_question
from agent_memory.synthesis import AnswerTable, answer_type, render_table, validate_table


def turn(n, text, *, session="s", minute=0):
    return Turn(turn_id=f"t{n}", request_id=f"r{n}", session_id=session, sequence=n,
                events=[Event(role="user", speaker="Inez", content=text,
                              occurred_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=minute))])


def hit(t):
    return Hit(content=t.events[0].content, score=.9, source_file="source", chunk_id=t.turn_id,
               metadata={"source_events": [_source_event(t, 0, t.events[0].content, 0)]})


def table(value="red", quote="red", source_id=1, target="colors"):
    return AnswerTable.model_validate({"slots": [{"target": target, "values": [
        {"value": value, "supports": [{"source_id": source_id, "quote": quote}]}]}]})


def test_all_conjuncts_rendered_and_quotes_checked():
    sources = [hit(turn(1, "I prefer red and blue."))]
    result = table()
    result.slots[0].values.append(table("blue", "blue").slots[0].values[0])
    validate_table(result, sources, ["colors"], "text")
    assert render_table(result) == "red; blue"


@pytest.mark.parametrize("result", [table(quote="green"), table(source_id=2), table(target="other")])
def test_invalid_provenance_or_lost_target_rejected(result):
    with pytest.raises(ValueError):
        validate_table(result, [hit(turn(1, "red"))], ["colors"], "text")


def test_duration_cannot_turn_into_event_enumeration():
    question = "After how many weeks did Inez visit again?"
    route = classify_question(question)
    plan = QueryPlan(required_info=["first visit date", "return date"],
                     collection_query=CollectionQuery(subjects=["Inez"], predicate="visits"))
    routed = apply_route(plan, route)
    assert route.kind == routed.query_type == "temporal"
    assert routed.required_info == plan.required_info
    assert answer_type(question) == "duration"
    with pytest.raises(ValueError, match="Duration"):
        validate_table(table("visit", "visit"), [hit(turn(1, "visit"))], ["colors"], "duration")


def test_bundle_closes_explanation_without_crossing_authorization_or_time():
    turns = [turn(1, "I finished a project."), turn(2, "What was it?"),
             turn(3, "A solar-powered pump."), turn(4, "Retracted secret."),
             turn(5, "Other session.", session="other"), turn(6, "Next day.", minute=1440)]
    sources = close_evidence_bundles([hit(turns[0])], turns, session_id="s",
                                    blocked_events={("t4", 0)}, budget=5000)
    events = sources[0].metadata["source_events"]
    assert {e["turn_id"] for e in events} == {"t1", "t2", "t3"}
    assert next(e for e in events if e["turn_id"] == "t2")["context_only"]
    assert not next(e for e in events if e["turn_id"] == "t3")["context_only"]


def test_date_difference_is_computed_and_cannot_invent_endpoints():
    source = hit(turn(1, "We met on 2024-03-01 and returned on 2024-03-15."))
    result = table("999 weeks", source.content)
    from agent_memory.synthesis import DateDifference
    result.slots[0].values[0].calculation = DateDifference(start="2024-03-01", end="2024-03-15", unit="weeks")
    validate_table(result, [source], ["colors"], "duration")
    assert render_table(result) == "2 weeks"
    result.slots[0].values[0].calculation = DateDifference(start="2024-02-23", end="2024-03-15", unit="weeks")
    with pytest.raises(ValueError, match="endpoints"):
        validate_table(result, [source], ["colors"], "duration")


def test_unresolved_slot_is_not_silently_treated_as_complete():
    result = AnswerTable.model_validate({"slots": [{"target": "colors", "missing": "No statement"}]})
    validate_table(result, [], ["colors"], "text")
    assert render_table(result) == "Not mentioned in the conversation"
    assert result.slots[0].missing


def test_context_premise_does_not_leak_into_final_answer():
    result = table("two years", "two years")
    context = table("2025-01-01", "two years", target="anchor").slots[0]
    context.role = "context"
    result.slots.append(context)
    validate_table(result, [hit(turn(1, "two years"))], ["colors", "anchor"], "duration")
    assert render_table(result) == "two years"


def test_question_can_link_context_but_cannot_establish_fact_alone():
    source = hit(turn(1, "Is it blue?"))
    source.metadata["source_events"].append(_source_event(turn(2, "It is red."), 0, "It is red.", 0))
    result = table("red", "Is it blue?")
    with pytest.raises(ValueError, match="alone"):
        validate_table(result, [source], ["colors"], "text")
    from agent_memory.synthesis import Support
    result.slots[0].values[0].supports.append(Support(source_id=1, quote="It is red."))
    validate_table(result, [source], ["colors"], "text")


def test_inverse_duration_only_establishes_start_year():
    from agent_memory.synthesis import DateDifference
    result = table("2020", "two years")
    result.slots[0].values[0].calculation = DateDifference(start="2023-01-01", end="2025-01-01", unit="years")
    validate_table(result, [hit(turn(1, "two years"))], ["colors"], "date")
    assert render_table(result) == "2023"
    assert result.slots[0].values[0].calculation is None
    result.slots[0].values[0].calculation = DateDifference(start="2022-01-01", end="2025-01-01", unit="years")
    with pytest.raises(ValueError, match="Start-year"):
        validate_table(result, [hit(turn(1, "two years"))], ["colors"], "date")

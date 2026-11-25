import json
from datetime import UTC, datetime, timedelta

import pytest

from agent_memory.grounding import (
    _source_event,
    close_evidence_bundles,
    source_payload,
)
from agent_memory.models import CollectionQuery, Event, Hit, QueryPlan, Turn
from agent_memory.reasoning import apply_route, classify_question
from agent_memory.synthesis import AnswerTable, Support, answer_type, render_table, validate_table


def turn(n, text, *, session="s", minute=0):
    return Turn(turn_id=f"t{n}", request_id=f"r{n}", session_id=session, sequence=n,
                events=[Event(role="user", speaker="Inez", content=text,
                              occurred_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=minute))])


def hit(t):
    return Hit(content=t.events[0].content, score=.9, source_file="source", chunk_id=t.turn_id,
               metadata={"source_events": [_source_event(t, 0, t.events[0].content, 0)]})


def table(value="red", quote="red", source_id=1, target="colors"):
    return AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": target, "values": [
        {"value": value, "supports": [{"source_id": source_id, "quote": quote}]}]}]})


def test_all_conjuncts_rendered_and_quotes_checked():
    sources = [hit(turn(1, "I prefer red and blue."))]
    result = table()
    result.slots[0].values.append(table("blue", "blue").slots[0].values[0])
    validate_table(result, sources, ["colors"], "text")
    assert render_table(result) == "red, blue"


@pytest.mark.parametrize("result", [table(quote="green"), table(source_id=2), table(target="other")])
def test_invalid_provenance_or_lost_target_rejected(result):
    with pytest.raises(ValueError):
        validate_table(result, [hit(turn(1, "red"))], ["colors"], "text")


def test_quote_may_repeat_the_speaker_label_the_event_only_names_once():
    # Real failure: the model cited a mid-utterance span and prefixed it with
    # "Caroline: ", which the stored event only carries at the start. Rejecting
    # that killed the whole answer call, so the question was lost at random.
    source = hit(turn(1, "Caroline: Thanks, Melanie. Now I'm passionate about creating a safe place."))
    result = table("safe place", "Caroline: Now I'm passionate about creating a safe place.")
    validate_table(result, [source], ["colors"], "text")
    assert render_table(result) == "safe place"


def test_quote_tolerates_width_case_and_whitespace_only_differences():
    source = hit(turn(1, "I prefer  RED and blue."))
    validate_table(table("red", "i   prefer red"), [source], ["colors"], "text")


@pytest.mark.parametrize("source_text, quote", [
    # Different speaker: borrowing another speaker's words is a provenance error.
    ("Melanie: Now I'm passionate about creating a safe place.",
     "Caroline: Now I'm passionate about creating a safe place."),
    # Same speaker, different words: a paraphrase is not a quote.
    ("Caroline: Now I'm passionate about creating a safe place.",
     "Caroline: Now I love building a safe place."),
])
def test_quote_still_rejects_other_speaker_or_paraphrase(source_text, quote):
    with pytest.raises(ValueError):
        validate_table(table("safe place", quote), [hit(turn(1, source_text))], ["colors"], "text")


def test_quote_ignores_markdown_emphasis_markers():
    # Measured regression (2026-11-13): 17 of 18 recorded exact-quote failures
    # were this alone. LongMemEval assistant turns are markdown; the stored event
    # reads "**Store Utensils by Task:** Organize ..." while the model, reading
    # rendered text, cites "Store Utensils by Task: Organize ...". Bold markers
    # are presentation, so the quote names the same words. Rejecting it killed
    # the whole answer (forced abstention) instead of accepting it.
    source = hit(turn(1, "**Store Utensils by Task:** Organize your utensils by task."))
    validate_table(
        table("organize utensils by task", "Store Utensils by Task: Organize your utensils by task."),
        [source], ["colors"], "text",
    )


def test_quote_ignores_inline_code_and_italic_markers():
    source = hit(turn(1, "Use the `_config` flag and set *verbose* mode."))
    validate_table(table("use config flag", "Use the config flag and set verbose mode."),
                   [source], ["colors"], "text")


def test_quote_does_not_ignore_hashtag_or_heading_markers():
    # '#' is meaningful in the evidence (hashtag names), so it is NOT stripped:
    # a quote must still carry it to match. Stripping it would let a quote match
    # a different token.
    source = hit(turn(1, "I joined the #PlankChallenge on social media."))
    validate_table(table("plank challenge", "I joined the #PlankChallenge on social media."),
                   [source], ["colors"], "text")
    with pytest.raises(ValueError):
        # Dropping the '#' no longer matches the stored token.
        validate_table(table("plank challenge", "I joined the PlankChallenge on social media."),
                       [source], ["colors"], "text")


def test_duration_routing_preserves_requested_operation():
    question = "After how many weeks did Inez visit again?"
    route = classify_question(question)
    plan = QueryPlan(required_info=["first visit date", "return date"],
                     collection_query=CollectionQuery(subjects=["Inez"], predicate="visits"))
    routed = apply_route(plan, route)
    assert route.kind == routed.query_type == "temporal"
    assert routed.required_info == plan.required_info
    assert answer_type(question) == "duration"
    # Quote provenance is not entailment. A short English unit regex cannot
    # prove duration semantics; preserve the requested operation for synthesis.


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


def test_duration_value_is_not_overwritten_from_unbound_calendar_endpoints():
    source = hit(turn(1, "We met on 2024-03-01 and returned on 2024-03-15."))
    result = table("999 weeks", source.content)
    original = result.model_dump()
    validate_table(result, [source], ["colors"], "duration")
    # Provenance checks do not establish the mathematical truth of the value.
    assert result.model_dump() == original
    assert render_table(result) == "999 weeks"


def test_unresolved_slot_is_not_silently_treated_as_complete():
    result = AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": "colors", "coverage": "unavailable", "missing": "No statement"}]})
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
    result.slots[0].values[0].supports.append(Support(source_id=1, quote="It is red."))
    validate_table(result, [source], ["colors"], "text")


def test_compact_source_payload_preserves_provenance_with_less_text():
    source = hit(turn(1, "A long factual statement about a project and its date."))
    full = source_payload(source)
    compact = source_payload(source, compact=True, include_time=False)
    assert "claim" not in compact and "claim" not in full
    assert compact["source_events"][0]["text"] == source.content
    assert len(json.dumps(compact, ensure_ascii=False)) <= len(json.dumps(full, ensure_ascii=False))


def test_evidence_bundle_expands_local_context_without_scanning_new_sessions():
    turns = [turn(1, "I finished a project."), turn(2, "It was a solar pump.", minute=5),
             turn(3, "Other session answer.", session="other", minute=5)]
    sources = close_evidence_bundles([hit(turns[0])], turns, session_id="s",
                                     blocked_events=set(), budget=5000)
    assert {event["turn_id"] for source in sources for event in source.metadata["source_events"]} == {"t1", "t2"}


@pytest.mark.parametrize(
    "question,expected",
    [
        # A time span inside a counting question filters the members; it is not
        # the unit of the answer.
        ("How many MCU films did I watch in the last 3 months?", "count"),
        ("How many different cuisines have I tried in the past few months?", "count"),
        ("How many engineers do I lead when I just started?", "count"),
        # A unit right after "how many" is genuinely a duration answer.
        ("How many days passed between the feedback and the test?", "duration"),
        ("How many weeks had passed since I recovered?", "duration"),
        # The question head asks for duration even without naming its unit.
        ("How long has Caroline had her friends?", "duration"),
        ("How long had I studied when I took the exam?", "duration"),
    ],
)
def test_answer_type_separates_count_from_duration(question, expected):
    assert answer_type(question) == expected

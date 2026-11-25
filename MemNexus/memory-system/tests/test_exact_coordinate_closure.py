"""Source hints and context closure respect a quote's explicit occurrence."""

from agent_memory.grounding import (
    _extraction_hint,
    _reference_intervals,
    _source_event,
    close_evidence_bundles,
)
from agent_memory.models import Event, Hit, QueryPlan, Turn


def turn(text):
    return Turn(turn_id="t", session_id="old", request_id="r", sequence=1,
                events=[Event(role="user", content=text)])


def test_scalar_and_member_hints_cannot_rebind_to_another_identical_occurrence():
    original = turn("blue then blue")
    reference = {"turn_id": "t", "event_index": 0, "quote": "blue", "start": 10, "end": 14}
    hit = Hit(content="blue", score=1, source_file="memory", chunk_id="m",
              metadata={"subject": "Alex", "predicate": "likes", "value": "blue", "evidence": [reference]})
    first = _source_event(original, 0, "blue", 0)
    second = _source_event(original, 0, "blue", 10)
    assert _extraction_hint(hit, first) == {}
    assert _extraction_hint(hit, second)["spans"] == [[0, 4]]
    hit.metadata["semantics"] = {"cardinality": "many", "members": [
        {"value": "blue", "identity": "blue", "evidence": [reference]}]}
    assert _extraction_hint(hit, first) == {}
    assert _extraction_hint(hit, second)["members"][0]["spans"] == [[0, 4]]


def test_archived_closure_adds_only_the_explicit_second_quote_interval():
    quote = "Recipe card. " + "Preparation background. " * 40 + "Ingredients: eggs and flour."
    text = quote + " The private intervening note is not part of either quote. " + quote
    original = turn(text)
    start = text.rindex(quote)
    reference = {"turn_id": "t", "event_index": 0, "quote": quote,
                 "start": start, "end": start + len(quote)}
    seed = _source_event(original, 0, "Recipe card.", start)
    hit = Hit(content=seed["text"], score=1, source_file="memory", chunk_id="m", metadata={
        "source_support_status": "authorized_session_quote", "source_events": [seed], "evidence": [reference]})
    result = close_evidence_bundles([hit], [original], session_id="current", blocked_events={("t", 0)},
        query="Which ingredients does the recipe use?", plan=QueryPlan(route="both"), budget=5000,
        authorized_quotes={("t", 0): [reference]})
    events = result[0].metadata["source_events"]
    assert any("eggs and flour" in item["text"] for item in events)
    assert all(start <= item["start"] < item["end"] <= start + len(quote) for item in events)
    assert all(text[item["start"]:item["end"]] == item["text"] for item in events)


def test_invalid_position_grant_does_not_fall_back_to_a_literal_elsewhere():
    original = turn("Public note. Secret allocation is room 9.")
    seed = _source_event(original, 0, "Public note.", 0)
    hit = Hit(content=seed["text"], score=1, source_file="memory", chunk_id="m", metadata={"source_events": [seed]})
    wrong = {"turn_id": "t", "event_index": 0, "quote": "Secret allocation is room 9.", "start": 0, "end": 28}
    result = close_evidence_bundles([hit], [original], session_id="current", blocked_events=set(),
        query="What is the allocation?", plan=QueryPlan(route="both"), budget=5000,
        authorized_quotes={("t", 0): [wrong]})
    assert result[0].metadata["source_events"] == [seed]


def test_legacy_ambiguous_reference_keeps_overlapping_literal_positions():
    original = turn("aaa")
    positions = _reference_intervals({"turn_id": "t", "event_index": 0, "quote": "aa"}, {"t": original})
    assert positions == [(0, 2), (1, 3)]

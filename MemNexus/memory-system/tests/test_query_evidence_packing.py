"""Evidence visibility and authorization checks, not benchmark accuracy tests."""
from datetime import UTC, datetime, timedelta

from agent_memory.grounding import (
    _chunks,
    _cost,
    _source_event,
    close_evidence_bundles,
    collect_raw_records,
)
from agent_memory.models import Event, Hit, QueryPlan, Turn
from agent_memory.prompt_codec import evidence_catalog


def turn(identifier, text, *, session="s", minute=0):
    return Turn(turn_id=identifier, session_id=session, request_id=identifier, sequence=1,
                events=[Event(role="assistant", content=text,
                              occurred_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=minute))])


def source(t, text=None, start=0):
    text = t.events[0].content if text is None else text
    return Hit(content=text, score=.8, source_file=t.turn_id, chunk_id=t.turn_id,
               metadata={"status": "active", "source_events": [_source_event(t, 0, text, start)]})


def test_chunking_is_lossless_and_keeps_numbered_lines_whole():
    text = "Intro. Another sentence.\n" + "".join(f"{i}. Choice number {i}, with explanation.\n" for i in range(1, 70))
    chunks = _chunks(text)
    assert "".join(piece for _, piece in chunks) == text
    assert all(text[start:start + len(piece)] == piece and len(piece) <= 700 for start, piece in chunks)
    assert all(piece.endswith("\n") for _, piece in chunks)
    for long_text in ("x" * 2500, "word " * 700):
        assert "".join(piece for _, piece in _chunks(long_text)) == long_text


def test_historical_closure_uses_only_authorized_quotes():
    anchor = turn("old", "I visited Lisbon.", session="old")
    neighbor = turn("detail", "PRIVATE medical note. The Lisbon visit was in May.", session="old")
    foreign = turn("foreign", "Private details in a different session.", session="foreign")
    allowed = {(anchor.turn_id, 0): ["I visited Lisbon."],
               (neighbor.turn_id, 0): ["The Lisbon visit was in May."],
               (foreign.turn_id, 0): [foreign.events[0].content]}
    seed = source(anchor)
    without = close_evidence_bundles([seed], [anchor, neighbor, foreign], session_id="new",
                                    blocked_events=set(), budget=5000, query="When did I visit Lisbon?")
    assert without == [seed]
    result = close_evidence_bundles([seed], [anchor, neighbor, foreign], session_id="new",
                                   blocked_events=set(), budget=5000, query="When did I visit Lisbon?",
                                   authorized_quotes=allowed)
    text = " ".join(e["text"] for h in result for e in h.metadata["source_events"])
    assert "May" in text
    assert "PRIVATE" not in text and "different session" not in text
    blocked = close_evidence_bundles([seed], [anchor, neighbor], session_id="new",
                                    blocked_events={(neighbor.turn_id, 0)}, budget=5000,
                                    query="When did I visit Lisbon?", authorized_quotes=allowed)
    assert all("May" not in e["text"] for h in blocked for e in h.metadata["source_events"])


def test_ordinal_selects_middle_of_recalled_event_with_same_budget_and_offsets():
    text = "Saved settings:\n" + "".join(f"{i}. Setting value {i}; this is a complete item.\n" for i in range(1, 90))
    t = turn("list", text)
    seed = source(t, text[:80])
    budget = _cost(seed) + 1200
    result = close_evidence_bundles([seed], [t], session_id="s", blocked_events=set(),
                                    budget=budget, query="What was the 27th setting you recommended?")
    # Check the exact provider evidence catalog, not merely a source's claim.
    texts = [event["text"] for s in evidence_catalog(result)["sources"] for event in s["source_events"]]
    assert any("27. Setting value 27; this is a complete item." in s for s in texts)
    assert result[0].chunk_id == seed.chunk_id and result[0].content == seed.content
    assert seed.metadata["source_events"] == [_source_event(t, 0, text[:80], 0)]
    for e in result[0].metadata["source_events"]:
        assert text[e["start"]:e["end"]] == e["text"]
    assert sum(_cost(h) for h in result) <= budget


def test_neighbor_reads_relevant_budget_section_without_unrelated_prefix_reservation():
    anchor = turn("anchor", "Can you repeat the retreat campaign plan?")
    campaign = turn("campaign", "Campaign introduction.\n" + "General discussion.\n" * 70
                    + "Budget:\nInfluencer marketing: 1800 credits.\n" + "Other notes.\n" * 50)
    seed = source(anchor)
    budget = _cost(seed) + 1300
    result = close_evidence_bundles([seed], [anchor, campaign], session_id="s", blocked_events=set(),
                                    budget=budget, query="How much was allocated for influencer marketing?",
                                    plan=QueryPlan(required_info=["influencer marketing allocation"]))
    assert any("Influencer marketing: 1800 credits." in e["text"]
               for e in result[0].metadata["source_events"])
    assert all(campaign.events[0].content[e["start"]:e["end"]] == e["text"]
               for e in result[0].metadata["source_events"] if e["turn_id"] == "campaign")
    assert sum(_cost(h) for h in result) <= budget


def test_partial_overlap_cannot_hide_remainder_of_a_list_item():
    text = "1. introductory item\n2. " + "detailed item " * 20 + "COMPLETE_TAIL.\n3. unrelated item\n"
    t = turn("overlap", text)
    seed = source(t, text[:60])
    result = close_evidence_bundles([seed], [t], session_id="s", blocked_events=set(),
                                    budget=2500, query="What is the 2nd item?")
    assert any("COMPLETE_TAIL." in e["text"] for e in result[0].metadata["source_events"])


def test_scope_time_and_lifecycle_fences_apply_to_self_and_neighbor_expansion():
    anchor = turn("anchor", "Public quote. Private same-event remainder.")
    foreign = turn("foreign", "Foreign confidential remainder.", session="elsewhere")
    blocked = turn("blocked", "Blocked confidential remainder.")
    future = turn("future", "Future confidential remainder.", minute=10)
    old = turn("old", "Older-session-time confidential remainder.", minute=-1440)
    seed = source(anchor, "Public quote.")
    result = close_evidence_bundles([seed], [anchor, foreign, blocked, future, old],
                                    session_id="s", blocked_events={("blocked", 0), ("anchor", 0)},
                                    budget=10000, query="confidential remainder")
    assert result == [seed]  # A blocked anchor cannot authorize local expansion.
    result = close_evidence_bundles([seed], [anchor, future, foreign, blocked, old],
                                    session_id="s", blocked_events={("blocked", 0)}, budget=10000,
                                    query="confidential remainder")
    assert {e["turn_id"] for e in result[0].metadata["source_events"]} == {"anchor", "future"}


def test_exhausted_budget_does_not_replace_existing_quotes():
    t = turn("t", "Visible original.\n" + "Additional discussion.\n" * 100)
    seed = source(t, "Visible original.")
    result = close_evidence_bundles([seed], [t], session_id="s", blocked_events=set(),
                                    budget=_cost(seed), query="additional discussion")
    assert result == [seed]


def test_evidence_bundle_adds_new_span_in_existing_event_within_budget():
    text = "Introduction.\n" + "Unrelated explanation.\n" * 65 + "Allocation: 250 credits.\n"
    t = turn("t", text)
    seed = source(t, "Introduction.")
    result = close_evidence_bundles([seed], [t], session_id="s", blocked_events=set(),
                                    budget=_cost(seed) + 1300, query="What was the allocation?")
    assert sum(_cost(hit) for hit in result) - _cost(seed) <= 1300
    assert result[0].chunk_id == seed.chunk_id
    assert result[0].metadata["source_events"][0] == seed.metadata["source_events"][0]
    assert any("Allocation: 250 credits." in e["text"] for e in result[0].metadata["source_events"])


def test_quantity_object_beats_procedure_and_unrelated_number_without_more_budget():
    text = ("Assembly note.\nParts:\n-7-9 bolts\n-12 washers\n"
            "1. Secure the bolts by following the bolt assembly procedure carefully.\n"
            "2. Inspect the bolts and clean the bolts before assembly.\n")
    t = turn("assembly", text)
    seed = source(t, "Assembly note.\n")
    budget = _cost(seed) + 520
    result = close_evidence_bundles([seed], [t], session_id="s", blocked_events=set(),
        budget=budget, query="How many bolts did the assembly need?", plan=QueryPlan(query_type="count"))
    visible = [e["text"] for s in evidence_catalog(result)["sources"] for e in s["source_events"]]
    assert any("7-9 bolts" in text for text in visible)
    # A short complete source may also contain another object. Its quantity
    # must remain attached to that object, not removed by an answer shortcut.
    assert all("12 washers" not in piece or "7-9 bolts" in piece for piece in visible)
    assert sum(_cost(h) for h in result) <= budget
    for e in result[0].metadata["source_events"]:
        assert text[e["start"]:e["end"]] == e["text"]


def test_quantity_with_written_number_and_plan_object():
    text = "Overview.\nCapacity: eleven pallets.\n1. Store pallets with pallets facing outward.\n"
    t = turn("capacity", text)
    seed = source(t, "Overview.\n")
    result = close_evidence_bundles([seed], [t], session_id="s", blocked_events=set(),
        budget=_cost(seed)+520, query="Confirm the pallet capacity.",
        plan=QueryPlan(query_type="count", subject="pallets", predicate="capacity"))
    assert any("eleven pallets" in e["text"] for e in result[0].metadata["source_events"])


def test_raw_count_recall_keeps_assistant_provenance_and_permission_boundaries(monkeypatch):
    current = turn("current", "The test recorded seven completed rounds.")
    foreign = turn("foreign", "The test recorded ten completed rounds.", session="other")
    future = turn("future", "The test recorded twelve completed rounds.", minute=10)
    blocked = turn("blocked", "The test recorded fifteen completed rounds.")
    question = turn("question", "How many rounds were completed?")
    plan = QueryPlan(query_type="count", as_of=current.events[0].occurred_at)
    query = "Looking back at our previous chat, how many rounds were completed?"
    monkeypatch.setenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "true")
    records = collect_raw_records(query, plan, [current, foreign, future, blocked, question], "s", {("blocked", 0)})
    assert {record["turn_id"] for record, _ in records} == {"current", "future"}
    record, text = records[0]
    assert record["turn_id"] == "current" and record["role"] == "assistant"
    assert record["evidence_grade"] == "secondary" and record["context_only"] is True
    assert text == current.events[0].content
    monkeypatch.setenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "false")
    assert collect_raw_records(query, plan, [current], "s", set()) == []
    user = current.model_copy(deep=True)
    user.events[0].role = "user"
    seed = source(user, "The test")
    # No neighbouring assistant turn is smuggled back in by local expansion.
    result = close_evidence_bundles([seed], [user, blocked], session_id="s", blocked_events=set(),
                                    budget=3000, query="completed rounds")
    assert all(e["turn_id"] == "current" for e in result[0].metadata["source_events"])

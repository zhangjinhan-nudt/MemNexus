"""Readback contract tests; fixture labels are never query-side retrieval keys."""

from agent_memory.grounding import (
    _extraction_hint,
    collect_raw_records,
    ground_evidence,
    match_evidence,
    source_payload,
)
from agent_memory.models import Event, Hit, QueryPlan, Turn


def source(text, *, session="old", role="assistant"):
    return Turn(turn_id="t", session_id=session, request_id="r", sequence=1,
                events=[Event(role=role, content=text)])


def ref(quote, **extra):
    return dict(turn_id="t", event_index=0, quote=quote, **extra)


def hit(references, *, authorized=True):
    return Hit(content="A discussion about household repairs", score=.8,
               source_file="fixture", chunk_id="m", metadata={
                   "origin": "indexed_sweep" if authorized else "compressed_memory",
                   "source_support_status": "authorized_session_quote" if authorized else "unresolved",
                   "evidence": references})


def read(query, t, h):
    return ground_evidence(query, QueryPlan(), [h], session_id="current", current_turns=[],
                           resolved_turns={"t": t}, blocked_events=set(), budget=9000).sources[0]


def test_fourth_quote_becomes_seed_before_the_existing_three_span_cap():
    quotes = ["A garden needs sunshine.", "The balcony has blue walls.",
              "A sewing machine can be repaired.", "The repair adhesive is maple resin."]
    t = source(" PRIVATE ".join(quotes))
    h = read("Which adhesive did you recommend for the repair?", t, hit([ref(q) for q in quotes]))
    assert h.content == quotes[-1]
    assert h.metadata["source_events"][0]["text"] == quotes[-1]
    assert len(h.metadata["source_events"]) == 3
    assert "PRIVATE" not in str(source_payload(h))


def test_matching_reaches_late_source_text_for_first_pass_and_archive_hits():
    intro = "General background. " * 80
    target = "The greenhouse contains exactly 12 orchids."
    t = source(intro + target)
    for authorized in (False, True):
        h = read("How many orchids are in the greenhouse?", t, hit([ref(t.events[0].content)], authorized=authorized))
        e = h.metadata["source_events"][0]
        assert target in e["text"]
        assert e["text"] == t.events[0].content[e["start"]:e["end"]]


def test_recorded_match_is_carried_and_invalid_annotation_cannot_expose_other_text():
    q = "The repair adhesive is maple resin."
    t = source(q + " Secret second recommendation: epoxy.")
    h = hit([ref(q)])
    h.metadata["matched_evidence"] = [{"turn_id": "t", "event_index": 0, "start": len(q),
                                      "end": len(t.events[0].content), "text": t.events[0].content[len(q):]}]
    selected = read("What is the second recommendation?", t, h)
    assert "epoxy" not in str(source_payload(selected))
    assert selected.metadata["source_events"][0]["text"] == q


def test_repeat_locations_and_explicit_offsets_are_exact_not_guessed_context():
    q = "The marker is violet."
    t = source(q + " Private intermediate information. " + q)
    found = match_evidence("What color is the marker?", QueryPlan(), [ref(q)], {"t": t})
    assert len(found) == 2
    assert {m["start"] for m in found} == {0, t.events[0].content.rindex(q)}
    start = t.events[0].content.rindex(q)
    explicit = match_evidence("marker", QueryPlan(), [ref(q, start=start, end=start + len(q))], {"t": t})
    assert len(explicit) == 1 and explicit[0]["start"] == start
    assert match_evidence("marker", QueryPlan(), [ref(q, start=1, end=len(q)+1)], {"t": t}) == []


def test_no_match_keeps_ordinary_semantic_candidate():
    t = source("The repair adhesive is maple resin.")
    h = read("Completely unrelated wording", t, hit([ref(t.events[0].content)]))
    assert h.metadata["source_events"][0]["text"] == t.events[0].content


def test_complete_member_survives_when_query_hits_tail_of_a_chunk_boundary():
    member = "The weekly session is hiking. This session happens every Sunday."
    t = source("background " * 60 + member + " Other background. " * 100)
    h = hit([ref(t.events[0].content)], authorized=False)
    h.metadata.update(subject="Alex", predicate="sessions", value="hiking", semantics={
        "cardinality": "many", "members": [{"value": "hiking", "evidence": [ref(member)]}]})
    selected = read("What session happens every Sunday?", t, h)
    assert any(member in e["text"] for e in selected.metadata["source_events"])
    assert any(_extraction_hint(selected, e).get("members") for e in selected.metadata["source_events"])


def test_event_read_scope_only_extends_the_existing_raw_pool_when_explicit():
    t = source("The repair adhesive is maple resin.")
    assert collect_raw_records("adhesive", QueryPlan(), [t], "current", set()) == []
    records = collect_raw_records("adhesive", QueryPlan(), [t], "current", set(), source_session_ids=("old",))
    assert records and records[0][0]["text"] == t.events[0].content
    assert collect_raw_records("adhesive", QueryPlan(), [t], "current", {("t", 0)}, source_session_ids=("old",)) == []

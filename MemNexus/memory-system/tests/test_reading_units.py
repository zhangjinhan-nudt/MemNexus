"""Complete original reading units, independent of any benchmark answers."""

from datetime import UTC, datetime

from agent_memory.grounding import _cost, _source_event, close_evidence_bundles, ground_evidence
from agent_memory.models import AnswerContract, AnswerOutput, Event, Hit, QueryPlan, Turn


def make_turn(identifier, text, *, sequence=1, session="s", role="assistant"):
    return Turn(turn_id=identifier, session_id=session, request_id=identifier, sequence=sequence,
                events=[Event(role=role, content=text, occurred_at=datetime(2025, 2, 1, tzinfo=UTC))])


def make_hit(turn, quote, *, start=0, rank=1):
    return Hit(content=quote, score=.5, source_file=turn.turn_id, chunk_id=turn.turn_id,
               metadata={"status": "active", "retrieval_rank": rank,
                         "evidence": [{"turn_id": turn.turn_id, "event_index": 0, "quote": quote,
                                       "start": start, "end": start+len(quote)}],
                         "source_events": [_source_event(turn, 0, quote, start)]})


def visible(hits):
    return "\n".join(e["text"] for h in hits for e in h.metadata["source_events"])


def test_header_and_complete_list_are_read_before_packing():
    intro = "Background notes. " * 70
    body = "Deliverables:\n1. Record observations.\n2. Compare the conditions.\n3. Release the anonymized measurements.\n"
    original = make_turn("proposal", intro+body)
    seed = make_hit(original, "Background notes.")
    result = ground_evidence("List the project deliverables.", QueryPlan(route="none", query_type="set"), [seed],
        session_id="s", current_turns=[original], resolved_turns={original.turn_id: original},
        blocked_events=set(), budget=1800)
    assert all(part in visible(result.sources) for part in ["Deliverables:", "1. Record", "2. Compare", "3. Release"])
    assert result.trace.context_tokens_estimate == sum(_cost(h) for h in result.sources) <= 1800
    assert result.trace.binding_units_selected == 0  # Original evidence needs no prior extracted binding.


def test_written_ordinal_keeps_the_list_heading_and_requested_item():
    text = "Device presets:\n" + "".join(f"{i}. Configuration {i}; use this complete preset.\n" for i in range(1, 30))
    original = make_turn("presets", text)
    seed = make_hit(original, "Device presets:\n")
    question = "Which was the fifth device preset?"
    plan = QueryPlan(query_type="set", answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="preset", question_span=question, member_positions=[5])]))
    result = close_evidence_bundles([seed], [original], session_id="s", blocked_events=set(), budget=1600,
        query=question, plan=plan)
    assert "Device presets:" in visible(result) and "5. Configuration 5;" in visible(result)
    assert "29. Configuration 29;" not in visible(result)
    for event in result[0].metadata["source_events"]:
        assert text[event["start"]:event["end"]] == event["text"]


def test_large_list_preserves_fitting_members_and_marks_source_partial():
    text = "Milestones:\n" + "".join(f"{i}. Complete milestone {i} with documented observations.\n" for i in range(1, 90))
    original = make_turn("milestones", text)
    result = close_evidence_bundles([make_hit(original, "Milestones:\n")], [original], session_id="s",
        blocked_events=set(), budget=700, query="List the milestones.", plan=QueryPlan(query_type="set"))
    assert _cost(result[0]) <= 700
    assert "1. Complete milestone 1" in visible(result)
    assert "89. Complete milestone 89" not in visible(result)
    assert result[0].metadata["reading_unit_partial"] is True


def test_matching_later_section_cannot_discard_original_supported_anchor():
    text = "The installation costs 24 credits. " + "Unrelated background. "*60 + "Delivery costs 8 credits."
    original = make_turn("costs", text)
    seed = make_hit(original, "The installation costs 24 credits.")
    result = close_evidence_bundles([seed], [original], session_id="s", blocked_events=set(), budget=1700,
                                  query="What are the installation and delivery costs?")
    assert "installation costs 24" in visible(result) and "Delivery costs 8" in visible(result)


def test_adjacent_speaker_does_not_create_a_shared_event_identity():
    request = make_turn("request", "Which adaptations did the studio propose?", sequence=1, role="user")
    reply = make_turn("reply", "Adaptations:\n1. Add acoustic panels.\n2. Improve lighting.", sequence=2)
    unrelated = make_turn("other", "Another studio proposed a swimming pool.", sequence=3)
    seed = make_hit(request, request.events[0].content)
    result = close_evidence_bundles([seed], [request, reply, unrelated], session_id="s", blocked_events=set(),
        budget=2000, query="Which adaptations did the studio propose?", plan=QueryPlan(query_type="set"))
    records = result[0].metadata["source_events"]
    assert {r["turn_id"] for r in records} == {"request", "reply"}
    assert {r["role"] for r in records} == {"user", "assistant"}
    assert all("event_identity" not in r for r in records)


def test_old_quote_cannot_read_a_different_list_or_private_neighbor():
    text = "Public alternatives:\n1. Oak.\n2. Pine.\nPRIVATE alternatives:\n1. Copper.\n2. Tin."
    allowed = text[:text.index("PRIVATE")]
    original = make_turn("old", text, session="old")
    seed = make_hit(original, "Public alternatives:\n")
    ref = {"turn_id": "old", "event_index": 0, "quote": allowed, "start": 0, "end": len(allowed)}
    result = close_evidence_bundles([seed], [original], session_id="current", blocked_events={("old", 0)},
        authorized_quotes={("old", 0): [ref]}, budget=2000,
        query="List the alternatives.", plan=QueryPlan(query_type="set"))
    assert "Pine" in visible(result) and "Copper" not in visible(result)
    assert all(e["end"] <= len(allowed) for e in result[0].metadata["source_events"])

"""One rank-based source budget; exact duplication is not word redundancy."""

import json

import pytest
from test_reading_units import make_hit, make_turn, visible

from agent_memory.grounding import _cost, ground_evidence, source_payload
from agent_memory.models import AnswerContract, AnswerOutput, QueryPlan


def read(hits, turns, *, budget=6000, max_sources=20):
    return ground_evidence("How did support improve the venue in the picture?", QueryPlan(route="none"), hits,
        session_id="s", current_turns=turns, resolved_turns={t.turn_id: t for t in turns},
        blocked_events=set(), budget=budget, max_sources=max_sources)


def test_ranked_relation_is_not_penalized_because_its_words_already_appear():
    texts = ["I discussed a picture, a venue, and support in three unrelated conversations.",
             "This picture shows the venue after the support enabled necessary repairs.",
             "Here is a completely different discussion with several novel words."]
    turns = [make_turn(str(i), text, sequence=i*10) for i, text in enumerate(texts)]
    hits = [make_hit(t, t.events[0].content, rank=i+1) for i, t in enumerate(turns)]
    hits[1].metadata["origin"] = "raw_fallback"
    result = read(hits, turns, max_sources=2)
    assert {hit.chunk_id for hit in result.sources} == {"0", "1"}
    assert "enabled necessary repairs" in visible(result.sources)


def test_source_interval_duplicates_do_not_consume_another_slot():
    t = make_turn("source", "The picture records repairs to the venue.")
    first = make_hit(t, t.events[0].content, rank=1)
    duplicate = make_hit(t, t.events[0].content, rank=2)
    duplicate.chunk_id = "another-memory-same-source"
    result = read([first, duplicate], [t])
    assert len(result.sources) == 1
    assert result.trace.selection_decisions[1]["reason"] == "source_intervals_already_visible"


def test_distinct_source_occurrences_are_not_deduplicated_by_shared_words():
    turns = [make_turn(str(i), "Support improved the venue.", sequence=i*10) for i in range(2)]
    hits = [make_hit(t, t.events[0].content, rank=i+1) for i, t in enumerate(turns)]
    assert len(read(hits, turns).sources) == 2


def test_actual_serialized_unit_budget_and_first_omission_are_auditable():
    turns = [make_turn(str(i), "Support improved the venue. "+str(i), sequence=i*10) for i in range(3)]
    hits = [make_hit(t, t.events[0].content, rank=i+1) for i, t in enumerate(turns)]
    initial = read(hits, turns)
    budget = _cost(initial.sources[0])
    result = read(hits, turns, budget=budget)
    assert len(result.sources) == 1
    assert result.trace.context_tokens_estimate == sum(_cost(h) for h in result.sources) <= budget
    assert result.trace.selection_decisions[1]["reason"] == "serialized_budget"
    assert result.trace.selection_decisions[1]["used_after"] == budget


def test_unknown_semantic_binding_does_not_filter_usable_original_text():
    t = make_turn("plain", "The support repaired the gallery roof visible in the picture.")
    result = read([make_hit(t, t.events[0].content)], [t])
    assert "gallery roof" in visible(result.sources)
    assert result.trace.binding_units_available == result.trace.binding_units_selected == 0


@pytest.mark.parametrize("compact", [False, True])
def test_exact_source_text_is_not_charged_again_as_a_claim(compact):
    original = "The original source remains visible. " * 30
    turn = make_turn("original", original)
    hit = make_hit(turn, original)
    before = hit.model_dump()
    payload = source_payload(hit, compact=compact)
    assert "claim" not in payload
    assert payload["source_events"][0]["text"] == original
    assert hit.model_dump() == before
    assert _cost(hit) == len(json.dumps(source_payload(hit), ensure_ascii=False)) + 16


@pytest.mark.parametrize("claim", ["The source describes the venue.", "Support improved the venue. "])
def test_a_distinct_summary_or_nonexact_claim_remains_available(claim):
    turn = make_turn("source", "Support improved the venue.")
    hit = make_hit(turn, turn.events[0].content)
    hit.content = claim
    payload = source_payload(hit)
    assert payload["claim"] == claim
    assert payload["source_events"][0]["text"] == turn.events[0].content


def test_claim_can_duplicate_a_later_source_record():
    first = make_turn("first", "The venue opened in spring.")
    second = make_turn("second", "Support improved the venue.", sequence=20)
    hit = make_hit(first, first.events[0].content)
    hit.metadata["source_events"] += make_hit(second, second.events[0].content).metadata["source_events"]
    hit.content = second.events[0].content
    payload = source_payload(hit)
    assert "claim" not in payload
    assert len(payload["source_events"]) == 2


def test_consumer_cost_applies_before_complete_reading_unit_construction():
    text = "Milestones:\n" + "".join(f"{i}. Deliver the observations. " + "Keep the record. "*14 + "\n"
                                     for i in range(1, 5))
    turn = make_turn("list", text)
    calls = []
    def consumer_cost(hit, source_id):
        calls.append(source_id)
        return 8
    result = ground_evidence("List the milestones.", QueryPlan(route="none", query_type="set"),
        [make_hit(turn, "Milestones:\n")], session_id="s", current_turns=[turn],
        resolved_turns={turn.turn_id: turn}, blocked_events=set(), budget=12, max_sources=3,
        source_cost=consumer_cost)
    assert all(f"{i}. Deliver the observations." in visible(result.sources) for i in range(1, 5))
    assert result.trace.context_tokens_estimate == 8
    assert 3 in calls and 1 in calls
    assert _cost(result.sources[0]) > 12


def test_consumer_cost_uses_selected_source_ids_in_final_packing():
    turns = [make_turn(str(i), f"The venue record is {i}.", sequence=10*i) for i in range(3)]
    hits = [make_hit(turn, turn.events[0].content, rank=i+1) for i, turn in enumerate(turns)]
    result = ground_evidence("What are the venue records?", QueryPlan(route="none"), hits,
        session_id="s", current_turns=turns, resolved_turns={t.turn_id: t for t in turns},
        blocked_events=set(), budget=6, max_sources=3, source_cost=lambda hit, source_id: source_id*2)
    assert [h.chunk_id for h in result.sources] == ["0", "1"]
    assert result.trace.context_tokens_estimate == 6
    assert [d["cost"] for d in result.trace.selection_decisions] == [2, 4, 6]


@pytest.mark.parametrize("max_sources", [2, 4, 5])
def test_planned_sum_fields_are_read_before_unbound_background(max_sources):
    question = "How many stored objects does Avery have in total?"
    predicates = [f"storage section {letter}" for letter in "ABCD"]
    plan = QueryPlan(route="none", answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="total", question_span=question, operator="sum", subjects=["Avery"],
        relation="stored objects", retrieval_relations=[("Avery", p) for p in predicates])]))
    turns = [make_turn(f"background-{i}", "Layout options:\n- Open shelving.\n- Enclosed cabinets.",
                       sequence=i*10) for i in range(4)]
    hits = [make_hit(t, t.events[0].content, rank=i+1) for i, t in enumerate(turns)]
    for index, predicate in enumerate(predicates, 1):
        t = make_turn(f"inventory-{index}", f"Avery stores {index} objects in section {index}.",
                      sequence=100+index*10, role="user")
        h = make_hit(t, t.events[0].content, rank=4+index)
        h.metadata.update(subject="Avery", predicate=predicate, value=str(index))
        turns.append(t)
        hits.append(h)
    before = [h.model_dump(mode="json") for h in hits]
    result = ground_evidence(question, plan, hits, session_id="s", current_turns=[],
        resolved_turns={t.turn_id: t for t in turns}, blocked_events=set(), budget=10000,
        max_sources=max_sources)
    expected = [f"inventory-{i}" for i in range(1, min(max_sources, 4)+1)]
    if max_sources > 4:
        expected.append("background-0")
    assert [h.chunk_id for h in result.sources] == expected
    assert result.trace.request_fields_available == 4
    assert result.trace.request_fields_selected == min(max_sources, 4)
    assert result.trace.binding_units_available == result.trace.binding_units_selected == 0
    originals = {t.turn_id: t.events[0] for t in turns}
    for hit in result.sources:
        for record in hit.metadata["source_events"]:
            original = originals[record["turn_id"]]
            assert original.content[record["start"]:record["end"]] == record["text"]
            assert record["role"] == original.role
    if max_sources > 4:
        assert result.sources[-1].metadata["source_events"][0]["text"] == turns[0].events[0].content
        background_decision = next(d for d in result.trace.selection_decisions if d["candidate_id"] == "background-0")
        assert not background_decision["deferred_known_unit"]
    assert [h.model_dump(mode="json") for h in hits] == before


def test_unbound_original_is_retained_when_planned_fields_have_no_extraction_match():
    question = "How many stored objects does Avery have in total?"
    plan = QueryPlan(route="none", answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="total", question_span=question, operator="sum", subjects=["Avery"],
        relation="stored objects", retrieval_relations=[("Avery", "inventory")])]))
    turn = make_turn("plain-record", "The inventory currently contains five objects.", role="user")
    result = ground_evidence(question, plan, [make_hit(turn, turn.events[0].content)],
        session_id="s", current_turns=[], resolved_turns={turn.turn_id: turn}, blocked_events=set(), budget=2000)
    assert [h.chunk_id for h in result.sources] == ["plain-record"]
    assert result.sources[0].metadata["source_events"][0]["text"] == turn.events[0].content
    assert result.trace.request_fields_available == result.trace.request_fields_selected == 0


@pytest.mark.parametrize("origin", ["indexed_sweep", "compressed_memory"])
def test_independent_channel_rank_is_not_pushed_behind_forty_high_score_candidates(origin):
    primary = [make_turn(f"primary-{i}", f"The venue has supporting document {i}.", sequence=i*10)
               for i in range(1, 41)]
    indexed = [make_turn(f"indexed-{i}", f"The picture records supporting document {i}.", sequence=500+i*10)
               for i in range(1, 3)]
    hits = [make_hit(t, t.events[0].content, rank=i) for i, t in enumerate(primary, 1)]
    for hit in hits:
        hit.score = .99
        hit.metadata["retrieval_channel_ranks"] = {"primary": hit.metadata["retrieval_rank"]}
    for rank, turn in enumerate(indexed, 1):
        hit = make_hit(turn, turn.events[0].content, rank=rank)
        hit.score = .01
        hit.metadata.update(origin=origin, retrieval_channel_ranks={"indexed_sweep": rank})
        hits.append(hit)
    result = read(hits, primary+indexed, max_sources=4)
    assert {hit.chunk_id for hit in result.sources} == {
        "indexed-1", "primary-1", "indexed-2", "primary-2"}
    assert all(item["upstream_rank"] <= 2 for item in result.trace.selection_decisions[:4])
    assert [item["rank"] for item in result.trace.selection_decisions[:4]] == [1, 2, 3, 4]
    assert all(item["selected"] for item in result.trace.selection_decisions[:4])
    assert sum(_cost(hit) for hit in result.sources) <= 6000


def test_semantic_source_is_not_demoted_for_lacking_lexical_overlap():
    source = make_turn("semantic", "Capoeira.", sequence=1)
    other = make_turn("other", "Support improves the venue in the picture.", sequence=20)
    hits = [make_hit(source, source.events[0].content, rank=9),
            make_hit(other, other.events[0].content, rank=1)]
    hits[0].metadata["retrieval_channel_ranks"] = {"semantic": 1}
    hits[1].metadata["retrieval_channel_ranks"] = {"semantic": 20}
    result = read(hits, [source, other], max_sources=1)
    assert result.sources[0].chunk_id == "semantic"
    decision = result.trace.selection_decisions[0]
    assert decision["rank"] == 1 and decision["upstream_rank"] == 1
    assert "rrf_score" not in decision and "source_rank" not in decision

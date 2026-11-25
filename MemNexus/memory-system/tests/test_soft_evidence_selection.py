"""Generic source coverage tests: no benchmark entity, ID, or gold answer."""
from copy import deepcopy

import pytest
from test_reading_units import make_hit, make_turn, visible

from agent_memory.grounding import ground_evidence
from agent_memory.models import AnswerContract, AnswerOutput, QueryPlan
from agent_memory.prompt_codec import compact_json, synthesis_catalog_receipt, synthesis_source_catalog


def setup(names=("Amber", "Violet")):
    a, b = names
    question = f"Which was installed first, {a} or {b}?"
    plan = QueryPlan(answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="order", question_span=question, operator="comparison",
        subjects=[a, b], subject_mode="joint", relation="installation")]))
    texts = [f"{a} and {b} installation options are available.",
             f"{a} was installed on March 5, 2024.",
             f"{b} was installed on April 7, 2024."]
    turns = [make_turn(str(i), text, sequence=10*i, role="user") for i, text in enumerate(texts)]
    hits = [make_hit(t, t.events[0].content, rank=i+1) for i, t in enumerate(turns)]
    return question, plan, turns, hits


def read(question, plan, turns, hits, **options):
    return ground_evidence(question, plan, hits, session_id="s", current_turns=[],
        resolved_turns={t.turn_id:t for t in turns}, blocked_events=set(),
        max_sources=2, budget=10000, **options)


@pytest.mark.parametrize("names", [("Amber", "Violet"), ("Orion", "Lyra"), ("甲设备", "乙设备")])
def test_empty_catalog_still_covers_literal_subjects_and_time(names):
    q, plan, turns, hits = setup(names)
    assert not plan.answer_contract.outputs[0].retrieval_relations
    result = read(q, plan, turns, hits)
    assert {h.chunk_id for h in result.sources} == {"1", "2"}
    assert "March 5" in visible(result.sources) and "April 7" in visible(result.sources)
    assert result.trace.request_fields_selected == 0
    assert result.trace.selection_coverage == "soft_locating_signals_not_answer_sufficiency"


def test_duplicate_channels_and_input_order_do_not_change_selection():
    q, plan, turns, hits = setup()
    before = read(q, plan, turns, hits)
    duplicate = hits[0].model_copy(deep=True)
    duplicate.chunk_id = "duplicate-memory"
    duplicate.metadata["retrieval_channel_ranks"] = {"semantic":1, "lexical":1, "symbolic":1}
    after = read(q, plan, turns, list(reversed(hits)) + [duplicate])
    assert [h.chunk_id for h in before.sources] == [h.chunk_id for h in after.sources]
    assert [h.metadata["source_events"] for h in before.sources] == [h.metadata["source_events"] for h in after.sources]


def test_unrelated_background_does_not_evict_comparison_premises():
    q, plan, turns, hits = setup()
    for i in range(12):
        turn = make_turn(f"other-{i}", f"The weather report discussed cloud cover {i}.", sequence=100+i*10)
        turns.append(turn)
        hits.append(make_hit(turn, turn.events[0].content, rank=1))
    result = read(q, plan, turns, hits)
    assert {h.chunk_id for h in result.sources} == {"1", "2"}


def test_unparsed_relative_relation_is_eligible_without_date_or_catalog():
    q, plan, turns, hits = setup()
    turns = turns[:1]
    turns[0].events[0].content = "Amber was installed before Violet."
    hits = [make_hit(turns[0], turns[0].events[0].content)]
    result = read(q, plan, turns, hits)
    assert "before Violet" in visible(result.sources)


def test_final_receipt_is_hash_of_actual_ordered_catalog():
    q, plan, turns, hits = setup()
    result = read(q, plan, turns, hits)
    import hashlib
    catalog = synthesis_source_catalog(result.sources, needs_time=True)
    receipt = synthesis_catalog_receipt(result.sources, needs_time=True)
    encoded = compact_json(catalog)
    assert receipt["sha256"] == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    assert receipt["serialized_chars"] == len(encoded)
    assert receipt["source_ids"] == [1, 2]
    assert result.trace.context_serialized_chars == result.trace.context_tokens_estimate
    assert result.trace.context_budget_unit == "serialized_characters"


def test_duplicate_view_has_one_selected_receipt():
    q, plan, turns, hits = setup()
    result = read(q, plan, turns, hits + [deepcopy(hits[1])])
    decisions = [d for d in result.trace.selection_decisions if d["candidate_id"] == "1" and d["selected"]]
    assert len(decisions) == 1
    assert len({d["final_source_id"] for d in result.trace.selection_decisions if d["selected"]}) == len(result.sources)

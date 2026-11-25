"""List addresses bind to the request; retrieval expansion cannot create them."""
import asyncio
import json

import pytest
from test_reading_units import make_hit, make_turn, visible
from test_request_binding_integration import ScriptedChat, prompt_data

from agent_memory.grounding import (
    _bm25,
    _list_member_positions,
    _span_scores,
    close_evidence_bundles,
    match_evidence,
)
from agent_memory.models import AnswerContract, AnswerOutput, QueryPlan, normalize_request_contract
from agent_memory.providers import CallableModel


def bound_plan(question, positions, *, operator="direct", question_span=None):
    return QueryPlan(answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="requested", question_span=question_span or question,
        operator=operator, member_positions=positions)]))


@pytest.mark.parametrize(("question", "expected"), [
    ("What was the first item?", {"1"}),
    ("What was the fifth item?", {"5"}),
    ("What was the 27th item?", {"27"}),
    ("What was the twenty-seventh item?", {"27"}),
    ("What was the fifth in the list?", {"5"}),
    ("What was the fifth item in the first list?", {"5"}),
    ("What was the fifth item in the list you gave me before my trip?", {"5"}),
    ("What was the first item I should prepare before leaving?", {"1"}),
    ("清单中的第27项是什么？", {"27"}),
    ("Which device did I install first?", set()),
    ("Where did I stay during my first trip?", set()),
    ("Which was my first trip?", set()),
    ("What items were in the first list you gave me?", set()),
    ("What was the fifth device preset?", set()),
])
def test_surface_fallback_requires_member_scope(question, expected):
    assert _list_member_positions(question, QueryPlan(query_type="set")) == expected


@pytest.mark.parametrize(("word", "position"), [("first", 1), ("fifth", 5), ("27th", 27), ("twenty-seventh", 27)])
def test_existing_contract_can_bind_an_arbitrary_domain_noun(word, position):
    question = f"Which was the {word} configuration?"
    plan = bound_plan(question, [position])
    normalized, notes = normalize_request_contract(question, plan)
    restored = QueryPlan.model_validate_json(normalized.model_dump_json())
    assert not notes
    assert restored.answer_contract.outputs[0].member_positions == [position]
    assert _list_member_positions(question, restored) == {str(position)}
    doc = "Configurations:\n" + "".join(f"{i}. Configuration {i}.\n" for i in range(1, 30))
    turn = make_turn("choices", doc)
    result = close_evidence_bundles([make_hit(turn, "Configurations:\n")], [turn],
        session_id="s", blocked_events=set(), budget=2000, query=question, plan=restored)
    assert f"{position}. Configuration {position}." in visible(result)
    assert "29. Configuration 29." not in visible(result)
    assert all(doc[e["start"]:e["end"]] == e["text"] for h in result for e in h.metadata["source_events"])


def test_unanchored_or_unmentioned_position_cannot_create_an_address():
    question = "Which was the fifth configuration?"
    assert _list_member_positions(question, bound_plan(question, [5], question_span="invented first item")) == set()
    assert _list_member_positions(question, bound_plan(question, [1, 5, 27])) == {"5"}
    assert _list_member_positions(question, bound_plan(question, [1], question_span="configuration")) == set()


@pytest.mark.parametrize("field", ["semantic_queries", "required_info", "keywords"])
def test_expanded_search_cannot_introduce_first_item_priority(field):
    question = "What did the release package contain?"
    texts = ["The release package contained a survey instrument and observations.",
             "1. Choose a comfortable chair."]
    turns = [make_turn(str(i), text) for i, text in enumerate(texts)]
    references = [{"turn_id": t.turn_id, "event_index": 0, "quote": t.events[0].content,
                   "start": 0, "end": len(t.events[0].content)} for t in turns]
    plan = QueryPlan(**{field: ["first review the release package"]})
    result = match_evidence(question, plan, references, {t.turn_id: t for t in turns})
    assert [record["turn_id"] for record in result] == ["0"]
    # Absence of positional intent is exactly ordinary scoring, not a filter.
    docs = [*texts, "The chair supports the release package review."]
    expanded = question + " first review the release package"
    assert _span_scores(expanded, docs, member_positions=_list_member_positions(question, plan)) == _bm25(expanded, docs)


def test_temporal_comparison_set_plan_retains_relevant_unnumbered_evidence():
    query = "Which device did I set up first, the thermostat or the mesh network?"
    plan = bound_plan(query, [], operator="comparison").model_copy(update={"query_type": "set"})
    docs = ["I set up my thermostat last month and the mesh network yesterday.", "1. Choose a comfortable chair."]
    scores = _span_scores(query, docs, member_positions=_list_member_positions(query, plan))
    assert scores == _bm25(query, docs)
    assert scores[0] > scores[1]


def test_first_list_reads_complete_collection_not_only_first_member():
    query = "What configurations were in the first list you gave me?"
    text = "Configurations:\n1. Amber.\n2. Bronze.\n3. Copper.\n"
    turn = make_turn("list", text)
    result = close_evidence_bundles([make_hit(turn, "Configurations:\n")], [turn], session_id="s",
        blocked_events=set(), budget=2000, query=query, plan=QueryPlan(query_type="set"))
    assert all(label in visible(result) for label in ("1. Amber", "2. Bronze", "3. Copper"))


def test_member_position_does_not_enable_another_container_position():
    question = "What was the fifth configuration in the first list?"
    assert _list_member_positions(question, bound_plan(question, [5])) == {"5"}


def test_no_contract_does_not_drop_numbered_relevant_candidates():
    question = "What was the fifth configuration?"
    docs = ["5. Configuration uses copper.", "1. An unrelated chair."]
    scores = _span_scores(question, docs, member_positions=_list_member_positions(question))
    assert scores == _bm25(question, docs)
    assert scores[0] > 0


def test_existing_planner_call_and_reader_keep_the_same_member_binding():
    question = "Which was the fifth configuration?"
    chat = ScriptedChat({"route": "both", "answer_contract": {"outputs": [{
        "output_id": "configuration", "question_span": question, "member_positions": [5],
    }]}})
    model = CallableModel(chat)
    plan = asyncio.run(model.plan(question, {}))
    assert len(chat.prompts) == 1
    assert '"member_positions"' in chat.prompts[0]
    plan, _ = normalize_request_contract(question, plan)
    assert _list_member_positions(question, plan) == {"5"}
    chat.responses.append({"slots": [{"target": question, "derivation": "No supplied sources.",
                                     "values": [], "missing": "No supplied sources."}]})
    asyncio.run(model.synthesize(question, plan, []))
    assert len(chat.prompts) == 2
    payload = prompt_data(chat.prompts[-1])
    assert payload["request_contract"]["outputs"][0]["member_positions"] == [5]
    # A legacy stored output without the field keeps the ordinary route.
    wire = plan.model_dump()
    del wire["answer_contract"]["outputs"][0]["member_positions"]
    assert QueryPlan.model_validate_json(json.dumps(wire)).answer_contract.outputs[0].member_positions == []

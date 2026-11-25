"""Compositional recall survives when an exact collection query is unavailable."""

import pytest

from agent_memory.models import AnswerContract, AnswerOutput, QueryPlan, normalize_request_contract
from agent_memory.reasoning import QuestionRoute, apply_route, bind_plan_entities, classify_question


def request(question, operator="count", **fields):
    return QueryPlan(answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="requested", question_span=question, operator=operator,
        subjects=["Mira"], relation="attended activities", qualifier="during the stated period", **fields,
    )]))


def test_temporal_count_keeps_coverage_without_an_exact_collection_query():
    question = "How many plants has Mira added during the last six months?"
    plan = request(question, count_unit="entity")
    normalized, _ = normalize_request_contract(question, plan)
    assert normalized.collection_query is None
    route = classify_question(question, ["Mira"])
    assert route.kind == "temporal" and route.recall_policy == "focused"
    result = apply_route(normalized, route)
    assert result.query_type == "temporal" and result.recall_policy == "recall_first"
    assert result.depth == 8 and result.query_entities == ["Mira"]
    assert result.collection_query is None and result.required_info == [question]
    assert result.answer_contract.outputs[0].qualifier == "during the stated period"
    assert plan.depth == 3 and plan.recall_policy == "focused"


@pytest.mark.parametrize("operator,expected_type", [("set", "set"), ("count", "count"), ("sum", "fact")])
def test_explicit_operator_supplies_coverage_when_surface_route_is_focused(operator, expected_type):
    question = "Report Mira's requested activity result for the stated period."
    plan, _ = normalize_request_contract(question, request(question, operator))
    route = QuestionRoute(kind="fact", recall_policy="focused", constraints=("year=2025",))
    result = apply_route(plan, route)
    assert result.collection_query is None and result.query_type == expected_type
    assert result.recall_policy == "recall_first" and result.depth == 8
    assert result.query_constraints == ["year=2025"] and result.query_entities == ["Mira"]


def test_multiple_requested_outputs_need_coverage_without_changing_their_conditions():
    question = "Give Mira's age and give Mira's home city."
    plan = QueryPlan(answer_contract=AnswerContract(outputs=[
        AnswerOutput(output_id="age", question_span="Give Mira's age", subjects=["Mira"], relation="age"),
        AnswerOutput(output_id="city", question_span="give Mira's home city", subjects=["Mira"], relation="home city"),
    ]))
    normalized, _ = normalize_request_contract(question, plan)
    result = apply_route(normalized, QuestionRoute(kind="fact", recall_policy="focused"))
    assert result.collection_query is None and result.recall_policy == "recall_first"
    assert result.depth == 8 and result.query_entities == ["Mira"]
    assert result.required_info == ["Give Mira's age", "give Mira's home city"]
    assert result.answer_contract == normalized.answer_contract


def test_known_subject_substitution_cannot_return_as_a_coverage_hint():
    question = "How many plants has Mira added during the last six months?"
    plan = request(question)
    plan.answer_contract.outputs[0].subjects = ["Kai"]
    bound, changes = bind_plan_entities(question, plan, ["Mira", "Kai"])
    normalized, _ = normalize_request_contract(question, bound)
    result = apply_route(normalized, classify_question(question, ["Mira", "Kai"]))
    assert changes == ["request_subjects:requested"]
    assert result.query_entities == ["Mira"] and result.collection_query is None
    assert result.answer_contract.outputs[0].subjects == []
    assert result.recall_policy == "recall_first"


def test_route_entities_have_priority_over_surviving_contract_subject_hints():
    question = "Which activities did Mira and her colleague attend during the stated period?"
    plan, _ = normalize_request_contract(question, request(question, "set"))
    result = apply_route(plan, QuestionRoute(kind="relation", recall_policy="recall_first",
                                            entities=("Mira", "Colleague"), constraints=("shared event",)))
    assert result.query_entities == ["Mira", "Colleague"]
    assert result.query_type == "relation" and result.query_constraints == ["shared event"]


@pytest.mark.parametrize("kind,policy,depth", [("fact", "focused", 3), ("temporal", "focused", 3),
                                              ("set", "recall_first", 8)])
def test_no_contract_keeps_existing_route_behavior(kind, policy, depth):
    route = QuestionRoute(kind=kind, recall_policy=policy, entities=("Mira",), constraints=("year=2025",))
    result = apply_route(QueryPlan(), route)
    assert result.query_type == kind and result.recall_policy == policy and result.depth == depth
    assert result.query_entities == ["Mira"] and result.query_constraints == ["year=2025"]


def test_unanchored_or_action_contract_does_not_add_coverage_requirements():
    invalid = request("An invented different question")
    normalized, _ = normalize_request_contract("When did Mira arrive?", invalid)
    route = QuestionRoute(kind="temporal", recall_policy="focused")
    result = apply_route(normalized, route)
    assert result.recall_policy == "focused" and result.depth == 3
    action = request("Make a plan")
    action.response_intent = "action_plan"
    result = apply_route(action, QuestionRoute(kind="fact", recall_policy="focused"))
    assert result.recall_policy == "focused" and result.depth == 3

"""One request IR supplies subject hints and compiled relation-catalog addresses."""

import pytest

from agent_memory.models import AnswerContract, AnswerOutput, QueryPlan, normalize_request_contract
from agent_memory.reasoning import apply_route, bind_plan_entities, classify_question


def proposal(question, **fields):
    output = {"output_id": "request", "question_span": question, "operator": "count", "subjects": ["user"],
              "relation": "participated in activities", "qualifier": "before the named event", **fields}
    return QueryPlan(semantic_queries=[question], answer_contract=AnswerContract(outputs=[AnswerOutput(**output)]))


@pytest.mark.parametrize("question", [
    "How many engineers did I supervise as Software Engineering Manager?",
    "How many charity events did I attend before the River Walk fundraiser?",
])
def test_capitalized_roles_and_event_titles_do_not_invent_query_subjects(question):
    for inventory in ([], ["user", "assistant"]):
        route = classify_question(question, inventory)
        assert route.entities == ()
        normalized, _ = normalize_request_contract(question, proposal(question))
        result = apply_route(normalized, route)
        assert result.query_entities == ["user"]
        assert result.collection_query is None
        assert result.semantic_queries == [question]


def test_only_literal_known_entities_are_retained_with_word_and_cjk_boundaries():
    route = classify_question("What did JOSÉ SILVA tell Anna about the Winter Forum?", ["José Silva", "Ann"])
    assert route.entities == ("José Silva",)
    assert classify_question("张三推荐了什么？", ["张三", "李四"]).entities == ("张三",)


def test_bound_request_subjects_precede_real_explicit_speakers_without_discarding_them():
    question = "What did Mira and Kai discuss about my activities before the named event?"
    normalized, _ = normalize_request_contract(question, proposal(question))
    result = apply_route(normalized, classify_question(question, ["Mira", "Kai", "user"]))
    assert result.query_entities == ["user", "Mira", "Kai"]
    assert result.semantic_queries == [question]


def test_catalog_alias_pairs_survive_without_an_exact_collection_partition():
    question = "How many activities did I attend before the named event?"
    plan = proposal(question, relation_refs=[2, 1, 2, 99], count_unit="event")
    catalog = [("user", "attended", "many", None), ("user", "participated in", "many", None)]
    normalized, notes = normalize_request_contract(question, plan, catalog)
    output = normalized.answer_contract.outputs[0]
    assert normalized.collection_query is None
    assert output.retrieval_relations == [("user", "participated in"), ("user", "attended")]
    assert output.relation_refs == [2, 1] and output.relation == "participated in activities"
    assert notes == ["request_contract:unknown_relation_ref:request"]
    assert plan.answer_contract.outputs[0].retrieval_relations == []
    round_trip = QueryPlan.model_validate_json(normalized.model_dump_json())
    repeated, _ = normalize_request_contract(question, round_trip)
    assert repeated == normalized  # Reader re-normalization has no catalog.


def test_catalog_recompilation_overwrites_preexisting_internal_pairs():
    question = "How many activities did I attend before the named event?"
    plan = proposal(question, relation_refs=[1], retrieval_relations=[("invented actor", "invented relation")])
    result, _ = normalize_request_contract(question, plan, [("user", "attended", "many", None)])
    assert result.answer_contract.outputs[0].retrieval_relations == [("user", "attended")]
    empty, _ = normalize_request_contract(question, result, [])
    assert empty.answer_contract.outputs[0].retrieval_relations == []
    assert empty.answer_contract.outputs[0].relation_refs == []
    assert empty.semantic_queries == [question]


def test_only_real_catalog_subject_predicate_pairs_are_compiled():
    question = "How many activities did I attend before the named event?"
    plan = proposal(question, relation_refs=[1, 2, 3, 4, 5])
    result, _ = normalize_request_contract(question, plan, [
        ("user", "attended", "many", None), ("user", None), {"predicate": "invented"}, ("", "x"),
        ("user", "attended", "one", "different event"),
    ])
    assert result.answer_contract.outputs[0].retrieval_relations == [("user", "attended")]
    assert result.collection_query is None  # Qualifier still cannot be projected away.


def test_wrong_known_subject_cannot_survive_in_compiled_relation_hints():
    question = "How many activities did Mira attend before the named event?"
    plan = proposal(question, subjects=["Kai"], relation_refs=[1], retrieval_relations=[("Kai", "attended")])
    bound, changes = bind_plan_entities(question, plan, ["Mira", "Kai"])
    normalized, _ = normalize_request_contract(question, bound)
    output = normalized.answer_contract.outputs[0]
    assert changes == ["request_subjects:request"]
    assert output.subjects == [] and output.relation_refs == [] and output.retrieval_relations == []
    assert apply_route(normalized, classify_question(question, ["Mira", "Kai"])).query_entities == ["Mira"]


def test_old_contract_receipts_read_without_new_compilation_metadata():
    output = AnswerOutput.model_validate({"output_id": "old", "question_span": "What happened?",
                                         "required_inputs": ["old premise"]})
    assert output.retrieval_relations == [] and output.required_inputs == ["old premise"]

"""Request provenance and retrieval projections never establish factual truth."""
from datetime import UTC, datetime

from agent_memory.models import (
    AnswerContract,
    AnswerOutput,
    CollectionQuery,
    Evidence,
    FactMember,
    FactSemantics,
    Memory,
    QueryPlan,
    normalize_request_contract,
)


def plan_for(question, **output):
    return QueryPlan(
        semantic_queries=[question], required_info=["old retrieval suggestion"],
        answer_contract=AnswerContract(outputs=[AnswerOutput(
            output_id="q0", question_span=question, **output,
        )]),
    )


def test_source_projection_is_complete_and_independent_of_recall_channel():
    quote = Evidence(turn_id="turn", event_index=0, quote="My red and blue folders")
    memory = Memory(
        memory_id="m", session_id="s", scope_id="s", content="Folders", subject="Mira",
        predicate="owns", value="red and blue", evidence=[quote],
        semantics=FactSemantics(cardinality="many", members=[
            FactMember(value="red", identity="red folder", evidence=[quote]),
        ], event_identity="folder purchase"),
        valid_from=datetime(2030, 1, 2, tzinfo=UTC),
    )
    first, second = memory.source_metadata(), memory.source_metadata()
    assert first == second
    assert first["valid_from"] == "2030-01-02T00:00:00Z"
    assert {"evidence", "semantics", "status", "recorded_at", "revision", "scope_id"} <= first.keys()
    assert not {"origin", "score", "matched_views", "rank_score"} & first.keys()
    first["semantics"]["members"][0]["evidence"][0]["quote"] = "corrupted"
    first["evidence"].clear()
    assert memory.evidence[0].quote == quote.quote
    assert second["semantics"]["members"][0]["evidence"][0]["quote"] == quote.quote


def test_contract_absent_or_action_plan_leaves_existing_behavior_untouched():
    missing = QueryPlan(required_info=["original"], collection_query=CollectionQuery(
        subjects=["Mira"], predicate="owns"))
    same, notes = normalize_request_contract("Anything?", missing, [])
    assert same is missing and notes == []
    action = plan_for("Design a plan", required_inputs=["legacy context"])
    action.response_intent = "action_plan"
    same, notes = normalize_request_contract("different question", action, [])
    assert same is action and notes == []


def test_fabricated_question_span_disables_interpretation_without_refusing_qa():
    query = "How many staff report to the Research Director?"
    plan = plan_for("How many staff report to the Finance Director?", subjects=["Mira"])
    old = plan.model_dump()
    result, notes = normalize_request_contract(query, plan)
    assert result.answer_contract is None
    assert result.semantic_queries == plan.semantic_queries
    assert result.required_info == plan.required_info
    assert result.route == "both"
    assert notes == ["request_contract:unanchored_output:q0"]
    assert plan.model_dump() == old


def test_invalid_subrequest_does_not_hide_the_rest_of_original_question():
    query = "Which clubs did Mira join, and when?"
    plan = QueryPlan(answer_contract=AnswerContract(outputs=[
        AnswerOutput(output_id="clubs", question_span="Which clubs did Mira join", operator="set"),
        AnswerOutput(output_id="invented", question_span="What was the membership fee?"),
    ]))
    result, notes = normalize_request_contract(query, plan)
    assert result.answer_contract is None
    assert result.collection_query is None
    assert result.required_info == [query]
    assert notes == [
        "request_contract:unanchored_output:invented",
        "request_contract:partial_interpretation_omitted",
    ]
    restored = QueryPlan.model_validate(result.model_dump())
    repeated, _ = normalize_request_contract(query, restored)
    assert repeated.model_dump() == result.model_dump()
    assert len(plan.answer_contract.outputs) == 2


def test_long_partially_anchored_query_preserves_existing_retrieval_hints():
    query = "Compare both designs. " + "Include their documented limitations. " * 20
    plan = QueryPlan(
        semantic_queries=["design comparison"],
        required_info=["designs", "documented limitations"],
        answer_contract=AnswerContract(outputs=[
            AnswerOutput(output_id="designs", question_span="both designs"),
            AnswerOutput(output_id="fees", question_span="Compare licensing fees"),
        ]),
    )
    result, _ = normalize_request_contract(query, plan)
    assert result.answer_contract is None
    assert result.required_info == plan.required_info
    assert result.semantic_queries == plan.semantic_queries


def test_background_premises_do_not_become_answer_obligations():
    query = "Roughly how long has the router been in use?"
    plan = plan_for(query, operator="duration", required_inputs=["Exact first configuration timestamp"])
    result, notes = normalize_request_contract(query, plan)
    assert result.answer_contract.outputs[0].required_inputs == []
    assert result.required_info == [query]
    assert "request_contract:legacy_premises_omitted:q0" in notes
    assert plan.answer_contract.outputs[0].required_inputs == ["Exact first configuration timestamp"]


def test_role_and_event_qualifiers_survive_without_becoming_collection_gates():
    query = "Which teams did Mira manage while acting as Research Director?"
    plan = plan_for(query, operator="set", subjects=["Mira"], relation="managed",
                    qualifier="while acting as Research Director", count_unit="entity")
    plan.subject, plan.predicate = "Mira", "managed"
    plan.query_entities = ["Mira"]
    result, _ = normalize_request_contract(query, plan)
    assert result.answer_contract.outputs[0].qualifier == "while acting as Research Director"
    assert result.collection_query is None
    assert result.semantic_queries == [query]
    assert result.subject == "Mira" and result.query_entities == ["Mira"]


def test_catalog_refs_are_bounded_hints_not_fact_or_entity_certificates():
    query = "Which outings did Mira attend?"
    plan = plan_for(query, operator="set", subjects=["Mira"], relation="went to",
                    relation_refs=[2, 2, 99], count_unit="event")
    catalog = [("Noel", "owns", "one", None), ("Mira", "attended", "many", None)]
    result, notes = normalize_request_contract(query, plan, catalog)
    output = result.answer_contract.outputs[0]
    assert output.relation_refs == [2]
    assert output.relation == "went to"
    assert result.collection_query.predicate == "attended"
    assert notes == ["request_contract:unknown_relation_ref:q0"]
    # Repeating the source-less reader normalization must not erase a mapping
    # already validated against the planner's catalog or change its scope.
    again, notes = normalize_request_contract(query, result)
    assert again.model_dump() == result.model_dump()
    assert notes == []


def test_aliases_needing_multiple_partitions_keep_ordinary_retrieval():
    query = "How many outings did Mira attend?"
    plan = plan_for(query, operator="count", subjects=["Mira"], relation="attended outings",
                    relation_refs=[1, 2], count_unit="event")
    result, _ = normalize_request_contract(query, plan, [
        ("Mira", "attended", "many", None), ("Mira", "participated in", "many", None),
    ])
    assert result.collection_query is None
    assert result.answer_contract.outputs[0].relation_refs == [1, 2]
    assert result.semantic_queries == [query]
    again, _ = normalize_request_contract(query, result)
    assert again.model_dump() == result.model_dump()


def test_planned_assertions_and_exclusions_are_not_silently_reset():
    query = "Which planned workshops does Mira want, excluding the beginner class?"
    qualifier = "planned workshops excluding the beginner class"
    plan = plan_for(query, operator="set", subjects=["Mira"], relation="workshops",
                    qualifier=qualifier, count_unit="value")
    plan.collection_query = CollectionQuery(
        subjects=["Mira"], predicate="workshops", qualifier=qualifier,
        assertions=["planned"], exclude_values=["beginner class"], projection="category",
    )
    result, _ = normalize_request_contract(query, plan)
    assert result.collection_query == plan.collection_query
    result.collection_query.exclude_values.clear()
    assert plan.collection_query.exclude_values == ["beginner class"]


def test_incompatible_rich_partition_is_not_replaced_with_default_filters():
    query = "Which workshops did Mira complete?"
    plan = plan_for(query, operator="set", subjects=["Mira"], relation="completed workshops",
                    count_unit="value")
    plan.collection_query = CollectionQuery(subjects=["Mira"], predicate="workshops", assertions=["planned"])
    result, notes = normalize_request_contract(query, plan)
    assert result.collection_query is None
    assert result.answer_contract is not None
    assert "request_contract:collection_optimization_not_representable" in notes


def test_multi_subject_modes_are_not_all_intersections():
    query = "List the clubs of Mira and Noel."
    for mode, operation in [("each", "group"), ("joint", "union"), ("common", "intersection")]:
        plan = plan_for(query, operator="set", subjects=["Mira", "Noel"], subject_mode=mode,
                        relation="clubs", count_unit="value")
        result, _ = normalize_request_contract(query, plan)
        assert result.collection_query.operation == operation


def test_unknown_count_unit_does_not_become_entity_or_event_count():
    query = "How many groups did Mira mention?"
    result, _ = normalize_request_contract(query, plan_for(
        query, operator="count", subjects=["Mira"], relation="mentioned groups"))
    assert result.collection_query is None
    assert result.answer_contract.outputs[0].count_unit is None


def test_long_request_is_preserved_without_a_truncated_condition():
    query = "Which records match " + "a contextual qualification " * 22 + "but exclude withdrawn entries?"
    plan = plan_for(query)
    result, notes = normalize_request_contract(query, plan)
    assert result.answer_contract.outputs[0].question_span == query
    assert result.required_info == []
    assert result.semantic_queries == [query]
    assert notes == ["request_contract:long_span_not_truncated_for_retrieval"]


def test_multiple_outputs_do_not_reuse_one_unrelated_partition():
    query = "Which clubs did Mira join and how old was she?"
    plan = QueryPlan(collection_query=CollectionQuery(subjects=["Mira"], predicate="clubs"),
                     answer_contract=AnswerContract(outputs=[
        AnswerOutput(output_id="clubs", question_span="Which clubs did Mira join", operator="set",
                     subjects=["Mira"], relation="clubs", count_unit="value"),
        AnswerOutput(output_id="age", question_span="how old was she", operator="direct"),
    ]))
    result, _ = normalize_request_contract(query, plan)
    assert result.collection_query is None
    assert result.required_info == ["Which clubs did Mira join", "how old was she"]


def test_request_normalization_does_not_restore_a_rebound_entity_gate():
    query = "What did I order?"
    plan = plan_for(query, subjects=["Unverified catalog name"], relation="ordered")
    plan.subject = "Authenticated speaker"
    plan.query_entities = ["Authenticated speaker"]
    plan.query_constraints = ["historical order"]
    result, _ = normalize_request_contract(query, plan)
    assert result.subject == plan.subject
    assert result.query_entities == plan.query_entities
    assert result.query_constraints == plan.query_constraints

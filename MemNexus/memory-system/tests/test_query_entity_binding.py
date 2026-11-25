"""Original-query entities constrain recall proposals, not evidence speakers."""

import asyncio

import pytest

from agent_memory.models import CollectionQuery, Evidence, Memory, QueryPlan, Session
from agent_memory.reasoning import apply_route, bind_plan_entities, classify_question
from agent_memory.retrieval import Retriever
from agent_memory.settings import Settings


def test_substitution_is_removed_without_rewriting_the_question_or_facts():
    question = "What agency is Morgan considering?"
    proposed = QueryPlan(
        subject="Riley",
        predicate="agency",
        semantic_queries=["What agency is Riley considering?", "adoption agency"],
        keywords=["Riley", "adoption", "agency"],
        required_info=["Riley's chosen agency"],
    )
    bound, changes = bind_plan_entities(question, proposed, ["Morgan", "Riley"])
    assert bound.semantic_queries == ["adoption agency"]
    assert bound.keywords == ["adoption", "agency"]
    assert bound.required_info == []
    assert bound.subject is None and bound.predicate is None
    assert set(changes) == {"semantic_queries", "keywords", "required_info", "symbolic_subject"}
    assert proposed.subject == "Riley"  # Original proposal remains auditable.


@pytest.mark.parametrize(
    ("query", "variants", "known"),
    [
        ("What did Morgan and Riley buy?", ["Morgan purchases", "Riley purchases"], ["Morgan", "Riley"]),
        ("Who is Morgan's sister?", ["Morgan's sister Riley", "sister"], ["Morgan", "Riley"]),
        ("Where did I travel?", ["Riley travel"], ["Morgan", "Riley"]),
        ("Where did Anna travel?", ["Ann travel"], ["Ann", "Riley"]),
        ("When did Morgan`s mother die?", ["Morgan's mother death"], ["Morgan", "Riley"]),
        ("Which dogs did Morgan adopt?", ["Morgan's dog Pixie", "Pixie adoption"], ["Morgan", "Riley"]),
        ("What app did Morgan plan?", ["dog-walking and pet-care app feature"], ["Morgan", "Riley"]),
    ],
)
def test_supported_expansions_and_implicit_subjects_remain(query, variants, known):
    plan = QueryPlan(semantic_queries=variants)
    bound, changes = bind_plan_entities(query, plan, known)
    assert bound == plan and not changes


def test_entity_matching_keeps_multiword_nonascii_and_word_boundaries():
    plan, changes = bind_plan_entities(
        "Where did JOSÉ SILVA travel?",
        QueryPlan(semantic_queries=["Riley travels", "José Silva travels"]),
        ["José Silva", "Riley", "José Silva"],
    )
    assert plan.semantic_queries == ["José Silva travels"] and changes == ["semantic_queries"]
    plan, changes = bind_plan_entities(
        "张三推荐了什么？", QueryPlan(semantic_queries=["李四推荐的作品"]), ["张三", "李四"]
    )
    assert plan.semantic_queries == [] and changes == ["semantic_queries"]


def test_collection_cannot_replace_an_explicit_actor():
    query = "Which movies did Morgan watch in 2023?"
    plan = QueryPlan(collection_query=CollectionQuery(subjects=["Riley"], predicate="movie"))
    bound, changes = bind_plan_entities(query, plan, ["Morgan", "Riley"])
    routed = apply_route(bound, classify_question(query, ["Morgan", "Riley"]))
    assert routed.collection_query is None and changes == ["collection_subjects"]
    assert routed.query_entities == ["Morgan"]
    assert "year=2023" in routed.query_constraints
    plan.collection_query.subjects = ["Morgan"]
    routed = apply_route(plan, classify_question(query, ["Morgan", "Riley"]))
    assert routed.query_entities == ["Morgan"]


@pytest.mark.parametrize(
    ("query", "subjects"),
    [
        ("How many Marvel movies did I re-watch?", ["user"]),
        ("How many engineers do I lead as Software Engineer Manager?", ["user"]),
        ("How often did the Chiefs play the Jaguars?", ["Kansas City Chiefs", "Jacksonville Jaguars"]),
        ("What names did we give the Radiation Amplified zombie?", ["Radiation Amplified"]),
    ],
)
def test_collection_retains_resolved_subjects_instead_of_capitalisation_guesses(query, subjects):
    plan = QueryPlan(collection_query=CollectionQuery(subjects=subjects, predicate="event"))
    bound, changes = bind_plan_entities(query, plan, ["user", "assistant"])
    routed = apply_route(bound, classify_question(query))
    assert not changes
    assert routed.query_entities == subjects


def test_retrieval_binds_before_embedding_but_keeps_third_party_evidence():
    class Planner:
        async def plan(self, query, context):
            return QueryPlan(
                semantic_queries=["What agency is Riley considering?"],
                keywords=["Riley", "agency"],
                required_info=["Riley's agency"],
            )

    class Embedder:
        def __init__(self):
            self.inputs = []

        async def encode(self, texts):
            self.inputs.extend(texts)
            return [[1.0, 0.0] for _ in texts]

    def memory(identifier, subject, text):
        return Memory(
            memory_id=identifier, session_id="session", scope_id="session", subject=subject,
            predicate="agency", value=text, content=text,
            evidence=[Evidence(turn_id=identifier, event_index=0, quote=text)],
        )

    embedder = Embedder()
    question = "What agency is Morgan considering?"
    result = asyncio.run(Retriever(Settings(), Planner(), embedder).search(
        question, Session(session_id="session"),
        [memory("m", "Morgan", "Morgan is considering an agency."),
         memory("r", "Riley", "Riley says Morgan prefers the Elm agency.")], [],
        known_speakers=["Morgan", "Riley"],
    ))
    assert result.plan.query_entities == ["Morgan"]
    assert result.plan.semantic_queries == []
    assert question in embedder.inputs
    assert "What agency is Riley considering?" not in embedder.inputs
    assert {hit.metadata["memory_id"] for hit in result.results} == {"m", "r"}
    assert any(warning.startswith("query_entity_binding:") for warning in result.warnings)

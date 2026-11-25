from agent_memory.models import QueryPlan
from agent_memory.retrieval import Retriever


def test_original_query_and_all_three_rewrites_are_retained():
    plan = QueryPlan(semantic_queries=["first", "second", "third"])
    assert Retriever._semantic_queries(" original ", plan) == ["original", "first", "second", "third"]
    assert plan.semantic_queries == ["first", "second", "third"]


def test_original_query_is_not_encoded_twice():
    plan = QueryPlan(semantic_queries=["original", " original ", " "])
    assert Retriever._semantic_queries("original", plan) == ["original"]


def test_original_query_recovers_evidence_missed_by_rewrite():
    import asyncio
    from types import SimpleNamespace
    from agent_memory.settings import Settings
    retriever = Retriever(Settings())
    async def vectors(queries, memories, namespace, generation):
        return [[1., 0.] if q == "original relationship" else [0., 1.] for q in queries] + [[1., 0.]], 0
    retriever._projected_vectors = vectors
    try:
        result, _ = asyncio.run(retriever._semantic_recall(
            "original relationship", QueryPlan(semantic_queries=["broad topic"]),
            [SimpleNamespace(memory_id="evidence", version=1)], None, None))
        assert result.scores[("evidence", 1)] == 1.0
        assert len(result.matches) == 1
    finally:
        retriever.close()

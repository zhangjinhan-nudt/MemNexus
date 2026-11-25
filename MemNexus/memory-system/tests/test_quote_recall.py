from agent_memory.models import Evidence, Memory, QueryPlan
from agent_memory.reasoning import indexed_memory_hits


def memory(name, content):
    return Memory(memory_id=name, session_id="s", scope_id="s", subject="cook",
                  predicate="recipe", value=content, content=content,
                  evidence=[Evidence(turn_id=name, event_index=0, quote=content)])


def test_quote_detail_participates_before_candidate_cutoff():
    memories = [memory("a", "A waffle cooking method."), memory("b", "Breakfast instructions.")]
    question = "How much flour does the waffle recipe use?"
    plan = QueryPlan()
    old = indexed_memory_hits(question, plan, memories, [], limit=1)
    assert old[0].metadata["memory_id"] == "a"
    matches = {"b": [{"text": "The waffle recipe uses 200 grams of flour."}]}
    new = indexed_memory_hits(question, plan, memories, [], limit=1, matched_evidence=matches)
    assert new[0].metadata["memory_id"] == "b"
    assert new[0].metadata["matched_evidence"] == matches["b"]


def test_duplicate_quotes_do_not_accumulate_rank_and_empty_map_is_compatible():
    memories = [memory("a", "A waffle recipe."), memory("b", "Breakfast instructions.")]
    query = "How much flour for waffles?"
    one = {"b": [{"text": "Waffles use flour."}]}
    repeated = {"b": one["b"] * 20}
    get = lambda m: [(h.chunk_id, h.score) for h in indexed_memory_hits(
        query, QueryPlan(), memories, [], matched_evidence=m)]
    assert get(one) == get(repeated)
    assert get({}) == get(None)


def test_topic_entity_words_remain_searchable_when_memory_subject_is_the_speaker():
    memories = [memory("a", "The assistant provided a list earlier."),
                memory("b", "The assistant provided woodworking jobs for retired carpenters.")]
    plan = QueryPlan(query_entities=["woodworking jobs for retired carpenters"])
    result = indexed_memory_hits("Which woodworking jobs for retired carpenters did you list?",
                                 plan, memories, [], limit=1)
    assert result[0].metadata["memory_id"] == "b"


def test_index_topic_can_rank_a_short_quote_but_is_not_published_as_evidence():
    from agent_memory.grounding import ground_evidence, source_payload
    from agent_memory.models import Event, Hit, Turn
    t = Turn(turn_id="t", session_id="old", request_id="r", sequence=1,
             events=[Event(role="assistant", content="Use rice flour.\nThe kitchen has a bowl.")])
    hits = [Hit(content=text, score=.5, source_file="memory", chunk_id=str(i), metadata={
        "source_support_status": "authorized_session_quote", "origin": "indexed_sweep",
        "evidence": [{"turn_id": "t", "event_index": 0, "quote": text}],
        **({"retrieval_hint": "Cedar dumpling dough recipe uses rice flour."} if i == 1 else {})})
        for i, text in [(0, "The kitchen has a bowl."), (1, "Use rice flour.")]]
    result = ground_evidence("Which flour did the cedar dumpling dough recipe use?", QueryPlan(), hits,
        session_id="now", current_turns=[], resolved_turns={"t": t}, blocked_events=set(),
        max_sources=1, budget=5000)
    hit, = result.sources
    assert hit.chunk_id == "1"
    assert "retrieval_hint" not in hit.metadata
    assert "Cedar" not in str(source_payload(hit))

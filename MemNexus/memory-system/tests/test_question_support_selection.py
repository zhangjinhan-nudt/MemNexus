"""A question may locate evidence without satisfying a requested fact.

Use ordinary aliases and distinct source dates; no benchmark entities or answers.
"""

from datetime import timedelta

import pytest

from agent_memory.grounding import _ReadingCandidate, _reading_support, ground_evidence
from agent_memory.models import AnswerContract, AnswerOutput, QueryPlan
from agent_memory.prompt_codec import synthesis_source_cost
from test_reading_units import make_hit, make_turn, visible


QUESTION = "Which water sports has Neri tried?"


def plan():
    return QueryPlan(route="none", answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="activities", question_span=QUESTION, operator="set", subjects=["Neri"],
        relation="water sports tried",
    )]))


def read(turns, hits, *, budget=4000, max_sources=20):
    return ground_evidence(QUESTION, plan(), hits, session_id="s", current_turns=[],
        resolved_turns={t.turn_id: t for t in turns}, blocked_events=set(), budget=budget,
        max_sources=max_sources, source_cost=lambda hit, source_id: synthesis_source_cost(
            hit, source_id, needs_time=False))


@pytest.mark.parametrize("keep_question_flag", [True, False])
def test_pure_question_does_not_saturate_the_request_feature(keep_question_flag):
    question = make_turn("query-echo", QUESTION, role="user")
    fact = make_turn("fact", "Neri tried paddling on the lake.", sequence=20, role="user")
    hits = [make_hit(question, QUESTION), make_hit(fact, fact.events[0].content, rank=2)]
    if not keep_question_flag:
        hits[0].metadata["source_events"][0].pop("is_question", None)
    candidates = [_ReadingCandidate(h, h.chunk_id, i, i) for i, h in enumerate(hits, 1)]
    features, weights = _reading_support(candidates, QUESTION, plan())
    request_keys = [key for key in weights if key[1] == "request"]
    assert request_keys
    assert all(features[candidates[0].key].get(key, 0) == 0 for key in request_keys)
    assert any(features[candidates[1].key].get(key, 0) > 0 for key in request_keys)


def test_mixed_statement_and_question_still_contributes_its_asserted_fact():
    text = "Neri tried paddling last summer. Which water sports has Neri tried?"
    turn = make_turn("mixed", text, role="user")
    hit = make_hit(turn, text)
    candidate = _ReadingCandidate(hit, hit.chunk_id, 1, 1)
    features, weights = _reading_support([candidate], QUESTION, plan())
    assert any(features[candidate.key].get(key, 0) > 0 for key in weights if key[1] == "request")
    result = read([turn], [hit])
    assert "Neri tried paddling last summer." in visible(result.sources)


def test_lexically_unmatched_semantic_candidate_competes_by_relevance_and_real_cost():
    # A long question echoes the requested vocabulary but provides no answer.
    # The already retrieved short answer uses a different vocabulary entirely.
    question_text = "Which water sports has Neri tried " + "during a holiday " * 35 + "?"
    question = make_turn("question", question_text, role="user")
    semantic = make_turn("semantic", "She kayaked.", sequence=20, role="user")
    qhit = make_hit(question, question_text, rank=1)
    answer = make_hit(semantic, semantic.events[0].content, rank=2)
    answer.metadata["retrieval_channel_ranks"] = {"short.semantic": 2}
    assert set(QUESTION.lower().split()).isdisjoint(semantic.events[0].content.lower().split())
    # Use the exact consumer serialization, not a hand-assigned candidate cost.
    question_only = read([question], [qhit])
    assert len(question_only.sources) == 1
    budget = synthesis_source_cost(question_only.sources[0], 1, needs_time=False)
    result = read([question, semantic], [qhit, answer], budget=budget)
    assert "She kayaked." in visible(result.sources)
    assert sum(synthesis_source_cost(h, i, needs_time=False)
               for i, h in enumerate(result.sources, 1)) <= budget


def test_question_only_source_remains_eligible_as_context():
    turn = make_turn("query-echo", QUESTION, role="user")
    result = read([turn], [make_hit(turn, QUESTION)])
    assert result.sources and QUESTION in visible(result.sources)
    # Zero request support is not a ban on a source and not an abstention rule.


def test_identical_words_at_distinct_dates_keep_both_original_events():
    turns = [make_turn("earlier", "Neri tried paddling.", sequence=1, role="user"),
             make_turn("later", "Neri tried paddling.", sequence=20, role="user")]
    turns[1].events[0].occurred_at += timedelta(days=60)
    hits = [make_hit(turn, turn.events[0].content, rank=i)
            for i, turn in enumerate(turns, 1)]
    result = read(turns, hits)
    records = [event for hit in result.sources for event in hit.metadata["source_events"]]
    assert {event["turn_id"] for event in records} == {"earlier", "later"}
    assert len({event["occurred_at"] for event in records}) == 2
    assert all(event["text"] == "Neri tried paddling." for event in records)

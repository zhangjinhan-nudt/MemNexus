"""Independent selector checks: comparisons, duplicate views and unit upgrades."""
from copy import deepcopy

import pytest
from test_reading_units import make_hit, make_turn

from agent_memory import grounding
from agent_memory.models import QueryPlan


def comparison_plan(question, subjects):
    return QueryPlan(route="none", answer_contract={"outputs": [{
        "output_id": "comparison", "question_span": question, "operator": "comparison",
        "subjects": subjects}]})


def candidate(name, text, rank=1):
    turn = make_turn(name, text, sequence=rank * 10)
    return grounding._ReadingCandidate(make_hit(turn, text, rank=rank), name, rank, rank)


@pytest.mark.parametrize("left,right", [("Alpha", "Beta"), ("Orchid", "Willow")])
def test_non_temporal_comparison_does_not_reward_an_unrelated_date(left, right):
    question = f"Which has more storage, {left} or {right}?"
    request = comparison_plan(question, [left, right])
    relevant = f"{left} has more storage than {right}."
    choices = [candidate("plain", relevant),
               candidate("irrelevant-date", relevant + " The sticker was printed on 2030-01-01.")]
    features, weights = grounding._reading_support(choices, question, request)
    value = lambda c: sum(weights[k] * v for k, v in features[c.key].items())
    assert value(choices[0]) >= value(choices[1])


def test_repeated_retrieval_view_does_not_change_other_candidates_scores():
    question = "Which has more storage, Alpha or Beta?"
    request = comparison_plan(question, ["Alpha", "Beta"])
    choices = [candidate("first", "Alpha has more storage than Beta."),
               candidate("second", "Beta has additional storage for archived images.")]
    features, weights = grounding._reading_support(choices, question, request)
    repeated = deepcopy(choices[1])
    repeated.parent_id = "another-retrieval-view"
    enlarged, new_weights = grounding._reading_support([*choices, repeated], question, request)
    assert new_weights == weights
    for c in choices:
        assert enlarged[c.key] == features[c.key]


@pytest.mark.parametrize("reverse", [False, True])
def test_local_exchange_can_upgrade_selected_variant_outside_lowest_contribution_pair(monkeypatch, reverse):
    # Compact A is initially efficient, but its full unit supplies three more
    # locating signals. The selected B/C each have lower tie priority than A.
    # The final improvement must replace A itself, not duplicate its parent.
    turns = [make_turn("A", "Anchor statement. Additional contextual details.", sequence=10),
             make_turn("B", "Independent statement B.", sequence=20),
             make_turn("C", "Independent statement C.", sequence=30)]
    hits = [make_hit(turn, turn.events[0].content, rank=i + 1) for i, turn in enumerate(turns)]
    question = "What do the recorded statements say?"
    plan = QueryPlan(route="none", answer_contract={"outputs": [{
        "output_id": "statements", "question_span": question}]})

    def units(sources, *args, include_neighbors=True, **kwargs):
        result = [source.model_copy(deep=True) for source in sources]
        if not include_neighbors:
            for hit in result:
                if hit.chunk_id == "A":
                    text = "Anchor statement."
                    hit.content = text
                    hit.metadata["source_events"][0].update(text=text, start=0, end=len(text))
        return result

    def locating_signals(candidates, *args):
        features = {}
        for c in candidates:
            signal = {c.parent_id: 1.0}
            if c.parent_id == "A" and c.variant == "context":
                signal.update(extra1=1.0, extra2=1.0, extra3=1.0)
            features[c.key] = signal
        return features, dict.fromkeys(["A", "B", "C", "extra1", "extra2", "extra3"], 1.0)

    def cost(hit, source_id):
        if hit.chunk_id != "A":
            return 1
        return 9 if "Additional" in hit.content else 2

    monkeypatch.setattr(grounding, "close_evidence_bundles", units)
    monkeypatch.setattr(grounding, "_reading_support", locating_signals)
    result = grounding.ground_evidence(question, plan, hits[::-1] if reverse else hits,
        session_id="s", current_turns=[], resolved_turns={t.turn_id: t for t in turns},
        blocked_events=set(), budget=10, max_sources=3, source_cost=cost)
    assert len(result.sources) == 2
    assert len({hit.chunk_id for hit in result.sources}) == 2
    assert "Additional" in next(h for h in result.sources if h.chunk_id == "A").content
    assert sum(cost(hit, i) for i, hit in enumerate(result.sources, 1)) == 10
    selected = [d for d in result.trace.selection_decisions if d["selected"]]
    assert next(d for d in selected if d["parent_id"] == "A")["reason"] == "selected_by_exchange"
    assert sum(d["final_cost"] for d in selected) == 10

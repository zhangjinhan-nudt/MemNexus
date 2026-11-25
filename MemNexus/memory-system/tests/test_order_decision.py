"""The order decision: what it may conclude, and what it must refuse.

The rule is meant to be arithmetic, not judgement, so the tests are about
warrants: two named candidates, dated events, one anchor, disjoint offsets. Each
test removes one warrant and expects the decision to be withheld.
"""

from __future__ import annotations

from agent_memory.reasoning import (
    GraphNode,
    decide_order_by_offset,
    names_other_candidate,
    offset_intervals,
    parse_order_candidates,
)


def node(node_id: str, text: str, occurred_at: str | None) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        subject=None,
        predicate=None,
        value=None,
        text=text,
        source_ids=[node_id],
        occurred_at=occurred_at,
    )


QUESTION = "Which item did I purchase first, the dog bed for Max or the training pads for Luna?"


def test_month_is_an_offset_range_not_a_calendar_month():
    # "a month ago" is 28-31 days back from the moment of speaking. Snapping it
    # to a calendar month (the ordering path) discards that quantity.
    assert offset_intervals("I got it a month ago.") == [(28, 31, "a month ago")]
    assert offset_intervals("I got it about a month ago.") == []
    assert offset_intervals("I got it two weeks ago.") == [(14, 14, "two weeks ago")]


def test_candidates_are_the_two_named_entities():
    assert parse_order_candidates(QUESTION) == (
        "the dog bed for Max",
        "the training pads for Luna",
    )


def test_a_question_without_two_candidates_is_not_an_order_question():
    assert parse_order_candidates("When did I buy the dog bed?") is None
    assert parse_order_candidates("Which item did I purchase first, the dog bed?") is None


def test_disjoint_offsets_decide_the_order():
    decision = decide_order_by_offset(
        QUESTION,
        [
            node("a", "I got a new dog bed for Max three weeks ago.", "2023-05-20T10:00:00+00:00"),
            node("b", "I got the training pads for Luna a month ago.", "2023-05-20T10:00:00+00:00"),
        ],
    )
    assert decision is not None
    assert decision["first"] == "the training pads for Luna"


def test_overlapping_offsets_withhold_the_decision():
    # A month is 28-31 days; four weeks is 28. They can be the same day, so the
    # evidence does not order them.
    assert (
        decide_order_by_offset(
            QUESTION,
            [
                node("a", "I got a new dog bed for Max four weeks ago.", "2023-05-20T10:00:00+00:00"),
                node("b", "I got the training pads for Luna a month ago.", "2023-05-20T10:00:00+00:00"),
            ],
        )
        is None
    )


def test_offsets_from_different_anchors_are_not_comparable():
    assert (
        decide_order_by_offset(
            QUESTION,
            [
                node("a", "I got a new dog bed for Max three weeks ago.", "2023-05-20T10:00:00+00:00"),
                node("b", "I got the training pads for Luna a month ago.", "2023-06-05T10:00:00+00:00"),
            ],
        )
        is None
    )


def test_an_undated_event_cannot_be_a_binding_target():
    # The dog bed is mentioned, but never dated: it cannot answer "which first".
    assert (
        decide_order_by_offset(
            QUESTION,
            [
                node("a", "I got a new dog bed for Max.", "2023-05-20T10:00:00+00:00"),
                node("b", "I got the training pads for Luna a month ago.", "2023-05-20T10:00:00+00:00"),
            ],
        )
        is None
    )


def test_two_dates_on_one_side_withhold_the_decision():
    # Two dog-bed events, two different offsets: the side itself is ambiguous.
    assert (
        decide_order_by_offset(
            QUESTION,
            [
                node("a1", "I got a new dog bed for Max three weeks ago.", "2023-05-20T10:00:00+00:00"),
                node("a2", "I got a new dog bed for Max two days ago.", "2023-05-20T10:00:00+00:00"),
                node("b", "I got the training pads for Luna a month ago.", "2023-05-20T10:00:00+00:00"),
            ],
        )
        is None
    )


def test_repeated_mentions_of_one_purchase_are_not_an_ambiguity():
    # Two mentions that agree on the offset describe one purchase.
    decision = decide_order_by_offset(
        QUESTION,
        [
            node("a1", "I got a new dog bed for Max three weeks ago.", "2023-05-20T10:00:00+00:00"),
            node("a2", "I told you, the dog bed for Max arrived three weeks ago.", "2023-05-20T10:00:00+00:00"),
            node("b", "I got the training pads for Luna a month ago.", "2023-05-20T10:00:00+00:00"),
        ],
    )
    assert decision is not None
    assert decision["first"] == "the training pads for Luna"


def _source(quote: str):
    from agent_memory.models import Hit

    return Hit(
        content=quote,
        score=0.9,
        source_file="source",
        chunk_id="m1",
        metadata={
            "source_events": [
                {
                    "text": quote,
                    "role": "user",
                    "occurred_at": "2023-05-20T10:00:00+00:00",
                    "quote": quote,
                }
            ]
        },
    )


def _table(value: str, quote: str):
    from agent_memory.synthesis import AnswerTable

    return AnswerTable.model_validate(
        {"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": QUESTION, "role": "answer", "values": [{
            "value": value,
            "supports": [{"source_id": 1, "quote": quote}],
        }]}]}
    )


DECIDED = [
    {
        "kind": "offset_order",
        "first": "the training pads for Luna",
        "second": "the dog bed for Max",
        "why": "test",
    }
]


def test_unbound_graph_hint_cannot_overwrite_supported_answer():
    from agent_memory.synthesis import validate_table

    table = _table("the dog bed for Max", "dog bed for Max")
    validate_table(table, [_source("dog bed for Max")], [QUESTION], "text", derivations=DECIDED)
    assert table.slots[0].values[0].value == "the dog bed for Max"


def test_answer_agreeing_with_the_decision_is_left_alone():
    from agent_memory.synthesis import validate_table

    table = _table("the training pads for Luna", "training pads for Luna")
    validate_table(
        table, [_source("training pads for Luna")], [QUESTION], "text", derivations=DECIDED
    )
    assert table.slots[0].values[0].value == "the training pads for Luna"


def test_without_a_decision_the_answer_is_left_alone():
    from agent_memory.synthesis import validate_table

    table = _table("the dog bed for Max", "dog bed for Max")
    validate_table(table, [_source("dog bed for Max")], [QUESTION], "text", derivations=[])
    assert table.slots[0].values[0].value == "the dog bed for Max"


def test_candidate_comparison_is_relative_not_absolute():
    first = "the narrator receiving their new phone case"
    second = "the narrator losing their phone charger"
    # Both answers say "phone"; only the verb tells them apart.
    assert names_other_candidate("losing their phone charger", first, second)
    assert not names_other_candidate("receiving their new phone case", first, second)

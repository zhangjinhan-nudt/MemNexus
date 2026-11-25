"""Interval algebra for stated dates: what "in January" or "two weeks ago" covers.

A date expression is not a point in time. "In January" covers a whole month;
"two weeks ago" at week precision covers a week bucket. Only disjoint spans have
a determined order; overlapping spans must stay undetermined rather than being
guessed from a shared edge. These properties are checkable by enumeration, so
they are asserted here instead of being inferred from benchmark scores.
"""

from datetime import date

import pytest

from agent_memory.reasoning import _interval_bounds, intervals_order


@pytest.mark.parametrize(
    "left,right,expected",
    [
        # Two exact days.
        (("2023-05-13", "day"), ("2023-05-15", "day"), -1),
        (("2023-05-15", "day"), ("2023-05-13", "day"), 1),
        (("2023-05-13", "day"), ("2023-05-13", "day"), 0),
        # A day inside a month is not ordered against that month.
        (("2023-05-15", "day"), ("2023-05-01", "month"), None),
        (("2023-05-15", "day"), ("2023-06-01", "month"), -1),
        (("2023-05-15", "day"), ("2023-04-01", "month"), 1),
        # Adjacent months are ordered; the same month is not.
        (("2023-04-01", "month"), ("2023-05-01", "month"), -1),
        # Any day inside May, at month precision, denotes the same span.
        (("2023-05-01", "month"), ("2023-05-31", "month"), 0),
        (("2023-05-01", "month"), ("2023-05-01", "month"), 0),
        # Week buckets: 2023-05-15 is a Monday, so the bucket runs to the 21st.
        (("2023-05-15", "week"), ("2023-05-22", "week"), -1),
        (("2023-05-15", "week"), ("2023-05-17", "day"), None),
        (("2023-05-15", "week"), ("2023-05-21", "day"), None),
        (("2023-05-15", "week"), ("2023-05-22", "day"), -1),
        # A week inside a month is not ordered against that month.
        (("2023-05-15", "week"), ("2023-05-01", "month"), None),
        (("2023-05-15", "week"), ("2023-06-01", "month"), -1),
    ],
)
def test_disjoint_spans_are_ordered_and_overlapping_are_not(left, right, expected):
    assert intervals_order(_interval_bounds(*left), _interval_bounds(*right)) == expected


def test_span_width_follows_precision():
    assert _interval_bounds("2023-05-15", "day") == (date(2023, 5, 15), date(2023, 5, 15))
    assert _interval_bounds("2023-05-15", "week") == (date(2023, 5, 15), date(2023, 5, 21))
    assert _interval_bounds("2023-05-01", "month") == (date(2023, 5, 1), date(2023, 5, 31))


@pytest.mark.parametrize(
    "stamp,expected_last",
    [
        ("2023-02-01", date(2023, 2, 28)),
        ("2024-02-01", date(2024, 2, 29)),
        ("2000-02-01", date(2000, 2, 29)),
        ("1900-02-01", date(1900, 2, 28)),
    ],
)
def test_february_length_follows_the_actual_calendar(stamp, expected_last):
    assert _interval_bounds(stamp, "month")[1] == expected_last


@pytest.mark.parametrize("precision", ["utterance", "explicit", "unknown"])
def test_telling_time_is_not_orderable(precision):
    # A turn timestamp dates when something was said, not when it happened.
    assert _interval_bounds("2023-05-15", precision) is None
    assert intervals_order(None, _interval_bounds("2023-05-15", "day")) is None


def test_malformed_stamp_is_not_orderable():
    assert _interval_bounds("not-a-date", "day") is None


def test_ledger_labels_week_as_week_not_month():
    from datetime import UTC, datetime

    from agent_memory.models import Event, Hit, Turn
    from agent_memory.reasoning import EventGraph, classify_question

    turn = Turn(
        turn_id="t1",
        session_id="s",
        sequence=1,
        request_id="t1",
        events=[
            Event(
                content="I bought the amp two weeks ago.",
                role="user",
                speaker="Nora",
                occurred_at=datetime(2023, 5, 27, tzinfo=UTC),
            )
        ],
    )
    from agent_memory.grounding import _source_event

    second = Turn(
        turn_id="t2",
        session_id="s",
        sequence=2,
        request_id="t2",
        events=[
            Event(
                content="I have had the amp for two weeks.",
                role="user",
                speaker="Nora",
                occurred_at=datetime(2023, 5, 27, tzinfo=UTC),
            )
        ],
    )
    source = Hit(
        content=turn.events[0].content,
        score=0.9,
        source_file="source",
        chunk_id="m1",
        metadata={"source_events": [
            _source_event(turn, 0, turn.events[0].content, 0),
            _source_event(second, 0, second.events[0].content, 0),
        ]},
    )
    graph = EventGraph.from_hits([source])
    graph.prompt_context("How long since the purchase?", classify_question("How long since the purchase?", []),
                         temporal_execution=True)
    ledger = next(item for item in graph.derivations if item.get("kind") == "anchored_date_ledger")
    assert any(
        "two weeks ago" in line and "week precision" in line for line in ledger["order"]
    ), ledger["order"]
    # Week precision must never be reported as month precision.
    assert not any("month precision" in line for line in ledger["order"])

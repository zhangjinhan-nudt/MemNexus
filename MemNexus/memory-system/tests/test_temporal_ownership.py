"""Shared temporal semantics: clocks, granularity and bound computation."""

from datetime import UTC, date, datetime

import pytest

from agent_memory.models import Hit
from agent_memory.reasoning import EventGraph, GraphNode, classify_question, decide_order_by_offset
from agent_memory.temporal import (
    anchored_dates,
    calendar_difference,
    question_time_window,
    resolve_time_references,
)

SOURCE = "2023-05-21T23:30:00-07:00"
QUESTION = "2023-06-21T09:00:00+08:00"


def test_source_and_question_clocks_resolve_the_same_words_differently():
    kwargs = {"source_time": SOURCE, "question_time": QUESTION}
    source = resolve_time_references("yesterday", **kwargs)[0]
    question = resolve_time_references("yesterday", clock="question", **kwargs)[0]
    assert (source.start, source.clock, source.anchor) == ("2023-05-20", "source", SOURCE)
    assert (question.start, question.clock) == ("2023-06-20", "question")
    assert source.end_exclusive == "2023-05-21"


def test_missing_source_clock_is_not_borrowed_from_question_or_sibling_iso_date():
    refs = resolve_time_references("It happened yesterday; the form says 2020-01-01.",
                                   question_time=QUESTION)
    assert [(r.start, r.clock) for r in refs] == [("2020-01-01", "absolute")]
    assert anchored_dates("yesterday", None, QUESTION) == []


def test_naive_timestamp_is_not_silently_assigned_a_timezone():
    assert resolve_time_references("yesterday", source_time="2023-05-21T23:30:00") == []
    assert resolve_time_references("yesterday", source_time="2023-05-21")[0].start == "2023-05-20"


@pytest.mark.parametrize("text, expected", [
    ("2024-02-29", "2024-02-29"),
    ("29th of February 2024", "2024-02-29"),
    ("March 15, 2021", "2021-03-15"),
    ("one week before 15 March 2021", "2021-03-08"),
])
def test_absolute_calendar_expressions_need_no_source_clock(text, expected):
    refs = resolve_time_references(text)
    assert len(refs) == 1
    assert (refs[0].start, refs[0].precision, refs[0].clock) == (expected, "day", "absolute")
    assert refs[0].anchor is None


def test_invalid_and_locale_ambiguous_dates_are_not_invented():
    for text in ("2023-02-29", "March 35, 2023", "3/8", "February 29, 2023"):
        assert resolve_time_references(text, source_time=SOURCE) == []


def test_month_precision_and_utterance_are_not_event_day_claims():
    refs = resolve_time_references("in March", source_time=SOURCE, include_utterance=True)
    month = next(r for r in refs if r.precision == "month")
    assert (month.start, month.end_exclusive) == ("2023-03-01", "2023-04-01")
    assert month.kind != "point"
    assert len([r for r in refs if r.kind == "utterance"]) == 1
    assert not any(r.kind == "utterance" for r in resolve_time_references("in March", source_time=SOURCE))


@pytest.mark.parametrize("text", ["about two days ago", "roughly a month ago", "around March 15, 2021"])
def test_approximation_has_no_fabricated_numeric_tolerance(text):
    ref = resolve_time_references(text, source_time=SOURCE)[0]
    assert ref.expression == text
    assert ref.precision == "approximate"
    assert ref.start is None and ref.end_exclusive is None
    assert question_time_window("What happened " + text + "?", datetime(2023, 6, 21, tzinfo=UTC)) is None


def test_duration_start_is_explicitly_distinct_from_an_event_point():
    refs = resolve_time_references("I have practiced for two weeks.", source_time=SOURCE)
    assert refs[0].kind == "interval_start@2023-05-21"
    assert refs[0].rule == "stated_duration_start"
    assert refs[0].expression == "for two weeks"


def test_graph_does_not_publish_unbound_pairwise_date_differences():
    text = "I moved on 2023-02-01. An unrelated invoice is dated 2023-02-08."
    hit = Hit(content=text, score=1, source_file="test", chunk_id="test", metadata={
        "source_events": [{"text": text, "occurred_at": SOURCE, "turn_id": "t1",
                           "event_index": 0, "role": "user", "speaker": "Nora"}]
    })
    graph = EventGraph.from_hits([hit])
    query = "How long did the move take?"
    derived = graph.deterministic_derivations(query, classify_question(query), temporal_execution=True)
    assert any(item["kind"] == "anchored_date_ledger" for item in derived)
    assert not any(item["kind"] == "date_differences" for item in derived)


def test_approximate_offsets_cannot_override_answer_with_a_hard_order():
    nodes = [GraphNode(node_id=str(i), subject=None, predicate=None, value=None,
                       source_ids=[str(i)], occurred_at=SOURCE, text=text)
             for i, text in enumerate(("I bought the guitar about three weeks ago.",
                                       "I bought the piano about a month ago."))]
    assert decide_order_by_offset("Which came first, the guitar or the piano?", nodes) is None


@pytest.mark.parametrize("start,end,unit,expected", [
    (date(2024, 2, 28), date(2024, 3, 1), "days", "2 days"),
    (date(2023, 1, 15), date(2023, 4, 15), "months", "3 months"),
    (date(2020, 2, 29), date(2024, 2, 29), "years", "4 years"),
])
def test_calendar_arithmetic_uses_actual_calendar(start, end, unit, expected):
    assert calendar_difference(start, end, unit) == expected


@pytest.mark.parametrize("start,end,unit", [
    (date(2023, 1, 15), date(2023, 2, 14), "months"),
    (date(2023, 1, 15), date(2023, 1, 23), "weeks"),
    (date(2023, 1, 15), date(2022, 1, 15), "years"),
])
def test_calendar_arithmetic_never_rounds_to_invent_a_whole_unit(start, end, unit):
    with pytest.raises(ValueError):
        calendar_difference(start, end, unit)

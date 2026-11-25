"""The question's own relative time window, resolved against the question date.

A question anchored to "a week ago" and a source anchored to "today" are anchored
to different days. These tests pin that the window is resolved from the question
date, that unresolvable or ambiguous questions yield nothing (so behaviour is
unchanged), and that a benchmark without a question date -- LoCoMo -- is
structurally untouched.
"""

from datetime import UTC, datetime

import pytest

from agent_memory.temporal import question_time_window


def at(*stamp):
    year, month, day = stamp
    return datetime(year, month, day, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize("question,as_of,expected", [
    # Retain the historical regression inputs, but not the old assumption that
    # their gold event being inside an ISO bucket proves that bucket's meaning.
    # Coarse elapsed language leaves the original question/as_of with the reader.
    ("Which book did I finish a week ago?", at(2023, 2, 7), None),
    ("I received a piece of jewelry last Saturday from whom?", at(2023, 3, 9),
     {"expression": "last Saturday", "start": "2023-03-04", "end_exclusive": "2023-03-05",
      "precision": "day", "as_of": "2023-03-09"}),
    ("What charity event did I participate in a month ago?", at(2023, 4, 18), None),
    ("What gardening-related activity did I do two weeks ago?", at(2023, 5, 5), None),
])
def test_relative_window_is_resolved_from_the_question_date(question, as_of, expected):
    assert question_time_window(question, as_of) == expected


def test_no_question_date_means_no_window():
    # LoCoMo supplies no Question date, so as_of is None and nothing is injected:
    # the prompt is byte-identical to before this change. This is the structural
    # guarantee that a LongMemEval fix cannot move LoCoMo.
    assert question_time_window("Which book did I finish a week ago?", None) is None


def test_naive_datetime_is_rejected():
    assert question_time_window(
        "Which book did I finish a week ago?", datetime(2023, 2, 7, 12, 0)
    ) is None


def test_questions_without_a_relative_expression_yield_nothing():
    for question in (
        "When did Caroline go to the LGBTQ support group?",
        "Who is Caroline a fan of?",
        "How many books did I read?",
    ):
        assert question_time_window(question, at(2023, 2, 7)) is None, question


def test_deictic_words_anchor_to_the_question_date():
    # "yesterday" is relative to the moment of asking, not to any source's own
    # utterance date -- the same anchor problem, one word shorter.
    window = question_time_window("What did Melanie paint yesterday?", at(2023, 2, 7))
    assert window == {"expression": "yesterday", "start": "2023-02-06",
                      "end_exclusive": "2023-02-07", "precision": "day",
                      "as_of": "2023-02-07"}


def test_absolute_dates_are_not_treated_as_a_relative_window():
    # An ISO date is anchored by the text itself; re-anchoring it to as_of would
    # invent a different span.
    assert question_time_window(
        "What happened on 2023-01-15?", at(2023, 2, 7)
    ) is None


def test_two_conflicting_windows_yield_nothing():
    # No single window to assert, so stay silent rather than pick one.
    assert question_time_window(
        "Did I read more last week or two weeks ago?", at(2023, 2, 7)
    ) is None


def test_explicit_exact_week_offset_is_a_day_not_a_calendar_bucket():
    # The wording itself now warrants a day, independently of any gold event.
    window = question_time_window("Which book did I finish exactly a week ago?", at(2023, 2, 7))
    assert window["precision"] == "day"
    assert (window["start"], window["end_exclusive"]) == ("2023-01-31", "2023-02-01")
    assert window["start"] <= "2023-01-31" < window["end_exclusive"]


def test_future_window_is_not_asserted():
    # Measured hazard: "I'm planning my meal prep next week, any suggestions?"
    # resolves "next week" to a window that has not happened yet. No recorded
    # event can fall inside it, so asserting it would push a correct answer into
    # an abstention. Such questions ask for advice, not for a past fact.
    assert question_time_window(
        "I'm planning my meal prep next week, any suggestions for new recipes?",
        at(2023, 5, 30),
    ) is None


@pytest.mark.parametrize("question", [
    # "since X" anchors the START of a period; the question asks for everything
    # completed from then until now, not just that one week.
    "How many total pieces have I completed since I started three weeks ago?",
    "How many weeks have I been taking classes since I began last month?",
    "Which events happened between the party last week and today?",
])
def test_period_anchoring_is_not_treated_as_a_filter(question):
    assert question_time_window(question, at(2023, 5, 30)) is None


def test_period_anchor_does_not_disable_a_plain_past_window():
    # The guard is about the question's own grammar, not about banning words:
    # a plain "last weekend" question still gets its window.
    assert question_time_window(
        "What game did I finally beat last weekend?", at(2023, 5, 23)
    ) is not None

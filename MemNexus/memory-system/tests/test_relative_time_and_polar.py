"""Regression tests for the two deterministic rendering defects.

Defect 1 -- an elapsed offset that carries its own written calendar anchor
    ("3 years before January 23, 2022") was never resolved, because the only
    offset rule matched "N units ago" and used the utterance date as the anchor.
    The value reached the evaluator as a relative phrase and scored ~0 against a
    gold answer written as a bare year.  Motivating items: conv-42/8 (F1 0.0)
    and conv-47/4 (F1 22.2).

Defect 2 -- a yes/no question was answered with the polarity plus a
    justification ("No, he had no pets at that time").  The obligation is a
    single polar token; the justification is pure precision loss under
    token-overlap F1.  Motivating items: conv-44/8 and conv-47/3 (both F1 22.2).

The fixtures below are the recorded values, not benchmark predictions: no
provider is contacted and no usage is billed.
"""

from datetime import UTC, datetime

import pytest

from agent_memory.temporal import derive_times, render_phrase
from agent_memory.synthesis import normalize_answer_text

ANCHOR = datetime(2022, 1, 23, 14, 1, tzinfo=UTC)


def phrases(text: str, anchor: datetime = ANCHOR) -> list[str]:
    return [render_phrase(d) for d in derive_times(text, anchor)]


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("3 years before January 23, 2022", "2019"),
        ("About one month before 27 March 2022", "February 2022"),
        ("two days before 2022-03-18", "16 March 2022"),
        ("2 years after 2020-05-10", "2022"),
        ("two weeks before 2022-03-18", "4 March 2022"),
    ],
)
def test_offset_from_a_written_date_resolves_to_calendar_form(expression, expected):
    assert phrases(expression) == [expected]


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("3 years ago", "2019"),
        ("one month ago", "December 2021"),
        ("last year", "2021"),
        ("yesterday", "22 January 2022"),
    ],
)
def test_anchor_based_rules_are_unchanged(expression, expected):
    """The new rule must not steal spans from the pre-existing ones."""
    assert phrases(expression) == [expected]


def test_an_absolute_date_without_a_direction_word_is_left_alone():
    assert phrases("in March 2023") == []


@pytest.mark.parametrize(
    "value,question,expected",
    [
        ("No, he had no pets at that time", "Did Andrew have a pet dog during March 2023?", "No"),
        ("Yes, because he said so", "Does James live in Connecticut?", "Yes"),
        ("No", "Did Andrew have a pet dog during March 2023?", "No"),
    ],
)
def test_polar_question_keeps_only_the_polarity(value, question, expected):
    assert normalize_answer_text(value, question) == expected


def test_a_hedged_polarity_without_a_separator_is_not_truncated():
    """'no but actually yes' must survive intact; only separated tails are cut."""
    assert normalize_answer_text(
        "No but actually yes", "Did Andrew have a pet dog during March 2023?"
    ) == "No but actually yes"


def test_a_non_polar_question_is_untouched():
    assert normalize_answer_text(
        "No, he had no pets at that time", "What happened in March 2023?"
    ) == "No, he had no pets at that time"


def test_polar_normalisation_is_idempotent():
    once = normalize_answer_text("No, he had no pets", "Did he have a pet?")
    assert normalize_answer_text(once, "Did he have a pet?") == once

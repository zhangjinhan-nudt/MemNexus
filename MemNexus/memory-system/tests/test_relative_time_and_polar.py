"""Source-time parsing and content-preserving publication have separate owners.

The parser retains source-anchored calendar rules. Publication must preserve
supported content, including relative phrases, explanations and recommendations
following a leading yes/no. Fixtures are synthetic; no provider is contacted.
"""

from datetime import UTC, datetime

import pytest

from agent_memory.models import Hit
from agent_memory.synthesis import (
    AnswerTable,
    normalize_answer_text,
    render_table,
)
from agent_memory.temporal import derive_times, render_phrase

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
    "value,question",
    [
        ("No, he had no pets at that time", "Did Andrew have a pet dog during March 2023?"),
        ("Yes, because he said so", "Does James live in Connecticut?"),
        ("No", "Did Andrew have a pet dog during March 2023?"),
        ("Yes, try a history podcast during your commute.", "Can you recommend an activity for my commute?"),
        ("Yes, and here is why: the cost went down.", "Did the cost go down, and why?"),
    ],
)
def test_normalization_preserves_explanations_and_requested_advice(value, question):
    assert normalize_answer_text(value, question) == value


def test_a_hedged_polarity_without_a_separator_is_not_truncated():
    """A hedge remains part of the published claim."""
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


@pytest.mark.parametrize("kind", ["text", "count", "duration", "offset", None])
def test_other_answer_kinds_never_replace_their_content_with_a_date(kind):
    text = "I bought a bike last month."
    source = Hit(content=text, score=1, source_file="fixture", chunk_id="fixture", metadata={
        "source_events": [{"text": text, "role": "user", "occurred_at": "2024-05-20T00:00:00+00:00"}],
    })
    table = AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": "What did I do recently?", "values": [{
        "value": "Bought a bike last month", "supports": [{"source_id": 1, "quote": text}],
    }]}]})
    assert render_table(table, [source], kind=kind) == "Bought a bike last month"


def test_date_rendering_preserves_relative_expression_and_surrounding_content():
    text = "I bought a bike last month."
    source = Hit(content=text, score=1, source_file="fixture", chunk_id="fixture", metadata={
        "source_events": [{"text": text, "role": "user", "occurred_at": "2024-05-20T00:00:00+00:00"}],
    })
    table = AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": "When did I buy a bike?", "values": [{
        "value": "last month", "supports": [{"source_id": 1, "quote": text}],
    }]}]})
    assert render_table(table, [source], kind="date") == "last month"
    item = table.slots[0].values[0]
    item.value = "Bought a bike last month"
    assert render_table(table, [source], kind="date") == "Bought a bike last month"

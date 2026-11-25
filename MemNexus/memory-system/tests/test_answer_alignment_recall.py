"""Surface normalization, evidence selection, and source-anchored relative time."""

from datetime import datetime

import pytest

from agent_memory.models import Hit
from agent_memory.synthesis import AnswerTable, normalize_answer_text, render_table
from agent_memory.temporal import derive_times, render_phrase

# --------------------------------------------------------------------------- #
# Surface canonicalisation
# --------------------------------------------------------------------------- #

def test_iso_date_is_displayed_as_a_written_date():
    assert normalize_answer_text("2023-01-29", "When?") == "29 January 2023"


def test_date_normalisation_is_idempotent():
    once = normalize_answer_text("2023-01-29", "When did she write?")
    assert normalize_answer_text(once, "When did she write?") == once


def test_lower_bound_is_not_expanded_from_the_question():
    assert (
        normalize_answer_text("at least 3", "How many dogs does James have?")
        == "at least 3"
    )


def test_lower_bound_without_a_noun_keeps_the_bound():
    assert normalize_answer_text("at least 3", "How many?") == "at least 3"


def test_lower_bound_with_noun_already_present_is_not_duplicated():
    assert (
        normalize_answer_text("at least 3 dogs", "How many dogs does James have?")
        == "at least 3 dogs"
    )


@pytest.mark.parametrize(
    "value",
    ["Max and Daisy", "Connecticut", "counseling or mental health", "taekwondo"],
)
def test_plain_answer_is_untouched(value):
    assert normalize_answer_text(value, "What are the names?") == value


def test_render_table_and_normalization_preserve_the_same_bound():
    table = AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": "q", "values": [
        {"value": "at least 3", "supports": [{"source_id": 1, "quote": "x"}]}]}]})
    assert render_table(table) == "at least 3"
    assert normalize_answer_text("at least 3", "How many dogs does James have?") == "at least 3"


def test_normalisation_collapses_duplicate_forms_before_joining():
    once = normalize_answer_text("2023-01-29", "When?")
    twice = normalize_answer_text(once, "When?")
    assert list(dict.fromkeys([once, twice])) == ["29 January 2023"]


# --------------------------------------------------------------------------- #
# Relative time resolution
# --------------------------------------------------------------------------- #

def _time_source(occurred_at: str, text: str) -> Hit:
    return Hit(content=text, score=0.9, source_file="source", chunk_id="t1",
               metadata={"source_events": [{"turn_id": "t1", "event_index": 0,
                                            "text": text, "occurred_at": occurred_at,
                                            "context_only": False}]})


def _value_table(value: str, quote: str) -> AnswerTable:
    return AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": "q", "values": [
        {"value": value, "supports": [{"source_id": 1, "quote": quote}]}]}]})


def test_parser_uses_the_utterance_but_publication_preserves_the_value():
    sources = [_time_source("2023-02-04T10:00:00+00:00", "I was in Bogota last summer.")]
    assert render_phrase(derive_times("last summer", datetime.fromisoformat("2023-02-04T10:00:00+00:00"))[0]) == "summer 2022"
    assert render_table(_value_table("last summer", "I was in Bogota last summer."), sources,
                        kind="date") == "last summer"


def test_publication_keeps_yesterday_without_a_second_time_resolver():
    sources = [_time_source("2023-01-26T10:00:00+00:00", "I got the letter yesterday.")]
    assert render_table(_value_table("yesterday", "I got the letter yesterday."), sources,
                        kind="date") == "yesterday"


def test_relative_value_is_kept_when_sources_disagree():
    first = _time_source("2023-02-04T10:00:00+00:00", "I was in Bogota last summer.")
    second = _time_source("2024-02-04T10:00:00+00:00", "Bogota last summer was hot.")
    table = AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": "q", "values": [
        {"value": "last summer", "supports": [{"source_id": 1, "quote": "I was in Bogota last summer."},
                                              {"source_id": 2, "quote": "Bogota last summer was hot."}]}]}]})
    assert render_table(table, [first, second], kind="date") == "last summer"


def test_absolute_value_is_not_rewritten():
    sources = [_time_source("2023-02-04T10:00:00+00:00", "I was in Bogota in June 2022.")]
    assert render_table(_value_table("June 2022", "I was in Bogota in June 2022."), sources,
                        kind="date") == "June 2022"


def test_multi_value_join_uses_commas_and_drops_duplicates():
    table = AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": "q", "values": [
        {"value": "taekwondo", "supports": [{"source_id": 1, "quote": "x"}]},
        {"value": "kickboxing", "supports": [{"source_id": 1, "quote": "x"}]},
        {"value": "taekwondo", "supports": [{"source_id": 1, "quote": "x"}]}]}]})
    assert render_table(table) == "taekwondo, kickboxing"
    assert "; " not in render_table(table)


# --- 2026-11-13 regression: a bound inside a longer answer is not the answer --


def test_lower_bound_inside_a_long_answer_is_not_collapsed():
    # Measured failure (LongMemEval preference): a multi-step cooking answer
    # containing "soak cashews for at least 4 hours" was rewritten to
    # "four (at least 4)", destroying three correct tips. The bound was a detail
    # inside the answer, not the answer itself.
    value = (
        "Monitor the milk temperature with a thermometer, aiming for 180F. "
        "Soak cashews for at least 4 hours or overnight, changing the water. "
        "Adjust water and blending time for desired thickness."
    )
    assert normalize_answer_text(value, "Any advice on getting better results?") == value


def test_lower_bound_with_a_leading_clause_is_not_collapsed():
    # Three words of remainder is already sentence content ("Monitor
    # temperature, hours"), so it must survive rather than be replaced.
    value = "Monitor temperature, at least 4 hours"
    assert normalize_answer_text(value, "How many dogs?") == value


def test_lower_bound_preserves_its_counted_noun_and_duration_unit():
    assert (
        normalize_answer_text("at least 3 dogs", "How many dogs does James have?")
        == "at least 3 dogs"
    )
    # Already carrying the counted form, so it is left alone rather than
    # restated into a duplicate.
    assert (
        normalize_answer_text("at least two dogs", "How many dogs?")
        == "at least two dogs"
    )
    assert normalize_answer_text("at least 3 years", "How long did I work?") == "at least 3 years"

"""Elapsed-time request metadata survives without publication-time arithmetic.

The question kind and decision time are available to generation. The source-bound
publisher validates provenance and preserves the model value; these tests do not
claim its numeric or temporal conclusion is correct.
"""

import pytest

from agent_memory.models import Hit
from agent_memory.runtime import _question_date, _reasoning_query
from agent_memory.synthesis import AnswerTable, answer_type, render_table, validate_table


def source(text: str, stamp: str) -> Hit:
    return Hit(
        content=text, score=0.9, source_file="source", chunk_id="c1",
        metadata={"source_events": [{
            "text": text, "occurred_at": stamp, "role": "user", "speaker": "Inez",
            "turn_id": "t1", "event_index": 0,
        }]},
    )


def table(value: str, quote: str) -> AnswerTable:
    return AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": "q", "values": [
        {"value": value, "supports": [{"source_id": 1, "quote": quote}]}]}]})


QUESTION = "How many days ago did I read the March 15th issue of The New Yorker?"
TEXT = "I finally got around to reading the March 15th issue of The New Yorker today"
STAMP = "2023-03-20T10:00:00+00:00"


@pytest.mark.parametrize("question", [
    "How many days ago did I meet Emma?",
    "How many weeks ago did I attend the friends and family sale?",
    "How many months ago did I book the Airbnb in San Francisco?",
    "How long ago did I start the pottery class?",
])
def test_elapsed_time_questions_retain_the_original_offset_metadata(question):
    # The candidate restores the original metadata; it does not execute date arithmetic.
    # Keep every legacy acceptance/rendering case below unchanged.
    assert answer_type(question) == "offset"


@pytest.mark.parametrize("question,expected", [
    # A length between two events keeps both endpoints in the evidence.
    ("How many days passed between the feedback and the test?", "duration"),
    ("How many weeks had passed since I recovered when I went on my 10th date?", "duration"),
    ("After how many weeks did Inez visit again?", "duration"),
    # Preserve the original legacy classifier and its duplicate regression input.
    ("How long since the purchase?", "offset"),
    # No published answer is recomputed by this metadata.
    ("How long since the purchase?", "offset"),
    # Filter spans and calendar lookups are unchanged.
    ("How many MCU films did I watch in the last 3 months?", "count"),
    ("When did Caroline go to the LGBTQ support group?", "date"),
    ("What did Melanie paint yesterday?", "text"),
])
def test_non_offset_questions_keep_their_kind(question, expected):
    assert answer_type(question) == expected


def test_point_offset_is_accepted_for_an_elapsed_time_question():
    # Regression: this exact form was rejected as "a duration answer must be a
    # length", which killed every question whose natural answer is "2 weeks ago".
    result = table("2 weeks ago", TEXT)
    validate_table(result, [source(TEXT, STAMP)], ["q"], "offset", as_of="2023-04-03")
    assert "weeks ago" in render_table(result)


def test_bare_number_is_a_valid_elapsed_time_answer():
    # LongMemEval gold is sometimes just "2"; the unit is implied by the question.
    result = table("2", TEXT)
    validate_table(result, [source(TEXT, STAMP)], ["q"], "offset", as_of="2023-04-03")
    assert render_table(result) == "2"


def test_decision_time_does_not_overwrite_the_model_answer():
    result = table("59 days", TEXT)
    original = result.model_dump()
    validate_table(result, [source(TEXT, STAMP)], ["q"], "offset", as_of="2023-04-01")
    assert result.model_dump() == original
    assert render_table(result) == "59 days"


def test_inclusive_day_convention_is_not_rewritten():
    # Gold accepts "13 days" for a 12-day gap; do not fight an accepted answer.
    result = table("13 days ago", TEXT)
    validate_table(result, [source(TEXT, STAMP)], ["q"], "offset", as_of="2023-04-01")
    assert render_table(result) == "13 days ago"


def test_without_decision_time_the_value_is_left_alone():
    result = table("59 days", TEXT)
    validate_table(result, [source(TEXT, STAMP)], ["q"], "offset")
    assert render_table(result) == "59 days"


def test_ambiguous_evidence_is_not_repaired():
    # One answer citing two differently dated events gives two offsets; picking
    # one of them would be a guess, so the published value is left untouched.
    second_text = "I read the March 15th issue before the trip"
    second = source(second_text, "2023-02-01T10:00:00+00:00")
    ambiguous = AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": "q", "values": [{
        "value": "59 days",
        "supports": [
            {"source_id": 1, "quote": TEXT},
            {"source_id": 2, "quote": second_text},
        ],
    }]}]})
    validate_table(ambiguous, [source(TEXT, STAMP), second], ["q"], "offset", as_of="2023-04-01")
    assert render_table(ambiguous) == "59 days"


def test_rendering_an_offset_does_not_turn_it_back_into_a_calendar_date():
    # Regression: render_table resolved every relative phrase against its own
    # utterance date, so a correct "26 days ago" was published as
    # "23 February 2022" -- the event date the question asked the distance from.
    result = table("26 days ago", TEXT)
    validate_table(result, [source(TEXT, STAMP)], ["q"], "offset", as_of="2023-04-15")
    assert render_table(result, [source(TEXT, STAMP)], kind="offset") == "26 days ago"


def test_decision_time_is_separated_from_the_question_text():
    query = "How many days ago did I meet Emma?\nQuestion date: 2023/04/20 (Thu) 09:29"
    as_of = _question_date(query)
    # Aware datetime at midday: relative-window derivation needs an aware anchor,
    # and callers that only need the day read str(as_of)[:10].
    assert as_of is not None and as_of.tzinfo is not None
    assert str(as_of)[:10] == "2023-04-20"
    assert _reasoning_query(query) == "How many days ago did I meet Emma?"
    # No metadata (LoCoMo): nothing to extract, nothing changes.
    assert _question_date("When did Caroline go to the support group?") is None


def test_observation_date_cannot_override_explicit_event_date():
    text = "I visited on 2023-05-01."
    result = table("20 days ago", text)
    validate_table(result, [source(text, "2023-05-20T10:00:00+00:00")], ["q"], "offset", as_of="2023-05-21")
    assert render_table(result) == "20 days ago"

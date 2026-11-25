"""Written dates bind exact calendar endpoints without guessing numeric locale."""

from datetime import UTC, date, datetime

import pytest

from agent_memory.models import Hit
from agent_memory.reasoning import _anchored_dates
from agent_memory.synthesis import AnswerTable, render_table, validate_table
from agent_memory.temporal import _parse_written_date, calendar_difference, derive_times

ANCHOR = "2023-03-29T06:00:00+00:00"


def points(text, anchor=ANCHOR):
    return [(stamp, precision) for stamp, _, precision, kind in _anchored_dates(text, anchor)
            if kind == "point"]


@pytest.mark.parametrize("text", [
    "on March 15", "on March 15th", "on 15 March", "on 15th March",
    "on the 15th of March", "on the 15TH OF MARCH",
])
def test_both_written_orders_use_source_year_only_when_year_is_missing(text):
    assert points(text) == [("2023-03-15", "day")]


@pytest.mark.parametrize("text", [
    "on March 15, 2021", "on March 15th 2021", "on 15 March 2021",
    "on 15th of March, 2021", "on March 15,2021",
])
def test_explicit_year_overrides_the_source_year(text):
    assert points(text) == [("2021-03-15", "day")]
    assert points(text, None) == [("2021-03-15", "day")]


@pytest.mark.parametrize("text", [
    "on 31st of April", "on February 29, 2023", "on 29 February 2023",
    "on 32 March 2024", "on March 15, 0000", "on March 15, 12345",
])
def test_invalid_explicit_dates_do_not_fall_back_to_the_source_year(text):
    assert points(text) == []


def test_leap_day_uses_the_explicit_calendar_year():
    assert points("on 29th February 2024") == [("2024-02-29", "day")]
    assert points("on February 29", "2024-03-10T00:00:00+00:00") == [("2024-02-29", "day")]


def test_missing_year_without_a_source_anchor_stays_unresolved():
    assert points("on 15th of March", None) == []
    assert _parse_written_date("15th of March") is None


def test_month_precision_is_not_promoted_to_an_exact_day():
    assert points("in March") == []
    assert points("in March 2021") == []
    assert any(precision == "month" for _, _, precision, _ in _anchored_dates("in March", ANCHOR))
    assert all(precision != "month" for _, _, precision, _ in _anchored_dates("on 15th March", ANCHOR))


def test_existing_offset_parser_reuses_written_date_grammar():
    assert _parse_written_date("March 2021") == date(2021, 3, 1)
    value = derive_times("one week before 15th of March 2021", datetime(2023, 3, 29, tzinfo=UTC))
    assert len(value) == 1
    assert (value[0].start, value[0].precision) == ("2021-03-08", "day")


def source(text, index):
    return Hit(content=text, score=0.9, source_file="source", chunk_id=f"c{index}",
               metadata={"source_events": [{"text": text, "occurred_at": ANCHOR,
                   "role": "user", "speaker": "user", "turn_id": f"t{index}", "event_index": 0}]})


def interval_answer(sources):
    return AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": "How many days between the two gifts?",
        "values": [{"value": "7 days", "supports": [
            {"source_id": index, "quote": hit.content} for index, hit in enumerate(sources, 1)
        ]}]}]})


def test_mixed_written_orders_support_exact_interval_without_changing_quotes():
    sources = [source("I bought the first gift on March 8.", 1),
               source("I bought the second gift on the 15th of March.", 2)]
    endpoints = [date.fromisoformat(points(hit.content)[0][0]) for hit in sources]
    assert calendar_difference(*endpoints, "days") == "7 days"
    table = interval_answer(sources)
    original_supports = table.slots[0].values[0].supports.copy()
    validate_table(table, sources, [table.slots[0].target], "duration")
    assert render_table(table) == "7 days"
    assert table.slots[0].values[0].supports == original_supports


def test_written_endpoint_parser_cannot_replace_an_explicit_year_with_source_year():
    texts = ["I bought the first gift on March 8, 2021.",
             "I bought the second gift on 15th March 2021."]
    endpoints = [date.fromisoformat(points(text)[0][0]) for text in texts]
    assert endpoints == [date(2021, 3, 8), date(2021, 3, 15)]
    assert calendar_difference(*endpoints, "days") == "7 days"


def test_ambiguous_slash_date_remains_unresolved():
    assert points("on the 3/8") == []
    assert _parse_written_date("3/8") is None


def test_invented_date_quote_still_fails_validation():
    sources = [source("I bought the first gift on March 8.", 1),
               source("I bought the second gift on 15th March.", 2)]
    table = interval_answer(sources)
    table.slots[0].values[0].supports[1].quote = "I bought the second gift on 16th March."
    with pytest.raises(ValueError, match="exact quote"):
        validate_table(table, sources, [table.slots[0].target], "duration")

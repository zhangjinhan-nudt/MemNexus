"""Independent clock, precision and range controls; no benchmark answers."""

from datetime import UTC, date, datetime, timedelta

import pytest

from agent_memory.temporal import derive_times, question_time_window, resolve_time_references


@pytest.mark.parametrize("stamp", [
    "2024-03-01T09:00:00+00:00",  # Leap-day crossing.
    "2024-01-02T00:30:00+14:00",  # Previous year; source calendar, not UTC day.
    "2022-12-31T23:30:00-07:00",
    "2025-08-31T11:00:00+00:00",  # Month length must not affect weeks.
])
@pytest.mark.parametrize("weeks", [1, 2, 9])
def test_exact_week_offsets_equal_elapsed_days_with_complete_spans(stamp, weeks):
    anchor = datetime.fromisoformat(stamp)
    phrase = f"exactly {weeks} weeks ago"
    text = f"The delivery arrived {phrase}; the invoice was later."
    ref, = resolve_time_references(text, source_time=anchor)
    day_ref, = resolve_time_references(f"{weeks * 7} days ago", source_time=anchor)
    expected = anchor.date() - timedelta(days=weeks * 7)
    assert (ref.start, ref.end_exclusive, ref.precision) == (
        expected.isoformat(), (expected + timedelta(days=1)).isoformat(), "day")
    assert (ref.start, ref.end_exclusive) == (day_ref.start, day_ref.end_exclusive)
    assert ref.expression == text[ref.span_start:ref.span_end] == phrase
    assert ref.anchor == anchor.isoformat() and ref.clock == "source"


@pytest.mark.parametrize("phrase,days", [("EXACTLY two DAYS ago", 2), ("exactly a week ago", 7)])
def test_exact_modifier_is_preserved_in_both_existing_temporal_views(phrase, days):
    anchor = datetime(2024, 3, 4, tzinfo=UTC)
    old_view, = derive_times(phrase, anchor)
    ref, = resolve_time_references(phrase, source_time=anchor)
    assert old_view.expression == ref.expression == phrase
    assert old_view.span_start == ref.span_start == 0
    assert old_view.span_end == ref.span_end == len(phrase)
    assert old_view.start == ref.start == (anchor.date() - timedelta(days=days)).isoformat()
    assert old_view.precision == ref.precision == "day"


def test_source_and_question_clocks_are_independent_for_exact_offsets():
    text = "exactly two weeks ago"
    source = "2024-03-01T00:30:00+14:00"
    question = "2024-03-20T23:30:00-07:00"
    kwargs = {"source_time": source, "question_time": question}
    a, = resolve_time_references(text, **kwargs)
    b, = resolve_time_references(text, clock="question", **kwargs)
    assert (a.start, a.clock, a.anchor) == ("2024-02-16", "source", source)
    assert (b.start, b.clock, b.anchor) == ("2024-03-06", "question", question)
    assert resolve_time_references(text, question_time=question) == []
    assert resolve_time_references(text, source_time="2024-03-01T09:00:00") == []


@pytest.mark.parametrize("modifier", ["about", "roughly", "approximately"])
def test_approximate_week_offsets_have_no_invented_tolerance(modifier):
    text = f"{modifier} two weeks ago"
    anchor = datetime(2024, 3, 1, tzinfo=UTC)
    ref, = resolve_time_references(text, source_time=anchor)
    assert ref.expression == text
    assert ref.start is None and ref.end_exclusive is None and ref.precision == "approximate"
    assert question_time_window(f"Which delivery arrived {text}?", anchor) is None


@pytest.mark.parametrize("phrase,precision,start,end", [
    ("exactly one month ago", "month", "2024-02-01", "2024-03-01"),
    ("exactly one year ago", "year", "2023-01-01", "2024-01-01"),
])
def test_months_and_years_are_not_reinterpreted_as_fixed_elapsed_days(phrase, precision, start, end):
    anchor = datetime(2024, 3, 31, tzinfo=UTC)
    ref, = resolve_time_references(phrase, source_time=anchor)
    assert (ref.expression, ref.precision, ref.start, ref.end_exclusive) == (phrase, precision, start, end)
    # The source's existing coarse representation is not an exact question filter.
    assert question_time_window(f"What happened {phrase}?", anchor) is None


@pytest.mark.parametrize("phrase", [
    "in the last month", "in the past month", "over the last week", "during the past year",
    "within the last month", "for the last month", "in last month", "in the previous month",
    "over the last 30 days", "in the past two weeks", "in the last 10 days",
])
def test_rolling_period_does_not_lease_its_substring_to_calendar_rules(phrase):
    anchor = datetime(2024, 3, 31, tzinfo=UTC)
    text = f"Several deliveries arrived {phrase}."
    assert derive_times(text, anchor) == []
    assert resolve_time_references(text, source_time=anchor) == []
    assert question_time_window(f"Which deliveries arrived {phrase}?", anchor) is None
    # Raw text stays with the caller. No invented 30/31-day or inclusive-today boundary.


def test_unresolved_range_does_not_erase_an_independent_exact_source_date():
    text = "It was busy in the last month; the parcel arrived on 2024-03-08."
    refs = resolve_time_references(text, source_time="2024-03-31")
    assert [(r.expression, r.start) for r in refs] == [("2024-03-08", "2024-03-08")]


@pytest.mark.parametrize("phrase,start,end,precision", [
    ("last month", "2024-02-01", "2024-03-01", "month"),
    ("in the last calendar month", "2024-02-01", "2024-03-01", "month"),
    ("during the previous calendar month", "2024-02-01", "2024-03-01", "month"),
    ("last week", "2024-03-18", "2024-03-25", "week"),
    ("in the last calendar week", "2024-03-18", "2024-03-25", "week"),
    ("last calendar year", "2023-01-01", "2024-01-01", "year"),
    ("yesterday", "2024-03-30", "2024-03-31", "day"),
    ("last Saturday", "2024-03-30", "2024-03-31", "day"),
])
def test_explicit_calendar_windows_and_day_offsets_remain_available(phrase, start, end, precision):
    anchor = datetime(2024, 3, 31, tzinfo=UTC)
    window = question_time_window(f"Which delivery arrived {phrase}?", anchor)
    assert (window["start"], window["end_exclusive"], window["precision"]) == (start, end, precision)


@pytest.mark.parametrize("phrase", ["two weeks ago", "one month ago", "three years ago"])
def test_coarse_elapsed_mentions_are_not_iso_question_filters(phrase):
    anchor = datetime(2024, 3, 31, tzinfo=UTC)
    assert resolve_time_references(phrase, source_time=anchor)  # Existing source view is retained.
    assert question_time_window(f"Which delivery arrived {phrase}?", anchor) is None


def test_exact_question_day_does_not_borrow_a_source_or_nearby_event_date():
    anchor = datetime(2024, 3, 31, tzinfo=UTC)
    window = question_time_window("Which delivery arrived exactly two weeks ago?", anchor)
    assert (window["start"], window["end_exclusive"]) == ("2024-03-17", "2024-03-18")
    assert not window["start"] <= "2024-03-14" < window["end_exclusive"]
    assert question_time_window("Which delivery arrived 14 days ago?", anchor)["start"] == "2024-03-17"


@pytest.mark.parametrize("question", [
    "What happened since exactly two weeks ago?",
    "What arrived after exactly two days ago?",
    "Was more delivered last week or two weeks ago?",
    "Which deliveries arrived in the last month rather than last week?",
])
def test_open_or_competing_periods_are_not_reduced_to_one_filter(question):
    assert question_time_window(question, datetime(2024, 3, 31, tzinfo=UTC)) is None


def test_quoted_clock_and_underflow_are_not_resolved_using_the_current_clock():
    anchor = datetime(2024, 3, 31, tzinfo=UTC)
    assert resolve_time_references('She wrote "exactly two weeks ago" on an unknown date.', source_time=anchor) == []
    assert derive_times("exactly 1000 weeks ago", datetime(2, 1, 1, tzinfo=UTC)) == []
    ref, = resolve_time_references("exactly 1000 weeks ago", source_time="9997-12-31")
    assert date.fromisoformat(ref.start) < date(9997, 12, 31)

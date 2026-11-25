"""Auditable calendar derivations anchored to the source utterance.

Intervals are half-open. No extraction time or machine clock is used. Vague
durations are deliberately not converted to fabricated event dates. These are
surface calendar rules, not a general temporal reasoning or date parser model.
"""

import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal


@dataclass(frozen=True)
class TemporalDerivation:
    expression: str
    start: str
    end_exclusive: str
    precision: str
    rule: str
    anchor: str
    span_start: int
    span_end: int

    def dump(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TimeReference:
    """A parsed time mention, not a binding between a question and an event.

    Calendar bounds are half-open. Approximate expressions retain their words
    without fabricated tolerance bounds. ``clock`` states which supplied clock
    was used; absolute dates are independent of both clocks. Provenance to the
    containing event remains with the caller.
    """

    expression: str
    start: str | None
    end_exclusive: str | None
    precision: str
    kind: str
    anchor: str | None
    clock: str
    rule: str
    span_start: int
    span_end: int

    def dump(self) -> dict:
        return asdict(self)


def _month(d: date, offset: int) -> date:
    year, month = divmod(d.year * 12 + d.month - 1 + offset, 12)
    return date(year, month + 1, 1)


_MONTH_ALT = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_MONTH_LOOKUP = {name.casefold(): index + 1 for index, name in enumerate(_MONTH_ALT)}
_MONTH_PATTERN = "|".join(_MONTH_ALT)

# "3 years before January 23, 2022" and "about one month before 27 March 2022"
# both carry their own calendar anchor inside the value.  The utterance date is
# irrelevant for these, so they need their own rule rather than the anchor-based
# ``elapsed_units_ago`` one.
_WRITTEN_DATE = re.compile(
    r"\b(?:(?P<iso>\d{4}-\d{2}-\d{2})"
    rf"|(?P<month_a>{_MONTH_PATTERN})\s+(?P<day_a>\d{{1,2}})(?:st|nd|rd|th)?\b"
    r"(?:\s*,?\s*(?P<year_a>\d{4})\b)?(?!\s*,?\s*\d)"
    rf"|(?P<day_b>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<month_b>{_MONTH_PATTERN})\b"
    r"(?:\s*,?\s*(?P<year_b>\d{4})\b)?(?!\s*,?\s*\d)"
    rf"|(?P<month_c>{_MONTH_PATTERN})\s+(?P<year_c>\d{{4}})\b)",
    re.IGNORECASE,
)
_OFFSET_FROM_DATE = re.compile(
    r"\b(?P<amount>\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?P<unit>days?|weeks?|months?|years?)\s+"
    r"(?P<direction>before|after|prior to|earlier|later)\b"
    # The written date is this operator's explicit operand. A greedy sentence
    # tail can borrow an unrelated later date and hide further time mentions.
    r"\s+(?:the\s+)?" + _WRITTEN_DATE.pattern,
    re.IGNORECASE,
)


def _parse_written_date(text: str, default_year: int | None = None) -> date | None:
    """A calendar date written in prose, e.g. ``January 23, 2022``."""
    match = _WRITTEN_DATE.search(text)
    if match is None:
        return None
    parts = match.groupdict()
    try:
        if parts["iso"]:
            return date.fromisoformat(parts["iso"])
        if parts["month_a"]:
            month = _MONTH_LOOKUP[parts["month_a"].casefold()]
            year = int(parts["year_a"]) if parts["year_a"] is not None else default_year
            return date(year, month, int(parts["day_a"])) if year is not None else None
        if parts["month_b"]:
            month = _MONTH_LOOKUP[parts["month_b"].casefold()]
            year = int(parts["year_b"]) if parts["year_b"] is not None else default_year
            return date(year, month, int(parts["day_b"])) if year is not None else None
        if parts["year_c"]:
            month = _MONTH_LOOKUP[parts["month_c"].casefold()]
            return date(int(parts["year_c"]), month, 1)
    except (ValueError, KeyError):
        return None
    return None


def _shift(base: date, unit: str, amount: int, sign: int) -> tuple[date, date, str]:
    """Apply a calendar offset, returning (start, end_exclusive, precision)."""
    unit = unit.lower()
    if unit.startswith("year"):
        start = date(base.year + sign * amount, 1, 1)
        return start, date(start.year + 1, 1, 1), "year"
    if unit.startswith("month"):
        start = _month(base, sign * amount)
        return start, _month(base, sign * amount + 1), "month"
    if unit.startswith("week"):
        start = base + timedelta(weeks=sign * amount)
        return start, start + timedelta(weeks=1), "week"
    start = base + timedelta(days=sign * amount)
    return start, start + timedelta(days=1), "day"


# Named seasons use a northern-hemisphere calendar convention. They are coarse
# intervals, not evidence of an exact event date.
_SEASON_START = {"spring": 3, "summer": 6, "autumn": 9, "winter": 12}


def _season_bounds(season: str, year: int) -> tuple[date, date]:
    start_month = _SEASON_START[season]
    if start_month == 12:
        return date(year, 12, 1), date(year + 1, 3, 1)
    return date(year, start_month, 1), date(year, start_month + 3, 1)


def _season_span(season: str, day: date, direction: str) -> tuple[date, date] | None:
    """The season interval a phrase such as ``last summer`` points at.

    ``this`` is the season that contains the utterance or the next one if the
    utterance falls between seasons; ``last``/``past`` and ``next`` step one
    season from there. Keeping the step relative to the *containing* season is
    what makes "last summer" spoken in February mean the previous June-August
    rather than the upcoming one.
    """
    spans = sorted(_season_bounds(season, year) for year in (day.year - 1, day.year, day.year + 1))
    containing = next(((start, end) for start, end in spans if start <= day < end), None)
    if containing is None:
        containing = next(((start, end) for start, end in spans if start > day), spans[-1])
    index = spans.index(containing)
    if direction in {"last", "past"}:
        return spans[index - 1] if index > 0 else None
    if direction in {"next", "coming"}:
        return spans[index + 1] if index + 1 < len(spans) else None
    return containing


def render_phrase(derivation: "TemporalDerivation") -> str:
    """Calendar phrase for one derivation, at the precision the source used."""
    if derivation.precision == "year":
        return derivation.start[:4]
    if derivation.precision == "month":
        return date.fromisoformat(derivation.start).strftime("%B %Y")
    if derivation.precision == "season":
        season = derivation.rule.rsplit("_", 1)[-1]
        return f"{season} {derivation.start[:4]}"
    if derivation.precision == "day":
        day = date.fromisoformat(derivation.start)
        return f"{day.day} {day.strftime('%B %Y')}"
    return ""


# A period ending around the moment of speaking is not the previous calendar
# bucket. Claim its whole span before the bare "last month/week/year" rules can
# reinterpret a substring. Without an endpoint/inclusivity convention, keep the
# original words rather than invent a rolling-window boundary. Explicit
# "calendar" periods do not match this pattern and retain their calendar rules.
_ROLLING_PERIOD = re.compile(
    r"\b(?:(?:in|within|over|during|for)\s+(?:the\s+)?|the\s+)"
    r"(?:last|past|previous)\s+"
    r"(?:(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
    r"(?:days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)


def derive_times(text: str, anchor: datetime | None) -> list[TemporalDerivation]:
    # Absolute expressions do not need a clock. Relative rules below run only
    # with a supplied aware clock in the supported calendar range.
    day = anchor.date() if anchor is not None and anchor.tzinfo is not None else None
    if day is not None and not 2 <= day.year <= 9997:
        day = None
    found = []
    occupied: list[tuple[int, int]] = [match.span() for match in _ROLLING_PERIOD.finditer(text)]
    quoted = [
        match.span() for match in re.finditer(r'(?<!\w)["“\u2018\x27][^"”\u2019\x27]+["”\u2019\x27]', text)
    ]
    numbers = {
        "a": 1,
        "an": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def add(match, start, end, precision, rule):
        if any(match.start() < b and match.end() > a for a, b in occupied):
            return
        if rule != "explicit_iso_date" and any(a <= match.start() < b for a, b in quoted):
            # A quoted speaker can have a different reference time.
            return
        occupied.append(match.span())
        found.append(
            TemporalDerivation(
                match.group(),
                start.isoformat(),
                end.isoformat(),
                precision,
                rule,
                anchor.isoformat() if anchor is not None else "",
                match.start(),
                match.end(),
            )
        )

    # An offset that carries its own written anchor ("3 years before January
    # 23, 2022", "about one month before 27 March 2022") must be claimed
    # BEFORE the bare ISO-date rule below, otherwise the ISO span is occupied
    # first and the offset silently survives into the rendered answer, where it
    # scores ~0 against a gold answer written as a bare year or month.
    for match in _OFFSET_FROM_DATE.finditer(text):
        raw = match.group("amount").casefold()
        amount = int(raw) if raw.isdigit() else numbers[raw]
        if amount > 1000:
            continue
        direction = match.group("direction").casefold()
        sign = -1 if direction in {"before", "prior to", "earlier"} else 1
        anchor_date = _parse_written_date(match.group())
        if anchor_date is None:
            continue
        try:
            start, end, precision = _shift(anchor_date, match.group("unit"), amount, sign)
        except (ValueError, OverflowError):
            continue
        if precision == "week":
            # ``render_phrase`` has no week form; a week precision would render
            # as an empty string and the offset would survive into the answer.
            # An offset from a named day resolves to that day, so report it as
            # one.  Only the new rule reaches here, so the anchor-based
            # ``last week`` path keeps its existing behaviour.
            precision, end = "day", start + timedelta(days=1)
        add(match, start, end, precision, "offset_from_written_date")
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            value = date(*map(int, match.groups()))
        except ValueError:
            continue
        if value == date.max:
            continue
        add(match, value, value + timedelta(days=1), "day", "explicit_iso_date")
    if day is None:
        return found
    for pattern, offset in ((r"\byesterday\b|昨天", -1), (r"\btoday\b|今天", 0), (r"\btomorrow\b|明天", 1)):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = day + timedelta(days=offset)
            add(match, value, value + timedelta(days=1), "day", "source_relative_day")
    for pattern, offset in (
        (r"\b(?:last(?: calendar)?|previous calendar) year\b|去年", -1),
        (r"\bthis year\b|今年", 0),
        (r"\bnext year\b|明年", 1),
    ):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            add(
                match,
                date(day.year + offset, 1, 1),
                date(day.year + offset + 1, 1, 1),
                "year",
                "source_relative_year",
            )
    for pattern, offset in (
        (r"\b(?:last(?: calendar)?|previous calendar) month\b|上个月|上月", -1),
        (r"\bthis month\b|本月", 0),
        (r"\bnext month\b|下个月|下月", 1),
    ):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            add(match, _month(day, offset), _month(day, offset + 1), "month", "source_relative_month")
    monday = day - timedelta(days=day.weekday())
    for pattern, offset in (
        (r"\b(?:last(?: calendar)?|previous calendar) week\b|上周(?!末)", -1),
        (r"\bthis week\b|本周(?!末)", 0),
        (r"\bnext week\b|下周(?!末)", 1),
    ):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = monday + timedelta(weeks=offset)
            add(match, start, start + timedelta(weeks=1), "week", "iso_monday_week")
    for match in re.finditer(r"\blast weekend\b|上周末", text, re.IGNORECASE):
        start = monday - timedelta(days=2)
        add(match, start, start + timedelta(days=2), "weekend", "previous_iso_weekend")
    weekdays = {
        name: index
        for index, name in enumerate(
            ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        )
    }
    for match in re.finditer(
        r"\b(last|next) (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", text, re.IGNORECASE
    ):
        direction, name = match.groups()
        target = weekdays[name.lower()]
        offset = (
            -((day.weekday() - target) % 7 or 7)
            if direction.lower() == "last"
            else ((target - day.weekday()) % 7 or 7)
        )
        value = day + timedelta(days=offset)
        add(match, value, value + timedelta(days=1), "day", "strict_previous_or_next_weekday")
    for match in re.finditer(
        r"\b(?P<exact>exactly\s+)?(?P<amount>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?P<unit>days?|weeks?|months?|years?)\s+ago\b",
        text,
        re.IGNORECASE,
    ):
        raw, unit = match.group("amount", "unit")
        amount = int(raw) if raw.isdigit() else numbers[raw.lower()]
        if amount > 1000:
            continue
        unit = unit.lower()
        if unit.startswith("year"):
            if day.year <= amount:
                continue
            start, end, precision = date(day.year - amount, 1, 1), date(day.year - amount + 1, 1, 1), "year"
        elif unit.startswith("month"):
            if (day.year - 1) * 12 + day.month - 1 < amount:
                continue
            start, end, precision = _month(day, -amount), _month(day, 1 - amount), "month"
        elif unit.startswith("week"):
            # An explicit exact offset is elapsed days from this source's
            # calendar date, not an ISO-week bucket. Keep the modifier in the
            # expression/span so consumers can audit why precision is a day.
            base = day if match.group("exact") else monday
            if base.toordinal() <= amount * 7:
                continue
            start = base - timedelta(weeks=amount)
            if match.group("exact"):
                end, precision = start + timedelta(days=1), "day"
            else:
                end, precision = start + timedelta(weeks=1), "week"
        else:
            if day.toordinal() <= amount:
                continue
            start = day - timedelta(days=amount)
            end, precision = start + timedelta(days=1), "day"
        add(match, start, end, precision, "elapsed_units_ago")
    for match in re.finditer(
        r"\b(last|this|next|past|coming)\s+(summer|winter|spring|fall|autumn)\b", text, re.IGNORECASE
    ):
        direction, name = match.group(1).lower(), match.group(2).lower()
        season = "autumn" if name == "fall" else name
        span = _season_span(season, day, direction)
        if span is not None:
            add(match, span[0], span[1], "season", f"source_relative_{season}")
    return found


# Rules whose span is fixed by the text itself, not by the moment of asking: an
# ISO date is absolute, and an offset written against a stated date ("3 years
# before January 2020") resolves against that date. Neither belongs in a window
# that is defined relative to `as_of`.
_ANCHORED_RULES = {"explicit_iso_date", "offset_from_written_date"}

# "since three weeks ago" and "between X and Y" use the relative expression to
# anchor the START of a period the question then asks about in full. Restricting
# the answer to that one week would contradict the question, so these are left
# alone: the window is not a filter here.
_PERIOD_ANCHOR = re.compile(r"\b(?:since|between|before|after|prior to|following)\b", re.IGNORECASE)


def question_time_window(question: str, as_of: datetime | None) -> dict | None:
    """Resolve the question's own relative time expression against ``as_of``.

    "Which book did I finish a week ago?" is unanswerable from evidence alone:
    a source that says "I finished it today" is anchored to *its own* utterance
    date, while the question is anchored to the moment it was asked. When the two
    differ by days, a reader that never learns the question date treats the
    utterance date as "now" and rejects the very event that answers it.

    This is the same missing anchor the elapsed-time answer type needed, applied
    to entity questions. It reuses ``derive_times`` rather than adding a second
    time parser, so the question and the evidence are read by one calendar.

    Returns ``None`` unless exactly one relative window is named, the window lies
    in the past, and the question uses it as a filter. A coarse elapsed offset
    ("two weeks ago") does not specify an ISO calendar week; only an explicit
    calendar period or a day-precision offset can supply a hard question window.
    """
    if as_of is None or as_of.tzinfo is None:
        return None
    if _PERIOD_ANCHOR.search(question or "") or _ROLLING_PERIOD.search(question or ""):
        # The relative phrase anchors a period, it does not bound the answer.
        return None
    derivations = resolve_time_references(question, question_time=as_of, clock="question")
    if any(item.rule == "elapsed_units_ago" and item.precision != "day" for item in derivations):
        # Preserve as_of and the complete question for the reader; do not turn
        # a coarse mention into an asserted calendar filter. This also leaves
        # approximate offsets unresolved instead of assigning a tolerance.
        return None
    windows = [item for item in derivations if item.clock == "question"
               and item.rule != "stated_duration_start"
               and item.start is not None and item.end_exclusive is not None]
    distinct = {(item.start, item.end_exclusive) for item in windows}
    if len(distinct) != 1:
        return None
    window = windows[0]
    if window.end_exclusive > as_of.date().isoformat():
        # A window that has not started yet cannot contain a recorded event.
        # "my meal prep next week" asks for advice, not for a past fact, and
        # asserting the window would push the reader to abstain.
        return None
    return {
        "expression": window.expression,
        "start": window.start,
        "end_exclusive": window.end_exclusive,
        "precision": window.precision,
        "as_of": as_of.date().isoformat(),
    }


def relative_time_expressions(text: str) -> list[dict[str, str]]:
    # The full calendar derivation remains in temporal.py; this ledger only
    # exposes explicit relative expressions for the answer model.
    expressions = re.findall(
        r"\b(?:last|next|this)\s+(?:year|month|week(?:end)?|Saturday|Sunday|Monday|Tuesday|Wednesday|Thursday|Friday)\b",
        text,
        re.IGNORECASE,
    )
    return [{"kind": "relative_time", "expression": value} for value in expressions]


_AMOUNT = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty)"
_UNIT = r"(?:days?|weeks?|months?|years?|hours?|minutes?)"
_DURATION_RE = re.compile(
    rf"\b(?:{_AMOUNT}\s+{_UNIT}\s+(?:before|after|ago|since|prior to|later|earlier|into)"
    rf"|(?:for|over|during)\s+(?:the\s+)?(?:past|last|previous)?\s*{_AMOUNT}\s+{_UNIT}"
    rf"|(?:started|began|lasted|took)\s+(?:about\s+)?{_AMOUNT}\s+{_UNIT})\b",
    re.IGNORECASE,
)


def duration_expressions(text: str) -> list[str]:
    """Explicit duration phrases as written by the speaker.

    Durations are quoted, never recomputed: the speaker's own wording is the
    only warrant for "four weeks", and recomputing from anchors would invent
    precision the conversation never stated.
    """
    return [match.group(0).strip() for match in _DURATION_RE.finditer(text)]


_FOR_NOW_RE = re.compile(
    rf"\bfor\s+({_AMOUNT})\s+({_UNIT})\b(?=\s*(?:now|so far|to date|already|[.,;!?\)]|$))",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30,
}


def dedup_dates(out: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for stamp, expression, precision, kind in out:
        key = (stamp, kind)
        if key not in seen:
            seen.add(key)
            unique.append((stamp, expression, precision, kind))
    return sorted(unique)


_MONTH_LAST_DAY = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                   7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

# How wide a stated date is. A date expression is not a point in time: "in
# January" covers a month, "two weeks ago" covers (at week precision) a whole
# week. Comparing such expressions as if they were points is what produced
# "a month ago vs two weeks ago is incomparable" (wrong in one direction) and
# "the start of a two-week span is the event date" (wrong in the other).
_PRECISION_SPAN_DAYS = {"day": 0, "week": 6, "month": None}


def interval_bounds(stamp: str, precision: str):
    """Inclusive [first, last] day this expression can denote, or None."""
    from datetime import date as _date
    from datetime import timedelta as _timedelta

    if precision in {"utterance", "explicit"}:
        # A turn timestamp dates the telling, not the event; not orderable.
        return None
    try:
        first = _date.fromisoformat(stamp)
    except ValueError:
        return None
    span = _PRECISION_SPAN_DAYS.get(precision)
    if precision == "month":
        year, month = first.year, first.month
        last_day = _MONTH_LAST_DAY[month]
        if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            last_day = 29
        # Any day inside the month denotes the whole month: normalise the start.
        return _date(year, month, 1), _date(year, month, last_day)
    if span is None:
        return None
    return first, first + _timedelta(days=span)


def intervals_order(left, right):
    """Order of two spans: -1 (left first), 1 (right first), 0 (same), None (unknown).

    Allen's interval algebra restricted to what the evidence warrants: only
    disjoint spans have a determined order. Overlapping spans are left
    undetermined rather than guessed from a shared boundary.
    """
    if left is None or right is None:
        return None
    left_first, left_last = left
    right_first, right_last = right
    if left_last < right_first:
        return -1
    if right_last < left_first:
        return 1
    if left_first == right_first and left_last == right_last:
        return 0
    return None


# A relative expression ("two weeks ago") is an offset from the moment of
# speaking, not a calendar span. Snapping it to a calendar month -- which
# `derive_times` does for ordering -- throws away exactly the quantity that
# makes it comparable with another relative expression. The offset lives in its
# own space, with the unit's width as uncertainty: a month is 28-31 days, a week
# is 7, a day is 1.
_OFFSET_UNIT_DAYS = {"day": (1, 1), "week": (7, 7), "month": (28, 31), "year": (365, 366)}
_OFFSET_AGO = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    r"|fifteen|twenty|thirty)\s+(day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)
# Articles count as one: "a month ago" is a duration, not an indefinite article.
# Kept separate from _NUMBER_WORDS so that adding it here cannot silently change
# how "for N units now" is parsed elsewhere.
_OFFSET_QUANTITY = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30,
}


def offset_intervals(text: str) -> list[tuple[int, int, str]]:
    """Day-offset interval for every "N units ago" in the text."""
    out = []
    for match in _OFFSET_AGO.finditer(text or ""):
        if _APPROXIMATE_PREFIX.search((text or "")[:match.start()]):
            # "About" provides no numerical tolerance. Unit widths alone do
            # not license a hard ordering decision over an approximate offset.
            continue
        raw = match.group(1).casefold()
        count = int(raw) if raw.isdigit() else _OFFSET_QUANTITY.get(raw, 0)
        if not count or count > 1000:
            continue
        low, high = _OFFSET_UNIT_DAYS[match.group(2).casefold()]
        out.append((low * count, high * count, match.group(0)))
    return out


def difference_allowed(left: str, right: str) -> bool:
    """Quantity compatibility only; it never establishes event-role binding."""
    if left == right == "point":
        return True
    prefix = "interval_start@"
    return (
        left.startswith(prefix)
        and right.startswith(prefix)
        and left[len(prefix):] == right[len(prefix):]
    )


_APPROXIMATE_PREFIX = re.compile(
    r"\b(?:about|approximately|around|roughly|nearly|almost)\s*$", re.IGNORECASE
)


def _supplied_clock(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    try:
        anchor = datetime.fromisoformat(value) if isinstance(value, str) else value
    except ValueError:
        return None
    if not isinstance(anchor, datetime):
        return None
    if not 2 <= anchor.year <= 9997:
        return None
    if anchor.tzinfo is not None:
        return anchor
    # A legacy date-only string is an explicit calendar anchor. A timestamp
    # without a timezone is ambiguous and follows derive_times' rejection.
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return anchor.replace(tzinfo=UTC)
    return None


def resolve_time_references(
    text: str,
    *,
    source_time: datetime | str | None = None,
    question_time: datetime | str | None = None,
    clock: Literal["source", "question"] = "source",
    include_utterance: bool = False,
) -> list[TimeReference]:
    """One calendar view for source evidence, question windows and validation.

    A missing source clock never falls back to question time or a date merely
    mentioned in the text. Absolute dates need no clock. Relative and incomplete
    expressions require the explicitly selected clock. Results identify time
    mentions only: callers must bind their roles before executing an operation.
    """
    if clock not in {"source", "question"}:
        raise ValueError("clock must be source or question")
    anchor = _supplied_clock(source_time if clock == "source" else question_time)
    records: list[TimeReference] = []
    occupied: list[tuple[int, int]] = []

    def append(expression, start, end, precision, kind, rule, a, b, *, absolute=False):
        approximate = _APPROXIMATE_PREFIX.search(text[:a])
        if approximate:
            a = approximate.start()
            expression = text[a:b]
            start = end = None
            precision = "approximate"
        records.append(TimeReference(
            expression, start, end, precision, kind,
            None if absolute or anchor is None else anchor.isoformat(),
            "absolute" if absolute else clock, rule, a, b,
        ))
        occupied.append((a, b))

    for item in derive_times(text, anchor):
        absolute = item.rule in _ANCHORED_RULES
        if anchor is None and not absolute:
            continue
        if item.rule == "explicit_iso_date":
            kind = "point"
        elif absolute:
            kind = "point" if item.precision == "day" else "interval"
        else:
            tag = "start" if item.precision == "day" else item.precision
            kind = f"interval_{tag}@{anchor.date().isoformat()}"
        append(item.expression, item.start, item.end_exclusive, item.precision,
               kind, item.rule, item.span_start, item.span_end, absolute=absolute)

    for match in _WRITTEN_DATE.finditer(text):
        if any(match.start() < b and match.end() > a for a, b in occupied):
            continue
        value = _parse_written_date(match.group(), anchor.year if anchor else None)
        if value is None or value == date.max:
            continue
        precision = "month" if match.group("month_c") else "day"
        try:
            end = _month(value, 1) if precision == "month" else value + timedelta(days=1)
        except (ValueError, OverflowError):
            continue
        absolute = bool(match.group("iso") or match.group("year_a")
                        or match.group("year_b") or match.group("year_c"))
        append(match.group(), value.isoformat(), end.isoformat(), precision,
               "point" if precision == "day" else "interval", "written_calendar_date",
               match.start(), match.end(), absolute=absolute)

    if anchor is not None:
        stamp = anchor.date().isoformat()
        for match in _FOR_NOW_RE.finditer(text):
            raw = match.group(1).casefold()
            amount = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw, 0)
            unit = match.group(2).casefold()
            if not 0 < amount <= 1000 or not unit.startswith(("day", "week")):
                continue
            try:
                start, _end, _precision = _shift(anchor.date(), unit, amount, -1)
            except (ValueError, OverflowError):
                continue
            append(match.group(), start.isoformat(), (start + timedelta(days=1)).isoformat(),
                   "day", f"interval_start@{stamp}", "stated_duration_start",
                   match.start(), match.end())
        # A month name supplies month precision, never an invented day. Avoid
        # reinterpreting the month fragment of an invalid written day/date.
        written_spans = [m.span() for m in _WRITTEN_DATE.finditer(text)]
        for match in re.finditer(
            r"\b(?:in|during)\s+(" + _MONTH_PATTERN + r")\b(?!\s+\d)", text, re.IGNORECASE
        ):
            if any(match.start() < b and match.end() > a for a, b in occupied + written_spans):
                continue
            start = date(anchor.year, _MONTH_LOOKUP[match.group(1).casefold()], 1)
            append(match.group(), start.isoformat(), _month(start, 1).isoformat(), "month",
                   f"interval_month@{stamp}", "source_year_month", match.start(), match.end())
        if include_utterance:
            records.append(TimeReference("utterance date", stamp,
                (anchor.date() + timedelta(days=1)).isoformat(), "day", "utterance",
                anchor.isoformat(), clock, "utterance_clock", 0, 0))
    # Preserve mentions with different roles and offsets, even on the same day.
    return sorted(records, key=lambda item: (item.span_start, item.span_end, item.kind))


def anchored_dates(
    text: str, occurred_at: str | None, fallback_anchor: str | None = None,
) -> list[tuple[str, str, str, str]]:
    """Legacy tuple view of the shared source-time parser.

    ``fallback_anchor`` remains accepted for old callers but is deliberately
    unused: a question timestamp is not a missing source timestamp.
    """
    refs = resolve_time_references(text, source_time=occurred_at, include_utterance=True)
    return dedup_dates([
        (item.start, item.expression, item.precision, item.kind)
        for item in refs if item.start is not None
    ])


def calendar_difference(start: date, end: date, unit: str) -> str:
    """Compute over already bound exact endpoints; never select those endpoints.

    Calendar months/years require matching day components. Coarse intervals,
    approximate dates and event-role binding are the caller's responsibility;
    passing arbitrary dates does not establish that the result answers a query.
    """
    if end < start:
        raise ValueError("Time difference must have ordered endpoints")
    if unit == "years":
        if (start.month, start.day) != (end.month, end.day):
            raise ValueError("Whole-year difference requires equal month and day")
        number = end.year - start.year
    elif unit == "months":
        if start.day != end.day:
            raise ValueError("Whole-month difference requires equal day of month")
        number = (end.year - start.year) * 12 + end.month - start.month
    elif unit in {"days", "weeks"}:
        days = (end - start).days
        if unit == "weeks" and days % 7:
            raise ValueError("Do not round an inexact interval to whole weeks")
        number = days // 7 if unit == "weeks" else days
    else:
        raise ValueError("Unsupported calendar difference unit")
    return f"{number} {unit}"

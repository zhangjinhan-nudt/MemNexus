"""Auditable calendar derivations anchored to the source utterance.

Intervals are half-open. No extraction time or machine clock is used. Vague
durations are deliberately not converted to fabricated event dates. These are
surface calendar rules, not a general temporal reasoning or date parser model.
"""

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta


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


def _month(d: date, offset: int) -> date:
    year, month = divmod(d.year * 12 + d.month - 1 + offset, 12)
    return date(year, month + 1, 1)


_MONTH_ALT = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_MONTH_LOOKUP = {name.casefold(): index + 1 for index, name in enumerate(_MONTH_ALT)}

# "3 years before January 23, 2022" and "about one month before 27 March 2022"
# both carry their own calendar anchor inside the value.  The utterance date is
# irrelevant for these, so they need their own rule rather than the anchor-based
# ``elapsed_units_ago`` one.
_WRITTEN_DATE = re.compile(
    r"\b(?:(?P<iso>\d{4}-\d{2}-\d{2})"
    r"|(?P<month_a>[A-Za-z]+)\s+(?P<day_a>\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(?P<year_a>\d{4})"
    r"|(?P<day_b>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month_b>[A-Za-z]+)\s*,?\s*(?P<year_b>\d{4})"
    r"|(?P<month_c>[A-Za-z]+)\s+(?P<year_c>\d{4}))",
)
_OFFSET_FROM_DATE = re.compile(
    r"\b(?P<amount>\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?P<unit>days?|weeks?|months?|years?)\s+"
    r"(?P<direction>before|after|prior to|earlier|later)\b"
    r"(?P<rest>[^.;|]*)",
    re.IGNORECASE,
)


def _parse_written_date(text: str) -> date | None:
    """A calendar date written in prose, e.g. ``January 23, 2022``."""
    match = _WRITTEN_DATE.search(text)
    if match is None:
        return None
    parts = match.groupdict()
    try:
        if parts["iso"]:
            return date.fromisoformat(parts["iso"])
        if parts["year_a"]:
            month = _MONTH_LOOKUP[parts["month_a"].casefold()]
            return date(int(parts["year_a"]), month, int(parts["day_a"]))
        if parts["year_b"]:
            month = _MONTH_LOOKUP[parts["month_b"].casefold()]
            return date(int(parts["year_b"]), month, int(parts["day_b"]))
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


# Seasons are the coarsest interval a speaker uses without naming a month.
# Northern-hemisphere convention, the one LoCoMo conversations assume.
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


def derive_times(text: str, anchor: datetime | None) -> list[TemporalDerivation]:
    if anchor is None or anchor.tzinfo is None:
        return []
    day = anchor.date()
    if not 2 <= day.year <= 9997:
        return []
    found = []
    occupied: list[tuple[int, int]] = []
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
                anchor.isoformat(),
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
        anchor_date = _parse_written_date(match.group("rest"))
        if anchor_date is None:
            continue
        start, end, precision = _shift(anchor_date, match.group("unit"), amount, sign)
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
    for pattern, offset in ((r"\byesterday\b|昨天", -1), (r"\btoday\b|今天", 0), (r"\btomorrow\b|明天", 1)):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = day + timedelta(days=offset)
            add(match, value, value + timedelta(days=1), "day", "source_relative_day")
    for pattern, offset in (
        (r"\blast year\b|去年", -1),
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
        (r"\blast month\b|上个月|上月", -1),
        (r"\bthis month\b|本月", 0),
        (r"\bnext month\b|下个月|下月", 1),
    ):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            add(match, _month(day, offset), _month(day, offset + 1), "month", "source_relative_month")
    monday = day - timedelta(days=day.weekday())
    for pattern, offset in (
        (r"\blast week\b|上周(?!末)", -1),
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
        r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten) (days?|weeks?|months?|years?) ago\b",
        text,
        re.IGNORECASE,
    ):
        raw, unit = match.groups()
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
            if monday.toordinal() <= amount * 7:
                continue
            start = monday - timedelta(weeks=amount)
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

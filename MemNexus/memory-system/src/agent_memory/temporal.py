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
    return found

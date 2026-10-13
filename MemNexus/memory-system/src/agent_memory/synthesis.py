"""One source-bound answer table for factual and compositional questions.

Quote checks prove provenance, not entailment. Keep that distinction explicit.
The renderer cannot silently discard an accepted item; unresolved slots remain
in the receipt. No benchmark categories, answers, or domain aliases live here.
"""

import re
from datetime import date
from typing import Literal

from pydantic import Field

from .models import Contract, Hit


class Support(Contract):
    source_id: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=700)


class DateDifference(Contract):
    start: date
    end: date
    unit: Literal["days", "weeks", "years"]

    def value(self) -> str:
        if self.end < self.start:
            raise ValueError("Time difference must have ordered endpoints")
        if self.unit == "years":
            if (self.start.month, self.start.day) != (self.end.month, self.end.day):
                raise ValueError("Whole-year difference requires equal month and day")
            number = self.end.year - self.start.year
        else:
            days = (self.end - self.start).days
            if self.unit == "weeks" and days % 7:
                raise ValueError("Do not round an inexact interval to whole weeks")
            number = days // 7 if self.unit == "weeks" else days
        return f"{number} {self.unit}"


class SupportedValue(Contract):
    value: str = Field(min_length=1, max_length=700)
    supports: list[Support] = Field(min_length=1, max_length=8)
    calculation: DateDifference | None = None


class AnswerSlot(Contract):
    target: str = Field(min_length=1, max_length=500)
    role: Literal["answer", "context"] = "answer"
    values: list[SupportedValue] = Field(default_factory=list, max_length=30)
    missing: str = Field(default="", max_length=500)


class AnswerTable(Contract):
    slots: list[AnswerSlot] = Field(min_length=1, max_length=20)


def answer_type(question: str) -> str:
    q = question.casefold()
    if re.search(r"how (?:many|long).*?(?:weeks?|days?|years?|months?|hours?)|相隔|多久", q):
        return "duration"
    if re.search(r"\b(?:when|what year|which year|what date)\b|何时|哪年", q):
        return "date"
    if re.search(r"how many|number of|几个|多少", q):
        return "count"
    return "text"


def validate_table(table: AnswerTable, sources: list[Hit], targets: list[str], kind: str) -> None:
    if [slot.target for slot in table.slots] != targets:
        raise ValueError("Return every requested target exactly once and in order")
    for slot in table.slots:
        if not slot.values and not slot.missing:
            raise ValueError("An empty target requires an explicit missing-evidence reason")
        for item in slot.values:
            dates = set()
            factual_support = False
            for support in item.supports:
                if support.source_id > len(sources):
                    raise ValueError("Unknown source number")
                events = sources[support.source_id - 1].metadata.get("source_events", [])
                matched = [e for e in events if support.quote in e["text"]]
                if not matched:
                    raise ValueError(f"Source {support.source_id} does not contain exact quote: {support.quote!r}")
                factual_support |= any(not e.get("context_only") for e in matched)
                for event in matched:
                    if event.get("context_only"):
                        continue
                    dates.add(str(event.get("occurred_at", ""))[:10])
                    dates.update(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", event["text"]))
                    for derivation in event.get("time_derivations", []):
                        # An interval boundary is not a known exact event date.
                        if derivation["precision"] == "day":
                            dates.add(derivation["start"])
            if not factual_support:
                raise ValueError("A question/context alone cannot establish a fact; cite at least one original factual utterance")
            if item.calculation:
                calc = item.calculation
                if kind == "date" and calc.unit == "years" and slot.role == "answer":
                    # Inverse duration: only the resulting year is warranted,
                    # not an invented exact day/month for the start event.
                    numbers = {word: n for n, word in enumerate(
                        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))}
                    durations = {int(raw) if raw.isdigit() else numbers[raw.casefold()]
                                 for ref in item.supports
                                 for raw in re.findall(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten) years?\b", ref.quote, re.IGNORECASE)}
                    if calc.end.isoformat() not in dates or calc.end.year-calc.start.year not in durations:
                        raise ValueError("Start-year derivation requires an anchored end date and quoted year duration")
                    item.value = str(calc.start.year)
                    # Do not publish manufactured month/day as a proven event date.
                    item.calculation = None
                    continue
                if kind != "duration" or slot.role != "answer":
                    raise ValueError("Date difference only answers a duration query")
                if not {item.calculation.start.isoformat(), item.calculation.end.isoformat()} <= dates:
                    raise ValueError("Calculation endpoints need exact source-bound dates. For approximate year/week precision, set calculation=null and answer the supported relative duration, without invented endpoints.")
                item.value = item.calculation.value()
            if slot.role == "answer" and kind == "duration" and not re.search(r"\b(?:days?|weeks?|months?|years?|hours?|minutes?)\b|天|周|月|年", item.value, re.IGNORECASE):
                raise ValueError("Duration answers must express a duration, not an event name")
            if slot.role == "answer" and kind == "count" and not re.search(r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|none)\b", item.value, re.IGNORECASE):
                raise ValueError("Count answers must contain a number or a supported bound")


def render_table(table: AnswerTable) -> str:
    values = list(dict.fromkeys(item.value.strip() for slot in table.slots if slot.role == "answer" for item in slot.values))
    return "; ".join(values) if values else "Not mentioned in the conversation"

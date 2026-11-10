"""Deterministic temporal rescue: answer a "when" question from evidence dates.

Measured motivation (result/bench150_conv48-50_20261023): MemNexus loses to
SimpleMem only on the temporal category (0.501 vs 0.648, n=25).  Failure
inspection shows the dominant shape is a *wrong abstention*: the question is
date-seeking ("When did Dave start his car maintenance shop?"), the retrieved
evidence literally contains the date, yet the contract synthesiser abstains
("Not mentioned in the conversation.").  On a non-adversarial question an
abstention scores F1 = 0, so any quote-grounded replacement is weakly dominant.

This stage is fully deterministic -- no model call.  It scans the grounded
evidence for date-like spans, scores each span by how strongly the containing
event matches the question's subject words and by span/subject proximity, and
returns the best span as the answer with the containing event text as the
quote.  Because the quote is the verbatim event text, the answer can never
introduce anything the evidence does not literally contain.

Degradation contract: no date, no subject overlap, or any error returns None
and the pipeline behaves exactly as with the flag off.  Never fires when the
question routed through the negative gate (adversarial), so a correct
abstention is never overwritten.
"""

from __future__ import annotations

import re

from .llm_reader import _norm

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_MONTH_DAY_YEAR = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I)
_DAY_MONTH_YEAR = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b", re.I)
_MONTH_YEAR = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", re.I)
_RELATIVE = re.compile(
    r"\b((?:a few|several|some)?\s*(?:days?|weeks?|months?|years?|time)\s+before\s+"
    r"(?:" + "|".join(MONTHS) + r")\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}|\d{4}-\d{2}-\d{2})",
    re.I)
_DATE_SPANS = (_ISO, _MONTH_DAY_YEAR, _DAY_MONTH_YEAR, _MONTH_YEAR, _RELATIVE)

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "was", "were", "are", "be", "been", "do", "does", "did", "what", "when",
    "which", "who", "whom", "how", "where", "why", "her", "his", "their",
    "its", "it", "she", "he", "they", "him", "them", "that", "this", "with",
    "from", "by", "about", "into", "over", "after", "before", "during", "s",
    "first", "ever", "exactly", "specific", "tell", "know", "according",
}


def _subject_words(query: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9']+", (query or "").casefold()))
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _spans(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for pattern in _DATE_SPANS:
        for match in pattern.finditer(text):
            out.append((match.start(), match.end(), match.group(0).strip()))
    # de-overlap: keep the longest span per region
    out.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, str]] = []
    for span in out:
        if any(span[0] < end and start < span[1] for start, end, _ in kept):
            continue
        kept.append(span)
    return kept


def _proximity_bonus(text: str, span: tuple[int, int, str], subjects: set[str]) -> float:
    """Reward a date that sits next to a question-subject mention."""
    if not subjects:
        return 0.0
    center = (span[0] + span[1]) / 2
    best = None
    for match in re.finditer(r"[a-z0-9']+", text.casefold()):
        if match.group(0) in subjects:
            distance = abs(match.start() - center)
            if best is None or distance < best:
                best = distance
    if best is None:
        return 0.0
    if best <= 80:
        return 1.0
    if best <= 200:
        return 0.5
    return 0.0


def rescue_temporal(query: str, sources) -> dict | None:
    """Return {"answer", "quote", "source_ids"} for a date question, else None."""
    subjects = _subject_words(query)
    if not subjects or not sources:
        return None
    best: tuple[float, dict] | None = None
    for index, source in enumerate(sources, 1):
        events = (getattr(source, "metadata", {}) or {}).get("source_events") or []
        for event in events:
            text = str(event.get("text") or "")
            if not text:
                continue
            words = set(re.findall(r"[a-z0-9']+", text.casefold()))
            overlap = len(subjects & words)
            if overlap == 0:
                continue
            occurred = str(event.get("occurred_at", ""))[:10]
            for span in _spans(text):
                # A date already stated in the question cannot be the answer.
                if _norm(span[2]) in _norm(query):
                    continue
                score = overlap + _proximity_bonus(text, span, subjects)
                if occurred and occurred in span[2]:
                    score += 1.0
                if best is None or score > best[0]:
                    best = (score, {
                        "answer": span[2],
                        "quote": text,
                        "source_ids": [index],
                        "verified": True,
                    })
    if best is None or best[0] < 2.0:
        return None
    return best[1]


def is_date_question(query: str) -> bool:
    q = (query or "").casefold()
    return bool(re.search(r"\b(?:when|what year|which year|what date)\b|何时|哪年", q))

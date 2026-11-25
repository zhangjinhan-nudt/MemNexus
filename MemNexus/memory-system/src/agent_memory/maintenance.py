"""Evidence-bound maintenance decisions; projections never become new facts."""

import math
import re
from collections import defaultdict
from datetime import datetime

from .evidence_policy import source_assertion_spans
from .models import Memory


def fact_key(memory: Memory) -> tuple:
    return (memory.scope_type, memory.scope_id, memory.subject, memory.predicate,
            memory.semantics.event_key, memory.semantics.event_identity)


def same_fact(left: Memory, right: Memory) -> bool:
    """Wording may differ, but assertions, scope and validity must agree exactly."""
    return fact_key(left) == fact_key(right) and all(
        getattr(left, name) == getattr(right, name)
        for name in ("value", "kind", "assertion", "valid_from", "valid_to", "semantics")
    )


def consolidation_key(memory: Memory) -> tuple:
    return fact_key(memory) + tuple(
        getattr(memory, name) for name in ("value", "kind", "assertion", "valid_from", "valid_to")
    ) + (memory.semantics.model_dump_json(),)


def _literal_pattern(value: str) -> str:
    # Case/spacing are presentation differences; substrings are not identities.
    return r"(?<!\w)" + r"\s+".join(re.escape(part) for part in value.split()) + r"(?!\w)"


def _english_transition(source: Memory, target: Memory, quote: str) -> str | None:
    """Recognize an explicit directional edit, not recency or general entailment.

    Unsupported paraphrases remain readable as separate active facts. This
    changes canonical lifecycle only when the quoted clause names both values
    in their old/new roles and assigns the change to the same subject.
    """
    subject = _literal_pattern(source.subject)
    old, new = _literal_pattern(target.value), _literal_pattern(source.value)
    field = _literal_pattern(re.sub(r"[_-]+", " ", source.predicate))
    # The subject directly performs the change, or its explicitly possessed
    # field changes. Do not borrow a subject from a preceding reporting clause.
    head = rf"^(?:correction\s*:\s*|actually,\s*)?(?P<subject>{subject})"
    actor = rf"{head}(?:['’]s\s+{field})?\s+"
    verb = r"(?:changed|switched|moved|updated|adjusted|renamed|revised)"
    transition = rf"{actor}(?:(?:has|had|was|is|has\s+been|had\s+been)\s+)?{verb}\b[^;!?]*?\bfrom\s+(?P<old>{old})\s+to\s+(?P<new>{new})"
    correction = rf"{actor}(?:is|was)\s+(?P<correction>not)\s+(?P<old>{old})\s*,?\s*but\s+(?P<new>{new})"
    for left, right in source_assertion_spans(quote):
        clause = quote[left:right].strip()
        for pattern, action in ((transition, "supersede"), (correction, "correct")):
            match = re.search(pattern, clause, re.IGNORECASE)
            if match is None:
                continue
            # Literal names/values such as May or 'not available' are data,
            # not grammatical operators. Mask only the matched argument spans.
            remainder = list(clause)
            for name in match.groupdict():
                start, end = match.span(name)
                remainder[start:end] = " " * (end - start)
            remainder = "".join(remainder)
            if re.search(r"\b(?:if|unless|whether|assuming|supposing|would|could|might|may|should|will)\b", remainder, re.IGNORECASE):
                continue
            # In a correction, the matched 'not OLD but NEW' operator is also
            # masked; any additional negation must not silently invert it.
            if not re.search(r"\b(?:not|never|no|without|cannot)\b|\b\w+n['’]t\b", remainder, re.IGNORECASE):
                return action
    return None


def correction_relation(source: Memory, target: Memory, quotes: list[str]) -> str | None:
    """Only explicit old/new transitions qualify, never question or hypothesis text."""
    if (fact_key(source) != fact_key(target) or source.value == target.value
            or source.assertion not in {"stated", "observed"}):
        return None
    for quote in quotes:
        english = _english_transition(source, target, quote)
        if english:
            return english
        if not all(value in quote for value in (source.subject, source.value, target.value)):
            continue
        if re.search(r"[?？]|如果|假如|假设|可能|是否|是不是", quote):
            continue
        if re.search(r"说错|记错|更正|纠正|不是.+(?:而是|是)", quote):
            return "correct"
        if re.search(r"改为|改成|改名|调整为|变更为|从现在|以后|不再", quote):
            return "supersede"
    return None


def retention(memory: Memory, now: datetime) -> dict:
    """Explainable age/evidence heuristic, not a confidence or truth probability."""
    age_days = max(0.0, (now - memory.recorded_at).total_seconds() / 86400)
    half_life = {"event": 30, "fact": 180, "preference": 120, "procedure": 365}[memory.kind]
    reinforcement = min(3, len({(e.turn_id, e.event_index) for e in memory.evidence}))
    strength = math.exp(-math.log(2) * age_days / (half_life * max(1, reinforcement)))
    expired = memory.valid_to is not None and memory.valid_to <= now
    action = "keep"
    # An ended validity interval excludes current use, not historical recall.
    # Archiving is an explicit lifecycle decision, never a consequence of expiry.
    if memory.status == "active" and strength < 0.25:
        action = "review_archive"
    return {
        "memory_id": memory.memory_id,
        "version": memory.version,
        "status": memory.status,
        "strength": round(strength, 4),
        "age_days": round(age_days, 2),
        "half_life_days": half_life,
        "evidence_reinforcement": reinforcement,
        "expired": expired,
        "recommendation": action,
        "measurement": "age_evidence_heuristic_not_truth_probability",
        "recoverable": memory.status == "archived" or action != "keep",
    }


def summary_groups(memories: list[Memory], now: datetime) -> list[dict]:
    """Bounded extractive summaries with explicit versioned provenance at each leaf."""
    buckets = defaultdict(list)
    for memory in memories:
        if (
            memory.tier != "long"
            or memory.status != "active"
            or (memory.valid_from and memory.valid_from > now)
            or (memory.valid_to and memory.valid_to <= now)
        ):
            continue
        path = memory.hierarchy_path or [memory.kind, memory.subject, memory.predicate]
        for depth in range(1, len(path) + 1):
            buckets[(memory.scope_type, memory.scope_id, tuple(path[:depth]))].append(memory)
    groups = []
    for (scope_type, scope_id, path), items in sorted(buckets.items()):
        selected, used = [], 0
        for memory in sorted(items, key=lambda m: (m.subject, m.predicate, m.memory_id)):
            text = f"{memory.subject} · {memory.predicate}：{memory.value}"
            if len(selected) >= 12 or used + len(text) > 1600:
                continue
            selected.append(
                {
                    "text": text,
                    "memory_ref": f"{memory.memory_id}:{memory.version}",
                    "assertion": memory.assertion,
                    "evidence_count": len(memory.evidence),
                }
            )
            used += len(text)
        groups.append(
            {
                "path": list(path),
                "level": len(path),
                "scope_type": scope_type,
                "scope_id": scope_id,
                "memory_refs": [f"{m.memory_id}:{m.version}" for m in items],
                "mode": "extractive_summary",
                "summary": "\n".join(item["text"] for item in selected),
                "summary_items": selected,
                "omitted_count": len(items) - len(selected),
                "is_derived_view": True,
                "usable_as_extraction_evidence": False,
                "updated_at": now.isoformat(),
            }
        )
    return groups

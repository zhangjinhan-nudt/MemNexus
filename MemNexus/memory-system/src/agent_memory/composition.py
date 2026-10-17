"""Source-bound collection queries. No provider calls, gold answers, or domain dictionaries.

The executor proves coverage of its input, not semantic completeness of natural text.
Shared-entity/embedding edges never establish membership or event identity.
"""

from collections import defaultdict
from datetime import UTC, datetime
import os
from typing import Literal

from pydantic import Field

from .evidence_policy import question_only
from .models import CollectionQuery, Contract, Hit, Memory, Turn


def key(value: str) -> str:
    return " ".join(value.casefold().split())


class AnswerAtom(Contract):
    subject: str
    value: str
    identity: str | None = None
    event_key: str | None = None
    event_identity: str | None = None
    quantity: int | None = Field(default=None, ge=0)
    source_ids: list[int] = Field(min_length=1)
    quotes: list[str] = Field(default_factory=list)
    support_kind: Literal["extracted", "adjudicated"] = "extracted"
    reason: str = "typed extraction with checked source"


class SourceCandidate(Contract):
    subject: str
    value: str = Field(min_length=1, max_length=500)
    normalized_value: str | None = Field(default=None, min_length=1, max_length=500)
    quote: str = Field(min_length=1, max_length=16000)
    identity: str | None = None
    event_key: str | None = None
    event_identity: str | None = None
    quantity: int | None = Field(default=None, ge=0)
    assertion: Literal["stated", "observed", "planned"] = "stated"
    status: Literal["accepted", "rejected", "unresolved"]
    reason: str = Field(min_length=1, max_length=500)


class SourceDecision(Contract):
    source_id: int = Field(ge=1)
    candidates: list[SourceCandidate] = Field(default_factory=list, max_length=80)
    # An empty source still needs an explicit disposition; never implies exhaustive extraction.
    reason: str = Field(min_length=1, max_length=500)


class CollectionExtraction(Contract):
    sources: list[SourceDecision] = Field(max_length=120)


class AnswerItem(Contract):
    item_id: str
    subjects: list[str]
    value: str
    rule: Literal[
        "member", "union", "intersection", "count", "count_lower_bound", "explicit_total", "explicit_empty"
    ] = "member"
    # Intersection/count need all premises; union can have alternative membership proofs.
    premises: list[AnswerAtom]


class Coverage(Contract):
    source_spans_examined: int = 0
    source_processing: str = "checked_quotes_only"
    extraction_completeness: str = "not_proven"
    indexed_rows: int = 0
    enumeration_complete: bool = False
    operations_complete: bool = False
    budget_exhausted: bool = False
    unresolved: list[str] = Field(default_factory=list)
    output_item_ids: list[str] = Field(default_factory=list)
    output_complete: bool = False


class AnswerIR(Contract):
    query: CollectionQuery
    snapshot: list[int] = Field(default_factory=list)
    atoms: list[AnswerAtom] = Field(default_factory=list)
    items: list[AnswerItem] = Field(default_factory=list)
    excluded: list[dict] = Field(default_factory=list)
    normalizations: list[dict] = Field(default_factory=list)
    source_reviews: list[dict] = Field(default_factory=list)
    explicit_empty: dict[str, AnswerAtom] = Field(default_factory=dict)
    coverage: Coverage = Field(default_factory=Coverage)
    mode: Literal["indexed", "source_adjudication"] = "indexed"


def execute(ir: AnswerIR) -> AnswerIR:
    """Evaluate the bounded algebra, preserving all premises for joins/counts."""
    query = ir.query
    subjects = {key(s): s for s in query.subjects}
    excluded = {key(v) for v in query.exclude_values}
    groups: dict[str, dict[str, AnswerAtom]] = {s: {} for s in subjects}
    for atom in ir.atoms:
        subject = key(atom.subject)
        if subject not in groups or key(atom.value) in excluded:
            ir.excluded.append({"subject": atom.subject, "value": atom.value, "reason": "query_filter"})
            continue
        if query.event_key is not None and atom.event_key != query.event_key:
            ir.coverage.unresolved.append("event_qualification_unresolved")
            continue
        value_key = (
            atom.identity
            if query.operation == "count" and query.count_unit == "entity" and atom.identity is not None
            else key(atom.value)
        )
        if query.operation == "count" and query.count_unit == "event" and atom.event_identity is not None:
            value_key = atom.event_identity
        if query.operation == "count" and atom.quantity is not None:
            value_key = f"explicit_total:{atom.quantity}"
        old = groups[subject].get(value_key)
        if old:
            old.source_ids = sorted(set(old.source_ids + atom.source_ids))
            old.quotes = list(dict.fromkeys(old.quotes + atom.quotes))
            if old.identity != atom.identity:
                old.identity = None
        else:
            groups[subject][value_key] = atom.model_copy(deep=True)

    ir.items = []
    if query.operation == "count":
        premises = [a for group in groups.values() for a in group.values()]
        totals = [a for a in premises if a.quantity is not None]
        if len({a.quantity for a in totals}) > 1:
            ir.coverage.unresolved.append("conflicting_explicit_totals")
        if query.count_unit == "event":
            identities = [a.event_identity for a in premises]
        elif query.count_unit == "entity":
            identities = [a.identity for a in premises]
        else:
            identities = [key(a.value) for a in premises]
        if len(query.subjects) == 1 and totals and len({a.quantity for a in totals}) == 1:
            ir.items.append(
                AnswerItem(
                    item_id="explicit_count",
                    subjects=query.subjects,
                    value=str(totals[0].quantity),
                    premises=totals,
                    rule="explicit_total",
                )
            )
        elif (
            not ir.coverage.enumeration_complete
            or ir.coverage.unresolved
            or any(x is None for x in identities)
        ):
            ir.coverage.unresolved.append("exact_count_not_established")
            if premises and all(x is not None for x in identities) and not totals:
                ir.items.append(
                    AnswerItem(
                        item_id="count_lower_bound",
                        subjects=query.subjects,
                        value=str(len(set(identities))),
                        premises=premises,
                        rule="count_lower_bound",
                    )
                )
        elif premises:
            ir.items.append(
                AnswerItem(
                    item_id="count",
                    subjects=query.subjects,
                    value=str(len(set(identities))),
                    premises=premises,
                    rule="count",
                )
            )
        else:
            ir.coverage.unresolved.append("absence_not_established")
    elif query.operation == "intersection":
        common = set.intersection(*(set(group) for group in groups.values()))
        for value_key in sorted(common):
            premises = [group[value_key] for group in groups.values()]
            ir.items.append(
                AnswerItem(
                    item_id=f"intersection:{value_key}",
                    subjects=query.subjects,
                    value=premises[0].value,
                    premises=premises,
                    rule="intersection",
                )
            )
        if not common:
            ir.coverage.unresolved.append("empty_intersection_is_not_proof_of_absence")
    else:
        for subject, group in groups.items():
            for value_key, atom in sorted(group.items()):
                ir.items.append(
                    AnswerItem(
                        item_id=f"{subject}:{value_key}",
                        subjects=[subjects[subject]],
                        value=atom.value,
                        premises=[atom],
                    )
                )
        # Union can use any one of several subject proofs, while keeping provenance.
        if query.operation == "union":
            merged = {}
            for item in ir.items:
                value_key = key(item.value)
                if value_key not in merged:
                    merged[value_key] = item
                else:
                    merged[value_key].subjects.extend(item.subjects)
                    merged[value_key].premises.extend(item.premises)
                    merged[value_key].rule = "union"
            ir.items = list(merged.values())
    for subject, group in groups.items():
        if not group and subject not in ir.explicit_empty:
            ir.coverage.unresolved.append(f"no_supported_members:{subjects[subject]}")
    if ir.explicit_empty and (query.operation == "intersection" or set(ir.explicit_empty) == set(subjects)):
        ir.items = [
            AnswerItem(
                item_id="explicit_empty",
                subjects=query.subjects,
                value="0" if query.operation == "count" else "none",
                rule="explicit_empty",
                premises=list(ir.explicit_empty.values()),
            )
        ]
        ir.coverage.unresolved = [
            r
            for r in ir.coverage.unresolved
            if r
            not in {
                "absence_not_established",
                "empty_intersection_is_not_proof_of_absence",
                "exact_count_not_established",
            }
        ]
    elif query.operation == "group":
        for subject, proof in ir.explicit_empty.items():
            ir.items.append(
                AnswerItem(
                    item_id=f"empty:{subject}",
                    subjects=[subjects[subject]],
                    value="none",
                    rule="explicit_empty",
                    premises=[proof],
                )
            )
    ir.coverage.unresolved = list(dict.fromkeys(ir.coverage.unresolved))
    ir.coverage.operations_complete = True
    return ir


def indexed_answer(
    query: CollectionQuery,
    memories: list[Memory],
    turns: dict[str, Turn],
    *,
    snapshot: tuple[int, int],
    pending: int,
    limit: int = 120,
    as_of: datetime | None = None,
) -> tuple[AnswerIR, list[Hit]]:
    """Fold typed membership operations from an authorized relation partition.

    Operations stay in canonical storage, so retracting a removal or a mistaken
    replacement restores the underlying supported members on the next query.
    """
    coverage = Coverage(
        indexed_rows=len(memories),
        enumeration_complete=len(memories) <= limit,
        budget_exhausted=len(memories) > limit,
    )
    if pending:
        coverage.unresolved.append(f"unprocessed_turns:{pending}")
    if coverage.budget_exhausted:
        coverage.unresolved.append("index_budget_exhausted")
    ir = AnswerIR(query=query, snapshot=list(snapshot), coverage=coverage)
    sources = []
    states: dict[tuple, dict[str, AnswerAtom]] = defaultdict(dict)
    closed: dict[tuple, AnswerAtom] = {}

    def observed_at(memory):
        dates = [
            turns[e.turn_id].events[e.event_index].occurred_at
            for e in memory.evidence
            if e.turn_id in turns and e.event_index < len(turns[e.turn_id].events)
        ]
        return memory.valid_from or max(dates, default=datetime.min.replace(tzinfo=UTC))

    for memory in sorted(memories, key=lambda m: (observed_at(m), m.recorded_at, m.memory_id))[:limit]:
        semantics = memory.semantics
        if as_of is not None and observed_at(memory) > as_of:
            continue
        if query.event_key is not None and semantics.event_key != query.event_key:
            if semantics.event_key is None:
                coverage.unresolved.append(f"event_identity_unknown:{memory.memory_id}")
            continue
        if memory.assertion not in query.assertions:
            ir.excluded.append({"memory_id": memory.memory_id, "reason": "assertion_type"})
            continue
        records = []
        references = memory.evidence + [e for member in semantics.members for e in member.evidence]
        seen = set()
        for ref in references:
            turn = turns.get(ref.turn_id)
            if turn is None or ref.event_index >= len(turn.events):
                coverage.unresolved.append(f"missing_source:{memory.memory_id}")
                continue
            event = turn.events[ref.event_index]
            allow_assistant = os.getenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "false").lower() in {"1", "true", "yes"}
            if ref.quote not in event.content or (event.role == "assistant" and not allow_assistant) or question_only(event.content):
                coverage.unresolved.append(f"invalid_source:{memory.memory_id}")
                continue
            marker = (ref.turn_id, ref.event_index, ref.quote)
            if marker in seen:
                continue
            seen.add(marker)
            records.append(
                {
                    "turn_id": ref.turn_id,
                    "event_index": ref.event_index,
                    "text": ref.quote,
                    "speaker": event.speaker,
                    "role": event.role,
                    "occurred_at": event.occurred_at.isoformat(),
                    "context_only": False,
                    "time_derivations": [],
                }
            )
        if len(seen) != len({(e.turn_id, e.event_index, e.quote) for e in references}):
            continue
        source_id = len(sources) + 1
        sources.append(
            Hit(
                content=memory.content,
                score=1,
                source_file=f"[memory:{memory.memory_id}]",
                chunk_id=f"{memory.memory_id}:{memory.version}",
                metadata={
                    "memory_id": memory.memory_id,
                    "version": memory.version,
                    "revision": memory.revision,
                    "subject": memory.subject,
                    "predicate": memory.predicate,
                    "value": memory.value,
                    "evidence": [e.model_dump() for e in references],
                    "source_events": records,
                    "source_support_status": "resolved",
                    "origin": "relation_index",
                    "status": memory.status,
                    "assertion": memory.assertion,
                },
            )
        )
        coverage.source_spans_examined += len(records)
        if semantics.cardinality == "unknown":
            coverage.unresolved.append(f"legacy_semantics:{memory.memory_id}")
            continue
        partition = (
            memory.scope_type,
            memory.scope_id,
            key(memory.subject),
            semantics.event_key,
            semantics.event_identity,
        )
        state = states[partition]
        if semantics.cardinality == "one" and state and key(memory.value) not in state:
            coverage.unresolved.append(f"unresolved_scalar_conflict:{memory.subject}:{memory.predicate}")
            continue
        if semantics.operation == "replace_set":
            state.clear()
            closed[partition] = AnswerAtom(
                subject=memory.subject,
                value="explicit set snapshot",
                source_ids=[source_id],
                quotes=[e.quote for e in memory.evidence],
            )
        elif partition in closed:
            closed[partition].source_ids.append(source_id)
            closed[partition].quotes.extend(e.quote for e in memory.evidence)
        members = [(m.value, m.identity, [e.quote for e in m.evidence]) for m in semantics.members]
        if semantics.cardinality == "one":
            members = [(memory.value, None, [e.quote for e in memory.evidence])]
        for value, identity, quotes in members:
            member_key = identity or key(value)
            if semantics.operation == "remove":
                matching = [k for k, a in state.items() if key(a.value) == key(value)]
                if identity is None and len(matching) > 1:
                    coverage.unresolved.append(f"ambiguous_removal_identity:{memory.memory_id}")
                    continue
                if identity is None and matching:
                    member_key = matching[0]
                state.pop(member_key, None)
                ir.excluded.append(
                    {
                        "subject": memory.subject,
                        "value": value,
                        "reason": "explicit_member_removal",
                        "source_ids": [source_id],
                    }
                )
                continue
            atom = AnswerAtom(
                subject=memory.subject,
                value=value,
                identity=identity,
                event_key=semantics.event_key,
                event_identity=semantics.event_identity,
                source_ids=[source_id],
                quotes=quotes,
            )
            if member_key in state:
                atom.source_ids = sorted(set(state[member_key].source_ids + atom.source_ids))
                atom.quotes = list(dict.fromkeys(state[member_key].quotes + atom.quotes))
            state[member_key] = atom
    ir.atoms = [atom for state in states.values() for atom in state.values()]
    for subject in query.subjects:
        partitions = [p for p in states if p[2] == key(subject)]
        if partitions and all(not states[p] and p in closed for p in partitions):
            proofs = [closed[p] for p in partitions]
            ir.explicit_empty[key(subject)] = AnswerAtom(
                subject=subject,
                value="explicit empty set",
                source_ids=sorted({sid for proof in proofs for sid in proof.source_ids}),
                quotes=[quote for proof in proofs for quote in proof.quotes],
            )
    return execute(ir), sources


def validate_extraction(result: CollectionExtraction, query: CollectionQuery, sources: list[Hit]) -> None:
    ids = [decision.source_id for decision in result.sources]
    if sorted(ids) != list(range(1, len(sources) + 1)):
        raise ValueError("Each supplied source must be examined exactly once")
    for decision in result.sources:
        source = sources[decision.source_id - 1]
        texts = [
            event["text"]
            for event in source.metadata.get("source_events", [])
            if not event.get("context_only")
        ]
        for candidate in decision.candidates:
            if not any(candidate.quote in text for text in texts):
                raise ValueError("Every candidate must quote original, non-context-only source text")
            if candidate.status == "accepted" and key(candidate.subject) not in {key(s) for s in query.subjects}:
                raise ValueError("Candidate subject must belong to the query")
            if candidate.status == "accepted" and candidate.quantity is not None and query.operation != "count":
                raise ValueError("Explicit quantities are only allowed for a count query")
            if candidate.status == "accepted" and candidate.assertion not in query.assertions:
                raise ValueError("Candidate modality is not allowed by the query")


def adjudicated_answer(
    query: CollectionQuery,
    result: CollectionExtraction,
    sources: list[Hit],
    *,
    snapshot: tuple[int, int],
    pending: int,
) -> AnswerIR:
    validate_extraction(result, query, sources)
    ir = AnswerIR(query=query, snapshot=list(snapshot), mode="source_adjudication")
    ir.coverage.source_processing = "selected_sources_examined_by_model"
    ir.coverage.source_spans_examined = len(
        {
            (event.get("turn_id"), event.get("event_index"), event["text"])
            for source in sources
            for event in source.metadata.get("source_events", [])
            if not event.get("context_only")
        }
    )
    # Examining every selected source is not exhaustive retrieval of the conversation.
    if pending:
        ir.coverage.unresolved.append(f"unprocessed_turns:{pending}")
    for decision in result.sources:
        ir.source_reviews.append({"source_id": decision.source_id, "reason": decision.reason})
        for candidate in decision.candidates:
            if candidate.status == "accepted":
                ir.atoms.append(
                    AnswerAtom(
                        subject=candidate.subject,
                        value=candidate.normalized_value or candidate.value,
                        identity=candidate.identity,
                        event_key=candidate.event_key,
                        event_identity=candidate.event_identity,
                        quantity=candidate.quantity,
                        source_ids=[decision.source_id],
                        quotes=[candidate.quote],
                        support_kind="adjudicated",
                        reason=candidate.reason,
                    )
                )
                if candidate.normalized_value and candidate.normalized_value != candidate.value:
                    # Preserve the semantic mapping as a model judgment, not a proven equivalence.
                    ir.normalizations.append(
                        {
                            "source_id": decision.source_id,
                            "reason": candidate.reason,
                            "original_value": candidate.value,
                            "normalized_value": candidate.normalized_value,
                        }
                    )
            else:
                ir.excluded.append({**candidate.model_dump(), "source_id": decision.source_id})
                if candidate.status == "unresolved":
                    ir.coverage.unresolved.append(candidate.reason)
    return execute(ir)


def render(ir: AnswerIR, language: str = "en") -> str:
    """Render members directly: no free-form writer can silently drop/add a member."""
    if not ir.items:
        text = (
            "当前证据不足以确定该结果，不能据此认定对话中不存在。"
            if language == "zh"
            else "The examined evidence does not establish this result; this does not prove absence."
        )
    else:
        groups = defaultdict(list)
        for item in ir.items:
            source_ids = sorted({sid for premise in item.premises for sid in premise.source_ids})
            label = item.value + "".join(f"【来源{sid}】" for sid in source_ids)
            if item.rule == "count_lower_bound":
                label = ("至少 " if language == "zh" else "at least ") + label
            group = ", ".join(item.subjects) if ir.query.operation == "group" else ""
            groups[group].append(label)
        text = "\n".join(
            (f"{subject}: " if subject else "") + "; ".join(values) + "."
            for subject, values in groups.items()
        )
        if ir.coverage.unresolved or not ir.coverage.enumeration_complete:
            text += (
                "\n以上为已支持的结果，尚不能保证穷尽。"
                if language == "zh"
                else "\nThese are the supported results; completeness is not established."
            )
    ir.coverage.output_item_ids = [item.item_id for item in ir.items]
    ir.coverage.output_complete = True
    return text

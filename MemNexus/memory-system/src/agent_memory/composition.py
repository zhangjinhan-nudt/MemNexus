"""Checked original sources from an authorized collection relation partition."""

import os
from datetime import UTC, datetime

from .evidence_policy import assertion_supported, listed_label_supported
from .models import CollectionQuery, Hit, Memory, Turn


def collection_sources(
    query: CollectionQuery,
    memories: list[Memory],
    turns: dict[str, Turn],
    *,
    limit: int = 120,
) -> list[Hit]:
    """Resolve source quotes; callers enforce owner, session and lifecycle scope.

    Membership operations remain in canonical storage. Their original evidence
    goes to the same synthesis path as other facts; this helper makes no claim
    that the selected records enumerate a complete set or establish a count.
    ``limit`` bounds returned, source-checked records. Rejected candidates do
    not spend that output allowance.
    """
    if limit < 0:
        raise ValueError("Collection source limit must be non-negative")
    if limit == 0:
        return []
    sources = []

    def source_order(memory):
        # Preserve existing fact/source ordering; this is not a visibility clock.
        dates = [
            turns[e.turn_id].events[e.event_index].occurred_at
            for e in memory.evidence
            if e.turn_id in turns and e.event_index < len(turns[e.turn_id].events)
        ]
        return memory.valid_from or max(dates, default=datetime.min.replace(tzinfo=UTC))

    for memory in sorted(memories, key=lambda m: (source_order(m), m.recorded_at, m.memory_id)):
        semantics = memory.semantics
        if query.event_key is not None and semantics.event_key != query.event_key:
            continue
        if memory.assertion not in query.assertions:
            continue
        records = []
        references = memory.evidence + [e for member in semantics.members for e in member.evidence]
        seen = set()
        for ref in references:
            turn = turns.get(ref.turn_id)
            if turn is None or ref.event_index >= len(turn.events):
                continue
            event = turn.events[ref.event_index]
            allow_assistant = os.getenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "true").lower() in {"1", "true", "yes"}
            if ref.quote not in event.content or (event.role == "assistant" and not allow_assistant):
                continue
            label_values = [memory.value] + [member.value for member in semantics.members if ref in member.evidence]
            if not assertion_supported(event.content, ref.quote, start=ref.start, end=ref.end) and not any(
                listed_label_supported(event.content, ref.quote, memory.predicate, value,
                                       start=ref.start, end=ref.end) for value in label_values
            ):
                continue
            start = ref.start if ref.start is not None else event.content.index(ref.quote)
            end = ref.end if ref.end is not None else start + len(ref.quote)
            if event.content[start:end] != ref.quote:
                continue
            marker = (ref.turn_id, ref.event_index, ref.quote, ref.start, ref.end)
            if marker in seen:
                continue
            seen.add(marker)
            records.append(
                {
                    "turn_id": ref.turn_id,
                    "event_index": ref.event_index,
                    "text": ref.quote,
                    "start": start,
                    "end": end,
                    "speaker": event.speaker,
                    "role": event.role,
                    "occurred_at": event.occurred_at.isoformat(),
                    "context_only": event.role == "assistant",
                    "evidence_grade": "secondary" if event.role == "assistant" else "primary",
                    "time_derivations": [],
                }
            )
        if len(seen) != len({(e.turn_id, e.event_index, e.quote, e.start, e.end) for e in references}):
            continue
        sources.append(
            Hit(
                content=memory.content,
                score=1,
                source_file=f"[memory:{memory.memory_id}]",
                chunk_id=f"{memory.memory_id}:{memory.version}",
                metadata={
                    **memory.source_metadata(),
                    "evidence": [e.model_dump(mode="json") for e in references],
                    "source_events": records,
                    "source_support_status": "resolved",
                    "origin": "relation_index",
                },
            )
        )
        if len(sources) >= limit:
            break
    return sources

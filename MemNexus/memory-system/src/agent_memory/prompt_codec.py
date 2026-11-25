"""Lossless prompt serialization. Local validators remain the source of truth.

No learned compressor, retrieval truncation, benchmark answers or identity
inference. Repeated evidence is sent once and addressed by source membership.
"""

import hashlib
import json
import re
from copy import deepcopy


def compact_json(value):
    """The one JSON representation used for provider data and its character budget."""
    def encode(item):
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if hasattr(item, "isoformat"):
            return item.isoformat()
        raise TypeError(f"Unsupported prompt data type: {type(item).__name__}")

    return json.dumps(value, ensure_ascii=False, default=encode, separators=(",", ":"))


# Each entry cost includes a comma; this remainder makes a nonempty catalog's
# summed costs exactly equal len(compact_json({"sources": entries})).
SOURCE_CATALOG_OVERHEAD = len(compact_json({"sources": []})) - 1


def compact_schema(schema):
    """Remove documentation, preserving validation keywords and field defaults."""
    value = deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            # Only schema nodes have these annotations. A property named title
            # in a properties map must survive.
            node.pop("title", None)
            node.pop("description", None)
            for key, child in node.items():
                if key in {"properties", "$defs", "patternProperties"}:
                    for nested in child.values():
                        visit(nested)
                elif key not in {"default", "enum", "const", "examples"}:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(value)
    return value


def evidence_entry(hit, source_id, *, include_time=True):
    from .grounding import source_payload

    payload = source_payload(hit, include_time=include_time)
    entry = {"source_id": source_id, "source_events": payload["source_events"]}
    if payload["attributes"]:
        entry["attributes"] = payload["attributes"]
    # Claims remain hints. Omit only exact duplication of a source text.
    if hit.content and hit.content not in {e["text"] for e in payload["source_events"]}:
        entry["claim_hint"] = hit.content
    return entry


def evidence_catalog(sources, source_ids=None, *, include_time=True):
    """Represent sources with a *single* directly addressable identifier.

    The former ``sources -> event_ids -> events`` indirection saved very little
    prompt space and caused models to confuse an event number with a source
    number.  Each source now carries its own lossless events.  This preserves
    speaker/time boundaries while making an evidence citation unambiguous:
    ``source_id`` plus an exact quote from that source.
    """
    allowed = None if source_ids is None else set(source_ids)
    entries = [evidence_entry(hit, source_id, include_time=include_time)
               for source_id, hit in enumerate(sources, 1)
               if allowed is None or source_id in allowed]
    return {"sources": entries}


def question_requires_time(question, as_of=None):
    """Existing literal time-request signal, shared by packing and serialization."""
    from .synthesis import answer_type
    from .temporal import question_time_window
    return (answer_type(question) in {"date", "duration", "offset"}
            or question_time_window(question, as_of) is not None
            or bool(re.search(r"\b(?:first|earliest|latest|before|after|previous|current|last|ago|when)\b",
                              question, re.IGNORECASE)))


def synthesis_request_context(question, plan, as_of=None):
    """The reader and its packer use the same temporal-payload decision."""
    from .models import normalize_request_contract
    from .synthesis import answer_type
    from .temporal import question_time_window

    kind = answer_type(question)
    normalized_plan, _ = normalize_request_contract(question, plan)
    request_contract = normalized_plan.answer_contract
    window = question_time_window(question, as_of)
    needs_time = question_requires_time(question, as_of)
    needs_time = needs_time or bool(request_contract and any(
        output.operator in {"time_point", "duration", "comparison"} for output in request_contract.outputs))
    return kind, request_contract, window, needs_time


def synthesis_request_payload(contract):
    """Serialize the existing reader contract without inapplicable hints.

    Stored contracts retain the optional field. An empty member-position list
    adds no request constraint, while a nonempty binding must reach the reader.
    """
    outputs = []
    for output in contract.outputs:
        record = output.model_dump(mode="json", exclude={"required_inputs", "relation_refs", "retrieval_relations"},
                                   exclude_none=True)
        if not record.get("member_positions"):
            record.pop("member_positions", None)
        outputs.append(record)
    return {"outputs": outputs}


def synthesis_source_entry(hit, source_id, *, needs_time):
    """Serialize exactly one source as the synthesis provider sends it."""
    from .temporal import resolve_time_references

    entry = evidence_entry(hit, source_id, include_time=False)
    if needs_time:
        for event in entry["source_events"]:
            refs = resolve_time_references(event["text"], source_time=event.get("occurred_at"))
            if refs:
                event["time_references"] = [ref.dump() for ref in refs]
    return entry


def synthesis_source_cost(hit, source_id, *, needs_time):
    """Exact source-entry character cost plus its catalog separator, not tokens."""
    return len(compact_json(synthesis_source_entry(hit, source_id, needs_time=needs_time))) + 1


def synthesis_source_catalog(sources, *, needs_time):
    """Final ordered citation namespace shared by generation and its receipt."""
    return {"sources": [synthesis_source_entry(hit, source_id, needs_time=needs_time)
                        for source_id, hit in enumerate(sources, 1)]}


def synthesis_catalog_receipt(sources, *, needs_time):
    encoded = compact_json(synthesis_source_catalog(sources, needs_time=needs_time))
    return {"sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "serialized_chars": len(encoded), "source_count": len(sources),
            "source_ids": list(range(1, len(sources) + 1)),
            "measurement": "exact_serialized_characters_not_provider_tokens"}

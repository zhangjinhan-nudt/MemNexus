"""Lossless prompt serialization. Local validators remain the source of truth.

No learned compressor, retrieval truncation, benchmark answers or identity
inference. Repeated evidence is sent once and addressed by source membership.
"""

import json
from copy import deepcopy


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


def evidence_catalog(sources, source_ids=None, *, include_time=True):
    """Represent sources with a *single* directly addressable identifier.

    The former ``sources -> event_ids -> events`` indirection saved very little
    prompt space and caused models to confuse an event number with a source
    number.  Each source now carries its own lossless events.  This preserves
    speaker/time boundaries while making an evidence citation unambiguous:
    ``source_id`` plus an exact quote from that source.
    """
    from .grounding import source_payload

    entries = []
    allowed = None if source_ids is None else set(source_ids)
    for source_id, hit in enumerate(sources, 1):
        if allowed is not None and source_id not in allowed:
            continue
        payload = source_payload(hit, include_time=include_time)
        entry = {"source_id": source_id, "source_events": payload["source_events"]}
        if payload["attributes"]:
            entry["attributes"] = payload["attributes"]
        # Claims remain hints. Omit only exact duplication of a source text.
        if hit.content and hit.content not in {e["text"] for e in payload["source_events"]}:
            entry["claim_hint"] = hit.content
        entries.append(entry)
    return {"sources": entries}

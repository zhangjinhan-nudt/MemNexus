"""Provenance joins and one bounded raw-evidence retrieval round.

Compressed memories are retrieval hints. Source utterances carry dates and
speakers; calendar deductions retain their source and precision. Candidates retain their retrieval ranks and exact source identities. One
serialized budget selects complete reading units; ranking never treats global
word coverage as semantic coverage. Scores are not truth probabilities.
"""

import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta, timezone
from itertools import pairwise

from .evidence_policy import (
    assertion_supported,
    question_only,
    source_assertion_scopes,
    source_assertion_spans,
    source_sentence_spans,
)
from .models import GroundingTrace, Hit, QueryPlan, Turn
from .reasoning import _retrieval_terms as normalized_terms
from .temporal import derive_times

STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "do",
    "does",
    "did",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "and",
    "or",
    "for",
    "with",
    "my",
    "me",
    "i",
    "it",
    "what",
    "when",
    "where",
    "who",
    "how",
    "which",
    "why",
    "would",
    "could",
    "should",
    "can",
    "have",
    "has",
    "had",
    "please",
    "tell",
    "about",
}


def _cjk_terms(text: str) -> set[str]:
    return {part[i : i + 2] for part in re.findall(r"[\u4e00-\u9fff]+", text)
            for i in range(max(1, len(part) - 1))}


def terms(text: str) -> set[str]:
    return (set(re.findall(r"[a-z0-9]+", text.casefold())) - STOP) | _cjk_terms(text)


def speaker_of(event) -> str | None:
    if event.speaker:
        return event.speaker
    # Compatibility with imported transcripts; record this as a text prefix,
    # not a model-inferred entity identity.
    match = re.match(r"^([A-Z][\w -]{0,60}):\s", event.content)
    return match.group(1) if match else None


def _extraction_hint(hit: Hit, event: dict) -> dict:
    """Project an extraction onto its visible original span, never a truth label.

    Offsets are local to this source event's unchanged text. They do not add a
    second citation namespace. Third-party assertions are allowed: the memory's
    subject is not required to equal the speaker. A known attribution conflict
    suppresses the extracted hint, while keeping the original evidence visible.
    """
    metadata = hit.metadata
    if metadata.get("attribution_conflict") or not metadata.get("subject") or not metadata.get("predicate"):
        return {}

    def spans(references):
        found = set()
        for reference in references:
            if (reference.get("turn_id"), reference.get("event_index")) != (
                event.get("turn_id"), event.get("event_index")
            ):
                continue
            quote = reference.get("quote", "")
            if not quote:
                continue
            start, end = reference.get("start"), reference.get("end")
            if start is not None or end is not None:
                origin = event.get("start")
                if (type(start) is int and type(end) is int and type(origin) is int
                        and 0 <= start - origin < end - origin <= len(event["text"])
                        and event["text"][start - origin:end - origin] == quote):
                    found.add((start - origin, end - origin))
                continue
            offset = event["text"].find(quote)
            if offset >= 0:
                found.add((offset, offset + len(quote)))
        return [list(span) for span in sorted(found)]

    semantics = metadata.get("semantics") or {}
    members = []
    for member in semantics.get("members", []):
        locations = spans(member.get("evidence", []))
        if locations:
            item = {"value": member["value"], "spans": locations}
            if member.get("identity"):
                item["identity"] = member["identity"]
            members.append(item)
    locations = spans(metadata.get("evidence", []))
    if not locations and not members:
        return {}
    hint = {"subject": metadata["subject"], "predicate": metadata["predicate"]}
    if members:
        hint["members"] = members
        hint["operation"] = semantics.get("operation", "assert")
    elif semantics.get("cardinality") != "many" and locations:
        hint["value"] = metadata.get("value", "")
        hint["spans"] = locations
    else:
        # A visible collection-level quote does not expose its omitted members.
        return {}
    for key in ("event_key", "event_identity"):
        if semantics.get(key):
            hint[key] = semantics[key]
    return hint


def source_payload(hit: Hit, *, compact: bool = False, include_time: bool = True) -> dict:
    # Full provenance stays in the receipt. The model gets the source once,
    # without repeated IDs, offsets and duplicate calendar anchors.
    source_events = []
    for event in hit.metadata.get("source_events", []):
        record = {key: event[key] for key in ("speaker", "role", "occurred_at", "text", "context_only")}
        if "turn_sequence" in event:
            record["turn_sequence"] = event["turn_sequence"]
        if hint := _extraction_hint(hit, event):
            record["extraction_hint"] = hint
        # Source quotes remain lossless even in compact mode; truncating them
        # would make a valid binding impossible to cite. Compact mode removes
        # duplicate metadata instead of cutting evidence spans.
        if include_time and event["time_derivations"]:
            record["time_derivations"] = [
                {key: value[key] for key in ("expression", "start", "end_exclusive", "precision", "rule")}
                for value in event["time_derivations"]
            ]
        source_events.append(record)
    support = hit.metadata.get("source_support_status", "unresolved")
    authorized_quote = support == "authorized_session_quote"
    payload = {
        "attributes": {
            key: "authorized_session_quote" if key == "origin" and authorized_quote else hit.metadata.get(key)
            for key in (
                "status",
                "assertion",
                "valid_from",
                "valid_to",
                "local_override",
                "origin",
                "attribution_conflict",
                "reading_unit_partial",
            )
            if hit.metadata.get(key) is not None
        },
        "source_events": source_events,
        "provenance": "resolved" if authorized_quote else support,
    }
    # The structured answer catalog already sends an identical source text
    # once. Keep its admission-cost proxy and the legacy payload consistent;
    # a genuinely compressed claim remains a separate retrieval hint.
    if hit.content not in {event["text"] for event in source_events}:
        payload["claim"] = hit.content[:500] if compact else hit.content
    return payload


def _cost(hit: Hit) -> int:
    # Match the runtime's existing conservative character proxy; actual
    # provider token counts remain separately measured by the provider.
    return len(json.dumps(source_payload(hit), ensure_ascii=False)) + 16


def _request_relation_units(hit: Hit, plan: QueryPlan) -> set[tuple[str, ...]]:
    """Source-visible catalog addresses, not verified semantic bindings.

    A relation about an actor can be stored under its reporter (e.g. attendance
    at another person's event). Match the planner's explicit catalog pair; do
    not infer aliases or equate the reporter with the requested actor.
    """
    visible = {
        (str(hint.get("subject") or "").strip().casefold(),
         str(hint.get("predicate") or "").strip().casefold())
        for event in hit.metadata.get("source_events", [])
        if (hint := _extraction_hint(hit, event))
    }
    return {
        (output.output_id, subject.strip().casefold(), predicate.strip().casefold())
        for output in (plan.answer_contract.outputs if plan.answer_contract else [])
        for subject, predicate in output.retrieval_relations
        if (subject.strip().casefold(), predicate.strip().casefold()) in visible
    }


def _request_units(hit: Hit, plan: QueryPlan) -> set[tuple[str, ...]]:
    """Visible member novelty for packing, not semantic qualification or recall.

    Exact indexed subject/relation matches can expose a useful coverage unit.
    Unknown aliases and unstructured evidence remain eligible at their upstream
    rank; no candidate is excluded for lacking this optional extraction hint.
    A source-event address or event category is never a domain event identity.
    """
    contract = plan.answer_contract
    if contract is None:
        return set()
    units = set()
    for output in contract.outputs:
        if output.operator not in {"set", "count"} or output.count_unit is None:
            continue
        subjects = {subject.strip().casefold() for subject in output.subjects}
        predicates = {output.relation.strip().casefold()} - {""}
        catalog_pairs = {(subject.strip().casefold(), predicate.strip().casefold())
                         for subject, predicate in output.retrieval_relations}
        # A representable collection hint is already a projection of this
        # request; it may carry the catalog spelling of the relation.
        collection = plan.collection_query
        if (len(contract.outputs) == 1 and collection is not None
                and {s.strip().casefold() for s in collection.subjects} == subjects):
            predicates.add(collection.predicate.strip().casefold())
        for event in hit.metadata.get("source_events", []):
            hint = _extraction_hint(hit, event)
            subject = str(hint.get("subject") or "").strip().casefold()
            predicate = str(hint.get("predicate") or "").strip().casefold()
            if subject not in subjects or (predicate not in predicates and (subject, predicate) not in catalog_pairs):
                continue
            if output.count_unit == "event":
                identities = [hint.get("event_identity")]
            else:
                members = hint.get("members", [])
                identities = [(member.get("identity") or member["value"]) if output.count_unit == "entity"
                              else member["value"] for member in members]
                if not members and output.count_unit == "value":
                    identities = [hint.get("value")]
                # An entity unit requires an extracted member; do not mistake
                # the speaker or whole scalar claim for a counted entity.
            state = tuple(str(hit.metadata.get(key) or "") for key in ("assertion", "status"))
            for identity in identities:
                if identity and str(identity).strip():
                    units.add((output.output_id, subject, output.count_unit,
                               str(identity).strip().casefold(), hint.get("operation", "assert"), *state))
    return units


@dataclass
class _ReadingCandidate:
    """Identity survives reading-unit construction and final selection."""

    hit: Hit
    parent_id: str
    rank: int
    upstream_rank: int
    variant: str = "context"

    @property
    def key(self):
        return (self.parent_id, self.variant, tuple(
            (str(event.get("turn_id", "")), event.get("event_index", -1),
             event.get("start", -1), event.get("end", -1))
            for event in self.hit.metadata.get("source_events", [])))


def _reading_support(candidates, query, plan):
    """Soft request coverage; scores locate evidence, never certify sufficiency.

    Request text remains usable without catalog bindings. Each output has equal
    weight, split over its observed locating signals. Repetition across retrieval
    channels cannot improve a source's score. Missing signals never filter it.
    """
    features = {candidate.key: {} for candidate in candidates}
    weights = {}
    outputs = plan.answer_contract.outputs if plan.answer_contract else []
    if not outputs:
        return features, weights
    # Questions may locate useful exchanges, but cannot themselves satisfy a
    # factual request. Keep them eligible as reading context; only their already
    # authorized assertion spans contribute this soft coverage signal.
    documents = [" ".join(event["text"][left:right]
                 for event in c.hit.metadata.get("source_events", []) if not event.get("is_question", False)
                 for left, right in source_assertion_spans(event["text"])) for c in candidates]
    # Score a source once even when several memories or views point to it.
    unique = list(dict.fromkeys(documents))
    from .prompt_codec import question_requires_time
    temporal_outputs = {o.output_id for o in outputs if o.operator in {"time_point", "duration"}
                        or question_requires_time(o.question_span or query)}
    temporal = bool(temporal_outputs)
    time_visible = {}
    if temporal:
        from .temporal import resolve_time_references
        for candidate in candidates:
            time_visible[candidate.key] = any(resolve_time_references(
                event["text"], source_time=event.get("occurred_at"))
                for event in candidate.hit.metadata.get("source_events", []))
    for output in outputs:
        # Invalid/absent contracts already fall back to the original query in
        # runtime; direct callers get the same conservative locating fallback.
        text = output.question_span if output.question_span and output.question_span in query else query
        scores = _span_scores(text, unique, member_positions=_list_member_positions(query, plan),
                              quantity_terms=_quantity_terms(query, plan))
        maximum = max(scores, default=0)
        topical = dict(zip(unique, [score / maximum if maximum else 0 for score in scores]))
        names = []
        subjects = list(dict.fromkeys(s for s in output.subjects if terms(s)))
        # Separate only explicitly requested subjects, never invented operands.
        facets = subjects if len(subjects) > 1 else [""]
        for facet in facets:
            key = (output.output_id, "request", facet)
            names.append(key)
            sought = terms(facet)
            for candidate, document in zip(candidates, documents):
                score = topical[document]
                if sought:
                    score *= len(sought & terms(document)) / len(sought)
                if output.output_id in temporal_outputs:
                    # Absence of a parsed date halves a locating signal, not
                    # eligibility: direct A-before-B statements still compete.
                    score *= (1.0 if time_visible[candidate.key] else 0.5)
                features[candidate.key][key] = score
        for kind, getter in (("member", _request_units), ("field", _request_relation_units)):
            values = {c.key: {unit for unit in getter(c.hit, plan) if unit[0] == output.output_id}
                      for c in candidates}
            for unit in sorted(set().union(*values.values())):
                key = (output.output_id, kind, *unit[1:])
                names.append(key)
                for candidate in candidates:
                    features[candidate.key][key] = float(unit in values[candidate.key])
        for name in names:
            weights[name] = 1.0 / (len(outputs) * len(names))
    return features, weights


def _query_rank_utility(candidate):
    """Keep query retrieval relevance after lexical coverage has saturated.

    Entity expansion and directory positions are not query relevance ranks.
    Repeated channels cannot add utility; use the best direct-query view once.
    """
    channels = candidate.hit.metadata.get("retrieval_channel_ranks") or {}
    direct = [value for name, value in channels.items()
              if name.rsplit(".", 1)[-1] in {"semantic", "lexical"}
              and type(value) is int and value > 0]
    if direct:
        return 1.0 / min(direct)
    if channels and all(name.rsplit(".", 1)[-1] in {"entity_link", "indexed_sweep"} for name in channels):
        return 0.0
    return 1.0 / max(1, candidate.upstream_rank)


def _coverage_value(candidates, features, weights):
    return sum(weight * max((features[c.key].get(key, 0.0) for c in candidates), default=0.0)
               for key, weight in weights.items())


def close_evidence_bundles(sources, turns, *, session_id, blocked_events, budget,
                           query="", plan=None, authorized_quotes=None,
                           source_cost: Callable[[Hit, int], int] | None = None, source_id: int = 1,
                           member_positions: set[str] | None = None, include_neighbors: bool = True):
    """Build source-bounded reading units before the one final budget selection.

    A unit is an original paragraph, a labelled list, or an addressed list item.
    One adjacent user/assistant exchange can supply reading context; proximity
    does not assert an event/identity edge. No neighbour-prefix reservation or
    second retrieval round spends a separate context budget.
    """
    authorized_quotes = authorized_quotes or {}
    by_id = {turn.turn_id: turn for turn in turns}
    ordered = [(turn, i) for turn in sorted(turns, key=lambda t: (t.session_id, t.sequence))
               for i in range(len(turn.events))]
    positions = {(turn.turn_id, i): position for position, (turn, i) in enumerate(ordered)}
    search = " ".join([query, *(plan.required_info if plan else []),
                       *(plan.semantic_queries if plan else []), *(plan.keywords if plan else [])]).strip()
    requested_ordinals = (_list_member_positions(query, plan) if member_positions is None
                          else member_positions)
    preserve_lists = bool(plan and plan.query_type == "set" and not requested_ordinals)

    def unit_cost(hit):
        return _cost(hit) if source_cost is None else source_cost(hit, source_id)

    def allowed_ranges(turn, index):
        key = turn.turn_id, index
        if turn.session_id == session_id and key not in blocked_events:
            return [(0, len(turn.events[index].content))]
        intervals = []
        for value in authorized_quotes.get(key, ()):
            ref = ({"turn_id": key[0], "event_index": key[1], "quote": value}
                   if isinstance(value, str) else value)
            if isinstance(ref, dict) and (ref.get("turn_id"), ref.get("event_index")) == key:
                intervals.extend(_reference_intervals(ref, by_id))
        return sorted(set(intervals))

    def records_for(turn, index, ranges, *, fallback=None, require_relevance=True):
        original = turn.events[index].content
        candidates = []
        for left, right in ranges:
            text = original[left:right]
            # Small original utterances are already coherent reading units.
            pieces = ([(0, text)] if len(text) <= 700 and not requested_ordinals
                      else _local_spans(text, preserve_lists=preserve_lists))
            for offset, piece in pieces:
                if piece.strip():
                    candidates.append((left + offset, piece))
        if not candidates:
            return [fallback] if fallback else []
        # Close the anchor's own linguistic scope before seeking another topical
        # window. Otherwise a short extracted noun phrase can survive beside an
        # unrelated sentence while its question/condition remains invisible.
        enclosing = []
        coherent = set()
        if fallback and not preserve_lists and not requested_ordinals:
            sentences = source_sentence_spans(original)
            touched = [span for span in sentences
                       if span['start'] < fallback['end'] and span['end'] > fallback['start']]
            if touched:
                left, right = touched[0]['start'], touched[-1]['end']
                if ((left < fallback['start'] or right > fallback['end'])
                        and any(start <= left < right <= end for start, end in ranges)):
                    enclosing = [(left, original[left:right])]
                    coherent = {piece for piece in candidates
                                if piece[0] <= left and right <= piece[0] + len(piece[1])}
                    candidates = list(dict.fromkeys([*enclosing, *candidates]))
        scores = _span_scores(search, [text for _, text in candidates],
                              quantity_terms=_quantity_terms(query, plan), member_positions=requested_ordinals)
        if fallback and len(candidates) > 1:
            # Exact already visible source bytes are not a missing detail.
            # This is interval identity, never global word-level novelty.
            for i, (left, piece) in enumerate(candidates):
                if not _uncovered_text(piece, left, [(fallback["start"], fallback["end"])]).strip():
                    scores[i] = 0
        bundles = {i for i, (_, text) in enumerate(candidates) if _list_bundle(text)}
        if preserve_lists:
            for i in bundles:
                start, text = candidates[i]
                contained = [j for j, (left, piece) in enumerate(candidates)
                             if start <= left and left + len(piece) <= start + len(text)]
                scores[i] = max([scores[i], *(scores[j] for j in contained)])
        order = sorted(range(len(candidates)), key=lambda i: (
            candidates[i] not in coherent and candidates[i] not in enclosing,
            candidates[i] in enclosing and bool(coherent) and candidates[i] not in coherent,
            -scores[i], i not in bundles, candidates[i][0]))
        if require_relevance and search and max(scores, default=0) <= 0 and not enclosing:
            return [fallback] if fallback else []
        # The final request budget is the only size ceiling. Oversized lists
        # retain whole fitting child units instead of arbitrary text prefixes.
        oversized_lists = []
        for i in order:
            start, text = candidates[i]
            record = _source_event(turn, index, text, start, assistant_focus(query))
            records = [record]
            if requested_ordinals and re.match(r"\s*\d+[.)]\s+", text):
                heading = _list_heading(original, start)
                if heading and any(left <= heading[0] < heading[1] <= right for left, right in ranges):
                    records.insert(0, _source_event(turn, index, original[heading[0]:heading[1]],
                                                    heading[0], assistant_focus(query)))
            probe = hit.model_copy(deep=True)
            probe.metadata["source_events"] = records
            if fallback and not any(record["start"] <= fallback["start"] < fallback["end"] <= record["end"]
                                    for record in records):
                probe.metadata["source_events"] = [fallback, *records]
            if (probe.metadata.get("origin") == "raw_fallback"
                    or probe.metadata.get("source_support_status") == "authorized_session_quote"):
                probe.content = records[0]["text"]
            if unit_cost(probe) <= budget:
                if any(left <= start < start + len(text) <= right for left, right in oversized_lists):
                    hit.metadata["reading_unit_partial"] = True
                return records
            if i in bundles:
                oversized_lists.append((start, start + len(text)))
        return [fallback] if fallback else []

    result = []
    for source in sources:
        hit = source.model_copy(deep=True)
        anchors = hit.metadata.get("source_events", [])
        records = []
        for anchor in anchors:
            key = anchor.get("turn_id"), anchor.get("event_index")
            turn = by_id.get(key[0])
            if turn is None or type(key[1]) is not int or not 0 <= key[1] < len(turn.events):
                records.append(anchor)
                continue
            # Initial exact quotes remain readable even if their surrounding
            # event is quarantined. They do not grant expansion permission.
            allowed = allowed_ranges(turn, key[1])
            enclosing = [(left, right) for left, right in allowed
                         if left <= anchor["start"] < anchor["end"] <= right]
            if enclosing:
                enclosing = allowed
            if not enclosing:
                enclosing = [(anchor["start"], anchor["end"])]
            chosen = records_for(turn, key[1], enclosing, fallback=anchor)
            if not any(
                    record["start"] <= anchor["start"] < anchor["end"] <= record["end"]
                    for record in chosen):
                probe = hit.model_copy(deep=True)
                probe.metadata["source_events"] = [anchor, *chosen]
                chosen = [anchor, *chosen] if unit_cost(probe) <= budget else [anchor]
            records.extend(chosen)
            if not include_neighbors or key in blocked_events or key not in positions:
                continue
            event = turn.events[key[1]]
            # One directly adjacent discourse unit, which can be an explicit
            # short clarification question and its answer. Its source records
            # remain independent: adjacency is not a relation assertion.
            neighbors = []
            for offset in (-1, 1):
                position = positions[key] + offset
                if not 0 <= position < len(ordered):
                    continue
                neighbor, other = ordered[position]
                adjacent = neighbor.events[other]
                if (neighbor.session_id != turn.session_id
                        or abs(neighbor.sequence - turn.sequence) > 1
                        or (neighbor.turn_id, other) in blocked_events or not _role_allowed(adjacent)
                        or event.occurred_at is None or adjacent.occurred_at is None
                        or abs((adjacent.occurred_at - event.occurred_at).total_seconds()) > 1800):
                    continue
                if (adjacent.role == "assistant" and plan and plan.response_intent == "action_plan"
                        and not assistant_focus(query)):
                    continue
                ranges = allowed_ranges(neighbor, other)
                if ranges:
                    is_question = question_only(adjacent.content)
                    proposed = records_for(neighbor, other, ranges, require_relevance=not (
                        is_question or (offset == 1 and question_only(event.content))))
                    # An assistant passage answers its preceding user turn.
                    # Keep that discourse parent ahead of a later follow-up;
                    # adjacency supplies context, never a factual relationship.
                    complete_exchange = (2 if offset == -1 and event.role == "assistant"
                                         and adjacent.role == "user" else 0)
                    if is_question and offset == 1 and position+1 < len(ordered):
                        reply_turn, reply_index = ordered[position+1]
                        reply = reply_turn.events[reply_index]
                        if (reply_turn.session_id == turn.session_id
                                and abs(reply_turn.sequence - turn.sequence) <= 2
                                and (reply_turn.turn_id, reply_index) not in blocked_events
                                and _role_allowed(reply) and not question_only(reply.content)
                                and reply.occurred_at is not None
                                and abs((reply.occurred_at-event.occurred_at).total_seconds()) <= 1800
                                and not (reply.role == "assistant" and plan
                                         and plan.response_intent == "action_plan" and not assistant_focus(query))):
                            reply_ranges = allowed_ranges(reply_turn, reply_index)
                            answer = records_for(reply_turn, reply_index, reply_ranges,
                                                 require_relevance=False) if reply_ranges else []
                            if proposed and answer:
                                proposed = [*proposed, *answer]
                                complete_exchange = True
                    if proposed:
                        neighbors.append((complete_exchange, proposed))
            if neighbors:
                relevance = _span_scores(search, [" ".join(r["text"] for r in group) for _, group in neighbors],
                                         member_positions=requested_ordinals)
                neighbor_records = neighbors[max(range(len(neighbors)),
                    key=lambda i: (neighbors[i][0], relevance[i]))][1]
                probe = hit.model_copy(deep=True)
                probe.metadata["source_events"] = _coalesce_source_events([*records, *neighbor_records], by_id, query)
                if unit_cost(probe) <= budget:
                    records.extend(neighbor_records)
        hit.metadata["source_events"] = _coalesce_source_events(records, by_id, query)
        # Raw/archival source excerpts are locating views of the same event;
        # do not serialize an obsolete prefix as a second competing claim.
        if hit.metadata["source_events"] and (hit.metadata.get("origin") == "raw_fallback"
                or hit.metadata.get("source_support_status") == "authorized_session_quote"):
            hit.content = hit.metadata["source_events"][0]["text"]
        result.append(hit)
    return result


def _list_heading(text, start):
    """The layout parent of an item; never a semantic event identity."""
    lines = list(re.finditer(r"(?m)^.*(?:\r?\n|$)", text))
    for position in range(len(lines)-1, -1, -1):
        line = lines[position]
        if line.start() >= start:
            continue
        body = line.group().strip()
        if not body or _LIST_MEMBER.match(line.group()):
            continue
        if body.rstrip("*_ ").endswith(":") or re.match(r"#{1,6}\s+\S", body):
            following = next((value for value in lines[position+1:] if value.group().strip()), None)
            if following is not None and _LIST_MEMBER.match(following.group()):
                return line.start(), line.end()
    return None


def _coalesce_source_events(records, source_turns, query):
    """Merge only overlapping exact source coordinates, not similar content."""
    unique = {}
    priorities = {}
    for priority, record in enumerate(records):
        key = record.get("turn_id"), record.get("event_index")
        turn = source_turns.get(key[0])
        start, end = record.get("start"), record.get("end")
        if (turn is None or type(key[1]) is not int or not 0 <= key[1] < len(turn.events)
                or type(start) is not int or type(end) is not int or not 0 <= start < end
                or turn.events[key[1]].content[start:end] != record.get("text")):
            continue
        unique.setdefault(key, []).append((start, end))
        priorities.setdefault(key, []).append((start, end, priority))
    result = []
    for (turn_id, index), spans in unique.items():
        intervals = []
        for start, end in sorted(spans):
            if intervals and start < intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
            else:
                intervals.append((start, end))
        turn = source_turns[turn_id]
        for start, end in intervals:
            priority = min(order for left, right, order in priorities[(turn_id, index)]
                           if left < end and start < right)
            result.append((priority, _source_event(turn, index, turn.events[index].content[start:end], start,
                                                   assistant_focus(query))))
    return [record for _, record in sorted(result, key=lambda pair: pair[0])]


def _span_covered(spans, start, end):
    """Whether the union of already visible intervals covers this whole span."""
    cursor = start
    for left, right in sorted(spans):
        if left > cursor:
            break
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def _uncovered_text(text, start, spans):
    """Ranking-only view: repeated visible words cannot buy another window."""
    pieces, cursor, end = [], start, start + len(text)
    for left, right in sorted(spans):
        if right <= cursor or left >= end:
            continue
        if left > cursor:
            pieces.append(text[cursor-start:left-start])
        cursor = max(cursor, min(right, end))
    if cursor < end:
        pieces.append(text[cursor-start:])
    return " ".join(pieces)


def _quantity_terms(query, plan=None):
    """Object of a requested quantity, used for ranking rather than inference.

    A topical procedural sentence is not as useful for a quantity request as
    a numeric assertion about the requested object. The question/plan supplies
    that object; neither a benchmark label nor an answer is consulted.
    """
    match = re.search(r"\b(?:how\s+many|how\s+much|number\s+of|quantity\s+of)\s+(.+?)"
                      r"(?=\s+(?:did|do|does|is|are|was|were|have|has|had|can|could|would|should|will)\b|[?.,;]|$)",
                      query, re.IGNORECASE)
    if match:
        return normalized_terms(match.group(1)) - terms("total number quantity amount")
    if plan and (plan.query_type == "count" or
                 (plan.collection_query and plan.collection_query.operation == "count")):
        return normalized_terms(" ".join([plan.subject or "", plan.predicate or ""])) - terms(
            "user total number quantity amount count")
    return set()


def _has_quantity(document):
    # A list ordinal is an address, not a quantity. Remove it before looking
    # for an actual numeric assertion (including ranges and written numbers).
    body = re.sub(r"(?m)^\s*(?:[-*•]\s*|\d+[.)]\s+)", "", document)
    return bool(re.search(r"\b\d+(?:[.,]\d+)?\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
                          r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
                          r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b",
                          body, re.IGNORECASE))


_ORDINAL_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                  "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                  "eighteenth", "nineteenth", "twentieth"]
_ORDINAL_TENS = dict(zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], range(20, 100, 10)))
_ORDINAL_EXPRESSION = (r"(?:\d+(?:st|nd|rd|th)|(?:" + "|".join(_ORDINAL_TENS)
    + r")[ -](?:" + "|".join(_ORDINAL_WORDS[:9]) + r")|" + "|".join(_ORDINAL_WORDS) + r")")


def _ordinal_positions(query):
    """Normalize ordinary ordinal spelling, without predicting answer content."""
    values = {int(value) for value in re.findall(r"\b(\d+)(?:st|nd|rd|th)\b", query, re.IGNORECASE)}
    words = _ORDINAL_WORDS
    remaining = query.casefold()
    tens = _ORDINAL_TENS
    composite = re.compile(r"\b(" + "|".join(tens) + r")[ -](" + "|".join(words[:9]) + r")\b")
    for match in composite.finditer(remaining):
        values.add(tens[match[1]] + words.index(match[2]) + 1)
    remaining = composite.sub(" ", remaining)
    values.update(i for i, word in enumerate(words, 1) if re.search(r"\b" + word + r"\b", remaining))
    values.update(int(value) for value in re.findall(r"第\s*(\d+)\s*(?:个|项|条|种|位)?", query))
    return {str(value) for value in values if value > 0}


def _list_member_positions(query, plan=None):
    """Resolve member addresses from the request, never from retrieval hints.

    The existing planner owns ambiguous noun semantics. Its optional binding
    is accepted only inside a literal question span with that ordinal. Legacy
    plans retain a small domain-independent fallback for explicit item/list
    syntax; a bare ordinal modifying an arbitrary noun is not enough. Missing
    intent removes a ranking bonus, never an evidence candidate or permission.
    """
    positions = set()
    if plan and plan.answer_contract:
        for output in plan.answer_contract.outputs:
            if output.question_span and output.question_span in query:
                positions.update(set(map(str, output.member_positions))
                                 & _ordinal_positions(output.question_span))
    member = r"(?:items?|entries|entry|elements?|members?|options?|steps?|points?|bullets?)"
    # The ordinal modifies the member, not its container: "first list of
    # items" has no address, while "fifth item in the first list" has 5.
    patterns = [
        r"\b(" + _ORDINAL_EXPRESSION + r")\s+" + member + r"\b",
        r"\b(" + _ORDINAL_EXPRESSION + r")(?:\s+one)?\s+(?:in|on|of|from)\s+"
        r"(?:(?:the|that|this|your|my)\s+)?(?:list|enumeration)\b",
        r"(第\s*\d+\s*(?:项|条))",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            positions.update(_ordinal_positions(match[1]))
    return positions


def _span_scores(query, documents, *, quantity_terms=None, member_positions=()):
    # Search text contributes topical relevance only. Positional semantics
    # are resolved from the original question/contract by the caller.
    ordinals = set(member_positions)
    normalized = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", query, flags=re.IGNORECASE)
    scores = _bm25(normalized, documents)
    quantity_bonus = max(scores, default=0) + 1
    for i, document in enumerate(documents):
        if quantity_terms and quantity_terms & normalized_terms(document) and _has_quantity(document):
            scores[i] += quantity_bonus
    if not ordinals:
        return scores
    ordinal_bonus = max(scores, default=0) + 1
    for i, document in enumerate(documents):
        numbered = set(re.findall(r"(?m)^\s*(\d+)[.)]\s+", document))
        if ordinals & numbered:
            # Explicit item identity precedes topical overlap with other items.
            scores[i] += ordinal_bonus
    return scores


def _reference_intervals(reference, resolved_turns):
    """Exact authorized quote positions; a locator never expands source access."""
    turn = resolved_turns.get(reference.get("turn_id"))
    index, quote = reference.get("event_index"), reference.get("quote", "")
    if turn is None or type(index) is not int or not 0 <= index < len(turn.events) or not quote:
        return []
    text = turn.events[index].content
    start, end = reference.get("start"), reference.get("end")
    if start is not None or end is not None:
        return [(start, end)] if (type(start) is int and type(end) is int
            and 0 <= start < end <= len(text) and text[start:end] == quote) else []
    # Legacy quotes can occur more than once. Retain exact occurrences rather
    # than attaching unrelated surrounding text to an assumed first occurrence.
    positions = []
    left = text.find(quote)
    while left >= 0:
        positions.append((left, left + len(quote)))
        left = text.find(quote, left + 1)
    return positions


def match_evidence(query, plan, references, resolved_turns):
    """Rank exact spans within caller-authorized references in one memory.

    Scores order spans only; they are neither cross-memory probabilities nor
    assertions of truth. All retained text is cut from its original event.
    """
    records = {}
    for reference_index, reference in enumerate(references):
        for start, _ in _reference_intervals(reference, resolved_turns):
            for relative, text in _chunks(reference["quote"]):
                left = start + relative
                key = (reference["turn_id"], reference["event_index"], left, left + len(text))
                records.setdefault(key, {"turn_id": key[0], "event_index": key[1], "start": key[2],
                                         "end": key[3], "text": text, "reference_index": reference_index})
    # Nested duplicate references do not replace a complete fitting quote with
    # a context-free phrase merely because the phrase is shorter.
    candidates, covered_end = [], {}
    for item in sorted(records.values(), key=lambda s: (s["turn_id"], s["event_index"], s["start"], -s["end"])):
        key = item["turn_id"], item["event_index"]
        if item["end"] <= covered_end.get(key, -1):
            continue
        candidates.append(item)
        covered_end[key] = item["end"]
    search = " ".join([query, *plan.required_info, *plan.semantic_queries, *plan.keywords])
    scores = _span_scores(search, [item["text"] for item in candidates],
                          quantity_terms=_quantity_terms(query, plan),
                          member_positions=_list_member_positions(query, plan))
    return sorted((dict(item, score=score) for item, score in zip(candidates, scores) if score > 0),
                  key=lambda item: (-item["score"], item["reference_index"], item["start"]))


def _missing(hits: list[Hit], required: list[str]) -> list[str]:
    available = (
        set().union(
            *(
                terms(
                    hit.content
                    + " "
                    + " ".join(item["text"] for item in hit.metadata.get("source_events", []))
                )
                for hit in hits
            )
        )
        if hits
        else set()
    )
    return [slot for slot in required if terms(slot) and len(terms(slot) & available) < len(terms(slot)) / 2]


_ASSISTANT_FOCUS_RE = re.compile(
    r"\b(?:assistant|you\s+(?:say|said|recommend|suggest|mention|tell|told|advise|propose|think)"
    r"|your\s+(?:recommendation|suggestion|advice|answer|response))\b"
    r"|助手|你(?:之前|以前|刚才|上次|刚刚)?\s*(?:说|建议|推荐|提到)",
    re.IGNORECASE,
)


def assistant_focus(query: str) -> bool:
    """Whether the question is about something the assistant said.

    This decides *which* voice carries the answer, not whether a turn may be
    recalled. A question about the assistant's own advice has the assistant
    turn as its primary evidence; every other question treats it as secondary.
    """
    return bool(_ASSISTANT_FOCUS_RE.search(query))


def _source_event(turn: Turn, index: int, text: str, start: int, assistant_is_primary: bool = False) -> dict:
    event = turn.events[index]
    anchor = event.occurred_at.astimezone(timezone(timedelta(minutes=event.source_utc_offset_minutes)))
    is_question = not assertion_supported(event.content, text, start=start, end=start + len(text))
    scopes = [{"start": scope["start"] - start, "end": scope["end"] - start,
               "operators": [{**operator, "start": operator["start"] - start,
                              "end": operator["end"] - start} for operator in scope["operators"]]}
              for scope in source_assertion_scopes(event.content)
              if scope["start"] < start + len(text) and scope["end"] > start]
    # Two orthogonal properties, previously collapsed into one flag:
    #   assertoric  - a question asserts nothing, so it cannot support a fact;
    #   speaker     - an assistant turn is a real assertion but second-hand.
    # `context_only` keeps its historical meaning (not a user fact) for
    # compatibility, while `evidence_grade` is what validation now consumes.
    if is_question:
        grade = "none"
    elif event.role == "assistant":
        grade = "primary" if assistant_is_primary else "secondary"
    else:
        grade = "secondary" if assistant_is_primary else "primary"
    return {
        "turn_id": turn.turn_id,
        "event_index": index,
        "turn_sequence": turn.sequence,
        "speaker": speaker_of(event),
        "role": event.role,
        "occurred_at": anchor.isoformat(),
        "text": text,
        "start": start,
        "end": start + len(text),
        "context_only": event.role == "assistant" or is_question,
        "is_question": is_question,
        "evidence_grade": grade,
        # Scope coordinates can extend outside the authorized text, but never
        # carry neighboring text into the public Hit, receipt or model payload.
        "assertion_scopes": scopes,
        "time_derivations": [item.dump() for item in derive_times(text, anchor)],
    }


_LIST_MEMBER = re.compile(r"(?m)^(?P<indent>[ \t]*)(?:\d+[.)]|[-*•])[ \t]+\S")


def _list_member_ranges(text: str) -> list[tuple[int, int]]:
    """Exact outer member ranges; indented children remain with their parent.

    A blank-separated paragraph at the member's indentation ends the member.
    This is a layout boundary, not a claim that a pronoun has been resolved.
    """
    markers = list(_LIST_MEMBER.finditer(text))
    ranges = []
    for index, marker in enumerate(markers):
        left = marker.start()
        if ranges and left < ranges[-1][1]:
            continue
        indent = len(marker["indent"].expandtabs(4))
        right = next((other.start() for other in markers[index + 1:]
                      if len(other["indent"].expandtabs(4)) <= indent), len(text))
        for paragraph in re.finditer(r"\r?\n[ \t]*\r?\n(?P<indent>[ \t]*)(?P<body>\S)", text[left:right]):
            if len(paragraph["indent"].expandtabs(4)) <= indent:
                right = left + paragraph.start("body")
                break
        ranges.append((left, right))
    return ranges


def _chunks(text: str, limit: int = 700) -> list[tuple[int, str]]:
    """Lossless bounded spans that keep a fitting list member intact.

    Keep the existing greedy packing, but do not cut inside a fitting member.
    Oversized members still split at sentence/whitespace boundaries; their tail
    cannot be joined to the next member's title. Prose behavior is unchanged.
    """
    if limit < 1:
        raise ValueError("Chunk limit must be positive")
    members = _list_member_ranges(text)
    protected = []
    for left, right in members:
        heading = re.search(r"(?m)^[ \t]*(?P<title>[^\r\n]*\S)[ \t]*\r?\n(?:[ \t]*\r?\n)*\Z",
                            text[:left])
        if heading:
            title = heading["title"].strip()
            labelled = title.rstrip("*_ ").endswith(":") or re.match(r"#{1,6}\s+\S", title)
            if (labelled and not _LIST_MEMBER.match(title)
                    and right - heading.start() <= limit):
                # A section heading is useful only with its following item.
                # Keep this fitting pair together instead of leaving a search
                # keyword at the end of the preceding background-only chunk.
                left = heading.start()
        protected.append((left, right))
    members = protected
    boundaries = [m.end() for m in re.finditer(r"\n+|(?<=[.!?。！？])(?<!\d\.)\s+(?=[A-Z\"“'])", text)]
    boundaries.append(len(text))
    if members:
        boundaries = sorted(set(boundaries) | {edge for member in members for edge in member})
        boundaries = [edge for edge in boundaries if not any(
            left < edge < right and right - left <= limit for left, right in members)]
    chunks, start = [], 0
    while start < len(text):
        upper = min(len(text), start + limit)
        for left, right in members:
            if left < start < right:
                # This chunk starts in an oversized member's continuation.
                upper = min(upper, right)
                break
            if start < left < upper and right - left > limit:
                upper = left
                break
        ends = [end for end in boundaries if start < end <= upper]
        end = max(ends) if ends else upper
        if not ends and upper < len(text):
            whitespace = max(text.rfind(" ", start, upper), text.rfind("\t", start, upper))
            if whitespace > start:
                end = whitespace + 1
        chunks.append((start, text[start:end]))
        start = end
    return chunks


def _list_bundle(text):
    lines = [line for line in text.splitlines() if line.strip()]
    return (len(lines) >= 3 and lines[0].rstrip().endswith(":")
            and all(re.match(r"\s*(?:\d+[.)]\s+|[-*•]\s+)", line) for line in lines[1:]))


def _local_spans(text, *, preserve_lists=False):
    """Small exact sentence/list units for expanding an already visible event.

    Its surrounding discourse is already present, so an extra relevant line
    need not pay for another full 700-character window of unrelated content.
    """
    boundaries = [0, *(m.end() for m in re.finditer(r"\n+|(?<=[.!?。！？])(?<!\d\.)\s+(?=[A-Z\"“'])", text))]
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    # Sentence boundaries inside one fitting member must not turn its trailing
    # description into an anonymous candidate beside the next member's title.
    members = _list_member_ranges(text)
    boundaries = sorted(set(boundaries) | {edge for member in members for edge in member})
    boundaries = [edge for edge in boundaries
                  if not any(left < edge < right for left, right in members)]
    units = [(left, right) for left, right in pairwise(boundaries) if text[left:right].strip()]
    spans = []
    for i, (left, right) in enumerate(units):
        # A section label alone cannot answer its own question. Keep its first
        # content line with it, e.g. "Budget:\n* Advertising: ...", as one exact
        # span instead of letting the rare heading word outrank its value.
        if text[left:right].rstrip().endswith(":") and i + 1 < len(units):
            right = units[i + 1][1]
            # A heading scopes all adjacent members, not only the first one.
            # Keep that relation as one exact candidate; the caller's existing
            # serialized budget decides whether it fits. Individual members
            # remain candidates below for ordinal or budget-limited queries.
            j = i + 1
            while preserve_lists and j < len(units) and re.match(
                r"\s*(?:\d+[.)]\s+|[-*•]\s+)", text[units[j][0]:units[j][1]]
            ):
                right = units[j][1]
                j += 1
            if j > i + 2:
                spans.append((left, text[left:right]))
                continue
            if right - left <= 700:
                spans.append((left, text[left:right]))
                continue
        spans.extend((left + offset, piece) for offset, piece in _chunks(text[left:right]))
    return spans


def _bm25(query: str, documents: list[str]) -> list[float]:
    query_terms = terms(query)
    if not query_terms:
        return [0.0] * len(documents)
    bags = [Counter(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", doc.casefold())) for doc in documents]
    for doc, bag in zip(documents, bags):
        bag.update(_cjk_terms(doc))
    matches = [sorted(query_terms.intersection(bag)) for bag in bags]
    frequencies = Counter(term for matched in matches for term in matched)
    weights = {token: math.log(1 + (len(bags) - freq + 0.5) / (freq + 0.5))
               for token, freq in frequencies.items()}
    lengths = [sum(bag.values()) for bag in bags]
    mean = max(1, sum(lengths) / max(1, len(lengths)))
    scores = []
    for bag, length, matched in zip(bags, lengths, matches):
        score = 0.0
        for token in matched:
            idf = weights[token]
            tf = bag[token]
            score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / mean))
        scores.append(score)
    return scores


@dataclass
class GroundedResult:
    sources: list[Hit]
    trace: GroundingTrace


def collect_raw_records(query, plan, turns, session_id, blocked_events, *, source_session_ids=()):
    records = []
    focus = assistant_focus(query)
    for turn in turns:
        if turn.session_id != session_id and turn.session_id not in source_session_ids:
            continue
        for index, event in enumerate(turn.events):
            if (turn.turn_id, index) in blocked_events or not source_assertion_spans(event.content):
                continue
            if not _role_allowed(event):
                continue
            if event.role == "assistant" and plan.response_intent == "action_plan" and not focus:
                # A new execution plan is constrained by user-adopted facts,
                # not an older assistant proposal. This explicit intent is
                # distinct from recalling a count in a prior conversation.
                continue
            # Retrieval availability and premise authority are separate. A
            # count request can concern an assistant assertion without saying
            # "you said". Keep its original role/grade; the answer contract
            # decides whether it can establish a personal fact or preference.
            for offset, text in _chunks(event.content):
                records.append((_source_event(turn, index, text, offset, focus), text))
    return records


def _role_allowed(event):
    return event.role != "assistant" or os.getenv(
        "MEMORY_ALLOW_ASSISTANT_EVIDENCE", "true").lower() in {"1", "true", "yes"}


def record_key(record):
    return f"{record['turn_id']}:{record['event_index']}:{record['start']}"


def _records_already_visible(records, selected, source_turns):
    """Only exact original spans can establish redundant evidence visibility."""
    def span(record):
        turn_id, index = record.get("turn_id"), record.get("event_index")
        turn = source_turns.get(turn_id) if isinstance(turn_id, str) and turn_id else None
        start, end, text = record.get("start"), record.get("end"), record.get("text")
        if (turn is None or type(index) is not int or not 0 <= index < len(turn.events)
                or type(start) is not int or type(end) is not int
                or not isinstance(text, str) or not text or start < 0 or end <= start
                or end - start != len(text) or turn.events[index].content[start:end] != text):
            return None
        return (turn_id, index), start, end

    visible = {}
    for hit in selected:
        for record in hit.metadata.get("source_events", []):
            known = span(record)
            if known is not None:
                identity, start, end = known
                visible.setdefault(identity, []).append((start, end))
    if not records:
        return False
    for record in records:
        known = span(record)
        if known is None:
            return False
        identity, cursor, end = known
        for left, right in sorted(visible.get(identity, [])):
            if left > cursor:
                break
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if cursor < end:
            return False
    return True


def ground_evidence(
    query: str,
    plan: QueryPlan,
    hits: list[Hit],
    *,
    session_id: str,
    current_turns: list[Turn],
    resolved_turns: dict[str, Turn],
    blocked_events: set[tuple[str, int]],
    max_sources: int = 20,
    budget: int = 5000,
    candidate_limit: int = 120,
    semantic_scores: dict[str, float] | None = None,
    source_session_ids: tuple[str, ...] = (),
    context_turns: list[Turn] | None = None,
    authorized_quotes: dict | None = None,
    source_cost: Callable[[Hit, int], int] | None = None,
) -> GroundedResult:
    started = time.perf_counter()
    member_positions = _list_member_positions(query, plan)
    trace = GroundingTrace(enabled=True, initial_sources=len(hits))
    known_speakers = {speaker_of(event) for turn in current_turns for event in turn.events} - {None}
    query_people = {
        person
        for person in known_speakers
        if re.search(r"(?<!\w)" + re.escape(person) + r"(?!\w)", query, re.IGNORECASE)
    }
    hydrated = []
    for hit in hits:
        denied = [
            reference
            for reference in hit.metadata.get("evidence", [])
            if (reference.get("turn_id"), reference.get("event_index")) in blocked_events
        ]
        ephemeral_raw = hit.metadata.get("predicate") == "recent_message" and str(
            hit.metadata.get("memory_id", "")
        ).startswith("raw_")
        if denied and ephemeral_raw:
            trace.rejected_source_refs += len(denied)
            continue
        item = hit.model_copy(deep=True)
        authorized_quote = hit.metadata.get("source_support_status") == "authorized_session_quote"
        source_events = []
        references = hit.metadata.get("evidence", [])
        intervals = [_reference_intervals(ref, resolved_turns) for ref in references]
        proposed = hit.metadata.get("matched_evidence")
        if proposed is None:
            proposed = match_evidence(query, plan, references, resolved_turns)
        matched = {}
        # Recheck recorded choices against the current exact references. A
        # stale/fabricated annotation never adds new readable source material.
        for rank, span in enumerate(proposed):
            left, right = span.get("start"), span.get("end")
            if type(left) is not int or type(right) is not int or left >= right:
                continue
            for ref_index, ref in enumerate(references):
                if (ref.get("turn_id"), ref.get("event_index")) != (span.get("turn_id"), span.get("event_index")):
                    continue
                if not any(a <= left < right <= b for a, b in intervals[ref_index]):
                    continue
                text = resolved_turns[ref["turn_id"]].events[ref["event_index"]].content
                if text[left:right] == span.get("text") and ref_index not in matched:
                    matched[ref_index] = (rank, span)
        ordered_refs = sorted(enumerate(references), key=lambda pair: (
            matched[pair[0]][0] if pair[0] in matched else len(proposed) + pair[0]))
        for ref_index, reference in ordered_refs:
            key = reference.get("turn_id"), reference.get("event_index")
            turn = resolved_turns.get(key[0])
            index = key[1]
            quote = reference.get("quote", "")
            if (
                turn is None
                or type(index) is not int
                or not 0 <= index < len(turn.events)
                or not quote
                or quote not in turn.events[index].content
                or not intervals[ref_index]
            ):
                trace.rejected_source_refs += 1
                continue
            event = turn.events[index]
            offset = intervals[ref_index][0][0]
            # Full small utterances retain subject/coreference context only in
            # the active session. Cross-session evidence never expands scope.
            # A consumed candidate can share an exact quote with its active
            # successor. The already-eligible memory may expose that quote,
            # but quarantined events never expand to their surrounding text.
            if turn.session_id == session_id and len(event.content) <= 700 and key not in blocked_events:
                text, offset = event.content, 0
            elif ref_index in matched:
                span = matched[ref_index][1]
                offset, text = span["start"], span["text"]
                # A collection's member hint needs its own complete evidence.
                # Expand only inside the already authorized quotation and the
                # existing window size; recompute coverage after this choice.
                right = offset + len(text)
                for member in (hit.metadata.get("semantics") or {}).get("members", []):
                    for member_ref in member.get("evidence", []):
                        if (member_ref.get("turn_id"), member_ref.get("event_index")) != key:
                            continue
                        for left, end in _reference_intervals(member_ref, resolved_turns):
                            a, b = min(offset, left), max(right, end)
                            if (left < right and offset < end and b - a <= 700
                                    and any(lo <= a < b <= hi for lo, hi in intervals[ref_index])):
                                offset, right = a, b
                            elif (left < right and offset < end and end - left <= 700
                                  and any(lo <= left < end <= hi for lo, hi in intervals[ref_index])
                                  and normalized_terms(query) & normalized_terms(event.content[left:end])
                                      & normalized_terms(text)):
                                # A 700-character window may include a member's
                                # tail plus unrelated prose. Keep the fitting
                                # complete member rather than its broken tail.
                                offset, right = left, end
                text = event.content[offset:right]
            elif authorized_quote:
                windows = _chunks(quote)
                if len(windows) == 1:
                    text = quote
                else:
                    search = " ".join([query, *plan.required_info, *plan.semantic_queries, *plan.keywords])
                    scores = _span_scores(search, [piece for _, piece in windows],
                        quantity_terms=_quantity_terms(query, plan), member_positions=member_positions)
                    relative, text = windows[max(range(len(windows)), key=scores.__getitem__)]
                    offset += relative
            else:
                text = quote[:700]
            record = _source_event(turn, index, text, offset)
            same_event = [x for x in source_events if (x["turn_id"], x["event_index"]) == key]
            if not same_event or not _span_covered(
                [(x["start"], x["end"]) for x in same_event], record["start"], record["end"]
            ):
                source_events.append(record)
        # Bound evidence bundles without pretending omitted spans are verified.
        source_events = source_events[:3]
        if authorized_quote and source_events:
            item.content = source_events[0]["text"]
        item.metadata["source_events"] = source_events
        item.metadata["source_support_status"] = "resolved" if source_events else "unresolved"
        if authorized_quote and source_events:
            item.metadata["source_support_status"] = "authorized_session_quote"
        item.metadata["origin"] = hit.metadata.get("origin", "compressed_memory")
        subject = item.metadata.get("subject")
        conflicts = []
        for record in source_events:
            speaker = record["speaker"]
            text = re.sub(r"^[^:\n]{1,60}:\s+", "", record["text"])
            if (
                subject in known_speakers
                and speaker in known_speakers
                and subject != speaker
                and re.match(r"(?i)^(?:I\b|my\b|I['’]m\b)", text)
                and not re.search(r"(?<!\w)" + re.escape(subject) + r"(?!\w)", text, re.IGNORECASE)
            ):
                conflicts.append(speaker)
        if conflicts:
            trace.attribution_conflicts += 1
            item.metadata["attribution_conflict"] = True
            # Use attested original text in place of the disputed compression.
            item.content = " ".join(record["text"] for record in source_events)
            for field in ("subject", "predicate", "value"):
                item.metadata.pop(field, None)
        hydrated.append(item)

    trace.resolved_source_events = sum(len(hit.metadata["source_events"]) for hit in hydrated)
    trace.initial_missing_info = _missing(hydrated, plan.required_info)
    temporal_query = bool(
        re.search(
            r"\bwhen\b|\b(year|month|date|days?|weeks?)\b|何时|什么时候|哪年|几月|日期|多久",
            query,
            re.IGNORECASE,
        )
    )
    # Raw evidence is consulted at most once for any answer that has unresolved
    # slots, an unresolved source, a temporal question, or weak lexical coverage.
    all_content = " ".join(hit.content for hit in hydrated)
    uncovered = terms(query) - terms(all_content)
    supplement = bool(
        plan.route != "none"
        and max_sources
        and (
            not hydrated
            or source_session_ids
            or trace.initial_missing_info
            or trace.attribution_conflicts
            or temporal_query
            or plan.recall_policy == "recall_first"
            or len(uncovered) > len(terms(query)) / 2
        )
    )
    candidates = []
    if supplement:
        trace.supplemental_rounds = 1
        queries = list(dict.fromkeys([query, *plan.semantic_queries[:3], *trace.initial_missing_info[:3]]))
        trace.supplemental_queries = queries
        records = collect_raw_records(query, plan, current_turns, session_id, blocked_events,
                                      source_session_ids=source_session_ids)
        scores = _span_scores(" ".join(queries), [text for _, text in records],
                              quantity_terms=_quantity_terms(query, plan), member_positions=member_positions)
        if semantic_scores is not None:
            # Reciprocal rank fusion avoids treating BM25 and cosine as
            # calibrated probabilities. RRF constant 60 is frozen, not fitted.
            lexical_order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
            semantic_order = sorted(
                (
                    i
                    for i, (record, _) in enumerate(records)
                    if semantic_scores.get(record_key(record), -1) >= 0.35
                ),
                key=lambda i: (-semantic_scores[record_key(records[i][0])], i),
            )
            fused = [0.0] * len(records)
            for order in (lexical_order, semantic_order):
                for rank, index in enumerate(order[:candidate_limit], 1):
                    fused[index] += 120 / (60 + rank)
            scores = fused
            trace.semantic_fallback_used = True
        for (record, text), score in zip(records, scores):
            if score <= 0:
                continue
            if (
                query_people
                and record["speaker"] not in query_people
                and not any(
                    re.search(r"(?<!\w)" + re.escape(person) + r"(?!\w)", text, re.IGNORECASE)
                    for person in query_people
                )
            ):
                continue
            candidates.append(
                (
                    score,
                    Hit(
                        content=text,
                        score=score / (1 + score),
                        source_file=f"[turn:{record['turn_id']}]",
                        chunk_id=f"raw:{record['turn_id']}:{record['event_index']}:{record['start']}",
                        metadata={
                            "origin": "raw_fallback",
                            "source_support_status": "resolved",
                            "source_events": [record],
                            "scope_type": "session",
                            "scope_id": next(turn.session_id for turn in current_turns
                                             if turn.turn_id == record["turn_id"]),
                            "status": "active",
                            "assertion": "inferred" if record["context_only"] else "stated",
                            "evidence": [
                                {
                                    "turn_id": record["turn_id"],
                                    "event_index": record["event_index"],
                                    "quote": text,
                                }
                            ],
                        },
                    ),
                )
            )
        candidates.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        candidates = candidates[:candidate_limit]
        trace.supplemental_candidates = len(candidates)

    # Preserve independent retrieval-view ranks. A raw score is not on the
    # same scale as a compressed-memory, vector or symbolic-match score.
    ranked = []
    locating_scores = _span_scores(query + " " + " ".join(plan.required_info), [
        hit.content + " " + str(hit.metadata.get("retrieval_hint") or "")
        + " " + " ".join(record["text"] for record in hit.metadata.get("source_events", []))
        for hit in hydrated], member_positions=member_positions)
    locating_score = {id(hit): score for hit, score in zip(hydrated, locating_scores)}
    fallback_ranks = {id(hit): index + 1 for index, hit in enumerate(sorted(
        hydrated, key=lambda h: (-h.score, -locating_score[id(h)], h.chunk_id)))}
    for hit in hydrated:
        channels = hit.metadata.get("retrieval_channel_ranks") or {}
        ranks = [value for value in channels.values() if type(value) is int and value > 0]
        pooled_rank = hit.metadata.get("retrieval_rank")
        fallback_rank = (pooled_rank if type(pooled_rank) is int and pooled_rank > 0
                         else fallback_ranks[id(hit)])
        rank = min(ranks, default=fallback_rank)
        ranked.append((rank, 0, hit.chunk_id, hit))
    ranked.extend((rank, 1, hit.chunk_id, hit) for rank, (_, hit) in enumerate(candidates, 1))
    ranked.sort(key=lambda item: item[:3])
    source_turns = {**resolved_turns, **{turn.turn_id: turn for turn in current_turns},
                    **{turn.turn_id: turn for turn in (context_turns or [])}}
    constructed = close_evidence_bundles([item[3] for item in ranked], list(source_turns.values()),
        session_id=session_id, blocked_events=blocked_events, budget=budget, query=query, plan=plan,
        authorized_quotes=authorized_quotes,
        source_cost=source_cost, source_id=max_sources, member_positions=member_positions)
    # Use the best rank already earned in a retrieval channel. The local
    # word match located the readable span; it must not demote a semantically
    # retrieved answer just because the original uses different vocabulary.
    reading_candidates = [_ReadingCandidate(hit, original[2], position, original[0])
                          for position, (original, hit) in enumerate(zip(ranked, constructed), 1)]
    # A complete source unit and its adjacent discourse context are alternate
    # representations of one candidate, not extra recall paths or model calls.
    if plan.answer_contract:
        compact = close_evidence_bundles([item[3] for item in ranked], list(source_turns.values()),
            session_id=session_id, blocked_events=blocked_events, budget=budget, query=query, plan=plan,
            authorized_quotes=authorized_quotes, source_cost=source_cost, source_id=max_sources,
            member_positions=member_positions, include_neighbors=False)
        for candidate, hit in zip(list(reading_candidates), compact):
            if source_payload(hit) != source_payload(candidate.hit):
                reading_candidates.append(_ReadingCandidate(hit, candidate.parent_id, candidate.rank,
                                                            candidate.upstream_rank, "compact"))
    # Identical retrieval views have one candidate identity and one receipt.
    unique_candidates = {}
    for candidate in reading_candidates:
        unique_candidates.setdefault(candidate.key, candidate)
    reading_candidates = list(unique_candidates.values())
    constructed = [candidate.hit for candidate in reading_candidates]
    unit_cache = {id(hit): _request_units(hit, plan) for hit in constructed}
    field_cache = {id(hit): _request_relation_units(hit, plan) for hit in constructed}
    binding_cache = {id(hit): {("member", *unit) for unit in unit_cache[id(hit)]}
                     or {("field", *relation) for relation in field_cache[id(hit)]}
                     for hit in constructed}
    trace.binding_units_available = len(set().union(*unit_cache.values())) if unit_cache else 0
    trace.request_fields_available = len(set().union(*field_cache.values())) if field_cache else 0
    features, weights = _reading_support(reading_candidates, query, plan)
    selected_candidates = []
    selected, used = [], 0
    covered = {}
    decisions = []
    pending = list(reading_candidates)
    costs = {}

    def cost(candidate, source_id):
        key = candidate.key, source_id
        if key not in costs:
            costs[key] = (_cost(candidate.hit) if source_cost is None
                          else source_cost(candidate.hit, source_id))
        return costs[key]

    def marginal(candidate):
        return sum(weight * max(0.0, value - covered.get(key, 0.0))
                   for key, value in features[candidate.key].items()
                   if (weight := weights.get(key, 0.0)))

    while pending:
        # Once request signals saturate, the same selector falls back to the
        # original cross-view rank. Unknown/paraphrased evidence stays eligible.
        candidate = min(pending, key=lambda c: (
            -marginal(c) / max(1, cost(c, len(selected) + 1)),
            -_query_rank_utility(c) / max(1, cost(c, len(selected) + 1)) if weights else 0,
            c.rank, c.variant != "context", c.key))
        pending.remove(candidate)
        hit = candidate.hit
        records = hit.metadata.get("source_events", [])
        entry_cost = cost(candidate, len(selected) + 1)
        gain = marginal(candidate)
        reason = None
        if any(c.parent_id == candidate.parent_id for c in selected_candidates):
            reason = "alternate_reading_unit"
        elif records and _records_already_visible(records, selected, source_turns):
            reason = "source_intervals_already_visible"
        elif len(selected) >= max_sources:
            reason = "source_limit"
        elif used + entry_cost > budget:
            reason = "serialized_budget"
        if reason is None:
            selected_candidates.append(candidate)
            selected.append(hit)
            used += entry_cost
            for key, value in features[candidate.key].items():
                covered[key] = max(covered.get(key, 0.0), value)
        decisions.append({"candidate_id": hit.chunk_id, "parent_id": candidate.parent_id,
                          "variant": candidate.variant, "rank": candidate.rank,
                          "upstream_rank": candidate.upstream_rank,
                          "cost": entry_cost, "used_after": used, "selected": reason is None,
                          "estimated_coverage_gain": gain,
                          "deferred_known_unit": bool(binding_cache[id(hit)]) and gain == 0,
                          "reason": reason or "selected"})
    # One bounded local exchange can recover an expensive joint premise. It
    # must improve the same set objective, preserve source permissions, and
    # re-cost every entry at its final ID; it never triggers another retrieval.
    current_value = _coverage_value(selected_candidates, features, weights)
    current_relevance = sum(_query_rank_utility(c) / max(1, cost(c, i))
                            for i, c in enumerate(selected_candidates, 1)) if weights else 0.0
    accepted_keys = {c.key for c in selected_candidates}
    rejected = [c for c in reading_candidates if c.key not in accepted_keys]
    best = None
    # The selected set stays fixed during this bounded exchange search.
    removable = sorted(range(len(selected_candidates)), key=lambda i: (
        current_value - _coverage_value(selected_candidates[:i] + selected_candidates[i+1:],
                                        features, weights), -selected_candidates[i].rank))[:2]
    removals = [(i,) for i in removable]
    if len(removable) == 2:
        removals.append(tuple(removable))
    for candidate in rejected:
        if not selected_candidates or _records_already_visible(
                candidate.hit.metadata.get("source_events", []), selected, source_turns):
            continue
        same_parent = [i for i, c in enumerate(selected_candidates) if c.parent_id == candidate.parent_id]
        exchanges = removals
        if same_parent:
            # Upgrading a reading unit replaces its own alternative first;
            # otherwise mutual exclusion can accidentally lock the short form.
            required = same_parent[0]
            exchanges = [(required,), *[(required, i) for i in removable if i != required]]
        for remove in exchanges:
            proposal = [c for i, c in enumerate(selected_candidates) if i not in remove] + [candidate]
            if len({c.parent_id for c in proposal}) != len(proposal):
                continue
            value = _coverage_value(proposal, features, weights)
            relevance = sum(_query_rank_utility(c) / max(1, cost(c, i))
                            for i, c in enumerate(proposal, 1)) if weights else 0.0
            if value < current_value - 1e-12 or (abs(value-current_value) <= 1e-12
                                                 and relevance <= current_relevance + 1e-12):
                continue
            total = sum(cost(c, i) for i, c in enumerate(proposal, 1))
            if len(proposal) <= max_sources and total <= budget:
                score = (value, relevance, -total, -candidate.rank)
                if best is None or score > best[0]:
                    best = (score, proposal, total)
    if best is not None:
        _, selected_candidates, used = best
        selected = [c.hit for c in selected_candidates]
    final_keys = {c.key for c in selected_candidates}
    decisions_by_key = {(d["parent_id"], d["variant"], d["rank"]): d for d in decisions}
    for candidate in reading_candidates:
        decision = decisions_by_key[(candidate.parent_id, candidate.variant, candidate.rank)]
        final = candidate.key in final_keys
        if final != decision["selected"]:
            decision["reason"] = "selected_by_exchange" if final else "replaced_by_coverage"
        decision["selected"] = final
    for source_id, candidate in enumerate(selected_candidates, 1):
        candidate.hit.metadata["selection_rank"] = candidate.rank
        decision = decisions_by_key[(candidate.parent_id, candidate.variant, candidate.rank)]
        decision["final_source_id"] = source_id
        decision["final_cost"] = cost(candidate, source_id)
    trace.selection_decisions = decisions
    covered_units = set().union(*(unit_cache[id(h)] for h in selected)) if selected else set()
    covered_fields = set().union(*(field_cache[id(h)] for h in selected)) if selected else set()
    raw_count = sum(h.metadata.get("origin") == "raw_fallback" for h in selected)
    trace.binding_units_selected = len(covered_units)
    trace.request_fields_selected = len(covered_fields)
    trace.raw_sources_selected = raw_count
    trace.indexed_sources_selected = sum(hit.metadata.get("origin") == "indexed_sweep" for hit in selected)
    trace.final_sources = len(selected)
    trace.final_missing_info = _missing(selected, plan.required_info)
    trace.temporal_derivations = sum(
        len(record["time_derivations"])
        for hit in selected
        for record in hit.metadata.get("source_events", [])
    )
    trace.context_tokens_estimate = used
    trace.context_serialized_chars = used
    trace.elapsed_ms = (time.perf_counter() - started) * 1000
    for hit in selected:
        hit.metadata.pop("retrieval_hint", None)
    return GroundedResult(selected, trace)


ANSWER_REASONING_CONTRACT = """
Evidence reasoning contract:
- source_events are original utterances with speaker and observation date. A
  compressed memory is an index; check its claim against the original utterance.
- Preserve the speaker/subject distinction. First-person statements refer to the
  recorded speaker; another person's related fact does not answer this question.
- Use source-anchored time_derivations, calendar arithmetic and explicit counting.
  Keep year/month/week precision; do not invent a day or equate recording time,
  event time and a fact's validity. A duration alone does not prove an event date.
- Answer supported parts directly. Only abstain on indispensable missing facts;
  lexical missing_info is a retrieval heuristic, not a proof of missing evidence.
  The planner may propose unnecessary slots: missing optional background must
  not veto an answer to the user's actual question when that answer is supported.
- For explicit likely/inference questions, give a qualified inference with its
  premises. Do not silently promote an inference, plan or assistant suggestion
  into an observed fact. Ordinary background knowledge is not personal memory.
- If evidence conflicts, state the conflict. Never infer an absent actor or event
  just because an alternative choice contains familiar words.
- Reply concisely in the requested language and attach existing source numbers.
"""

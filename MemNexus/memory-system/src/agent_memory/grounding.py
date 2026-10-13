"""Provenance joins and one bounded raw-evidence retrieval round.

Compressed memories are retrieval hints. Source utterances carry dates and
speakers; calendar deductions retain their source and precision. Candidate
selection uses BM25 plus greedy uncovered-query-term gain and redundancy
penalties. Scores are retrieval heuristics, never truth probabilities.
"""

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta, timezone

from .evidence_policy import question_only
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


def terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.casefold())) - STOP
    for part in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(part[i : i + 2] for i in range(max(1, len(part) - 1)))
    return words


def speaker_of(event) -> str | None:
    if event.speaker:
        return event.speaker
    # Compatibility with imported transcripts; record this as a text prefix,
    # not a model-inferred entity identity.
    match = re.match(r"^([A-Z][\w -]{0,60}):\s", event.content)
    return match.group(1) if match else None


def source_payload(hit: Hit) -> dict:
    # Full provenance stays in the receipt. The model gets the source once,
    # without repeated IDs, offsets and duplicate calendar anchors.
    source_events = []
    for event in hit.metadata.get("source_events", []):
        record = {key: event[key] for key in ("speaker", "role", "occurred_at", "text", "context_only")}
        if event["time_derivations"]:
            record["time_derivations"] = [
                {key: value[key] for key in ("expression", "start", "end_exclusive", "precision", "rule")}
                for value in event["time_derivations"]
            ]
        source_events.append(record)
    return {
        "claim": hit.content,
        "attributes": {
            key: hit.metadata.get(key)
            for key in (
                "status",
                "assertion",
                "valid_from",
                "valid_to",
                "local_override",
                "origin",
                "attribution_conflict",
            )
            if hit.metadata.get(key) is not None
        },
        "source_events": source_events,
        "provenance": hit.metadata.get("source_support_status", "unresolved"),
    }


def _cost(hit: Hit) -> int:
    # Match the runtime's existing conservative character proxy; actual
    # provider token counts remain separately measured by the provider.
    return len(json.dumps(source_payload(hit), ensure_ascii=False)) + 16


def close_evidence_bundles(sources, turns, *, session_id, blocked_events, budget):
    """Attach local discourse context without crossing session/time/lifecycle boundaries.

    Round-robin nearest neighbours prevents one hit consuming the whole budget.
    Context is evidence to reason over, never an asserted identity/event edge.
    """
    sources = [hit.model_copy(deep=True) for hit in sources]
    ordered = [(turn, i) for turn in turns if turn.session_id == session_id for i in range(len(turn.events))]
    positions = {(turn.turn_id, i): n for n, (turn, i) in enumerate(ordered)}
    seen = {(e["turn_id"], e["event_index"]) for hit in sources for e in hit.metadata.get("source_events", [])}
    seeds = [(hit, positions[(e["turn_id"], e["event_index"])]) for hit in sources
             for e in hit.metadata.get("source_events", []) if (e["turn_id"], e["event_index"]) in positions]
    used = sum(_cost(hit) for hit in sources)
    for offset in (-1, 1, -2, 2, 3):
        for hit, position in seeds:
            n = position + offset
            if not 0 <= n < len(ordered):
                continue
            turn, i = ordered[n]
            identity = turn.turn_id, i
            anchor_turn, anchor_i = ordered[position]
            event, anchor = turn.events[i], anchor_turn.events[anchor_i]
            if identity in seen or identity in blocked_events:
                continue
            if event.occurred_at is None or anchor.occurred_at is None:
                continue
            if abs((event.occurred_at - anchor.occurred_at).total_seconds()) > 1800:
                continue
            record = _source_event(turn, i, event.content[:700], 0)
            previous = _cost(hit)
            hit.metadata["source_events"].append(record)
            cost = _cost(hit) - previous
            if used + cost > budget:
                hit.metadata["source_events"].pop()
                continue
            used += cost
            seen.add(identity)
    for hit in sources:
        hit.metadata["source_events"].sort(key=lambda e: positions.get((e["turn_id"], e["event_index"]), -1))
    return sources


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


def _source_event(turn: Turn, index: int, text: str, start: int) -> dict:
    event = turn.events[index]
    anchor = event.occurred_at.astimezone(timezone(timedelta(minutes=event.source_utc_offset_minutes)))
    return {
        "turn_id": turn.turn_id,
        "event_index": index,
        "speaker": speaker_of(event),
        "role": event.role,
        "occurred_at": anchor.isoformat(),
        "text": text,
        "start": start,
        "end": start + len(text),
        "context_only": event.role == "assistant" or question_only(event.content),
        "time_derivations": [item.dump() for item in derive_times(text, anchor)],
    }


def _chunks(text: str, limit: int = 700) -> list[tuple[int, str]]:
    # Exact non-overlapping character spans. Large user messages cannot hide
    # content after the first truncation boundary. No paraphrase is created.
    return [(offset, text[offset : offset + limit]) for offset in range(0, len(text), limit)]


def _bm25(query: str, documents: list[str]) -> list[float]:
    bags = [Counter(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", doc.casefold())) for doc in documents]
    query_terms = terms(query)
    # Add bigrams to support Chinese retrieval without model dependencies.
    for doc, bag in zip(documents, bags):
        bag.update(term for term in terms(doc) if re.search(r"[\u4e00-\u9fff]", term))
    frequencies = Counter(term for bag in bags for term in bag)
    lengths = [sum(bag.values()) for bag in bags]
    mean = max(1, sum(lengths) / max(1, len(lengths)))
    scores = []
    for bag, length in zip(bags, lengths):
        score = 0.0
        for token in query_terms & bag.keys():
            idf = math.log(1 + (len(bags) - frequencies[token] + 0.5) / (frequencies[token] + 0.5))
            tf = bag[token]
            score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / mean))
        scores.append(score)
    return scores


@dataclass
class GroundedResult:
    sources: list[Hit]
    trace: GroundingTrace


def collect_raw_records(query, plan, turns, session_id, blocked_events):
    records = []
    for turn in turns:
        if turn.session_id != session_id:
            continue
        for index, event in enumerate(turn.events):
            if (turn.turn_id, index) in blocked_events or question_only(event.content):
                continue
            if plan.as_of and event.occurred_at > plan.as_of:
                continue
            if event.role == "assistant" and not re.search(
                r"recommend|suggest|assistant|you (?:say|said)|建议|推荐|你说", query, re.IGNORECASE
            ):
                continue
            for offset, text in _chunks(event.content):
                records.append((_source_event(turn, index, text, offset), text))
    return records


def record_key(record):
    return f"{record['turn_id']}:{record['event_index']}:{record['start']}"


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
    raw_limit: int = 3,
    budget: int = 5000,
    candidate_limit: int = 120,
    semantic_scores: dict[str, float] | None = None,
) -> GroundedResult:
    started = time.perf_counter()
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
        source_events = []
        for reference in hit.metadata.get("evidence", []):
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
            ):
                trace.rejected_source_refs += 1
                continue
            event = turn.events[index]
            offset = event.content.index(quote)
            # Full small utterances retain subject/coreference context only in
            # the active session. Cross-session evidence never expands scope.
            # A consumed candidate can share an exact quote with its active
            # successor. The already-eligible memory may expose that quote,
            # but quarantined events never expand to their surrounding text.
            if turn.session_id == session_id and len(event.content) <= 700 and key not in blocked_events:
                text, offset = event.content, 0
            else:
                text = quote[:700]
            record = _source_event(turn, index, text, offset)
            if not any((x["turn_id"], x["event_index"]) == key for x in source_events):
                source_events.append(record)
        # Bound evidence bundles without pretending omitted spans are verified.
        source_events = source_events[:3]
        item.metadata["source_events"] = source_events
        item.metadata["source_support_status"] = "resolved" if source_events else "unresolved"
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
        queries = [query, *trace.initial_missing_info[:3]]
        trace.supplemental_queries = queries
        records = collect_raw_records(query, plan, current_turns, session_id, blocked_events)
        scores = _bm25(" ".join(queries), [text for _, text in records])
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
                            "scope_id": session_id,
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

    # All original selected sources compete with the bounded raw candidates.
    # Weighted coverage gain favours missing query facets; Jaccard redundancy
    # and duplicate source-event penalties discourage repeated paraphrases.
    selected, used, raw_count = [], 0, 0
    covered: set[str] = set()
    seen_events: set[tuple] = set()
    covered_entities: set[str] = set()
    covered_entity_fields: set[str] = set()
    covered_predicates: set[str] = set()
    requested_entities = {name.casefold() for name in plan.query_entities}
    wanted = terms(query + " " + " ".join(plan.required_info))
    focus_terms = normalized_terms(query) - {
        term for entity in requested_entities for term in normalized_terms(entity)
    }

    def fields_and_entities(hit: Hit) -> tuple[set[str], set[str]]:
        subject = str(
            hit.metadata.get("subject")
            or " ".join(str(record.get("speaker") or "") for record in hit.metadata.get("source_events", []))
        ).casefold()
        entities = {
            entity
            for entity in requested_entities
            if re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", subject)
        }
        field = (
            normalized_terms(f"{hit.metadata.get('predicate') or ''} {hit.metadata.get('value') or ''}")
            & focus_terms
        )
        if not field:
            field = normalized_terms(hit.content) & focus_terms
        return entities, field

    if plan.recall_policy == "recall_first":
        # Reserve a bounded part of the prompt for the most relevant first-pass
        # evidence, then fill distinct subject/field slots from indexed facts.
        baseline = [hit for hit in hydrated if hit.metadata.get("origin") != "indexed_sweep"]
        baseline.sort(
            key=lambda hit: (
                -(
                    3 * len(fields_and_entities(hit)[0])
                    + 2 * len(fields_and_entities(hit)[1])
                    + len(normalized_terms(hit.content) & focus_terms)
                ),
                -hit.score,
                hit.chunk_id,
            )
        )
        seed_budget = min(budget, int(budget * 0.58))
        for hit in baseline:
            cost = _cost(hit)
            if len(selected) >= 8 or used + cost > seed_budget:
                continue
            selected.append(hit)
            used += cost
            seen_events.update(
                (record["turn_id"], record["event_index"]) for record in hit.metadata.get("source_events", [])
            )
            covered.update(terms(hit.content) & wanted)
            predicate = str(hit.metadata.get("predicate") or "").casefold()
            if predicate:
                covered_predicates.add(predicate)
            subject = str(hit.metadata.get("subject") or "").casefold()
            covered_entities.update(
                entity
                for entity in requested_entities
                if re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", subject)
            )
            hit_entities, fields = fields_and_entities(hit)
            if fields:
                covered_entity_fields.update(hit_entities)
    selected_ids = {id(hit) for hit in selected}
    pool = [(1.0 + hit.score, hit) for hit in hydrated if id(hit) not in selected_ids] + candidates
    while pool and len(selected) < max_sources:
        choices = []
        for index, (relevance, hit) in enumerate(pool):
            raw = hit.metadata.get("origin") == "raw_fallback"
            cost = _cost(hit)
            if used + cost > budget or (raw and raw_count >= raw_limit):
                continue
            content_terms = terms(hit.content)
            events = {(x["turn_id"], x["event_index"]) for x in hit.metadata.get("source_events", [])}
            gain = len((content_terms & wanted) - covered) / max(1, len(wanted))
            redundant = max(
                (
                    len(content_terms & terms(x.content)) / max(1, len(content_terms | terms(x.content)))
                    for x in selected
                ),
                default=0,
            )
            utility = relevance / (1 + relevance) + 2 * gain - 0.6 * redundant
            if plan.recall_policy == "recall_first":
                origin = hit.metadata.get("origin")
                if origin == "indexed_sweep":
                    utility += 0.2
                elif origin == "raw_fallback":
                    # The indexed fact already carries a validated source quote.
                    # Spend the bounded context on distinct facts before long
                    # raw utterances unless the latter cover new query facets.
                    utility -= 0.2
                subject_text = str(
                    hit.metadata.get("subject")
                    or " ".join(
                        str(record.get("speaker") or "") for record in hit.metadata.get("source_events", [])
                    )
                ).casefold()
                hit_entities = {
                    entity
                    for entity in requested_entities
                    if re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", subject_text)
                }
                utility += 1.5 * len(hit_entities - covered_entities) / max(1, len(requested_entities))
                field_entities, fields = fields_and_entities(hit)
                if fields:
                    utility += 0.9 * len(fields) / max(1, len(focus_terms))
                    utility += 1.1 * len(field_entities - covered_entity_fields)
                predicate = str(hit.metadata.get("predicate") or "").casefold()
                if plan.query_type in {"set", "count"} and predicate and predicate not in covered_predicates:
                    utility += 0.35
            if events and events <= seen_events:
                utility -= 0.6
            choices.append((utility, -cost, -index, index, hit, events))
        if not choices:
            break
        utility, _, _, index, hit, events = max(choices, key=lambda x: x[:3])
        if utility <= 0:
            break
        pool.pop(index)
        selected.append(hit)
        used += _cost(hit)
        raw_count += hit.metadata.get("origin") == "raw_fallback"
        seen_events.update(events)
        covered.update(terms(hit.content) & wanted)
        subject_text = str(
            hit.metadata.get("subject")
            or " ".join(str(record.get("speaker") or "") for record in hit.metadata.get("source_events", []))
        ).casefold()
        covered_entities.update(
            entity
            for entity in requested_entities
            if re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", subject_text)
        )
        hit_entities, fields = fields_and_entities(hit)
        if fields:
            covered_entity_fields.update(hit_entities)
        predicate = str(hit.metadata.get("predicate") or "").casefold()
        if predicate:
            covered_predicates.add(predicate)
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
    trace.elapsed_ms = (time.perf_counter() - started) * 1000
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

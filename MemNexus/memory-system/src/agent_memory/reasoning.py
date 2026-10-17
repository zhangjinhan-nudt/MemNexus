"""Question routing, event-graph reconstruction, and typed evidence gates.

The module is deliberately provider-free. It turns already-authorized memory
hits and their source events into a small, inspectable graph. The graph is used
for recall policy, bounded deterministic deductions, and negative evidence
checks before a model is allowed to answer.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from .models import Hit, Memory, QueryPlan

QUESTION_WORDS = {
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "How",
    "Which",
    "Would",
    "Could",
    "Can",
    "Did",
    "Does",
    "Is",
    "Are",
    "Do",
    "In",
    "On",
    "The",
    "Not",
    "Choose",
}
CITY_COUNTRY = {
    "paris": "France",
    "london": "United Kingdom",
    "boston": "United States",
    "new york": "United States",
    "jasper": "Canada",
    "rockies": "Canada",
    "tokyo": "Japan",
}
TYPE_TERMS = {
    "dj": {"dj", "disc jockey"},
    "band": {"band", "group", "artist", "performer"},
    "movie": {"movie", "film", "romantic drama"},
    "novella": {"novella", "novel", "book", "writing"},
    "country": {"country", "france", "canada", "japan", "united kingdom"},
    "agency": {"agency", "adoption"},
}
MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        1,
    )
}
RETRIEVAL_STOP = {
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "are",
    "was",
    "were",
    "did",
    "does",
    "do",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "have",
    "has",
    "both",
    "some",
    "their",
    "from",
    "about",
    "and",
}


@dataclass(frozen=True)
class QuestionRoute:
    kind: str
    recall_policy: str
    entities: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    requires_composition: bool = False
    negative_gate: bool = False


@dataclass
class GraphNode:
    node_id: str
    subject: str | None
    predicate: str | None
    value: str | None
    text: str
    source_ids: list[str] = field(default_factory=list)
    speaker: str | None = None
    occurred_at: str | None = None
    event_type: str = "fact"
    state: str = "observed"


@dataclass
class EventGraph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    derivations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_hits(cls, hits: list[Hit]) -> EventGraph:
        graph = cls()
        seen: set[tuple[str, str]] = set()
        for hit_index, hit in enumerate(hits, 1):
            metadata = hit.metadata
            memory_id = str(metadata.get("memory_id", hit.chunk_id))
            source_events = metadata.get("source_events", [])
            subject = _clean(metadata.get("subject"))
            predicate = _clean(metadata.get("predicate"))
            value = _clean(metadata.get("value"))
            key = (memory_id, hit.content)
            if key not in seen:
                graph.nodes.append(
                    GraphNode(
                        node_id=f"memory:{memory_id}:{hit_index}",
                        subject=subject,
                        predicate=predicate,
                        value=value,
                        text=hit.content,
                        source_ids=[str(hit_index)],
                        event_type=_event_type(hit.content, predicate),
                        state=_state(hit.content, metadata.get("assertion")),
                    )
                )
                seen.add(key)
            for event_index, event in enumerate(source_events):
                text = str(event.get("text", "")).strip()
                if not text:
                    continue
                event_key = (str(event.get("turn_id", "")), text)
                if event_key in seen:
                    continue
                speaker = _clean(event.get("speaker"))
                graph.nodes.append(
                    GraphNode(
                        node_id=f"source:{event.get('turn_id', 'unknown')}:{event_index}",
                        subject=speaker,
                        predicate=None,
                        value=None,
                        text=text,
                        source_ids=[str(hit_index)],
                        speaker=speaker,
                        occurred_at=event.get("occurred_at"),
                        event_type=_event_type(text, None),
                        state=_state(text, None),
                    )
                )
                seen.add(event_key)
        graph._link_shared_entities()
        return graph

    def _link_shared_entities(self) -> None:
        by_entity: dict[str, list[str]] = defaultdict(list)
        for node in self.nodes:
            for entity in _entities(node.text):
                by_entity[entity.casefold()].append(node.node_id)
        for entity, node_ids in by_entity.items():
            unique = list(dict.fromkeys(node_ids))
            for left, right in pairwise(unique):
                self.edges.append((left, "shares_entity", right))

    def deterministic_derivations(self, query: str, route: QuestionRoute) -> list[dict[str, Any]]:
        derived: list[dict[str, Any]] = []
        if route.kind in {"temporal", "count", "relation", "causal", "inference", "set"}:
            for node in self.nodes:
                for location, country in CITY_COUNTRY.items():
                    if re.search(r"(?<!\w)" + re.escape(location) + r"(?!\w)", node.text, re.IGNORECASE):
                        derived.append(
                            {
                                "kind": "bounded_geography",
                                "text": f"{location.title()} is in {country}.",
                                "source_ids": node.source_ids,
                                "precision": "fixed_knowledge_relation",
                            }
                        )
                for derivation in _time_derivations(node.text):
                    derived.append({"kind": "temporal", **derivation, "source_ids": node.source_ids})
        # Several memories and source records may describe the same event.
        # Counting graph nodes would fabricate an event count, so a numerical
        # deduction is emitted only by a future event identity resolver.
        self.derivations = _deduplicate_derivations(derived)
        return self.derivations

    def prompt_context(self, query: str, route: QuestionRoute, max_chars: int = 2600) -> str:
        derivations = self.deterministic_derivations(query, route)
        query_terms = _retrieval_terms(query) - {
            term for entity in route.entities for term in _retrieval_terms(entity)
        }
        requested = {entity.casefold() for entity in route.entities}

        def priority(node: GraphNode) -> tuple[int, int, int]:
            subject = (node.subject or "").casefold()
            related = int(
                any(re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", subject) for entity in requested)
            )
            field = len(_retrieval_terms(node.predicate or "") & query_terms)
            content = len(_retrieval_terms(node.text) & query_terms)
            return (related, field, content)

        ranked = sorted(self.nodes, key=priority, reverse=True)
        if route.kind in {"set", "relation", "count"}:
            distinct, repeated = [], []
            seen_fields = set()
            for node in ranked:
                field = ((node.subject or "").casefold(), (node.predicate or "").casefold())
                if node.predicate and field not in seen_fields:
                    distinct.append(node)
                    seen_fields.add(field)
                else:
                    repeated.append(node)
            ranked = distinct + repeated
        lines = [
            "STRUCTURED EVENT GRAPH (derived from authorized evidence; not independent proof):",
            (
                f"question_type={route.kind}; recall_policy={route.recall_policy}; "
                f"composition={route.requires_composition}"
            ),
        ]
        for node in ranked[:40]:
            attrs = ", ".join(
                part
                for part in (
                    f"subject={node.subject}" if node.subject else "",
                    f"predicate={node.predicate}" if node.predicate else "",
                    f"value={node.value[:120]}"
                    if node.value and node.value != node.text and len(node.value) <= 120
                    else "",
                    f"event_type={node.event_type}",
                    f"state={node.state}",
                    f"sources={','.join(node.source_ids)}",
                )
                if part
            )
            lines.append(f"- {attrs}: {node.text[:220]}")
            if len("\n".join(lines)) >= max_chars:
                break
        if derivations:
            lines.append("DETERMINISTIC DERIVATIONS:")
            for item in derivations[:20]:
                lines.append(f"- {item}")
        if route.constraints:
            lines.append("QUERY CONSTRAINTS: " + "; ".join(route.constraints))
        rendered = "\n".join(lines)
        return rendered[:max_chars]


@dataclass(frozen=True)
class NegativeGate:
    blocked: bool
    reason: str
    fallback_text: str = "Not mentioned in the conversation."
    supported_source_ids: tuple[str, ...] = ()


def classify_question(query: str, known_entities: list[str] | None = None) -> QuestionRoute:
    text = query.strip()
    lower = text.casefold()
    entities = _query_entities(text, known_entities or [])
    adversarial = bool(
        re.search(r"not mentioned|not in (?:the )?conversation|choose exactly one|select the correct", lower)
    )
    if adversarial:
        kind = "adversarial"
    elif re.search(
        r"\b(?:recommend|recommendation|suggest|suggestions?|advice|ideas?|inspiration)\b|建议|推荐|灵感",
        lower,
    ) and not re.search(r"\b(?:what did|what .* (?:recommend|suggest)|which .* (?:recommend|suggest))\b", lower):
        # Advice questions need a broad personal-context recall pass.  Treating
        # them as ordinary fact lookup over-ranks the nearest topical question
        # and loses the user's established preferences/ongoing activities.
        kind = "recommendation"
    elif re.search(r"how (?:many|long).*?(?:years?|months?|weeks?|days?|hours?)|相隔|多久", lower):
        kind = "temporal"
    elif re.search(r"\b(?:how many|number of|how often|times)\b|多少|几次|几个|几项", lower):
        kind = "count"
    elif re.search(
        r"when|what date|what year|which year|how long ago|how many years|哪年|何时|什么时候|多久|几月|日期", lower
    ):
        kind = "temporal"
    elif re.search(r"both|common|shared|between|relationship|共同|都|之间|关系", lower) or (
        len({entity.casefold() for entity in entities}) >= 2
        and re.search(r"\band\b|与|和", lower)
        and re.match(r"^(?:what|which|how|who)\b", lower)
    ):
        kind = "relation"
    elif re.search(r"why|reason|decide|choose|motivat|because|为何|为什么|原因|决定", lower):
        kind = "causal"
    elif re.search(
        r"would|likely|could|might|based on|what country|which country|可能|会不会|倾向|什么国家|哪个国家",
        lower,
    ):
        kind = "inference"
    elif re.search(
        r"names?|items?|which .*\?|what .* include|哪些|什么.*有|列表|所有|all"
        r"|^what\s+(?:are|were)\b|^what\s+[a-z]{4,}s\b|\bbesides\b",
        lower,
    ):
        kind = "set"
    else:
        kind = "fact"
    broad = kind != "fact"
    constraints = []
    for type_name, terms in TYPE_TERMS.items():
        if any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", lower) for term in terms):
            constraints.append(f"object_type={type_name}")
    years = re.findall(r"\b(?:19|20)\d{2}\b", lower)
    if years:
        constraints.append("year=" + ",".join(years))
    months = [name for name in MONTHS if re.search(r"\b" + name + r"\b", lower)]
    if months:
        constraints.append("month=" + ",".join(months))
    return QuestionRoute(
        kind=kind,
        recall_policy="recall_first"
        if kind in {"set", "count", "relation", "causal", "inference", "recommendation"}
        else "focused",
        entities=tuple(entities),
        constraints=tuple(constraints),
        requires_composition=broad,
        negative_gate=adversarial,
    )


def apply_route(plan: QueryPlan, route: QuestionRoute) -> QueryPlan:
    if plan.collection_query is not None:
        spec = plan.collection_query
        return plan.model_copy(update={
            "query_type": "temporal" if route.kind == "temporal" else "count" if spec.operation == "count" else "set",
            "recall_policy": "recall_first", "query_entities": spec.subjects,
            "required_info": plan.required_info or [f"{s}: {spec.predicate}" for s in spec.subjects],
            "depth": max(8, plan.depth),
        })
    depth = max(plan.depth, 8 if route.recall_policy == "recall_first" else plan.depth)
    return plan.model_copy(
        update={
            "query_type": route.kind,
            "recall_policy": route.recall_policy,
            "query_entities": list(route.entities),
            "query_constraints": list(route.constraints),
            "depth": min(20, depth),
        }
    )


def indexed_memory_hits(
    query: str,
    route: QuestionRoute,
    memories: list[Memory],
    selected_hits: list[Hit],
    *,
    limit: int = 32,
) -> list[Hit]:
    """Retrieve additional authorized facts for compositional questions.

    A caller must apply owner, scope, lifecycle and time eligibility before
    passing memories. This function only ranks; every quote is rechecked by
    the existing grounding layer before reaching the answer model.
    """
    if limit <= 0:
        return []
    already = {hit.metadata.get("memory_id") for hit in selected_hits}
    entities = tuple(dict.fromkeys(entity.casefold() for entity in route.entities))
    query_terms = _retrieval_terms(query) - {term for entity in entities for term in _retrieval_terms(entity)}
    entity_tokens = {token for entity in entities for token in _retrieval_terms(entity)}
    seed_pairs = [
        (
            str(hit.metadata.get("subject") or "").casefold(),
            {pair for pair in _retrieval_pairs(hit.content) if not set(pair) & entity_tokens},
        )
        for hit in selected_hits
    ]
    pair_frequency = Counter(pair for memory in memories for pair in _retrieval_pairs(memory.content))
    scored = []
    for memory in memories:
        if memory.memory_id in already:
            continue
        text = f"{memory.subject} {memory.predicate} {memory.content} {memory.value}"
        terms = _retrieval_terms(text)
        subject_text = memory.subject.casefold()
        related = [
            entity
            for entity in entities
            if re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", subject_text)
        ]
        if entities and not related:
            continue
        overlap = len(query_terms & terms)
        useful_seed_pairs = (
            set().union(
                *(
                    pairs
                    for subject, pairs in seed_pairs
                    if len(entities) == 1
                    or not any(
                        re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", subject) for entity in related
                    )
                )
            )
            if seed_pairs
            else set()
        )
        bridge = sum(
            math.log1p(len(memories) / (1 + pair_frequency[pair]))
            for pair in _retrieval_pairs(memory.content) & useful_seed_pairs
            if not set(pair) & entity_tokens
        )
        if not related and overlap == 0 and bridge == 0:
            continue
        score = 4 * overlap + 2 * bridge + 2 * len(related)
        scored.append((score, memory, set(related)))
    scored.sort(key=lambda row: (-row[0], row[1].memory_id))

    chosen = []
    seen = set()
    for entity in entities:
        per_entity = 0
        for score, memory, _ in (row for row in scored if entity in row[2]):
            if memory.memory_id not in seen:
                chosen.append((score, memory))
                seen.add(memory.memory_id)
                per_entity += 1
            if per_entity >= 6:
                break
    for score, memory, _ in scored:
        if len(chosen) >= limit:
            break
        if memory.memory_id not in seen:
            chosen.append((score, memory))
            seen.add(memory.memory_id)
    hits = []
    for score, memory in chosen[:limit]:
        hits.append(
            Hit(
                content=memory.content,
                score=min(0.99, score / (score + 10)),
                source_file=f"[memory:{memory.memory_id}]",
                chunk_id=f"{memory.memory_id}:{memory.version}",
                metadata={
                    "origin": "indexed_sweep",
                    "memory_id": memory.memory_id,
                    "version": memory.version,
                    "revision": memory.revision,
                    "subject": memory.subject,
                    "predicate": memory.predicate,
                    "value": memory.value,
                    "status": memory.status,
                    "assertion": memory.assertion,
                    "scope_type": memory.scope_type,
                    "scope_id": memory.scope_id,
                    "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
                    "valid_to": memory.valid_to.isoformat() if memory.valid_to else None,
                    "evidence": [item.model_dump(mode="json") for item in memory.evidence],
                },
            )
        )
    return hits


def _retrieval_terms(text: str) -> set[str]:
    result = set()
    for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.casefold()):
        if token in RETRIEVAL_STOP:
            continue
        token = token.replace("favourite", "favorite")
        if re.fullmatch(r"[a-z]+", token) and len(token) > 4:
            if token.endswith("ies"):
                token = token[:-3] + "y"
            elif token.endswith("ing") and len(token) > 6:
                token = token[:-3]
            elif token.endswith("ed") and len(token) > 5:
                token = token[:-2]
            elif token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
        result.add(token)
    return result


def _retrieval_pairs(text: str) -> set[tuple[str, str]]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", text.casefold()) if token not in RETRIEVAL_STOP]
    return set(pairwise(tokens))


def typed_negative_gate(query: str, route: QuestionRoute, graph: EventGraph) -> NegativeGate:
    if not route.negative_gate:
        return NegativeGate(False, "not_an_adversarial_question")
    target_entities = {item.casefold() for item in route.entities}
    type_requirements = [
        item.split("=", 1)[1] for item in route.constraints if item.startswith("object_type=")
    ]
    years = [item.split("=", 1)[1] for item in route.constraints if item.startswith("year=")]
    months = [item.split("=", 1)[1] for item in route.constraints if item.startswith("month=")]
    supported: list[str] = []
    for index, node in enumerate(graph.nodes, 1):
        text = node.text.casefold()
        subject_ok = not target_entities or any(
            entity in text or (node.subject and entity == node.subject.casefold())
            for entity in target_entities
        )
        type_ok = all(
            any(term in text for term in TYPE_TERMS.get(type_name, {type_name}))
            for type_name in type_requirements
        )
        year_ok = not years or any(
            year in text or (node.occurred_at and year in str(node.occurred_at)) for year in years
        )
        month_ok = not months or any(
            month in text or (node.occurred_at and _month_name(node.occurred_at) == month) for month in months
        )
        if subject_ok and type_ok and year_ok and month_ok:
            supported.append(str(index))
    if supported:
        return NegativeGate(False, "typed_evidence_supported", supported_source_ids=tuple(supported))
    constraints = ", ".join(route.constraints) or "no typed candidate"
    # An incomplete retrieved graph cannot establish absence in the conversation.
    return NegativeGate(False, f"not_found_in_examined_evidence; subject/type/time: {constraints}")


def _clean(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _entities(text: str) -> set[str]:
    return {match for match in re.findall(r"\b[A-Z][a-z]+\b", text) if match not in QUESTION_WORDS}


def _query_entities(query: str, known: list[str]) -> list[str]:
    explicit = [
        entity
        for entity in known
        if entity and re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", query, re.IGNORECASE)
    ]
    if explicit:
        return list(dict.fromkeys(explicit))
    return sorted(_entities(query))


def _event_type(text: str, predicate: str | None) -> str:
    lower = (text + " " + (predicate or "")).casefold()
    for label, pattern in (
        ("purchase", r"buy|bought|purchase|bought"),
        ("visit", r"visit|went to|travel|trip|road trip|camp"),
        ("performance", r"perform|concert|festival|band|music"),
        ("adoption", r"adopt|agency"),
        ("education", r"study|school|degree|education|career"),
        ("hobby", r"hobby|painting|reading|dance|writing"),
        ("decision", r"decide|choose|start|plan|want|consider"),
    ):
        if re.search(pattern, lower):
            return label
    return "fact"


def _state(text: str, assertion: str | None) -> str:
    lower = text.casefold()
    if assertion in {"planned", "hypothetical"} or re.search(
        r"plan|planning|thinking about|consider|would like|want to", lower
    ):
        return "planned"
    if re.search(r"started|began|took up|launched|bought|went|visited|adopted|painted|performed", lower):
        return "observed"
    return assertion or "observed"


def _time_derivations(text: str) -> list[dict[str, str]]:
    # The full calendar derivation remains in temporal.py; this ledger only
    # exposes explicit relative expressions for the answer model.
    expressions = re.findall(
        r"\b(?:last|next|this)\s+(?:year|month|week(?:end)?|Saturday|Sunday|Monday|Tuesday|Wednesday|Thursday|Friday)\b",
        text,
        re.IGNORECASE,
    )
    return [{"kind": "relative_time", "expression": value} for value in expressions]


def _month_name(value: str) -> str:
    match = re.search(r"-(\d{2})-", str(value))
    if not match:
        return ""
    month = int(match.group(1))
    return next((name for name, index in MONTHS.items() if index == month), "")


def _deduplicate_derivations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for item in items:
        key = repr(sorted(item.items()))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

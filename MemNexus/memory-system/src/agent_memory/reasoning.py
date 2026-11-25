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
from .temporal import (
    anchored_dates as _anchored_dates,
)
from .temporal import (
    dedup_dates as _dedup_dates,  # noqa: F401 - compatibility import for legacy callers
)
from .temporal import (
    difference_allowed as _difference_allowed,  # noqa: F401 - compatibility import
)
from .temporal import (
    duration_expressions as _duration_expressions,
)
from .temporal import (
    interval_bounds as _interval_bounds,  # noqa: F401 - compatibility import
)
from .temporal import (
    intervals_order,  # noqa: F401 - compatibility import
    offset_intervals,
)
from .temporal import (
    relative_time_expressions as _time_derivations,
)

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
        # Two key spaces share one set: a memory node and its own source
        # event can carry the same id and text, and without a namespace
        # prefix the second would be dropped as a duplicate.
        seen: set[tuple[str, ...]] = set()
        for hit_index, hit in enumerate(hits, 1):
            metadata = hit.metadata
            memory_id = str(metadata.get("memory_id", hit.chunk_id))
            source_events = metadata.get("source_events", [])
            subject = _clean(metadata.get("subject"))
            predicate = _clean(metadata.get("predicate"))
            value = _clean(metadata.get("value"))
            key = ("memory", memory_id, hit.content)
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
                event_key = ("source", str(event.get("turn_id", "")), text)
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
                        state=_state(text, None, event.get("role")),
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

    def deterministic_derivations(
        self, query: str, route: QuestionRoute, temporal_execution: bool = False
    ) -> list[dict[str, Any]]:
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
        if temporal_execution:
            # The order decision carries its own preconditions, so it must not be
            # gated by a separate list of question phrasings: "which item did I
            # purchase first" is the same question as "which event happened
            # first", and a hand-maintained pattern list only guarantees that
            # some phrasing is missed.
            order = decide_order_by_offset(query, self.nodes)
            if order is not None:
                derived.append(order)
        if temporal_execution and (route.kind == "temporal" or _ORDER_OR_DURATION_QUERY.search(query)):
            # Quoted durations and a date-sorted event ledger. Both are read
            # straight from the evidence: durations are the speaker's own
            # words, ordering is a sort over day-precision timestamps. The
            # model selects among them instead of doing calendar arithmetic.
            for node in self.nodes:
                for expression in _duration_expressions(node.text):
                    derived.append(
                        {
                            "kind": "duration_quoted",
                            "expression": expression,
                            "source_ids": node.source_ids,
                        }
                    )
            dated = []
            for node in self.nodes:
                for stamp, expression, precision, kind in _anchored_dates(
                    node.text, node.occurred_at
                ):
                    dated.append((stamp, expression, precision, kind, node))
            dated.sort(key=lambda item: (item[0], item[4].node_id))
            if len(dated) >= 2:
                derived.append(
                    {
                        "kind": "anchored_date_ledger",
                        "order": [
                            f"{stamp} ({expression}"
                            f"{_PRECISION_LABEL.get(precision, '')}"
                            f"{'' if kind == 'point' else ', span start' if kind.startswith('interval') else ''}"
                            f"): {node.text[:80]}"
                            for stamp, expression, precision, kind, node in dated[:14]
                        ],
                        "source_ids": [sid for _, _, _, _, node in dated for sid in node.source_ids][:14],
                    }
                )
                # The ledger contains date mentions, not bound endpoints. Never
                # publish all pairwise differences as answer candidates: a unique
                # numerical difference still need not answer the user's question.
        # Several memories and source records may describe the same event.
        # Counting graph nodes would fabricate an event count, so a numerical
        # deduction is emitted only by a future event identity resolver.
        self.derivations = _deduplicate_derivations(derived)
        return self.derivations

    def prompt_context(
        self, query: str, route: QuestionRoute, max_chars: int = 2600, temporal_execution: bool = False
    ) -> str:
        derivations = self.deterministic_derivations(query, route, temporal_execution)
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
        # A conclusion drawn from the whole evidence set is worthless if it is
        # appended after forty node lines and then cut by max_chars. Every
        # derivation that bears on the question now goes to the head with a
        # reserved budget; the node list -- a sample, not a conclusion -- takes
        # what is left. Measured before the fix: no non-order derivation survived
        # in 103 of 103 LME rows or 426 of 426 LoCoMo rows, while the hoisted
        # order conclusion survived 4 of 4.
        lines.extend(self._derivation_lines(derivations, max_chars, query))
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
        if route.constraints:
            lines.append("QUERY CONSTRAINTS: " + "; ".join(route.constraints))
        rendered = "\n".join(lines)
        return rendered[:max_chars]

    def _derivation_lines(
        self, derivations: list[dict[str, Any]], max_chars: int, query: str = ""
    ) -> list[str]:
        """Render the question-bearing derivations, bounded, most decisive first.

        Order matters more than completeness here. A decision about the whole
        evidence set (order, a computed total, an elapsed span) is worth more to
        the reader than the twentieth node, so it is emitted first and capped so
        the node list cannot evict it. Each kind gets its own small allowance:
        without caps, one ledger of fourteen rows would consume the whole block
        and crowd out the single difference the question needs.
        """
        if not derivations:
            return []
        reserved = max(240, int(max_chars * _DERIVATION_BUDGET_SHARE))
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for item in derivations:
            by_kind.setdefault(str(item.get("kind")), []).append(item)

        out: list[str] = []

        def emit(text: str) -> None:
            if len("\n".join(out + [text])) > reserved:
                return
            out.append(text)

        for item in by_kind.get("offset_order", []):
            emit(f"ORDER IS DETERMINED: {item.get('why')}")
        for item in by_kind.get("date_differences", []):
            for entry in (item.get("differences") or [])[:_DIFFERENCE_LINES]:
                emit(f"- ELAPSED: {entry}")
        for item in by_kind.get("duration_quoted", [])[:_DURATION_ITEMS]:
            emit(f"- LENGTH STATED IN SOURCE: {item.get('expression')}")
        for item in by_kind.get("anchored_date_ledger", []):
            for entry in (item.get("order") or [])[:_LEDGER_LINES]:
                emit(f"- DATED: {entry}")
        # Fixed knowledge relations reached the model before this change whenever
        # the node list left room; they must keep doing so, or an inference like
        # "Jolene bought the pendant in Paris" loses its only bridge to "France".
        # Scoped to questions that actually ask for a place: the derivation is
        # produced for six route kinds, and hoisting it unconditionally would push
        # "Paris is in France" into unrelated conversations.
        if _COUNTRY_QUESTION.search(query or ""):
            for item in by_kind.get("bounded_geography", []):
                emit(f"- INFERRED: {item.get('text')}")
        # `relative_time` and `temporal` are deliberately not rendered: they echo
        # time words the node text already carries, and rendering them would change
        # prompts even with temporal execution off, making the flag unmeasurable.
        return out


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
    outputs = (plan.answer_contract.outputs
               if plan.response_intent == "direct_answer" and plan.answer_contract else [])
    entities = list(route.entities)
    if outputs:
        # The bound request owns subject interpretation; the route contributes
        # only literal mentions from a caller-provided identity inventory.
        # Both remain recall hints, not evidence or source-speaker filters.
        proposed = [subject for output in outputs for subject in output.subjects] + entities
        entities, seen = [], set()
        for subject in proposed:
            subject = subject.strip()
            if subject and len(subject) <= 200 and subject.casefold() not in seen:
                entities.append(subject)
                seen.add(subject.casefold())
        entities = entities[:12]  # Existing QueryPlan entity-hint bound.
    if plan.collection_query is not None:
        spec = plan.collection_query
        return plan.model_copy(update={
            "query_type": "temporal" if route.kind == "temporal" else "count" if spec.operation == "count" else "set",
            "recall_policy": "recall_first", "query_entities": entities if outputs else spec.subjects,
            "query_constraints": list(route.constraints),
            "required_info": plan.required_info or [f"{s}: {spec.predicate}" for s in spec.subjects],
            "depth": max(8, plan.depth),
        })
    # Coverage demand belongs to the request, not to the optional exact-set
    # optimization. Qualifiers can make collection_query unrepresentable while
    # the ordinary retrieval path still needs several facts or operands.
    needs_coverage = len(outputs) > 1 or any(output.operator in {"set", "count", "sum"} for output in outputs)
    policy = "recall_first" if needs_coverage else route.recall_policy
    kind = route.kind
    if kind == "fact" and len(outputs) == 1 and outputs[0].operator in {"set", "count"}:
        kind = outputs[0].operator
    depth = max(plan.depth, 8 if policy == "recall_first" else plan.depth)
    return plan.model_copy(
        update={
            "query_type": kind,
            "recall_policy": policy,
            "query_entities": entities,
            "query_constraints": list(route.constraints),
            "depth": min(20, depth),
        }
    )


def bind_plan_entities(
    query: str, plan: QueryPlan, known_entities: list[str]
) -> tuple[QueryPlan, list[str]]:
    """Keep a planner's recall expansions from substituting an explicit entity.

    Only literal matches to authoritative source-speaker identities establish a
    conflict. Memory subjects include objects and categories, so are unsuitable
    as an identity inventory; capitalisation guesses are unsuitable too. An expansion may
    add related entities or omit names entirely, but must not replace every
    explicit entity with different known entities. This binds the query, not the
    evidence: third-party statements remain eligible for retrieval.
    """
    explicit = {entity.casefold() for entity in _known_entity_mentions(query, known_entities)}
    if not explicit:
        return plan, []

    def substitutes(text: str) -> bool:
        mentioned = {entity.casefold() for entity in _known_entity_mentions(text, known_entities)}
        return bool(mentioned and not mentioned & explicit)

    updates: dict[str, Any] = {}
    changed: list[str] = []
    for field_name in ("semantic_queries", "keywords", "required_info"):
        values = getattr(plan, field_name)
        retained = [text for text in values if not substitutes(text)]
        if retained == values:
            continue
        # The original question is already the lexical query, and is also the
        # semantic fallback when every proposed expansion has substituted it.
        updates[field_name] = retained
        changed.append(field_name)
    if plan.subject and substitutes(plan.subject):
        updates.update(subject=None, predicate=None)
        changed.append("symbolic_subject")
    if plan.collection_query and any(substitutes(s) for s in plan.collection_query.subjects):
        updates["collection_query"] = None
        changed.append("collection_subjects")
    if plan.answer_contract is not None:
        outputs = []
        for output in plan.answer_contract.outputs:
            if output.subjects and substitutes(" ".join(output.subjects)):
                # The same source-speaker constraint applies to the new request
                # interface. Otherwise normalization can recreate a discarded
                # collection hint and silently reintroduce the wrong person.
                output = output.model_copy(deep=True, update={
                    "subjects": [], "relation_refs": [], "retrieval_relations": [],
                })
                changed.append("request_subjects:" + output.output_id)
            outputs.append(output)
        updates["answer_contract"] = plan.answer_contract.model_copy(update={"outputs": outputs})
    return plan.model_copy(update=updates), changed


def indexed_memory_hits(
    query: str,
    plan: QueryPlan,
    memories: list[Memory],
    selected_hits: list[Hit],
    *,
    limit: int = 32,
    matched_evidence: dict[str, list[dict]] | None = None,
    existing_candidates: list[Hit] | None = None,
) -> list[Hit]:
    """Retrieve additional authorized facts for compositional questions.

    A caller must apply owner, scope, lifecycle and time eligibility before
    passing memories. The finalized plan owns the query's entity hints;
    supplemental recall must not reinterpret them through a second route.
    Every quote is rechecked by the existing grounding layer before generation.
    """
    if limit <= 0:
        return []
    # Broad recall candidates are an exclusion set, not query-expansion seeds.
    # Promoting every unselected candidate into a bridge spreads unrelated
    # phrase pairs into the supplemental ranking and crowds out direct matches.
    already = {hit.metadata.get("memory_id")
               for hit in [*selected_hits, *(existing_candidates or [])]}
    bound_pairs = list(dict.fromkeys(
        (subject.strip().casefold(), predicate.strip().casefold())
        for output in (plan.answer_contract.outputs if plan.answer_contract else [])
        for subject, predicate in output.retrieval_relations
    ))
    entities = tuple(dict.fromkeys(entity.casefold() for entity in plan.query_entities))
    # Entity hints may name a recipe, list, or other topic rather than the
    # stored subject (often the speaker). Keep their words in textual recall;
    # removing them here discarded the very topic that disambiguates details.
    query_terms = _retrieval_terms(query)
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
        # An authorized quote may contain the requested detail that compression
        # omitted. Score one supported span at a time: repeated quotations and
        # unrelated events must not accumulate an artificial relevance bonus.
        matches = (matched_evidence or {}).get(memory.memory_id, [])
        overlap = max([len(query_terms & terms), *(
            len(query_terms & (terms | _retrieval_terms(span["text"]))) for span in matches
        )])
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
        # A bridge is weighted coverage of this candidate, not an unbounded
        # reward for repeating many rare phrases from noisy first-pass hits.
        # Any shared pair still permits recall, including cross-person hops.
        candidate_pairs = {pair for pair in _retrieval_pairs(memory.content)
                           if not set(pair) & entity_tokens}
        pair_weights = {pair: math.log1p(len(memories) / (1 + pair_frequency[pair]))
                        for pair in sorted(candidate_pairs)}
        total_pair_weight = sum(pair_weights.values())
        bridge = (sum(weight for pair, weight in pair_weights.items() if pair in useful_seed_pairs)
                  / total_pair_weight if total_pair_weight else 0.0)
        bound = (memory.subject.strip().casefold(), memory.predicate.strip().casefold()) in bound_pairs
        if not bound and not related and overlap == 0 and bridge == 0:
            continue
        score = 4 * overlap + 2 * bridge + 2 * len(related)
        scored.append((score, memory, set(related)))
    scored.sort(key=lambda row: (-row[0], row[1].memory_id))

    chosen = []
    seen = set()
    # The planner already resolved these catalog addresses. Cover each field
    # within the SAME candidate cap before broad name/lexical matches crowd it
    # out. This is retrieval coverage, never actor equivalence or fact truth.
    for pair in bound_pairs:
        if len(chosen) >= limit:
            break
        for score, memory, _ in scored:
            if (memory.subject.strip().casefold(), memory.predicate.strip().casefold()) == pair:
                if memory.memory_id not in seen:
                    chosen.append((score, memory))
                    seen.add(memory.memory_id)
                break
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
    for rank, (score, memory) in enumerate(chosen[:limit], 1):
        hits.append(
            Hit(
                content=memory.content,
                score=min(0.99, score / (score + 10)),
                source_file=f"[memory:{memory.memory_id}]",
                chunk_id=f"{memory.memory_id}:{memory.version}",
                metadata={
                    **memory.source_metadata(),
                    "origin": "indexed_sweep",
                    "retrieval_rank": rank,
                    "retrieval_rank_stage": "indexed_sweep_before_grounding",
                    "retrieval_channel_ranks": {"indexed_sweep": rank},
                    **({"matched_evidence": matched_evidence[memory.memory_id]}
                       if matched_evidence and memory.memory_id in matched_evidence else {}),
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


def _typed_constraints_ok(
    text: str,
    subject: str | None,
    occurred_at,
    target_entities: set[str],
    type_requirements: list[str],
    years: list[str],
    months: list[str],
) -> bool:
    """Does one piece of evidence satisfy every typed constraint of the question?"""
    text = text.casefold()
    subject_ok = not target_entities or any(
        entity in text or (subject and entity == subject.casefold()) for entity in target_entities
    )
    type_ok = all(
        any(term in text for term in TYPE_TERMS.get(type_name, {type_name}))
        for type_name in type_requirements
    )
    year_ok = not years or any(
        year in text or (occurred_at and year in str(occurred_at)) for year in years
    )
    month_ok = not months or any(
        month in text or (occurred_at and _month_name(occurred_at) == month) for month in months
    )
    return subject_ok and type_ok and year_ok and month_ok


def typed_negative_gate(
    query: str,
    route: QuestionRoute,
    graph: EventGraph,
) -> NegativeGate:
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
        if _typed_constraints_ok(
            node.text, node.subject, node.occurred_at,
            target_entities, type_requirements, years, months,
        ):
            supported.append(str(index))
    if supported:
        return NegativeGate(False, "typed_evidence_supported", supported_source_ids=tuple(supported))
    constraints = ", ".join(route.constraints) or "no typed candidate"
    # Retrieved evidence can establish support, never absence in the full
    # conversation. Keep the diagnostic; it is not an answer veto.
    return NegativeGate(False, f"not_found_in_examined_evidence; subject/type/time: {constraints}")


def _clean(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _entities(text: str) -> set[str]:
    return {match for match in re.findall(r"\b[A-Z][a-z]+\b", text) if match not in QUESTION_WORDS}


def _query_entities(query: str, known: list[str]) -> list[str]:
    # Capitalized words may be roles, event titles or objects, not the queried
    # actor. They remain in the unchanged query but cannot invent a subject
    # constraint. Production callers supply known source-speaker identities.
    return _known_entity_mentions(query, known)


def _known_entity_mentions(text: str, known: list[str]) -> list[str]:
    """Literal, case-insensitive entity mentions; no inferred entity aliases."""
    found: dict[str, str] = {}
    for entity in known:
        entity = entity.strip()
        if not entity:
            continue
        # CJK names are normally adjacent to verbs/particles rather than spaces.
        left = "" if re.match(r"[\u3400-\u9fff]", entity[0]) else r"(?<!\w)"
        right = "" if re.match(r"[\u3400-\u9fff]", entity[-1]) else r"(?!\w)"
        if re.search(left + re.escape(entity) + right, text, re.IGNORECASE):
            found.setdefault(entity.casefold(), entity)
    return list(found.values())


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


def _state(text: str, assertion: str | None, speaker_role: str | None = None) -> str:
    if speaker_role == "assistant":
        # Second-hand: the assistant described an event, it did not witness one.
        # Calling this "observed" would let a suggestion masquerade as a fact.
        return "reported"
    lower = text.casefold()
    if assertion in {"planned", "hypothetical"} or re.search(
        r"plan|planning|thinking about|consider|would like|want to", lower
    ):
        return "planned"
    if re.search(r"started|began|took up|launched|bought|went|visited|adopted|painted|performed", lower):
        return "observed"
    return assertion or "observed"




# Ordering and duration questions do not always route as "temporal": the
# router sends "which happened first" to set/fact. The derivation trigger must
# look at the question form itself, not at the routed kind. Kept deliberately
# narrow: bare "before/after/since" also appears in ordinary multi-hop
# questions, and injecting a date ledger there adds noise without a warrant.
_ORDER_OR_DURATION_QUERY = re.compile(
    r"\b(?:how long|how much time|how many (?:days?|weeks?|months?|years?|hours?)"
    r"|passed between|days? between|which (?:event|one|of (?:these|them))[^.?]{0,60}?\bfirst\b"
    r"|who[^.?]{0,60}?\bfirst\b|what happened first|came first|before or after"
    r"|earlier[,:?]|first[,:?])\b",
    re.IGNORECASE,
)

# Share of the reasoning-context budget reserved for derivations. A conclusion
# computed over the whole evidence set is the most expensive thing in the prompt
# to lose, so it is bounded but guaranteed; the node list is a sample and takes
# whatever remains.
_DERIVATION_BUDGET_SHARE = 0.35
_LEDGER_LINES = 4
_DIFFERENCE_LINES = 3
_DURATION_ITEMS = 3
# A city-to-country inference only helps a question that asks where something is.
_COUNTRY_QUESTION = re.compile(
    r"\b(?:what|which)\s+(?:country|nation)\b|\bin\s+(?:what|which)\s+country\b", re.IGNORECASE
)




_PRECISION_LABEL = {
    "day": "",
    "week": ", week precision",
    "month": ", month precision",
}

_ORDER_QUESTION = re.compile(
    r"\b(?:which|who|what)\b[^?]{0,80}?\b(?:first|earlier|sooner)\b", re.IGNORECASE
)
_ORDER_TAIL = re.compile(
    r"\b(?:first|earlier|sooner)\b[\s,:;-]*(.+)$", re.IGNORECASE | re.DOTALL
)
_ORDER_SPLIT = re.compile(r"\s+or\s+|\s+-\s+|\s+vs\.?\s+", re.IGNORECASE)
_ORDER_STOP = {
    "the", "and", "for", "with", "was", "were", "did", "does", "what", "which", "who",
    "when", "how", "narrator", "user", "speaker", "had", "have", "has", "been", "from",
    "at", "by", "as", "but", "first", "earlier", "sooner", "happened", "occurred", "new",
    "losing", "lose", "lost", "receiving", "receive", "received", "getting", "got", "get",
    "purchase", "purchased", "buy", "bought", "item", "their", "they", "them", "this",
    "that", "its", "his", "her",
}


def parse_order_candidates(question: str) -> tuple[str, str] | None:
    """The two entities a "which happened first, X or Y" question names."""
    if not _ORDER_QUESTION.search(question or ""):
        return None
    match = _ORDER_TAIL.search(question)
    if not match:
        return None
    tail = re.split(r"\n\s*Question date", match.group(1), maxsplit=1)[0].rstrip("?. ").strip()
    parts = _ORDER_SPLIT.split(tail, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = parts[0].strip(), parts[1].strip()
    return (left, right) if left and right else None


def _order_words(text: str) -> set[str]:
    return {
        word[:4]
        for word in re.findall(r"[a-z0-9]+", (text or "").casefold())
        if len(word) > 2 and word not in _ORDER_STOP
    }


# For comparing an answer against two candidates the verbs matter: "losing"
# and "receiving" are what distinguish them, so this word list keeps verbs and
# drops only the scaffolding every candidate shares.
_MENTION_STOP = _ORDER_STOP - {
    "losing", "lose", "lost", "receiving", "receive", "received", "getting",
    "got", "get", "purchase", "purchased", "buy", "bought", "new",
}


def _mention_words(text: str) -> set[str]:
    return {
        word[:4]
        for word in re.findall(r"[a-z0-9]+", (text or "").casefold())
        if len(word) > 2 and word not in _MENTION_STOP
    }


def names_other_candidate(answer: str, first: str, second: str) -> bool:
    """Whether an answer matches the rejected candidate better than the decision.

    Sharing a head noun is not agreement: "losing their phone charger" and
    "receiving their new phone case" both say "phone", so the comparison has to
    be relative rather than absolute.
    """
    text = _mention_words(answer)
    return len(text & _mention_words(second)) > len(text & _mention_words(first))


def _bind_order(phrase: str, candidates: list[tuple[str, object]]) -> list:
    """Events a candidate can mean, by content-word overlap.

    Only events that state a date are admissible: an undated event cannot answer
    "which happened first", so letting it compete would be a false option rather
    than a binding.
    """
    wanted = _order_words(phrase)
    if not wanted:
        return []
    scored = sorted(((len(wanted & _order_words(text)), item) for text, item in candidates), key=lambda x: -x[0])
    if not scored or scored[0][0] == 0:
        return []
    best_score = scored[0][0]
    return [item for score, item in scored if score == best_score]


def decide_order_by_offset(question: str, nodes: list) -> dict[str, Any] | None:
    """Order two named events when their offsets from one anchor are disjoint.

    Every condition is a warrant, not a heuristic knob:

    * the question names exactly two candidates;
    * each binds to an event that states a date, by a strictly best overlap;
    * the two bindings share one anchor, so the offsets are commensurable;
    * the offset intervals do not overlap.

    When all four hold the order is arithmetic. When any fails the function
    returns None and the model keeps the decision.
    """
    pair = parse_order_candidates(question)
    if pair is None:
        return None
    dated: list[tuple[str, dict]] = []
    for node in nodes:
        if not node.occurred_at:
            continue
        values = offset_intervals(node.text)
        if not values:
            continue
        dated.append(
            (
                node.text,
                {
                    "anchor": str(node.occurred_at)[:10],
                    "offsets": values,
                    "source_ids": list(node.source_ids),
                    # Identity is the event, not the hit that carried it: two
                    # distinct events inside one hit must stay distinguishable or
                    # a candidate pair would look like one reused event.
                    "node_id": node.node_id,
                    "text": node.text,
                },
            )
        )
    if len(dated) < 2:
        return None
    left = _bind_order(pair[0], dated)
    right = _bind_order(pair[1], dated)
    if not left or not right:
        return None
    # A tie is acceptable only when every tied event states the same offset:
    # two mentions of one purchase are one date, so the ambiguity cannot change
    # the answer. A tie across different offsets is a genuine ambiguity.
    for bound in (left, right):
        if len({value[:2] for item in bound for value in item["offsets"]}) != 1:
            return None
        if len({item["anchor"] for item in bound}) != 1:
            return None
    if {item["node_id"] for item in left} & {item["node_id"] for item in right}:
        return None
    if {item["anchor"] for item in left} != {item["anchor"] for item in right}:
        return None
    left_low = max(value[0] for item in left for value in item["offsets"])
    left_high = min(value[1] for item in left for value in item["offsets"])
    right_low = max(value[0] for item in right for value in item["offsets"])
    right_high = min(value[1] for item in right for value in item["offsets"])
    if left_high < right_low:
        first, second = pair[1], pair[0]
        first_low, first_high, second_low, second_high = right_low, right_high, left_low, left_high
    elif right_high < left_low:
        first, second = pair[0], pair[1]
        first_low, first_high, second_low, second_high = left_low, left_high, right_low, right_high
    else:
        return None
    return {
        "kind": "offset_order",
        "first": first,
        "second": second,
        "why": (
            f"both events are dated from {min(item['anchor'] for item in left)}; "
            f"'{first}' is {first_low}-{first_high} days before it and "
            f"'{second}' is {second_low}-{second_high} days before it, "
            f"so '{first}' came first"
        ),
        "source_ids": [sid for item in left + right for sid in item["source_ids"]][:8],
    }


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

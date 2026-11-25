"""Shared contracts. Changes here should be reviewed by all module owners."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Scope(Contract):
    tenant_id: str = Field(min_length=1, max_length=100)
    owner_id: str = Field(min_length=1, max_length=100)


class Event(Contract):
    role: Literal["user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=100000)
    speaker: str | None = Field(default=None, min_length=1, max_length=200)
    occurred_at: datetime = Field(default_factory=utcnow)
    source_utc_offset_minutes: int = Field(default=0, ge=-840, le=840)
    answer_id: str | None = Field(default=None, min_length=1, max_length=100)
    answer_context: AnswerContext | None = None

    @model_validator(mode="before")
    @classmethod
    def preserve_source_offset(cls, value):
        if isinstance(value, dict) and "source_utc_offset_minutes" not in value:
            original = value.get("occurred_at")
            if isinstance(original, str):
                try:
                    original = datetime.fromisoformat(original)
                except ValueError:
                    return value
            if isinstance(original, datetime) and original.utcoffset() is not None:
                value = {
                    **value,
                    "source_utc_offset_minutes": int(original.utcoffset().total_seconds() // 60),
                }
        return value

    @field_validator("occurred_at")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at must include timezone")
        return value.astimezone(UTC)


class TurnInput(Contract):
    request_id: str = Field(min_length=1, max_length=100)
    events: list[Event] = Field(min_length=1, max_length=30)

    def prompt_dump(self) -> dict:
        """Model input excludes UI provenance snapshots and receipt identifiers."""
        return self.model_dump(mode="json", exclude={"events": {"__all__": {"answer_id", "answer_context"}}})


class Turn(TurnInput):
    turn_id: str
    session_id: str
    sequence: int


class Session(Contract):
    session_id: str
    topic: str = ""
    project_id: str | None = None
    goal: str = ""
    plan: list[str] = Field(default_factory=list)
    pending_tasks: list[str] = Field(default_factory=list)
    summary: str = ""
    processed_sequence: int = 0
    revision: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="before")
    @classmethod
    def legacy_updated_at(cls, value):
        if isinstance(value, dict) and "updated_at" not in value and "created_at" in value:
            return {**value, "updated_at": value["created_at"]}
        return value


class Evidence(Contract):
    turn_id: str
    event_index: int = Field(ge=0)
    quote: str = Field(min_length=1, max_length=100000)
    # Half-open positions in the original Python/Unicode string, never bytes or
    # offsets into normalized text. Old quote-only records remain valid.
    start: int | None = Field(default=None, ge=0, strict=True)
    end: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def paired_positions(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("Evidence start and end must be supplied together")
        if self.start is not None and self.end - self.start != len(self.quote):
            raise ValueError("Evidence positions must span the exact quote length")
        return self


class FactMember(Contract):
    """An atomic collection value; identity is optional, never a graph-node count."""

    value: str = Field(min_length=1, max_length=500)
    identity: str | None = Field(default=None, min_length=1, max_length=200)
    evidence: list[Evidence] = Field(min_length=1, max_length=30)


class FactSemantics(Contract):
    cardinality: Literal["unknown", "one", "many"] = "unknown"
    operation: Literal["assert", "add", "remove", "replace_set"] = "assert"
    members: list[FactMember] = Field(default_factory=list, max_length=80)
    # A domain-event qualifier, distinct from the source utterance's ID/time.
    event_key: str | None = Field(default=None, min_length=1, max_length=200)
    event_identity: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def collection_operations(self):
        if self.cardinality != "many" and (self.members or self.operation != "assert"):
            raise ValueError("Collection members/operations require cardinality=many")
        if self.cardinality == "many" and not self.members and self.operation != "replace_set":
            raise ValueError("A collection assertion needs at least one sourced member")
        return self


class Candidate(Contract):
    content: str = Field(min_length=1, max_length=3000)
    kind: Literal["fact", "preference", "event", "procedure"] = "fact"
    subject: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=2000)
    keywords: list[str] = Field(default_factory=list, max_length=30)
    evidence: list[Evidence] = Field(min_length=1, max_length=30)
    assertion: Literal["stated", "observed", "planned", "hypothetical", "inferred"] = "stated"
    durable: bool = False
    scope_type: Literal["session", "project", "user"] = "session"
    # Organizational hints; never owner/permission controls or mandatory query gates.
    hierarchy_path: list[str] = Field(default_factory=list, max_length=3)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    semantics: FactSemantics = Field(default_factory=FactSemantics)

    @field_validator("valid_from", "valid_to")
    @classmethod
    def aware_validity(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("valid times must include timezone")
        return value.astimezone(UTC) if value else None


class Memory(Candidate):
    memory_id: str
    version: int = 1
    revision: int = 1
    session_id: str
    scope_id: str
    tier: Literal["short", "long"] = "short"
    status: Literal["active", "superseded", "retracted", "pending", "archived"] = "active"
    supersedes: str | None = None
    recorded_at: datetime = Field(default_factory=utcnow)

    def source_metadata(self) -> dict:
        """A fresh, channel-independent fact projection; never retrieval policy.

        Different recall paths must carry the same source semantics. JSON-mode
        serialization also prevents a consumer from mutating stored members or
        evidence through shared nested objects.
        """
        return self.model_dump(mode="json", include={
            "memory_id", "version", "revision", "tier", "assertion", "kind",
            "subject", "predicate", "value", "semantics", "durable", "scope_type",
            "scope_id", "status", "valid_from", "valid_to", "hierarchy_path",
            "recorded_at", "evidence",
        })


class EntityMemory(Contract):
    entity_id: str
    subject: str
    scope_type: Literal["session", "project", "user"]
    scope_id: str
    summary: str
    facts: list[Memory]
    pending_facts: list[Memory] = Field(default_factory=list)
    conflict_predicates: list[str] = Field(default_factory=list)
    updated_at: datetime


class StatePatch(Contract):
    goal: str | None = None
    plan: list[str] | None = None
    pending_tasks: list[str] | None = None


class ReferenceFact(Contract):
    """Canonical field hints only; these records cannot serve as extraction evidence."""

    subject: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=2000)
    scope_type: Literal["project", "user"]
    cardinality: Literal["unknown", "one", "many"] = "unknown"
    event_key: str | None = None
    event_identity: str | None = None
    assertion: Literal["stated", "observed", "planned", "hypothetical", "inferred"] = "stated"
    status: Literal["active", "superseded"] = "active"
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class ExtractionWindow(Contract):
    session: Session
    new_turns: list[Turn]
    context_turns: list[Turn]
    reference_fields: list[ReferenceFact] = Field(default_factory=list, max_length=8)


class ExtractionResult(Contract):
    candidates: list[Candidate] = Field(default_factory=list, max_length=40)
    summary: str = Field(default="", max_length=3000)
    state_patch: StatePatch = Field(default_factory=StatePatch)


class CollectionQuery(Contract):
    """Small query algebra, proposed by the existing planner, executed locally."""

    subjects: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(min_length=1, max_length=12)
    predicate: str = Field(min_length=1, max_length=100)
    operation: Literal["union", "group", "intersection", "count"] = "union"
    count_unit: Literal["value", "entity", "event"] = "value"
    projection: Literal["value", "category"] = "value"
    qualifier: str = Field(default="", max_length=500)
    event_key: str | None = Field(default=None, max_length=200)
    exclude_values: list[str] = Field(default_factory=list, max_length=20)
    assertions: list[Literal["stated", "observed", "planned"]] = Field(
        default_factory=lambda: ["stated", "observed"], min_length=1, max_length=3
    )

    @model_validator(mode="after")
    def distinct_subjects(self):
        if len({s.strip().casefold() for s in self.subjects}) != len(self.subjects):
            raise ValueError("Query subjects must be distinct")
        if self.operation == "intersection" and len(self.subjects) < 2:
            raise ValueError("Intersection requires at least two subjects")
        return self


class AnswerOutput(Contract):
    """An obligation anchored to the user's question, not a proposed answer."""

    output_id: str = Field(min_length=1, max_length=40)
    question_span: str = Field(min_length=1, max_length=1000)
    operator: Literal["direct", "set", "count", "sum", "time_point", "duration", "comparison"] = "direct"
    subjects: list[str] = Field(default_factory=list, max_length=8)
    subject_mode: Literal["single", "each", "joint", "common"] = "single"
    relation: str = Field(default="", max_length=200)
    qualifier: str = Field(default="", max_length=500)
    count_unit: Literal["value", "entity", "event"] | None = None
    # Positions of requested members within an enumerated answer/list, not
    # event chronology or the ordinal of a whole list/session. Grounding
    # intersects these hints with literal ordinals in question_span.
    member_positions: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list, max_length=20)
    # Request-local, one-based catalog positions: retrieval suggestions, not
    # proof that the corresponding facts satisfy the requested relation.
    relation_refs: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list, max_length=60)
    # Internal compilation of catalog addresses; omitted from planner/reader
    # wire schemas. These pairs aid recall/packing, never certify a fact.
    retrieval_relations: list[tuple[str, str]] = Field(default_factory=list, max_length=60)
    # Retained only for old receipts. New planning wire omits guessed premises.
    required_inputs: list[str] = Field(default_factory=list, max_length=6)


class AnswerContract(Contract):
    outputs: list[AnswerOutput] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({o.output_id for o in self.outputs}) != len(self.outputs):
            raise ValueError("Output IDs must be unique")
        return self


class QueryPlan(Contract):
    route: Literal["none", "short", "long", "both"] = "both"
    response_intent: Literal["direct_answer", "action_plan"] = "direct_answer"
    semantic_queries: list[Annotated[str, Field(max_length=1000)]] = Field(default_factory=list, max_length=3)
    keywords: list[Annotated[str, Field(max_length=200)]] = Field(default_factory=list, max_length=30)
    problem_breakdown: list[Annotated[str, Field(max_length=500)]] = Field(
        default_factory=list, max_length=12
    )
    required_info: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list, max_length=20)
    information_gathering: list[Annotated[str, Field(max_length=700)]] = Field(
        default_factory=list, max_length=20
    )
    integration_steps: list[Annotated[str, Field(max_length=700)]] = Field(
        default_factory=list, max_length=12
    )
    action_template: Literal["none", "general_action_plan", "incident_action_plan"] = "none"
    action_elements: list[Annotated[str, Field(max_length=300)]] = Field(default_factory=list, max_length=40)
    depth: int = Field(default=3, ge=1, le=20)
    subject: str | None = None
    predicate: str | None = None
    temporal_mode: Literal["current", "history"] = "current"
    as_of: datetime | None = None
    # Deterministic query routing, filled after the model plan and therefore
    # never trusted as a permission or truth decision by itself.
    # ``recommendation`` is produced by reasoning.classify_question (see the
    # recall_policy branch there), so it must be accepted here — otherwise the
    # value survives model_copy(update=...) unchecked and only explodes later
    # in SQLiteStore.save_answer, taking down the whole run.
    query_type: Literal[
        "fact", "temporal", "relation", "causal", "set", "count",
        "inference", "recommendation", "adversarial",
    ] = "fact"
    recall_policy: Literal["focused", "recall_first"] = "focused"
    query_entities: list[Annotated[str, Field(max_length=200)]] = Field(default_factory=list, max_length=12)
    query_constraints: list[Annotated[str, Field(max_length=300)]] = Field(
        default_factory=list, max_length=12
    )
    collection_query: CollectionQuery | None = None
    # One request interpretation, distinct from factual evidence and optimized
    # retrieval hints. Missing/invalid contracts do not prevent ordinary QA.
    answer_contract: AnswerContract | None = None

    @field_validator("query_type", mode="before")
    @classmethod
    def known_query_type(cls, value: object) -> object:
        """Any future router label degrades to ``fact`` instead of raising.

        A single unknown label used to abort an entire benchmark run from
        inside a persistence call, far away from the router that produced it.
        Degrading keeps the run alive and stays on the generic answering path.
        """
        if value is None or value == "":
            return "fact"
        if isinstance(value, str):
            allowed = {
                "fact", "temporal", "relation", "causal", "set", "count",
                "inference", "recommendation", "adversarial",
            }
            return value if value in allowed else "fact"
        return value

    @field_validator("as_of")
    @classmethod
    def aware_as_of(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("as_of must include timezone")
        return value.astimezone(UTC) if value else None


def normalize_request_contract(
    question: str,
    plan: QueryPlan,
    relation_catalog: list | None = None,
) -> tuple[QueryPlan, list[str]]:
    """Anchor request spans and project bounded retrieval hints, without refusal.

    This checks addresses and representability, not natural-language entailment.
    The original question remains the retrieval/reader input. In particular, a
    real catalog reference does not establish that its relation qualifies.
    Entity gates already reconciled by the caller are never overwritten here.
    """
    if plan.response_intent != "direct_answer" or plan.answer_contract is None:
        return plan, []
    notes = []
    outputs = []
    for output in plan.answer_contract.outputs:
        span = output.question_span.strip()
        if not span or span not in question:
            notes.append(f"request_contract:unanchored_output:{output.output_id}")
            continue
        refs = list(dict.fromkeys(output.relation_refs))
        relations = list(output.retrieval_relations)
        if relation_catalog is not None:
            valid = [ref for ref in refs if 1 <= ref <= len(relation_catalog)]
            if valid != refs:
                notes.append(f"request_contract:unknown_relation_ref:{output.output_id}")
            refs = valid
            # Recompile from this request's authorized catalog, overwriting
            # any old or externally supplied projection. A reader-only repeat
            # without the catalog retains the already compiled pairs.
            relations = []
            for ref in refs:
                entry = relation_catalog[ref - 1]
                if (isinstance(entry, (tuple, list)) and len(entry) >= 2
                        and all(isinstance(value, str) and value.strip() for value in entry[:2])):
                    pair = (entry[0], entry[1])
                    if pair not in relations:
                        relations.append(pair)
        if output.required_inputs:
            notes.append(f"request_contract:legacy_premises_omitted:{output.output_id}")
        outputs.append(output.model_copy(deep=True, update={
            "question_span": span, "relation_refs": refs, "retrieval_relations": relations, "required_inputs": [],
        }))
    if not outputs:
        # Preserve the old retrieval plan when the interpretation is unusable;
        # do not turn a malformed interpretation into a missing-evidence claim.
        return plan.model_copy(deep=True, update={"answer_contract": None}), notes
    if len(outputs) != len(plan.answer_contract.outputs):
        # A valid address for one sub-request does not certify that the
        # remaining contract represents the whole question. Keeping only
        # surviving outputs narrows downstream entity/coverage bindings, and
        # a second normalization used to erase the full-question fallback.
        # Decline this optimization as a whole. Ordinary retrieval/synthesis
        # still receive the original question; this is never a refusal.
        notes.append("request_contract:partial_interpretation_omitted")
        required = [question] if len(question) <= 500 else list(plan.required_info)
        return plan.model_copy(deep=True, update={
            "answer_contract": None, "collection_query": None, "required_info": required,
        }), notes
    required = list(dict.fromkeys(o.question_span for o in outputs if len(o.question_span) <= 500))
    if any(len(o.question_span) > 500 for o in outputs):
        notes.append("request_contract:long_span_not_truncated_for_retrieval")
    collection = None
    if len(outputs) == 1:
        collection = _request_collection_projection(outputs[0], plan.collection_query, relation_catalog)
    if plan.collection_query is not None and collection is None:
        notes.append("request_contract:collection_optimization_not_representable")
    update = {"answer_contract": AnswerContract(outputs=outputs), "collection_query": collection,
              "required_info": required}
    return plan.model_copy(deep=True, update=update), notes


def _request_collection_projection(
    output: AnswerOutput,
    previous: CollectionQuery | None,
    catalog: list | None,
) -> CollectionQuery | None:
    """Project only a representable set request; unknown is ordinary retrieval.

    Rich existing collection hints may be retained only when the new request
    describes the same scope. Their assertions/exclusions/category projection
    must not be silently replaced by defaults.
    """
    if output.operator not in {"set", "count"}:
        return None
    if output.relation_refs and catalog is None and previous is None:
        # Reader-side normalization has no catalog. An unresolved/multi-alias
        # retrieval mapping must not become an exact partition on a later pass.
        return None
    subjects = list(dict.fromkeys(s.strip() for s in output.subjects if s.strip()))
    if (not subjects or any(len(s) > 200 for s in subjects)
            or len({s.casefold() for s in subjects}) != len(subjects)):
        return None
    if output.subject_mode == "single" and len(subjects) != 1:
        return None
    if output.subject_mode == "common":
        if len(subjects) < 2:
            return None
        operation = "intersection"
    elif output.subject_mode == "each" and output.operator == "set":
        operation = "group"
    else:
        operation = "count" if output.operator == "count" else "union"
    predicate = output.relation.strip()
    if output.relation_refs and catalog is not None:
        predicates = {predicate.strip() for _, predicate in output.retrieval_relations}
        # A single symbolic partition cannot represent a disjunction of aliases.
        if len(predicates) != 1:
            return None
        predicate = next(iter(predicates))
    if not predicate or len(predicate) > 100:
        return None
    if previous is not None:
        same_scope = (
            [s.casefold() for s in previous.subjects] == [s.casefold() for s in subjects]
            and (previous.predicate.casefold() == predicate.casefold()
                 or (output.relation_refs and catalog is None))
            and previous.operation == operation
            and previous.qualifier.strip() == output.qualifier.strip()
            and (output.count_unit is None or previous.count_unit == output.count_unit)
        )
        if same_scope:
            return previous.model_copy(deep=True)
        if (previous.qualifier or previous.event_key or previous.exclude_values
                or previous.projection != "value" or previous.assertions != ["stated", "observed"]):
            return None
    # A nontrivial role/state/time qualifier cannot safely be reconstructed as
    # exact predicate membership. Keep the original semantic retrieval instead.
    if output.qualifier.strip() or output.count_unit is None:
        return None
    return CollectionQuery(
        subjects=subjects, predicate=predicate, operation=operation,
        count_unit=output.count_unit,
    )


class Hit(Contract):
    content: str
    score: float = Field(ge=0, le=1)
    source_file: str
    chunk_id: str
    engine: str = "simple_memory"
    metadata: dict = Field(default_factory=dict)


class QueryStep(Contract):
    order: int = Field(ge=1)
    phase: Literal["planning", "short_retrieval", "long_retrieval", "filter", "selection"]
    action: str
    input_count: int = Field(ge=0)
    output_count: int = Field(ge=0)
    detail: str = ""


class ChannelCandidate(Contract):
    memory_id: str
    version: int
    revision: int
    subject: str
    predicate: str
    value: str
    content: str
    content_truncated: bool = False
    matched: bool
    selected: bool = False
    score: float | None = None
    reason: str = ""


class RetrievalChannel(Contract):
    view: Literal["semantic", "lexical", "symbolic"]
    tier: Literal["short", "long"] = "short"
    status: Literal["complete", "disabled", "skipped"]
    input_count: int = Field(default=0, ge=0)
    matched_count: int = Field(default=0, ge=0)
    selected_count: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(default=0, ge=0)
    detail: str = ""
    query_conditions: dict = Field(default_factory=dict)
    candidates: list[ChannelCandidate] = Field(default_factory=list, max_length=20)
    candidate_total: int = 0
    candidate_limit: int = 20
    candidates_truncated: bool = False


class RetrievalBudget(Contract):
    planned_depth: int
    required_info_count: int
    candidate_limit: int
    safety_cap: int
    target_k: int
    selected_k: int = 0
    token_limit: int
    used_tokens: int = 0
    covered_info: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    selection_policy: str = "slot_diversity_complete_evidence_bundles_within_budget"
    score_semantics: str = "relevance_not_truth_confidence"


class SearchResponse(Contract):
    results: list[Hit]
    # Internal recall envelope. Public search still exposes its top-k view;
    # answering budgets the attested source text, not a preselected summary.
    candidates: list[Hit] = Field(default_factory=list, exclude=True)
    plan: QueryPlan
    mode: str
    candidate_count: int
    selected_k: int
    context_tokens: int
    steps: list[QueryStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    snapshot_valid_until: datetime | None = None
    channels: list[RetrievalChannel] = Field(default_factory=list)
    dynamic_k: RetrievalBudget | None = None


class AccelerationTrace(Contract):
    enabled: bool
    cache_hit: bool
    cache_status: Literal["disabled", "miss", "hit", "shared"]
    strategy: str = "exact_context_generation_reuse"
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)
    context_tokens_before: int = Field(ge=0)
    context_tokens_after: int = Field(ge=0)
    avoided_model_calls: int = Field(ge=0)
    original_generation_ms: float | None = None
    token_measurement: str = "conservative_character_estimate"


class ModelCallTrace(Contract):
    """External model calls made by one answer request."""

    extract_calls: int = Field(default=0, ge=0)
    plan_calls: int = Field(default=0, ge=0)
    answer_calls: int = Field(default=0, ge=0)
    relation_calls: int = Field(default=0, ge=0)
    summary_calls: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)


class GroundingTrace(Contract):
    enabled: bool = False
    initial_sources: int = 0
    final_sources: int = 0
    indexed_candidates_considered: int = 0
    indexed_sources_selected: int = 0
    resolved_source_events: int = 0
    rejected_source_refs: int = 0
    attribution_conflicts: int = 0
    supplemental_rounds: int = 0
    supplemental_queries: list[str] = Field(default_factory=list)
    supplemental_candidates: int = 0
    semantic_fallback_used: bool = False
    raw_vector_inputs: int = 0
    raw_vector_reused: int = 0
    raw_sources_selected: int = 0
    subject_recall_candidates: int = 0
    subject_recall_added: int = 0
    initial_missing_info: list[str] = Field(default_factory=list)
    final_missing_info: list[str] = Field(default_factory=list)
    coverage_measurement: str = "lexical_support_not_entailment"
    binding_units_available: int = 0
    binding_units_selected: int = 0
    binding_unit_measurement: str = "visible_extraction_hints_not_qualified_set_completeness"
    # Query catalog addresses supported by visible extraction hints. These
    # counts measure locating coverage, not truth or answer completeness.
    request_fields_available: int = 0
    request_fields_selected: int = 0
    selection_decisions: list[dict] = Field(default_factory=list)
    temporal_derivations: int = 0
    # Legacy alias retained for older receipts/budget consumers. This value
    # has always used serialized characters, never provider token usage.
    context_tokens_estimate: int = 0
    context_serialized_chars: int = 0
    context_budget_unit: str = "serialized_characters"
    selection_coverage: str = "soft_locating_signals_not_answer_sufficiency"
    elapsed_ms: float = 0


class AnswerContext(Contract):
    """Server-attested snapshot for this answer, never a new extraction source."""

    sources: list[Hit] = Field(default_factory=list, max_length=120)
    citations: list[int] = Field(default_factory=list)
    query_steps: list[QueryStep] = Field(default_factory=list)
    plan: QueryPlan | None = None
    elapsed_ms: int = 0
    retrieval_mode: str = ""
    candidate_count: int = 0
    selected_k: int = 0
    acceleration: AccelerationTrace | None = None
    model_calls: ModelCallTrace = Field(default_factory=ModelCallTrace)
    run_events: list[dict] = Field(default_factory=list)
    channels: list[RetrievalChannel] = Field(default_factory=list)
    dynamic_k: RetrievalBudget | None = None
    grounding: GroundingTrace = Field(default_factory=GroundingTrace)
    reasoning: dict = Field(default_factory=dict)


class AnswerResponse(Contract):
    answer_id: str | None = None
    generated_text: str
    citations: list[int]
    sources: list[Hit]
    retrieval_count: int
    elapsed_ms: int
    plan: QueryPlan | None = None
    query_steps: list[QueryStep] = Field(default_factory=list)
    retrieval_mode: str = ""
    candidate_count: int = 0
    selected_k: int = 0
    context_tokens: int = 0
    warnings: list[str] = Field(default_factory=list)
    acceleration: AccelerationTrace | None = None
    model_calls: ModelCallTrace = Field(default_factory=ModelCallTrace)
    memory_updates: list[dict] = Field(default_factory=list)
    channels: list[RetrievalChannel] = Field(default_factory=list)
    dynamic_k: RetrievalBudget | None = None
    grounding: GroundingTrace = Field(default_factory=GroundingTrace)
    reasoning: dict = Field(default_factory=dict)


class EvolutionInput(Contract):
    action: Literal[
        "promote", "merge", "supersede", "correct", "retract", "defer", "archive", "restore", "activate"
    ]
    expected_version: int = Field(ge=1)
    expected_revision: int | None = Field(default=None, ge=1)
    target_id: str | None = None
    target_version: int | None = Field(default=None, ge=1)
    target_revision: int | None = Field(default=None, ge=1)
    effective_at: datetime | None = None

    @field_validator("effective_at")
    @classmethod
    def aware_effective(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("effective_at must include timezone")
        return value.astimezone(UTC) if value else None


class GroupSynthesis(Contract):
    summary: str = Field(min_length=1, max_length=3000)
    source_refs: list[str] = Field(min_length=1, max_length=100)


Event.model_rebuild()
TurnInput.model_rebuild()
Turn.model_rebuild()
ExtractionWindow.model_rebuild()

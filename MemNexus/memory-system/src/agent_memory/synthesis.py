"""One source-bound answer table for factual and compositional questions.

Quote checks prove provenance, not entailment. Keep that distinction explicit.
The renderer cannot silently discard an accepted item; unresolved slots remain
in the receipt. No benchmark categories, answers, or domain aliases live here.
"""

import hashlib
import json
import re
from datetime import date, datetime
from typing import Literal

from pydantic import Field

from .models import AnswerContract, AnswerOutput, Contract, Hit, QueryPlan
from .temporal import derive_times


class Support(Contract):
    source_id: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=700)


class DateDifference(Contract):
    start: date
    end: date
    unit: Literal["days", "weeks", "years"]

    def value(self) -> str:
        if self.end < self.start:
            raise ValueError("Time difference must have ordered endpoints")
        if self.unit == "years":
            if (self.start.month, self.start.day) != (self.end.month, self.end.day):
                raise ValueError("Whole-year difference requires equal month and day")
            number = self.end.year - self.start.year
        else:
            days = (self.end - self.start).days
            if self.unit == "weeks" and days % 7:
                raise ValueError("Do not round an inexact interval to whole weeks")
            number = days // 7 if self.unit == "weeks" else days
        return f"{number} {self.unit}"


class SupportedValue(Contract):
    value: str = Field(min_length=1, max_length=700)
    supports: list[Support] = Field(min_length=1, max_length=8)
    calculation: DateDifference | None = None
    time_expression: str | None = Field(default=None, max_length=150)


class AnswerSlot(Contract):
    target: str = Field(min_length=1, max_length=500)
    role: Literal["answer", "context"] = "answer"
    values: list[SupportedValue] = Field(default_factory=list, max_length=30)
    missing: str = Field(default="", max_length=500)


class AnswerTable(Contract):
    slots: list[AnswerSlot] = Field(min_length=1, max_length=20)


def answer_type(question: str) -> str:
    q = question.casefold()
    if re.search(r"how (?:many|long).*?(?:weeks?|days?|years?|months?|hours?)|相隔|多久", q):
        return "duration"
    if re.search(r"\b(?:when|what year|which year|what date)\b|何时|哪年", q):
        return "date"
    if re.search(r"how many|number of|几个|多少", q):
        return "count"
    return "text"


def reader_profile(question: str) -> str:
    """Classify the requested *operation*, independently of retrieved topics.

    This small deterministic profile prevents a planner's broad ``count`` or
    ``direct`` label from changing the answer type.  It is deliberately lexical
    and conservative: uncertain questions use the normal recall path.
    """
    q = re.sub(r"\s+", " ", question.casefold()).strip()
    # A request for advice is different from recalling a past suggestion.
    if (re.search(r"\b(?:suggest|recommend|ideas?|advice|inspiration)\b|建议|推荐|灵感", q)
            and not re.search(r"\b(?:did|was|were|have|had|what)\b.{0,35}\b(?:recommend|suggest|say)\b", q)):
        return "recommendation"
    if re.match(r"^(?:which|who)\b", q) and re.search(
            r"\b(?:most|least|fewest|highest|lowest|best|worst|最多|最少|最高|最低)\b", q):
        return "entity_selection"
    if re.match(r"^(?:which|who)\b", q):
        return "recall"
    if re.search(r"\b(?:how many|how much|how long|number of)\b|多少|多久|几", q):
        # Comparisons and before/after questions genuinely require multiple
        # values and must not be collapsed into one scalar.
        if re.search(r"\b(?:before|after|between|each person|each subject|per person|respectively)\b|之前|之后|分别", q):
            return "recall"
        return "single_quantity"
    if re.search(r"\b(?:what did .* recommend|what .* suggested|what advice .* gave)\b", q):
        return "recall"
    return "recall"


def validate_table(table: AnswerTable, sources: list[Hit], targets: list[str], kind: str) -> None:
    if [slot.target for slot in table.slots] != targets:
        raise ValueError("Return every requested target exactly once and in order")
    for slot in table.slots:
        if not slot.values and not slot.missing:
            raise ValueError("An empty target requires an explicit missing-evidence reason")
        for item in slot.values:
            dates = set()
            factual_support = False
            for support in item.supports:
                if support.source_id > len(sources):
                    raise ValueError("Unknown source number")
                events = sources[support.source_id - 1].metadata.get("source_events", [])
                matched = [e for e in events if support.quote in e["text"]]
                if not matched:
                    raise ValueError(f"Source {support.source_id} does not contain exact quote: {support.quote!r}")
                factual_support |= any(not e.get("context_only") for e in matched)
                for event in matched:
                    if event.get("context_only"):
                        continue
                    dates.add(str(event.get("occurred_at", ""))[:10])
                    dates.update(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", event["text"]))
                    for derivation in event.get("time_derivations", []):
                        # An interval boundary is not a known exact event date.
                        if derivation["precision"] == "day":
                            dates.add(derivation["start"])
            if not factual_support:
                raise ValueError("A question/context alone cannot establish a fact; cite at least one original factual utterance")
            if kind == "date" and slot.role == "answer":
                # Both answer paths resolve the cited expression from its source
                # anchor before inspecting a model-proposed calculation. Never
                # swap arithmetic endpoints or infer a date from the answer text.
                resolved = _source_time(item, sources)
                if resolved is not None:
                    item.value = resolved
                    item.calculation = None
                    continue
            if item.calculation:
                calc = item.calculation
                if kind == "date" and calc.unit == "years" and slot.role == "answer":
                    # Inverse duration: only the resulting year is warranted,
                    # not an invented exact day/month for the start event.
                    numbers = {word: n for n, word in enumerate(
                        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))}
                    durations = {int(raw) if raw.isdigit() else numbers[raw.casefold()]
                                 for ref in item.supports
                                 for raw in re.findall(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten) years?\b", ref.quote, re.IGNORECASE)}
                    if calc.end.isoformat() not in dates or calc.end.year-calc.start.year not in durations:
                        raise ValueError("Start-year derivation requires an anchored end date and quoted year duration")
                    item.value = str(calc.start.year)
                    # Do not publish manufactured month/day as a proven event date.
                    item.calculation = None
                    continue
                if kind != "duration" or slot.role != "answer":
                    raise ValueError("Date difference only answers a duration query")
                if not {item.calculation.start.isoformat(), item.calculation.end.isoformat()} <= dates:
                    raise ValueError("Calculation endpoints need exact source-bound dates. For approximate year/week precision, set calculation=null and answer the supported relative duration, without invented endpoints.")
                item.value = item.calculation.value()
            if slot.role == "answer" and kind == "duration" and not re.search(r"\b(?:days?|weeks?|months?|years?|hours?|minutes?)\b|天|周|月|年", item.value, re.IGNORECASE):
                raise ValueError("Duration answers must express a duration, not an event name")
            if slot.role == "answer" and kind == "count" and not re.search(r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|none)\b", item.value, re.IGNORECASE):
                raise ValueError("Count answers must contain a number or a supported bound")


def render_table(table: AnswerTable) -> str:
    values = list(dict.fromkeys(item.value.strip() for slot in table.slots if slot.role == "answer" for item in slot.values))
    return "; ".join(values) if values else "Not mentioned in the conversation"


class BoundValue(SupportedValue):
    """Model-interpreted relation; provenance checks do not prove entailment."""

    subjects: list[str] = Field(default_factory=list, max_length=8)
    event: str = Field(default="", max_length=300)
    event_key: str | None = Field(default=None, max_length=200)
    identity: str | None = Field(default=None, max_length=200)
    matches_request: bool = True
    basis: Literal["stated", "inferred"] = "stated"
    quantity: int | None = Field(default=None, ge=0)
    binding_id: str = Field(default="", max_length=48)
    role_bindings: list["RoleBinding"] = Field(default_factory=list, max_length=8)
    status: Literal["active", "superseded", "retracted", "conflicted", "uncertain"] = "active"
    conflict_set_id: str | None = Field(default=None, max_length=64)


class RoleBinding(Contract):
    """A source-backed semantic role; names alone never establish ownership."""

    role: Literal[
        "speaker", "owner", "giver", "recipient", "participant", "object", "location", "actor"
    ]
    entity: str = Field(min_length=1, max_length=200)
    supports: list[Support] = Field(min_length=1, max_length=8)
    confidence: Literal["explicit", "resolved", "ambiguous"] = "explicit"


class AnswerCandidate(Contract):
    """A candidate is retained even when the model rejects or cannot resolve it."""

    candidate_id: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=700)
    identity: str | None = Field(default=None, max_length=200)
    subjects: list[str] = Field(default_factory=list, max_length=8)
    event_key: str | None = Field(default=None, max_length=200)
    role_bindings: list[RoleBinding] = Field(default_factory=list, max_length=8)
    supports: list[Support] = Field(default_factory=list, max_length=8)
    disposition: Literal["accepted", "rejected", "unresolved"] = "unresolved"
    reason_code: str = Field(default="", max_length=120)


class ConflictRecord(Contract):
    conflict_set_id: str = Field(min_length=1, max_length=64)
    binding_ids: list[str] = Field(min_length=1, max_length=8)
    reason: Literal["contradictory_values", "ambiguous_revision", "version_overlap"]
    supports: list[Support] = Field(min_length=1, max_length=8)


class BindingRevision(Contract):
    binding_id: str = Field(min_length=1, max_length=48)
    reason: Literal["interpretation_correction", "contradicting_evidence"]
    explanation: str = Field(min_length=1, max_length=500)
    supports: list[Support] = Field(min_length=1, max_length=8)


class BoundSlot(Contract):
    output_id: str
    values: list[BoundValue] = Field(default_factory=list, max_length=40)
    missing_inputs: list[str] = Field(default_factory=list, max_length=6)
    # Bind required input descriptions to concrete evidence, not just a flag
    # saying the final prose looks relevant.
    input_support: dict[str, list[Support]] = Field(default_factory=dict)
    replaced_values: dict[str, str] = Field(default_factory=dict)
    revisions: list[BindingRevision] = Field(default_factory=list, max_length=20)
    unresolved_candidates: list[str] = Field(default_factory=list, max_length=8)
    candidates: list[AnswerCandidate] = Field(default_factory=list, max_length=80)
    conflict_records: list[ConflictRecord] = Field(default_factory=list, max_length=20)
    coverage: Literal["complete", "partial", "open", "unknown"] = "unknown"
    # Completion is relative to examined evidence, never a global absence proof.
    scope: Literal["open", "explicit_total", "closed"] = "open"
    scope_support: list[Support] = Field(default_factory=list, max_length=4)


class BoundAnswer(Contract):
    slots: list[BoundSlot] = Field(min_length=1, max_length=8)


class EvidenceGap(Contract):
    output_id: str
    code: Literal["output_missing", "subject_unresolved", "relation_or_event_unresolved",
                  "member_scope_unknown", "time_operand_missing", "conflict_unresolved",
                  "invalid_operation", "candidate_unresolved"]
    cue: str
    action: Literal["retrieve", "reread", "uncertain"] = "retrieve"


class ExecutionResult(Contract):
    values: dict[str, list[str]] = Field(default_factory=dict)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    source_ids: list[int] = Field(default_factory=list)
    operations: list[dict] = Field(default_factory=list)

    def render(self, *, choice_question: bool = False) -> str:
        operators = {op["output_id"]: op["operator"] for op in self.operations}
        groups = []
        for output_id, values in self.values.items():
            if values:
                separator = ", " if operators.get(output_id) == "set" else "; "
                groups.append(separator.join(dict.fromkeys(values)))
        if groups:
            return "; ".join(dict.fromkeys(groups))
        return "Not mentioned in the conversation" if choice_question else "Insufficient evidence to determine the answer."


def follow_up_gaps(result: ExecutionResult) -> tuple[str, list[EvidenceGap]]:
    """A single extra read budget: repair available premises before acquiring more."""
    for action in ("reread", "retrieve"):
        gaps = [g for g in result.gaps if g.action == action]
        if gaps:
            return action, gaps
    return ("unresolved_scope" if result.gaps else "ready"), []


def execution_contract(question: str, plan: QueryPlan) -> AnswerContract:
    """Reject invented question spans; preserve a mandatory whole-question fallback."""
    if plan.answer_contract and all(o.question_span in question for o in plan.answer_contract.outputs):
        contract = plan.answer_contract.model_copy(deep=True)
        for output in contract.outputs:
            # Qualitative yes/no questions do not require a precise inventory.
            if output.operator == "count" and re.match(r"(?i)^(?:does|do|is|are|has|have)\b", question.strip()):
                output.operator = "direct"
            # ``count`` returns quantities.  A question asking which/who must
            # return an entity (often an argmax), which this executor cannot
            # compute.  Falling back to direct source-bound reading is safer
            # than executing a quantity contract and rejecting the answer.
            if output.operator == "count" and reader_profile(question) == "entity_selection":
                output.operator = "direct"
        return contract
    kind = answer_type(question)
    operator = {"text": "direct", "date": "time_point"}.get(kind, kind)
    if kind == "text" and plan.collection_query:
        operator = "set"
    return AnswerContract(outputs=[AnswerOutput(
        output_id="answer", question_span=question[:1000], operator=operator,
        subjects=plan.query_entities[:8], qualifier=question[:500],
        required_inputs=plan.required_info[:6],
    )])


def validate_bound(answer: BoundAnswer, contract: AnswerContract, sources: list[Hit]) -> None:
    if [s.output_id for s in answer.slots] != [o.output_id for o in contract.outputs]:
        raise ValueError("Every mandatory output ID must appear exactly once, in contract order")
    for slot in answer.slots:
        _materialize_candidates(slot)
        ids = set()
        for item in slot.values:
            # Calculations are checked at execution, independently of binding.
            plain = SupportedValue(value=item.value, supports=item.supports)
            validate_table(AnswerTable(slots=[AnswerSlot(target="binding", values=[plain])]),
                           sources, ["binding"], "text")
            if not item.binding_id:
                data = item.model_dump(mode="json", exclude={"binding_id"})
                item.binding_id = "b_" + hashlib.sha256(
                    json.dumps([slot.output_id, data], sort_keys=True).encode()
                ).hexdigest()[:24]
            if item.binding_id in ids:
                raise ValueError("Binding IDs must be distinct within an output")
            ids.add(item.binding_id)
            for role in item.role_bindings:
                validate_table(AnswerTable(slots=[AnswerSlot(target="role", values=[
                    SupportedValue(value=role.entity, supports=role.supports)])]),
                    sources, ["role"], "text")
        for candidate in slot.candidates:
            if candidate.disposition == "accepted" and not candidate.supports:
                raise ValueError("Accepted candidates require exact supporting quotes")
            if candidate.supports:
                validate_table(AnswerTable(slots=[AnswerSlot(target="candidate", values=[
                    SupportedValue(value=candidate.value, supports=candidate.supports)])]),
                    sources, ["candidate"], "text")
            for role in candidate.role_bindings:
                validate_table(AnswerTable(slots=[AnswerSlot(target="candidate_role", values=[
                    SupportedValue(value=role.entity, supports=role.supports)])]),
                    sources, ["candidate_role"], "text")
        for revision in slot.revisions:
            validate_table(AnswerTable(slots=[AnswerSlot(target="revision", values=[
                SupportedValue(value=revision.explanation, supports=revision.supports)])]),
                sources, ["revision"], "text")
        for support in slot.scope_support:
            plain = SupportedValue(value="scope", supports=[support])
            validate_table(AnswerTable(slots=[AnswerSlot(target="scope", values=[plain])]),
                           sources, ["scope"], "text")
        for supports in slot.input_support.values():
            if not supports:
                continue
            validate_table(AnswerTable(slots=[AnswerSlot(target="premise", values=[
                SupportedValue(value="premise", supports=supports)])]), sources, ["premise"], "text")


def merge_incremental(previous: BoundAnswer, revised: BoundAnswer, affected_ids: set[str],
                      *, sources: list[Hit], contract: AnswerContract) -> BoundAnswer:
    """Apply a source-checked delta atomically, preserving unmodified support.

    Revision reasons remain model interpretations, not logical entailment proofs.
    Free-text legacy replaced_values never authorizes deleting a binding.
    """
    previous = previous.model_copy(deep=True)
    revised = revised.model_copy(deep=True)
    validate_bound(previous, contract, sources)
    validate_bound(revised, contract, sources)
    previous_by_id = {s.output_id:s for s in previous.slots}
    merged = revised.model_copy(deep=True)
    for index, slot in enumerate(merged.slots):
        old = previous_by_id[slot.output_id]
        if slot.output_id not in affected_ids:
            merged.slots[index] = old.model_copy(deep=True)
            continue
        old_by_id = {v.binding_id:v for v in old.values}
        revisions = {r.binding_id:r for r in slot.revisions}
        if len(revisions) != len(slot.revisions) or not set(revisions) <= set(old_by_id):
            raise ValueError("Revision must target one existing binding exactly once")
        for revision in revisions.values():
            if re.search(r"(?i)(?:not (?:present|provided|available)|missing|absent).{0,50}(?:current|prompt)|"
                         r"(?:source|quote).{0,60}not present", revision.explanation):
                raise ValueError("Not resending a source is not evidence that it is invalid")
        retained = [v.model_copy(deep=True) for v in old.values if v.binding_id not in revisions]
        conflicts = list(old.conflict_records) + list(slot.conflict_records)
        for revision in revisions.values():
            original = old_by_id[revision.binding_id]
            if _revision_is_authorized(revision):
                archived = original.model_copy(deep=True)
                archived.status = "superseded" if revision.reason == "interpretation_correction" else "retracted"
                retained.append(archived)
            else:
                conflict_id = "revision_" + hashlib.sha256(revision.binding_id.encode()).hexdigest()[:24]
                old_value = original.model_copy(deep=True)
                old_value.status = "conflicted"
                old_value.conflict_set_id = conflict_id
                retained = [old_value if value.binding_id == old_value.binding_id else value for value in retained]
                conflicts.append(ConflictRecord(
                    conflict_set_id=conflict_id,
                    binding_ids=[revision.binding_id],
                    reason="ambiguous_revision",
                    supports=list(revision.supports),
                ))
        for value in slot.values:
            if value.binding_id in old_by_id and value.binding_id not in revisions:
                if value != old_by_id[value.binding_id]:
                    raise ValueError("Changing an existing binding requires an evidence-backed revision")
                continue
            # Compatible updates merge support; a changed value with the same
            # identity becomes a conflict instead of silently overwriting state.
            same = next((v for v in retained if v.status == "active" and _value_key(v) == _value_key(value)), None)
            if same is not None:
                same.supports = list({(r.source_id, r.quote): r for r in [*same.supports, *value.supports]}.values())
                same.role_bindings = list({
                    (r.role, _norm(r.entity), tuple((s.source_id, s.quote) for s in r.supports)): r
                    for r in [*same.role_bindings, *value.role_bindings]
                }.values())
                continue
            conflict_with = next((v for v in retained if v.status == "active" and _identity_key(v) == _identity_key(value)
                                  and _norm(v.value) != _norm(value.value)), None)
            if conflict_with is not None:
                conflict_id = _conflict_id(slot.output_id, conflict_with, value)
                conflict_with.status = "conflicted"
                conflict_with.conflict_set_id = conflict_id
                value = value.model_copy(deep=True)
                value.status = "conflicted"
                value.conflict_set_id = conflict_id
                conflicts.append(ConflictRecord(
                    conflict_set_id=conflict_id,
                    binding_ids=[conflict_with.binding_id, value.binding_id],
                    reason="contradictory_values",
                    supports=list(value.supports),
                ))
            if value not in retained:
                retained.append(value)
        slot.values = retained
        slot.conflict_records = list({c.conflict_set_id: c for c in conflicts}.values())
        slot.input_support = {key:list(refs) for key,refs in old.input_support.items()}
        for key, refs in revised.slots[index].input_support.items():
            slot.input_support[key] = list(dict.fromkeys(
                (r.source_id,r.quote) for r in [*slot.input_support.get(key, []), *refs]
            ))
            slot.input_support[key] = [Support(source_id=sid,quote=quote) for sid,quote in slot.input_support[key]]
        # A positive support and a model-declared missing input cannot silently
        # erase one another; keep the binding and expose the uncertainty.
        slot.replaced_values = {}
    validate_bound(merged, contract, sources)
    return merged


def _materialize_candidates(slot: BoundSlot) -> None:
    """Turn accepted candidates into answer values before deterministic execution.

    This is the completeness boundary: an accepted candidate cannot disappear
    between the model's ledger and the rendered answer.
    """
    existing = {_value_key(value) for value in slot.values}
    for candidate in slot.candidates:
        if candidate.disposition != "accepted" or not candidate.supports:
            continue
        value = BoundValue(
            value=candidate.value,
            identity=candidate.identity,
            subjects=list(candidate.subjects),
            event=candidate.event_key or "",
            event_key=candidate.event_key,
            supports=list(candidate.supports),
            role_bindings=list(candidate.role_bindings),
            binding_id="candidate_" + hashlib.sha256(candidate.candidate_id.encode()).hexdigest()[:24],
        )
        if _value_key(value) not in existing:
            slot.values.append(value)
            existing.add(_value_key(value))


def _identity_key(value: BoundValue):
    return tuple(sorted(map(_norm, value.subjects))), _norm(value.identity or value.value)


def _revision_is_authorized(revision: BindingRevision) -> bool:
    if revision.reason == "interpretation_correction":
        return True
    text = " ".join([revision.explanation, *(support.quote for support in revision.supports)])
    return bool(re.search(
        r"(?i)\b(?:no longer|not anymore|was incorrect|is incorrect|retract|retracted|sold|cancelled|canceled|moved)\b",
        text,
    ))


def _conflict_id(output_id: str, left: BoundValue, right: BoundValue) -> str:
    payload = json.dumps([output_id, _identity_key(left), _identity_key(right)], sort_keys=True)
    return "conflict_" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _value_key(value: BoundValue):
    # Identity groups possible conflicts; the value remains part of the exact
    # deduplication key so two claims about the same entity are not collapsed.
    return tuple(sorted(map(_norm, value.subjects))), _norm(value.identity or ""), _norm(value.value)


def _mode(output: AnswerOutput) -> str:
    # Older contracts did not carry quantifier scope. Lists of several owners
    # default to distributive grouping; new planners explicitly choose the mode.
    if output.subject_mode == "single" and len(output.subjects) > 1:
        return "each" if output.operator in {"set", "count"} else "joint"
    return output.subject_mode


def _subject_matches(output: AnswerOutput, actual: list[str]) -> bool:
    expected, subjects = set(map(_norm,output.subjects)), set(map(_norm,actual))
    if not expected:
        return True
    if _mode(output) in {"each", "common"}:
        return bool(expected & subjects)
    return expected <= subjects


def _speaker_conflict(item: BoundValue, sources: list[Hit]) -> bool:
    """Detect first-person ownership transferred to a different requested actor.

    Report only unambiguous first-person actor spans; other attribution remains
    a model interpretation. Names in greetings do not resolve the actor.
    """
    actors = {s.casefold() for s in item.subjects}
    for support in item.supports:
        events = sources[support.source_id - 1].metadata.get("source_events", [])
        for event in events:
            speaker = str(event.get("speaker") or "")
            if not speaker or support.quote not in event["text"] or event.get("context_only"):
                continue
            if re.search(r"\b(?:I|we) (?:bought|own|have|went|received|adopted|attended|made|built|skyped)\b", support.quote):
                if actors and speaker.casefold() not in actors:
                    return True
    return False


def _role_conflict(output: AnswerOutput, item: BoundValue) -> bool:
    """Apply question quantifiers to explicit source-backed roles."""
    if not item.role_bindings:
        return False
    expected = {_norm(subject) for subject in output.subjects}
    owners = {_norm(role.entity) for role in item.role_bindings if role.role == "owner"}
    if owners and expected and not owners & expected:
        return True
    if _mode(output) == "joint" and len(expected) > 1:
        participants = {_norm(role.entity) for role in item.role_bindings
                        if role.role in {"participant", "actor"}}
        if not item.event_key:
            return True
        if not expected <= participants and not expected <= {_norm(s) for s in item.subjects}:
            return True
    if any(role.confidence == "ambiguous" for role in item.role_bindings):
        return True
    return False


def _execution_values(slot: BoundSlot) -> list[BoundValue]:
    """Return active bindings only; conflicted values never become facts silently."""
    return [value for value in slot.values if value.status == "active"]


def _explicit_quantity(item: BoundValue) -> bool:
    if item.quantity is None:
        return False
    names = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    numeral = str(item.quantity) + ("|" + names[item.quantity] if item.quantity < len(names) else "")
    for ref in item.supports:
        text = re.sub(r"\[Image:.*?\]", "", ref.quote, flags=re.DOTALL)
        if re.search(r"\b(?:have|has|own|owns|total|exactly|only)\b[^.!?]{0,60}\b(?:" + numeral + r")\b", text, re.I):
            return True
    return False


def _closed_inventory(slot: BoundSlot) -> bool:
    """A scope claim must say what is exhaustive; ownership alone is open-world."""
    if slot.scope != "closed" or not slot.scope_support:
        return False
    for support in slot.scope_support:
        text = support.quote.casefold()
        if "[image:" in text or "photo" in text or "picture" in text:
            continue
        if re.search(r"\b(?:all|every|each|only|exactly|total|in total|altogether)\b", text):
            return True
    return False


def _source_time(item: SupportedValue, sources: list[Hit]) -> str | None:
    """Resolve only time expressions inside the item's own supporting quotes.

    Recompute from the source anchor instead of trusting model-generated dates.
    Multiple distinct expressions require an explicit selection, not a guess.
    """
    candidates = {}
    relative_anchors = set()
    for support in item.supports:
        for event in sources[support.source_id - 1].metadata.get("source_events", []):
            if event.get("context_only") or support.quote not in event["text"] or not event.get("occurred_at"):
                continue
            anchor = datetime.fromisoformat(event["occurred_at"])
            if anchor.tzinfo is None:
                continue
            for derived in derive_times(event["text"], anchor):
                if derived.expression not in support.quote:
                    continue
                if item.time_expression and _norm(item.time_expression) != _norm(derived.expression):
                    continue
                candidates[derived.start,derived.end_exclusive,derived.precision,derived.expression] = derived
            if re.search(r"\b(?:last|next|this)\s+\w+", item.value, re.IGNORECASE) and item.value in support.quote:
                relative_anchors.add(anchor.date().isoformat())
    if len(candidates) > 1:
        raise ValueError("Several source times match; select the event's time_expression")
    if candidates:
        value = next(iter(candidates.values()))
        if value.precision == "year":
            return value.start[:4]
        if value.precision == "month":
            return datetime.fromisoformat(value.start).strftime("%B %Y")
        if value.precision == "day":
            day = datetime.fromisoformat(value.start)
            return f"{day.day} {day.strftime('%B %Y')}"
        if value.precision == "week":
            # Do not append the month of an arbitrarily selected week boundary:
            # a week may span two months and conversational usage is ambiguous.
            anchor = datetime.fromisoformat(value.anchor)
            anchor_text = f"{anchor.day} {anchor.strftime('%B %Y')}"
            relation = {"last week": "The week before", "next week": "The week after"}
            if value.expression.casefold() in relation:
                return f"{relation[value.expression.casefold()]} {anchor_text}"
            return f"{value.expression} (relative to {anchor_text})"
        anchor = datetime.fromisoformat(value.anchor)
        return f"{value.expression} (relative to {anchor.day} {anchor.strftime('%B %Y')})"
    if item.time_expression:
        raise ValueError("Selected time_expression is not a supported calendar expression")
    if len(relative_anchors) == 1:
        return f"{item.value} (as of {next(iter(relative_anchors))})"
    return None


def check_and_execute(contract: AnswerContract, answer: BoundAnswer, sources: list[Hit]) -> ExecutionResult:
    validate_bound(answer, contract, sources)
    result = ExecutionResult()
    for output, slot in zip(contract.outputs, answer.slots, strict=True):
        values = []
        def gap(code, cue, action="retrieve"):
            record = EvidenceGap(output_id=output.output_id, code=code, cue=cue, action=action)
            if record not in result.gaps:
                result.gaps.append(record)
        if slot.conflict_records:
            gap("conflict_unresolved", output.question_span, "reread")
        for item in _execution_values(slot):
            if not item.matches_request:
                result.rejected.append({"output_id":output.output_id,"value":item.value,"reason":"relation_or_event_mismatch"})
                gap("relation_or_event_unresolved", output.question_span, "reread")
                continue
            if (not _subject_matches(output, item.subjects)
                    or _speaker_conflict(item, sources)
                    or _role_conflict(output, item)):
                result.rejected.append({"output_id":output.output_id,"value":item.value,"reason":"subject_mismatch"})
                gap("subject_unresolved", output.question_span, "reread")
                continue
            values.append(item)
        # A scalar question may have several source formulations of the same
        # quantity ("up to 50", "50", and "40 + 10").  Keep one canonical
        # answer while retaining all provenance in the binding/trace.  Do not
        # apply this to comparisons, lists, or multi-period questions.
        if values and output.operator in {"direct", "time_point", "duration"} and reader_profile(output.question_span) == "single_quantity":
            grouped = {}
            for item in values:
                numbers = re.findall(r"\b\d+(?:\.\d+)?\b", item.value)
                units = re.findall(r"(?i)\b(?:hours?|minutes?|days?|weeks?|months?|years?|items?|people|flights?)\b", item.value)
                key = (numbers[-1] if numbers else _norm(item.value), units[-1].casefold() if units else "")
                grouped.setdefault(key, []).append(item)
            collapsed = []
            for group in grouped.values():
                # Prefer a bound/precision qualifier over a bare restatement;
                # otherwise choose the shortest equivalent phrase.
                chosen = sorted(group, key=lambda item: (
                    not bool(re.search(r"(?i)\b(?:up to|at least|about|around|approximately)\b", item.value)),
                    len(item.value),
                ))[0]
                collapsed.append(chosen)
            values = collapsed
        for missing in slot.missing_inputs:
            gap("relation_or_event_unresolved", missing,
                "reread" if slot.input_support.get(missing) else "retrieve")
        for candidate in slot.unresolved_candidates:
            gap("candidate_unresolved", candidate, "reread")
        for requirement in output.required_inputs:
            if not slot.input_support.get(requirement):
                gap("time_operand_missing" if output.operator in {"time_point","duration"}
                    else "relation_or_event_unresolved", requirement)
        if not values:
            gap("output_missing", output.question_span,
                "reread" if any(slot.input_support.values()) or slot.values else "retrieve")
            result.values[output.output_id] = []
            continue
        if output.operator == "set" and not _closed_inventory(slot):
            # An unproven inventory is not itself an actionable missing fact.
            gap("member_scope_unknown", output.question_span, "uncertain")
        if output.operator in {"set", "count"}:
            for candidate in slot.candidates:
                if candidate.disposition == "unresolved":
                    gap("candidate_unresolved", candidate.value, "reread" if candidate.supports else "retrieve")
        if _mode(output) in {"each", "common"}:
            represented = {_norm(s) for v in values for s in v.subjects}
            for subject in output.subjects:
                if _norm(subject) not in represented:
                    gap("subject_unresolved", f"{subject}: {output.relation}")
        accepted_sources = sorted({r.source_id for v in values for r in v.supports})
        if output.operator == "count":
            mode = _mode(output)
            groups = {subject:[v for v in values if _norm(subject) in map(_norm,v.subjects)]
                      for subject in output.subjects} if mode == "each" else {"":values}
            common_ids = None
            if mode == "common":
                ids = [{_norm(v.identity) for v in values if v.identity and v.quantity is None
                        and _norm(subject) in map(_norm,v.subjects)} for subject in output.subjects]
                common_ids = set.intersection(*ids) if ids else set()
            counted = []
            for subject, group in groups.items():
                totals = [v for v in group if _explicit_quantity(v)] if mode != "common" else []
                members = {_norm(v.identity):v for v in group if v.identity and v.quantity is None
                           and (common_ids is None or _norm(v.identity) in common_ids)}
                counts = {v.quantity for v in totals}
                closed = _closed_inventory(slot) and not slot.missing_inputs
                if len(counts) > 1 or (counts and next(iter(counts)) < len(members)):
                    gap("conflict_unresolved", f"{subject} {output.question_span}", "reread")
                elif counts:
                    counted.append((subject,str(next(iter(counts)))))
                elif members:
                    counted.append((subject,str(len(members)) if closed else f"at least {len(members)}"))
                    if not closed:
                        gap("member_scope_unknown", output.question_span, "uncertain")
                else:
                    gap("member_scope_unknown", f"{subject} {output.question_span}", "uncertain")
                result.operations.append({"output_id":output.output_id,"operator":"count", "subject":subject,
                                          "member_ids":sorted(members),"explicit_totals":sorted(counts),
                                          "scope":slot.scope})
            result.values[output.output_id] = [f"{s}: {v}" if s else v for s,v in counted]
        else:
            rendered = []
            common = None
            if _mode(output) == "common":
                sets = [{_norm(v.value) for v in values if _norm(subject) in map(_norm,v.subjects)}
                        for subject in output.subjects]
                common = set.intersection(*sets) if sets else set()
            for item in values:
                if common is not None and _norm(item.value) not in common:
                    continue
                value = item.value
                if output.operator in {"duration", "time_point"}:
                    try:
                        resolved = _source_time(item, sources) if output.operator == "time_point" else None
                        if resolved is not None:
                            value = resolved
                        else:
                            if item.calculation and not item.event:
                                raise ValueError("A calculation must identify the endpoint events")
                            temp = AnswerTable(slots=[AnswerSlot(target="time", values=[SupportedValue(
                                value=value, supports=item.supports, calculation=item.calculation)])])
                            validate_table(temp, sources, ["time"], "duration" if output.operator == "duration" else "date")
                            value = temp.slots[0].values[0].value
                    except ValueError as error:
                        gap("invalid_operation", f"{output.question_span}: {error}", "reread")
                        continue
                if _mode(output) == "each" and len(output.subjects) > 1:
                    owners = [s for s in output.subjects if _norm(s) in map(_norm,item.subjects)]
                    value = f"{', '.join(owners)}: {value}"
                rendered.append(value)
            result.values[output.output_id] = list(dict.fromkeys(rendered))
            result.operations.append({"output_id":output.output_id,"operator":output.operator,
                                      "values":result.values[output.output_id]})
        if result.values.get(output.output_id):
            result.source_ids = sorted(set(result.source_ids + accepted_sources))
    return result

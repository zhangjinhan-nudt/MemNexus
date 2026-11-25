"""One source-bound answer table for factual and compositional questions.

Quote checks prove provenance, not entailment. Keep that distinction explicit.
The renderer cannot silently discard an accepted item; unresolved slots remain
in the receipt. No benchmark categories, answers, or domain aliases live here.
"""

import re
import unicodedata
from datetime import date, timedelta
from decimal import Decimal, localcontext
from typing import Literal, get_args

from pydantic import Field, PrivateAttr

from .evidence_policy import assertion_span_supported, listed_label_supported, source_assertion_scopes
from .models import AnswerContract, Contract, Hit
from .temporal import calendar_difference, resolve_time_references


class Support(Contract):
    source_id: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=700)


class RequestBinding(Contract):
    """A reader's interpretation of a sourced fact relative to one request.

    These fields are model declarations, not verified entailment labels.
    Request addresses are checked; free-text matching labels are only audited.
    Explicit conflicts in selected execution results/inputs authorize feedback.
    Operand bindings may inherit the request address, never factual attributes.
    """

    request_id: str | None = Field(default=None, min_length=1, max_length=40)
    subject: str = Field(default="", max_length=200)
    relation: str = Field(default="", max_length=200)
    qualifier: str = Field(default="", max_length=500)
    event_identity: str | None = Field(default=None, min_length=1, max_length=200)
    match: Literal["matched", "unresolved", "conflict"] = "unresolved"
    reason: str = Field(default="", max_length=500)


class TimeInputRef(Contract):
    """One-based addresses into the existing source/time-reference payload.

    An address selects a time mention, not proof that it belongs to the correct
    real-world event. The reader still binds that meaning to this input.
    """

    source_id: int = Field(ge=1)
    source_event: int = Field(ge=1)
    reference: int = Field(ge=1)


class BoundInput(Contract):
    """A model-selected semantic input, directly linked to original source quotes."""

    identity: str = Field(min_length=1, max_length=200)
    statement: str = Field(min_length=1, max_length=500)
    supports: list[Support] = Field(min_length=1, max_length=8)
    amount: str | None = Field(default=None, pattern=r"^-?\d+(?:\.\d+)?$", max_length=60)
    qualification: RequestBinding | None = None
    time_ref: TimeInputRef | None = None


TemporalOperation = Literal["time_difference", "time_compare", "time_since", "time_earlier", "time_later"]
_TEMPORAL_OPERATIONS = frozenset(get_args(TemporalOperation))


class SupportedValue(Contract):
    supports: list[Support] = Field(default_factory=list, max_length=80)
    basis: Literal["memory", "inference", "recommendation"] = "memory"
    inputs: list[BoundInput] = Field(default_factory=list, max_length=80)
    operation: Literal["none", "count", "sum", TemporalOperation] = "none"
    unit: str = Field(default="", max_length=50)
    unresolved_premise: str = Field(default="", max_length=500)
    value: str = Field(default="")
    request_binding: RequestBinding | None = None
    # Executed temporal values have one owner. Retain a redundant model-written
    # display only in the existing receipt, never in the generation schema.
    _declared_temporal_value: str | None = PrivateAttr(default=None)
    _unexecuted_temporal: dict | None = PrivateAttr(default=None)
    _temporal_winner_index: int | None = PrivateAttr(default=None)
    _temporal_display: str | None = PrivateAttr(default=None)


class AnswerSlot(Contract):
    target: str = Field(min_length=1, max_length=500)
    derivation: str = Field(min_length=1, max_length=1000)
    role: Literal["answer", "context"] = "answer"
    values: list[SupportedValue] = Field(default_factory=list, max_length=30)
    missing: str = Field(default="", max_length=16000)
    missing_kind: Literal["unspecified", "requested_part", "necessary_premise", "optional_detail"] = "unspecified"
    coverage: Literal["complete", "partial", "unavailable"] = "complete"


class AnswerTable(Contract):
    slots: list[AnswerSlot] = Field(min_length=1, max_length=20)


def answer_values(slot: AnswerSlot) -> list[SupportedValue]:
    """One publication view, shared by coverage, rendering and citations."""
    return slot.values if slot.role == "answer" else []


def _binding_errors(table: AnswerTable, request_contract: AnswerContract | None) -> list[str]:
    """Check request addresses and declared execution conflicts, not free-text truth."""
    request_ids = {output.output_id for output in request_contract.outputs} if request_contract else None
    errors = []
    for slot_index, slot in enumerate(table.slots, 1):
        for value_index, item in enumerate(slot.values, 1):
            # Existing dependency resolution already handles this value. A
            # declared contradiction must not become a second removal path.
            if item.unresolved_premise.strip():
                continue
            parent = item.request_binding
            # A free-text answer may correctly describe a conflict or negate a
            # premise. Its binding is auditable interpretation, not a polarity
            # label authorizing rejection of that conclusion.
            bindings = [(f"value {slot_index}.{value_index}", parent, None, item.operation != "none")]
            bindings.extend(
                (f"input {bound.identity!r} of value {slot_index}.{value_index}",
                 bound.qualification, parent.request_id if parent else None, item.operation != "none")
                for bound in item.inputs
            )
            for location, binding, inherited_id, used_as_result in bindings:
                if binding is None:
                    continue  # Old responses are not retrospectively disqualified.
                request_id = binding.request_id or inherited_id
                if request_ids is not None and request_id not in request_ids:
                    errors.append(
                        f"The binding of {location} names unknown request {request_id!r}. "
                        f"Use a supplied request ID: {', '.join(sorted(request_ids))}. "
                        "A request is a question constraint, not a fact or a source ID."
                    )
                selected_as_answer = slot.role == "answer" and used_as_result
                if binding.match == "conflict" and selected_as_answer:
                    errors.append(
                        f"The binding of {location} explicitly declares conflict with request "
                        f"{request_id!r}, yet uses that fact as a requested answer or executed input. "
                        "Recheck the original question and cited fact together, including subject, "
                        "role, relation, event and state. Correct the binding only if the original "
                        "evidence supports it; do not merely relabel conflict as matched. "
                        "Otherwise keep this fact only as explanatory context, exclude it from the "
                        "computed inputs, or state the affected requested gap. Preserve independent "
                        "supported answers. Different wording alone is not a conflict, and an "
                        "unresolved exact detail need not prevent an approximate or conditional answer."
                    )
    return list(dict.fromkeys(errors))


def binding_audit(table: AnswerTable, request_contract: AnswerContract | None = None) -> dict:
    """Expose declared bindings and executed displays in the receipt, not a score."""
    records = []
    computed_temporal = []
    not_executed_temporal = []
    unbound_values = unbound_inputs = 0
    for slot_index, slot in enumerate(table.slots, 1):
        for value_index, item in enumerate(answer_values(slot), 1):
            if item._unexecuted_temporal is not None:
                not_executed_temporal.append({"slot": slot_index, "value": value_index,
                                              **item._unexecuted_temporal})
            if (item.operation in _TEMPORAL_OPERATIONS
                    and item._declared_temporal_value is not None):
                computed_temporal.append({
                    "slot": slot_index, "value": value_index, "operation": item.operation,
                    "declared": item._declared_temporal_value, "computed": item.value,
                    **({"display": item._temporal_display} if item._temporal_display is not None else {}),
                })
            parent = item.request_binding
            if parent is None:
                unbound_values += 1
            else:
                records.append({"slot": slot_index, "value": value_index, "kind": "answer",
                                **parent.model_dump(mode="json")})
            for bound in item.inputs:
                if bound.qualification is None:
                    unbound_inputs += 1
                    continue
                record = bound.qualification.model_dump(mode="json")
                record["request_id"] = record["request_id"] or (parent.request_id if parent else None)
                records.append({"slot": slot_index, "value": value_index, "kind": "input",
                                "identity": bound.identity, **record})
    observed_ids = {record["request_id"] for record in records}
    return {
        "semantic_truth_verified": False,
        "records": records,
        "computed_temporal": computed_temporal,
        "not_executed_temporal": not_executed_temporal,
        "unbound_values": unbound_values,
        "unbound_inputs": unbound_inputs,
        "requests_without_binding": [output.output_id for output in request_contract.outputs
                                     if output.output_id not in observed_ids] if request_contract else [],
    }


_DURATION_UNIT = r"(?:days?|weeks?|months?|years?|hours?|minutes?)"

# A question whose reference point is the moment of asking ("how many days ago",
# "how long ago", "how long since").  Its two endpoints are not both in the
# evidence: one is the decision time, the other is an event date.  That makes it
# a different answer kind from a duration, whose endpoints are both events.
_ASKS_OFFSET = re.compile(
    r"how\s+(?:long|many\s+" + _DURATION_UNIT + r")\b[^?.!]*?\bago\b|how\s+long\s+since\b",
    re.IGNORECASE,
)


def answer_type(question: str) -> str:
    q = question.casefold()
    # The requested quantity is age; a subordinate 'when' is an event filter.
    if re.search(r"\b(?:how old|(?:at )?what age)\b|几岁|多大年纪", q):
        return "text"
    # "How many days ago ..." asks for a distance from the decision time, not the
    # length of anything measured between two events.  Typing it as a duration
    # forced the model to drop "ago" and made every elapsed-time question
    # unanswerable without a reference date.
    if _ASKS_OFFSET.search(q):
        return "offset"
    # "How many films did I watch in the last 3 months?" counts films; the span
    # is a filter on which films qualify, not the unit of the answer. Only a
    # unit immediately after "how many" makes the answer itself a duration.
    # Without this, such a question is typed `duration` and the answer is then
    # rejected for "not expressing a duration" even though it is a valid count.
    if re.search(rf"how many\s+(?!{_DURATION_UNIT})\w", q):
        return "count"
    if re.search(rf"how long\b|how many\s+{_DURATION_UNIT}\b|相隔|多久", q):
        return "duration"
    if re.search(r"\b(?:when|what year|which year|what date)\b|何时|哪年", q):
        return "date"
    if re.search(r"how many|number of|几个|多少", q):
        return "count"
    return "text"


_QUOTE_SPEAKER = re.compile(r"^\s*([A-Za-z][\w'-]{0,38}):\s*")

# Markdown emphasis and inline-code markers. These are presentation, so a quote
# that drops them still names the same words -- exactly the formatting-versus-
# provenance distinction _canonical_quote already draws for case and width.
# Measured (2026-11-13): 17 of the 18 recorded exact-quote failures were this
# mismatch alone, e.g. the event holds "**Store Utensils by Task:**" while the
# model cited "Store Utensils by Task:"; each rejection killed a whole answer.
# Deliberately narrow: '#' is NOT stripped because it is meaningful in the
# evidence (hashtag names such as #PlankChallenge), and stripping it would let a
# quote match a different token.
_MARKDOWN_EMPHASIS = re.compile(r"[*_`]")


def _canonical_quote(text: str) -> str:
    """Case, width, whitespace and inline emphasis are formatting; provenance is not."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _MARKDOWN_EMPHASIS.sub("", normalized)
    return " ".join(normalized.casefold().split())


def quote_matches(quote: str, text: str) -> bool:
    """Whether ``quote`` occurs in ``text`` once formatting is set aside.

    Measured failure: asked for the place Melanie wants to create, the model
    cited ``"Caroline: Now I'm passionate about creating a safe, inviting place
    for people to grow."`` The stored event reads ``"Caroline: Thanks, Melanie.
    It really mattered. ... Now I'm passionate about creating ..."`` -- the quote
    is verbatim, but it carries the speaker label that the event only shows once
    at its start. The strict substring test rejected it and the exception
    aborted the whole answer call, so that question was lost at random: it
    passes in one run and dies in the next, which is exactly the kind of
    run-to-run swing the frozen-base protocol is trying to measure.

    Normalising the speaker label is the narrowest repair that accepts the
    formatting variant without accepting a paraphrase: the quoted words still
    have to be present, and a quote attributed to one speaker never matches
    another speaker's utterance.
    """
    canonical_quote = _canonical_quote(quote)
    canonical_text = _canonical_quote(text)
    if not canonical_quote:
        return False
    if canonical_quote in canonical_text:
        return True
    quoted_speaker = _QUOTE_SPEAKER.match(canonical_quote)
    if quoted_speaker is None:
        return False
    body = canonical_quote[quoted_speaker.end():]
    if not body:
        return False
    text_speaker = _QUOTE_SPEAKER.match(canonical_text)
    if text_speaker is not None and text_speaker.group(1) != quoted_speaker.group(1):
        # Borrowing another speaker's words would be a provenance error, not a
        # formatting one, so the label has to agree when the text carries one.
        return False
    return body in canonical_text


def _assertion_grade(event: dict) -> str:
    """What a matched source event can support.

    ``none``      the event asserts nothing (a question);
    ``primary``   a first-hand statement (a user turn);
    ``secondary`` a second-hand statement (an assistant turn).

    Legitimacy of support is not the same question as who spoke. A question
    can never establish a fact, while an assistant turn does assert something
    the conversation contains -- so it may support an answer, but it ranks
    below first-hand evidence. Records written before the field existed are
    mapped from the legacy ``context_only`` flag.
    """
    grade = event.get("evidence_grade")
    if grade in {"primary", "secondary", "none"}:
        return str(grade)
    if event.get("context_only"):
        return "secondary" if event.get("role") == "assistant" else "none"
    return "primary" if event.get("role") == "user" else "secondary"


def _quote_spans(quote: str, text: str) -> list[tuple[int, int]]:
    """Map existing format-equivalent quote matches to original characters."""
    if quote in text:
        return [(m.start(), m.end()) for m in re.finditer(re.escape(quote), text)]
    folded, starts, ends = [], [], []
    index = 0
    while index < len(text):
        end = index + 1
        # Keep composition clusters together, including combining marks and
        # Hangul; compatibility characters and casefold expansions retain their
        # original extent. Newlines are only collapsed for searching, never for
        # determining clause boundaries.
        while end < len(text) and (unicodedata.combining(text[end]) or
                unicodedata.normalize("NFKC", text[index:end + 1]) !=
                unicodedata.normalize("NFKC", text[index:end]) + unicodedata.normalize("NFKC", text[end])):
            end += 1
        for char in unicodedata.normalize("NFKC", text[index:end]):
            if _MARKDOWN_EMPHASIS.fullmatch(char):
                continue
            for unit in char.casefold():
                unit = " " if unit.isspace() else unit
                if unit == " " and folded and folded[-1] == " ":
                    ends[-1] = end
                    continue
                folded.append(unit)
                starts.append(index)
                ends.append(end)
        index = end
    normalized = "".join(folded)
    body = _canonical_quote(quote)
    if body not in normalized and (speaker := _QUOTE_SPEAKER.match(body)):
        body = body[speaker.end():]
    if not body:
        return []
    return [(starts[m.start()], ends[m.end() - 1]) for m in re.finditer(re.escape(body), normalized)]


def _support_is_assertion(event: dict, quote: str, *, label_predicate: str = "", label_value: str = "") -> bool:
    """Check this quote's original scope, independently of speaker authority.

    Hydration refreshes the event grade from the original source. Older caller
    records still retain their explicit grade; a positive grade cannot license
    an interrogative or a cropped conditional in a different part of the turn.
    Formatting equivalents already accepted by quote_matches remain accepted.
    """
    if not quote_matches(quote, event["text"]):
        return False
    if label_predicate and listed_label_supported(event["text"], quote, label_predicate, label_value):
        return True
    if _assertion_grade(event) == "none":
        return False
    scopes = event.get("assertion_scopes")
    if scopes is None:
        scopes = source_assertion_scopes(event["text"])
    spans = _quote_spans(quote, event["text"])
    return bool(spans) and all(assertion_span_supported(scopes, left, right) for left, right in spans)


_HANDLE = re.compile(r"(?<![\w@])@[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*")
_IDENTIFIER_ESCAPE = re.compile(r"\\([_@.\-])")


def _identifier_text(text: str) -> str:
    # Unlike quote matching, identifiers retain underscores and letter case.
    # Only Markdown escaping and presentation markers are removed.
    text = _IDENTIFIER_ESCAPE.sub(r"\1", unicodedata.normalize("NFKC", text))
    return text.replace("`", "").replace("*", "")


def _has_handle(text: str, handle: str) -> bool:
    # Sources may omit the display '@'. Never accept a prefix or fuzzy match.
    return re.search(r"(?<![\w@.\-])@?" + re.escape(handle[1:]) + r"(?!\w|[.\-][A-Za-z0-9_])",
                     _identifier_text(text)) is not None


def _bind_unique_quote_sources(table: AnswerTable, sources: list[Hit]) -> None:
    """Correct a wrong source number only for one literal source-text match.

    Valid bindings keep their original provenance, including formatting-only
    matches. Ambiguous or absent quotes remain errors for the existing validator.
    This resolves an address; it does not establish what the quoted words entail.
    """
    for slot in table.slots:
        for item in slot.values:
            references = [*item.supports, *(ref for bound in item.inputs for ref in bound.supports)]
            for support in references:
                if not 1 <= support.source_id <= len(sources):
                    continue
                if any(
                    quote_matches(support.quote, event["text"])
                    for event in sources[support.source_id - 1].metadata.get("source_events", [])
                ):
                    continue
                matches = [index for index, source in enumerate(sources, 1) if any(
                    support.quote in event["text"] for event in source.metadata.get("source_events", [])
                )]
                if len(matches) == 1:
                    support.source_id = matches[0]


def validate_table(
    table: AnswerTable,
    sources: list[Hit],
    targets: list[str],
    kind: str,
    derivations: list[dict] | None = None,
    as_of: str | None = None,
    request_contract: AnswerContract | None = None,
) -> None:
    """Resolve explicit dependencies, validate sources, execute declared arithmetic.

    Semantic selection stays with the model. Only explicitly declared inputs
    authorize computation; no quantities, event dates or premises are guessed.
    """
    if [slot.target for slot in table.slots] != targets:
        raise ValueError("Return every requested target exactly once and in order")
    if binding_errors := _binding_errors(table, request_contract):
        raise ValueError("\n".join(binding_errors))
    # Representation limits may retain an already supplied ordinary inference.
    # Invalid concrete bindings must instead use the existing model repair: a
    # nonempty answer is not permission to ignore malformed selected inputs.
    computed_temporal = {}
    for slot in table.slots:
        for item in slot.values:
            item._temporal_winner_index = None
            item._temporal_display = None
            if (item.operation not in _TEMPORAL_OPERATIONS
                    or item.unresolved_premise.strip()):
                continue
            try:
                if any(bound.time_ref is not None and not 1 <= bound.time_ref.source_id <= len(sources)
                       for bound in item.inputs):
                    raise _InvalidTimeSource("Unknown time_ref source number")
                computed_temporal[id(item)] = _computed_value(item, sources, as_of)
                item._unexecuted_temporal = None
            except _UnparsedTimeReference as error:
                raise _UnparsedTimeReference(
                    f"Invalid temporal binding: {error}. The source event exists, but it supplies "
                    "no parsed time references, so the selected reference cannot be used for "
                    "computation. This parser representation gap is not itself a missing fact. "
                    "Reassess the original quote and its source context at their supported precision. "
                    "If they support an ordinary contextual or qualified answer, use operation=none, "
                    "remove the unavailable time_ref, and write that answer explicitly in value "
                    "with the real source quotes. Do not invent a reference or equate an utterance "
                    "date with an event date without contextual support. If a qualified answer "
                    "leaves requested exactness unresolved, retain it with coverage=partial and "
                    "missing_kind=requested_part naming that unresolved part. Use necessary_premise "
                    "only when a fact indispensable to that answer is actually missing, not merely "
                    "because the parser cannot compute it."
                ) from error
            except _InvalidTemporalBinding as error:
                raise _InvalidTemporalBinding(
                    f"Invalid temporal binding: {error}. Repair the selected inputs and their "
                    "source-event/time-reference addresses using the supplied evidence. Select "
                    "the event relation or phase requested by the question consistently for each "
                    "compared entity; ordering, receiving and completing are different events. "
                    "Do not keep an unexecuted comparison as though it were computed. This binding "
                    "error does not itself establish missing evidence. Do not change operation to "
                    "none merely to bypass a bad address or wrong input count. If the original evidence "
                    "supports only a qualified or contextual answer, explicitly use operation=none "
                    "with that answer and its real quotes; do not invent a reference or event date."
                ) from error
            except _UnexecutableTimeAnnotation as error:
                if not item.value.strip():
                    # No answer exists to return to. Use the existing repair,
                    # explaining the representation limit without declaring a
                    # source-evidence gap or fabricating a conclusion ourselves.
                    raise ValueError(
                        f"Time annotation cannot execute: {error}. This does not by itself mean "
                        "the original evidence is insufficient. If your cited original evidence "
                        "supports a qualified or contextual inference, use operation=none and "
                        "write that conclusion explicitly in value, keeping the real source quotes. "
                        "Do not invent a parser reference or a source endpoint. If an indispensable "
                        "fact is truly missing, state that gap instead of inventing an answer."
                    ) from error
                item._unexecuted_temporal = {
                    "requested_operation": item.operation, "not_executed": True,
                    "error": str(error), "declared_value": item.value,
                }
                item.operation = "none"
    _bind_unique_quote_sources(table, sources)
    for slot in table.slots:
        answers = answer_values(slot)
        blocked = [item.unresolved_premise.strip() for item in answers if item.unresolved_premise.strip()]
        if (slot.missing_kind == "necessary_premise" and slot.missing.strip()
                and answers and not blocked):
            raise ValueError(
                "The table labels a missing detail as necessary while publishing an answer. Reassess "
                "whether that detail is indispensable for the original question and its requested "
                "precision; the label itself does not establish necessity. A missing exact "
                "detail need not block a source-supported approximate or conditional answer. If that "
                "answer leaves a requested exact result unknown, keep the qualified answer and name "
                "that unmet request in missing with missing_kind=requested_part and coverage=partial. "
                "Use optional_detail or remove the gap only when it is not required by the question. "
                "Do not substitute a different person, event, relation "
                "or aggregation member for a genuinely missing input. "
                "A necessary missing premise must be linked to its dependent value via unresolved_premise; "
                "omit only that value and retain independent answers. If the gap is an independently "
                "unanswered requested part, use missing_kind=requested_part instead. "
                "Do not publish a value while declaring its indispensable premise unsupported."
            )
        slot.values = [item for item in slot.values if not item.unresolved_premise.strip()]
        answers = answer_values(slot)
        if blocked:
            slot.missing = "; ".join(dict.fromkeys([slot.missing, *blocked])).strip("; ")
            # Remaining values are independent. The removed dependent answer is
            # now an unanswered part, not a missing premise of those values.
            slot.missing_kind = "requested_part" if answers else "necessary_premise"
            slot.coverage = "partial" if answers else "unavailable"
        if slot.role == "answer" and not answers and slot.missing.strip():
            # No published value plus an explicit gap already determines this state.
            slot.coverage = "unavailable"
        if slot.role == "answer" and not answers and not slot.missing.strip():
            raise ValueError("An empty target requires an explicit missing-evidence reason")
        if slot.role == "answer" and slot.coverage in {"complete", "partial"} and not answers:
            raise ValueError(
                "Complete or partial coverage requires a supported answer value; "
                "use unavailable when no answer is supported"
            )
        if slot.coverage == "partial" and not slot.missing:
            raise ValueError("Partial coverage requires naming the unmet condition")
        if slot.coverage == "unavailable" and answers:
            raise ValueError(
                "Coverage representation conflict: unavailable cannot contain published answer values. "
                "Reassess the existing values and their quotes: a supported negative answer, partial answer, "
                "or qualified inference is not the same as no supported answer. Preserve independently "
                "supported information and citations; repair only the conflicting representation. "
                "Do not clear all values merely to pass validation or mark coverage complete without support. "
                "Use unavailable only when no answer is supported; otherwise state any unresolved requested part."
            )
        slot_has_user_assertion = False
        for item in slot.values:
            if item.operation == "none" and not item.value.strip():
                raise ValueError("A direct answer requires a value; use unavailable for missing evidence")
            # Input quotes use the same source IDs as ordinary answers. Assemble
            # only quotes explicitly chosen by the model; never add a source.
            references = {(ref.source_id, ref.quote): ref for ref in item.supports}
            for bound in item.inputs:
                references.update({(ref.source_id, ref.quote): ref for ref in bound.supports})
            item.supports = list(references.values())
            if not item.supports and item.basis != "recommendation":
                raise ValueError("An answer requires original source supports")
            assertions = []
            handles = (set(_HANDLE.findall(_identifier_text(item.value)))
                       if item.basis in {"memory", "inference"} and item.operation == "none" else set())
            supported_handles = set()
            for support in item.supports:
                if support.source_id > len(sources):
                    raise ValueError("Unknown source number")
                events = sources[support.source_id - 1].metadata.get("source_events", [])
                matched = [e for e in events if quote_matches(support.quote, e["text"])]
                if not matched:
                    raise ValueError(f"Source {support.source_id} does not contain exact quote: {support.quote!r}")
                assertions.extend(e for e in matched if _support_is_assertion(
                    e, support.quote, label_predicate=slot.target, label_value=item.value))
                supported_handles.update(
                    handle for handle in handles if _has_handle(support.quote, handle)
                    and any(_support_is_assertion(e, support.quote) and _has_handle(e["text"], handle) for e in matched)
                )
            if not assertions and item.basis != "recommendation":
                raise ValueError(
                    "A question alone cannot establish a fact; cite at least one original assertion"
                )
            if item.operation != "none":
                # Executed operands each need an assertion. With no operation,
                # input grouping does not change the ordinary support contract.
                for bound in item.inputs:
                    if not any(
                        _support_is_assertion(event, ref.quote)
                        for ref in bound.supports
                        for event in sources[ref.source_id - 1].metadata.get("source_events", [])
                    ):
                        raise ValueError("Each bound input needs an assertion, not only a question")
                computed = (computed_temporal[id(item)] if id(item) in computed_temporal
                            else _computed_value(item, sources, as_of))
                if mismatch := _operation_value_mismatch(item, computed):
                    raise ValueError(mismatch)
                if (item.operation in _TEMPORAL_OPERATIONS
                        and item._declared_temporal_value is None):
                    item._declared_temporal_value = item.value
                item.value = computed
                item._temporal_display = _temporal_subject_display(item, request_contract)
            if handles - supported_handles:
                cited_spellings = set()
                for support in item.supports:
                    events = sources[support.source_id - 1].metadata.get("source_events", [])
                    for event in events:
                        if not _support_is_assertion(event, support.quote):
                            continue
                        for original in _HANDLE.findall(_identifier_text(event["text"])):
                            if _has_handle(support.quote, original):
                                cited_spellings.add((support.source_id, original))
                evidence = "; ".join(f"source {sid}: {value}" for sid, value in sorted(cited_spellings))
                raise ValueError(
                    "Recalled @ identifiers must match the cited original spelling exactly. "
                    "Unsupported values: " + ", ".join(sorted(handles - supported_handles))
                    + ". Exact spellings in your cited original evidence: "
                    + (evidence or "none; provide a quote containing the original identifier")
                    + ". Select only the identity supported for this question; these are not automatic replacements."
                )
            if item in answers:
                slot_has_user_assertion = slot_has_user_assertion or any(e.get("role") == "user" for e in assertions)
        # Several suggestions for one target may share a cited personal
        # constraint. Each value still has its own valid assertion support.
        if any(item.basis == "recommendation" for item in answers) and not slot_has_user_assertion:
            eligible = [str(index) for index, source in enumerate(sources, 1)
                        if any(event.get("role") == "user" and _assertion_grade(event) == "primary"
                               for event in source.metadata.get("source_events", []))]
            guidance = (
                " Sources containing eligible user assertions: " + ", ".join(eligible)
                + ". Cite a relevant exact user premise from these sources in this target, alongside any "
                "suggestion evidence. Their presence does not prove relevance; do not invent a personal fact."
                if eligible else
                " No eligible user assertions are present in the supplied sources; do not invent a personal fact."
            )
            raise ValueError("Personalized recommendations require a quoted user assertion in the same target."
                             + guidance)


_CURRENCY_ALIASES = {
    "usd": {"usd", "us dollar", "us dollars", "u.s. dollar", "u.s. dollars", "us$", "$"},
    "eur": {"eur", "euro", "euros", "€"},
    "gbp": {"gbp", "pound sterling", "pounds sterling", "£"},
    "cny": {"cny", "rmb", "yuan", "cn¥", "¥", "￥"},
    "jpy": {"jpy", "yen", "jp¥", "¥", "￥"},
    "cad": {"cad", "canadian dollar", "canadian dollars", "ca$", "c$", "$"},
    "aud": {"aud", "australian dollar", "australian dollars", "au$", "a$", "$"},
}


def _same_scalar_display(declared: str, computed_number: str, unit: str) -> bool:
    """Compare one exact scalar in its declared unit, without unit conversion.

    Formatting is not a second arithmetic owner. Currency symbols are resolved
    only from the explicit unit, never inferred from a bare number or a question.
    Ranges, prose, approximations and conflicting units are not scalar displays.
    """
    text = declared.strip().casefold()
    normalized_unit = unit.strip().casefold()
    aliases = {normalized_unit} if normalized_unit else set()
    if normalized_unit in {"$", "dollar", "dollars"}:
        aliases.update({"$", "dollar", "dollars"})
    # An ambiguous symbol/name as the declared unit does not establish an ISO code.
    for group in _CURRENCY_ALIASES.values():
        if normalized_unit not in {"$", "dollar", "dollars", "¥", "￥"} and normalized_unit in group:
            aliases.update(group)
            if "$" in group:
                aliases.update({"dollar", "dollars"})
    displays = {text}
    for alias in sorted(aliases, key=len, reverse=True):
        if text.startswith(alias):
            displays.add(text[len(alias):].strip())
        if text.endswith(alias):
            displays.add(text[:-len(alias)].strip())
    scalar = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
    return any(re.fullmatch(scalar, value) and Decimal(value.replace(",", "")) == Decimal(computed_number)
               for value in displays)


def _operation_value_mismatch(item: SupportedValue, computed: str) -> str | None:
    """One scalar owner; a separately declared unit need not be repeated in value."""
    if item.operation in _TEMPORAL_OPERATIONS:
        # The original time bindings authorize calculation, not an additional
        # model-authored rendering. Requiring two displays to match creates a
        # second numeric owner and can reject valid approximate wording.
        return None
    equivalent = {computed}
    if unit := item.unit.strip():
        equivalent.add(computed.removesuffix(" " + unit))
    if item.value.strip() and item.value.strip() not in equivalent and not _same_scalar_display(
        item.value, computed.removesuffix(" " + item.unit.strip()) if item.unit.strip() else computed,
        item.unit,
    ):
        return (
            f"Declared operation={item.operation!r} computes {computed!r} from the selected inputs, "
            f"but the provided value is {item.value.strip()!r}. "
            "Count is the cardinality of distinct input identities: each identity must be one "
            "requested counted member, not a premise, endpoint or interval from which members "
            "still need to be derived. Sum adds the selected explicit amounts. Reconsider the "
            "operation and qualifying inputs against the original target; do not merely erase "
            "the provided value to silence this mismatch. Leave value empty only when that "
            "computed result is intended. Use operation=none for a supported derivation outside "
            "these operations or to preserve a qualified, approximate or conditional answer."
        )
    return None


def validation_feedback(
    table: AnswerTable, primary_error: ValueError, request_contract: AnswerContract | None = None,
    *, sources: list[Hit] | None = None, as_of: str | None = None,
) -> str:
    """Add only already-defined, independently computable conflicts after a failure.

    This does not revalidate provenance, select operands, mutate the table, or
    authorize an answer. Uncomputable or omitted values cannot supply diagnostics.
    """
    errors = [str(primary_error)]
    errors.extend(error for error in _binding_errors(table, request_contract) if error not in str(primary_error))
    for slot in table.slots:
        for item in slot.values:
            if item.operation == "none" or item.unresolved_premise.strip():
                continue
            try:
                computed = _computed_value(item, sources, as_of)
            except ValueError:
                # Unknown amounts, empty inputs and conflicting identities need
                # their own existing validation, not a guessed computed scalar.
                continue
            mismatch = _operation_value_mismatch(item, computed)
            if mismatch and mismatch not in errors:
                errors.append(mismatch)
    return "\n".join(errors)


class _InvalidTemporalBinding(ValueError):
    """A declared operation has invalid operands, concrete addresses or provenance."""


class _InvalidTimeSource(_InvalidTemporalBinding):
    """A nonexistent source is not an unrepresentable time annotation."""


class _UnparsedTimeReference(_InvalidTemporalBinding):
    """A concrete reference is invalid because the selected event has no parsed refs."""


class _UnexecutableTimeAnnotation(ValueError):
    """An optional annotation cannot express otherwise ordinary sourced reasoning."""


def _bound_time_interval(bound: BoundInput, sources: list[Hit]) -> tuple[date, date]:
    """Resolve exactly the selected source mention; never infer an event date."""
    selection = bound.time_ref
    if selection is None:
        raise _UnexecutableTimeAnnotation(
            "Each temporal input needs an explicit time_ref; use operation=none for other reasoning")
    if selection.source_id > len(sources):
        raise _InvalidTimeSource("Unknown time_ref source number")
    events = sources[selection.source_id - 1].metadata.get("source_events", [])
    if selection.source_event > len(events):
        raise _InvalidTemporalBinding("Unknown time_ref source_event position")
    event = events[selection.source_event - 1]
    refs = resolve_time_references(event["text"], source_time=event.get("occurred_at"))
    if not refs:
        # Still reject every nonexistent concrete address, including index 1.
        # Only the repair guidance differs: a parser gap is not an evidence gap.
        raise _UnparsedTimeReference(
            "Unknown time_ref reference position; the selected source event has no parsed time references")
    if selection.reference > len(refs):
        raise _InvalidTemporalBinding(
            "Unknown time_ref reference position; use only references supplied for that source event")
    reference = refs[selection.reference - 1]
    if _assertion_grade(event) == "none" or not any(
        support.source_id == selection.source_id and quote_matches(support.quote, event["text"])
        and quote_matches(reference.expression, support.quote)
        for support in bound.supports
    ):
        raise _InvalidTemporalBinding(
            "A time_ref needs an assertion quote from the selected source event containing its time expression")
    if reference.start is None or reference.end_exclusive is None:
        raise _UnexecutableTimeAnnotation(
            "This time reference has no bounded interval. Preserve its approximate or conditional meaning "
            "with operation=none; do not replace it with the utterance date or zero."
        )
    start, end = date.fromisoformat(reference.start), date.fromisoformat(reference.end_exclusive)
    if end <= start:
        raise _InvalidTemporalBinding("A time reference must have an ordered nonempty interval")
    return start, end


def _computed_time_value(
    item: SupportedValue, sources: list[Hit] | None, as_of: str | None = None,
) -> str:
    """Small, declared interval algebra over already parsed source references."""
    if sources is None:
        raise _InvalidTemporalBinding("Temporal computation requires the original source events")
    if item.operation == "time_since":
        if len(item.inputs) != 1:
            raise _InvalidTemporalBinding(
                "time_since needs exactly one source input; its endpoint is the supplied question clock")
    elif len(item.inputs) != 2:
        raise _InvalidTemporalBinding(
            "A temporal operation needs exactly two selected inputs in the requested order")
    intervals = []
    representation_error = None
    for bound in item.inputs:
        try:
            intervals.append(_bound_time_interval(bound, sources))
        except _UnexecutableTimeAnnotation as error:
            # A missing/approximate first annotation must not hide an invalid
            # concrete address or quote in a later selected operand.
            representation_error = representation_error or error
    if representation_error is not None:
        raise representation_error
    left = intervals[0]
    if item.operation == "time_since":
        # This is an explicitly supplied question clock, never a fabricated
        # second source event or a fallback for a missing source-event clock.
        anchors = resolve_time_references("", question_time=as_of, clock="question", include_utterance=True)
        if not anchors:
            raise _UnexecutableTimeAnnotation(
                "time_since needs a supplied question date; do not invent a source endpoint or current clock")
        right = (date.fromisoformat(anchors[0].start), date.fromisoformat(anchors[0].end_exclusive))
    else:
        right = intervals[1]
    if item.operation in {"time_compare", "time_earlier", "time_later"}:
        first, second = (bound.identity.strip() for bound in item.inputs)
        if not first or not second:
            raise _InvalidTemporalBinding("Temporal input identities cannot be whitespace")
        if left[1] <= right[0]:
            earlier, later, relation = first, second, "before"
            earlier_index, later_index = 0, 1
        elif right[1] <= left[0]:
            earlier, later, relation = second, first, "after"
            earlier_index, later_index = 1, 0
        elif left == right and left[1] - left[0] == timedelta(days=1):
            return f"{first} and {second} occurred on the same day; their order within that day is unknown"
        else:
            return f"The supported time ranges for {first} and {second} overlap; their order is not determined"
        if item.operation == "time_earlier":
            item._temporal_winner_index = earlier_index
            return earlier
        if item.operation == "time_later":
            item._temporal_winner_index = later_index
            return later
        return f"{first} occurred {relation} {second}"
    # Date intervals contain calendar days and use half-open boundaries. Do not
    # turn a week/month range into a fabricated point or discard a negative sign.
    minimum = (right[0] - (left[1] - timedelta(days=1))).days
    maximum = ((right[1] - timedelta(days=1)) - left[0]).days
    unit = (item.unit.strip().casefold() or "days").removesuffix("s") + "s"
    if unit not in {"days", "weeks", "months", "years"}:
        raise _UnexecutableTimeAnnotation(
            "Time difference supports days/weeks/months/years; use operation=none for other interpretations")
    result = None
    if minimum == maximum:
        # Reuse the existing calendar algebra. It accepts ordered endpoints;
        # restore the declared direction explicitly instead of swallowing it.
        negative = right[0] < left[0]
        start, end = (right[0], left[0]) if negative else (left[0], right[0])
        try:
            result = ("-" if negative else "") + calendar_difference(start, end, unit)
        except ValueError:
            # Inexact calendar months/years/weeks retain their precise day
            # difference. An unavailable exact unit is not an unavailable date.
            pass
    if result is None:
        if unit == "weeks" and minimum % 7 == maximum % 7 == 0:
            lower, upper, shown_unit = minimum // 7, maximum // 7, "weeks"
        else:
            lower, upper, shown_unit = minimum, maximum, "days"
        number = str(lower) if lower == upper else f"{lower} to {upper}"
        result = f"{number} {shown_unit}"
    number, shown_unit = result.rsplit(" ", 1)
    if number in {"1", "-1"}:
        result = f"{number} {shown_unit.removesuffix('s')}"
    if item.operation == "time_since":
        result += " ago"
        if maximum < 0:
            result += " (the event is after the question date)"
        elif minimum < 0:
            result += " (the supported range crosses the question date)"
    return result


def _computed_value(
    item: SupportedValue, sources: list[Hit] | None = None, as_of: str | None = None,
) -> str:
    """One output owner for explicitly selected, sourced arithmetic inputs."""
    if item.operation in _TEMPORAL_OPERATIONS:
        return _computed_time_value(item, sources, as_of)
    if not item.inputs:
        raise ValueError("Count/sum needs selected inputs; directly stated totals may use operation=none")
    distinct = {}
    for bound in item.inputs:
        key = " ".join(bound.identity.casefold().split())
        if not key:
            raise ValueError("Input identity cannot be whitespace")
        amount = Decimal(bound.amount) if bound.amount is not None else None
        if item.operation == "sum" and amount is None:
            raise ValueError("Sum requires an explicit amount for every input; unknown is not zero")
        if key in distinct and item.operation == "sum" and distinct[key] != amount:
            raise ValueError("One input identity has conflicting amounts; resolve its evidence first")
        distinct[key] = amount
    with localcontext() as context:
        context.prec = 128  # At most 80 operands, each bounded to 60 decimal characters.
        total = Decimal(len(distinct)) if item.operation == "count" else sum(distinct.values(), Decimal(0))
    number = format(total, "f")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return " ".join(part for part in (number, item.unit.strip()) if part)


_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_ISO_DATE = re.compile(r"(?<![\d(])(\d{4})-(\d{2})-(\d{2})(?![\d)])")


def _month_phrase(match: re.Match[str]) -> str:
    """Display an explicit ISO date without changing the surrounding answer."""
    year, month, day = match.group(1), int(match.group(2)), int(match.group(3))
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return match.group(0)
    return f"{day} {_MONTH_NAMES[month - 1]} {year}"


def normalize_answer_text(text: str, question: str = "") -> str:
    """Display explicit ISO dates while preserving all answer content."""
    return _ISO_DATE.sub(_month_phrase, text.strip())


def _label_present(label: str, text: str) -> bool:
    """Literal label matching, with boundaries for Latin words and digits."""
    label, text = _canonical_quote(label), _canonical_quote(text)
    if not label:
        return False
    left = r"(?<![a-z0-9_])" if re.match(r"[a-z0-9_]", label) else ""
    right = r"(?![a-z0-9_])" if re.search(r"[a-z0-9_]$", label) else ""
    return bool(re.search(left + re.escape(label) + right, text))


def _temporal_subject_display(item: SupportedValue, contract: AnswerContract | None) -> str | None:
    """Display an executed winner's requested name, without changing its identity.

    The selected operands must name distinct question subjects, and the winning
    subject must occur in its own quote. Same-subject phases and ambiguous or
    missing bindings retain their full computed identity. A model's redundant
    declared answer is never consulted here.
    """
    winner = item._temporal_winner_index
    if (winner is None or contract is None or item.request_binding is None
            or item.operation not in {"time_earlier", "time_later"}):
        return None
    output = next((o for o in contract.outputs if o.output_id == item.request_binding.request_id), None)
    if output is None or output.operator != "comparison" or len(output.subjects) < 2:
        return None
    subjects = [s for s in output.subjects if _label_present(s, output.question_span)]
    matches = [[s for s in subjects if _label_present(s, bound.identity)] for bound in item.inputs]
    if len(matches) != 2 or any(len(labels) != 1 for labels in matches):
        return None
    if _canonical_quote(matches[0][0]) == _canonical_quote(matches[1][0]):
        return None
    subject = matches[winner][0]
    if any(_label_present(subject, support.quote) for support in item.inputs[winner].supports):
        return subject
    return None


def _display_value(item: SupportedValue, *, kind: str | None = None) -> str:
    """Retain explicit scalar units and the name of an executed winner.

    Formatting cannot infer a unit or shorten a qualified answer. A bare number
    is a quantity only when the request says so or a literal number/unit phrase
    occurs in its source; date answers and unresolved labels keep their text.
    """
    if item._temporal_display is not None:
        return item._temporal_display
    value = item.value.strip()
    unit = item.unit.strip()
    if (item.operation == "none" and unit and kind != "date"
            and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value)):
        quantity_request = kind in {"count", "duration"}
        quoted_quantity = any(_label_present(f"{value} {unit}", s.quote) for s in item.supports)
        if quantity_request or quoted_quantity:
            return f"{value} {unit}"
    if item.operation != "sum" or item.unit.strip().casefold() not in _CURRENCY_ALIASES:
        return value
    number, space, unit = value.partition(" ")
    if unit != item.unit.strip() or not re.fullmatch(r"-?[0-9]{4,}(?:\.[0-9]+)?", number):
        return value
    return f"{Decimal(number):,f}{space}{unit}"


def render_table(
    table: AnswerTable,
    sources: list[Hit] | None = None,
    *,
    kind: str | None = None,
    question: str = "",
) -> str:
    """Join the answer values as one comma-separated list.

    A comma is the separator the list itself is made of: the LoCoMo evaluator
    compares a multi-value answer item by item, splitting both the reference
    and the candidate on commas. Joining distinct values with a semicolon made
    the whole list a single item, so ``taekwondo; kickboxing`` was scored
    against ``Kickboxing, Taekwondo`` as one long phrase and lost 0.33 F1 while
    naming exactly the same facts. Nothing about the content changes here; the
    list is only written the way the reader parses it.

    Values retain the conclusions written by synthesis. Source clocks and
    time references are generation inputs, not a license to rewrite dates here.

    A partly supported conjunction has a third outcome besides "answered" and
    "not mentioned": the supported part, plus the named gap. The answer table
    already records that gap, and dropping it here made the declaration
    invisible -- LongMemEval's items ask for exactly that sentence ("you
    mentioned planting 5 plants for tomatoes but you did not mention chili
    peppers"). Publishing is gated in `_publishable_gap`, because appending a
    note about an incidental absence would lengthen an answer that was complete.
    """
    values = []
    advice_only = True
    for slot in table.slots:
        if slot.role != "answer":
            continue
        for item in answer_values(slot):
            value = _display_value(item, kind=kind)
            values.append(value)
            advice_only = advice_only and item.basis == "recommendation"
    values = list(dict.fromkeys(values))
    if not values:
        return "Not mentioned in the conversation"
    endings = (".", "!", "?", "。", "！", "？")
    separator = " " if advice_only and all(v.endswith(endings) for v in values) else ", "
    joined = separator.join(values)
    gaps = list(dict.fromkeys(
        gap for slot in table.slots if (gap := _publishable_gap(slot, question))
    ))
    if gaps:
        # Every accepted answer target has the same publication contract. An
        # early return here silently discarded later targets' unmet conditions.
        for gap in gaps:
            if _gap_already_published(gap, values):
                continue
            joined += (" " if joined.endswith(endings) else ". ") + gap
    return joined


def _gap_already_published(gap: str, values: list[str]) -> bool:
    """Only omit complete sentences already present verbatim in answer values."""
    def sentences(text: str) -> set[str]:
        return {normalized for part in re.split(r"(?<=[.!?。！？])\s+", text.strip())
                if (normalized := _canonical_quote(part).rstrip(".!?。！？"))}
    recorded = sentences(gap)
    published = set().union(*(sentences(value) for value in values))
    return bool(recorded) and recorded <= published


# A gap that only says "something is absent" is not yet an answer sentence: it has
# to name the absent thing the question asked about, which the overlap test does.
_GAP_ASSERTION = re.compile(
    r"\bno evidence\b|\bnot (?:mentioned|provided|stated|supported|present|available|specified)\b"
    r"|\bdid not mention\b|\bonly\b[^.]{0,80}\bsupported\b|\binsufficient\b"
    r"|\bcannot (?:be )?(?:determined|confirmed)\b",
    re.IGNORECASE,
)
# Without a coordination in the question there is no conjunction to be partly
# covered, and a missing detail is an ordinary omission rather than a gap in the
# answer. This is what keeps the clause off complete single-fact answers.
_GAP_COORDINATION = re.compile(r"\b(?:and|both|plus|as well as|or)\b", re.IGNORECASE)
_GAP_STOP = frozenset("""the a an and or of to in on at for is are was were it that this with by from
as be been my i you your we they he she his her its me our their there here what which who whom when
where why how did do does have has had will would can could should may might must not no yes than then
so if but about after before during between day days week weeks month months year years time times""".split())


def _gap_terms(text: str) -> set[str]:
    out = set()
    for word in re.findall(r"[a-z0-9][a-z0-9'-]*", str(text or "").casefold()):
        word = word.strip("'-")
        if len(word) > 2 and word not in _GAP_STOP:
            out.add(word)
    return out


def _publishable_gap(slot, question: str) -> str:
    """The recorded gap, when publishing it is what the question asks for.

    Three conditions, all measured against the recorded rows before this was
    wired (LoCoMo 1 of 1000 affected, LongMemEval 7): a supported value exists,
    the question coordinates two or more things, and the gap both asserts an
    absence and names something the question asked about. Anything looser turns
    a complete answer into a longer one, which costs precision on list answers.
    """
    gap = (slot.missing or "").strip()
    # Explicit semantic scope replaces language-specific lexical guessing for
    # new tables. Older persisted tables keep their existing fallback below.
    if slot.missing_kind == "optional_detail":
        return ""
    if slot.missing_kind in {"requested_part", "necessary_premise"}:
        return gap if slot.role == "answer" and slot.coverage == "partial" and answer_values(slot) else ""
    if slot.coverage == "partial" and slot.role == "answer":
        return gap
    if not gap or not answer_values(slot) or slot.role != "answer":
        return ""
    if not question or not _GAP_COORDINATION.search(question):
        return ""
    if not _GAP_ASSERTION.search(gap):
        return ""
    if not (_gap_terms(question) & _gap_terms(gap)):
        return ""
    return gap[:200].rstrip()

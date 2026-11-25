"""Binding addresses and declared contradictions, not a semantic truth oracle."""

from copy import deepcopy

import pytest

from agent_memory.models import AnswerContract, AnswerOutput, Hit
from agent_memory.synthesis import (
    AnswerTable,
    binding_audit,
    render_table,
    validate_table,
    validation_feedback,
)

QUESTION = "How many people were in Noel's team when Noel was a senior engineer?"
QUOTE = "As the engineering manager, Noel supervised four people."


def contract(question=QUESTION, **overrides):
    return AnswerContract(outputs=[AnswerOutput(
        output_id="q1", question_span=question, subjects=["Noel"],
        relation="team size", qualifier="while a senior engineer", **overrides,
    )])


def source(text=QUOTE):
    return Hit(content=text, score=1, source_file="fixture", chunk_id="one", metadata={
        "source_events": [{"text": text, "speaker": "Noel", "role": "user",
                           "occurred_at": "2034-06-03T12:00:00+00:00",
                           "context_only": False, "evidence_grade": "primary"}],
    })


def answer(*, binding=None, value="4", operation="none", inputs=None, question=QUESTION, quote=QUOTE, **slot):
    item = {"value": value, "supports": [{"source_id": 1, "quote": quote}], "operation": operation}
    if binding is not None:
        item["request_binding"] = binding
    if inputs is not None:
        item["inputs"] = inputs
    return AnswerTable.model_validate({"slots": [{"target": question,
        "derivation": "Use sourced facts only for the relation actually requested.", "values": [item], **slot}]})


def bound(*, qualification=None, identity="one", amount="4", quote=QUOTE):
    result = {"identity": identity, "statement": "Noel's team in the stated role.",
              "supports": [{"source_id": 1, "quote": quote}], "amount": amount}
    if qualification is not None:
        result["qualification"] = qualification
    return result


def check(table, *, request=None, text=QUOTE, kind="text"):
    validate_table(table, [source(text)], [table.slots[0].target], kind, request_contract=request)
    return render_table(table)


def test_legacy_answer_stays_accepted_with_or_without_request_contract():
    old = answer()
    assert check(deepcopy(old)) == check(deepcopy(old), request=contract()) == "4"
    audit = binding_audit(old, contract())
    assert audit["unbound_values"] == 1
    assert audit["requests_without_binding"] == ["q1"]
    assert audit["semantic_truth_verified"] is False


@pytest.mark.parametrize("subject,relation,qualifier", [
    ("N. Smith", "supervision", "lead developer"),
    ("Someone else", "unrelated property", "manager rather than senior engineer"),
])
def test_free_text_mismatch_is_not_turned_into_a_semantic_guard(subject, relation, qualifier):
    table = answer(binding={"request_id": "q1", "subject": subject,
                            "relation": relation, "qualifier": qualifier, "match": "matched"})
    assert check(table, request=contract()) == "4"
    # The second case is deliberately semantically wrong. Address validity is
    # not evidence that a reader's matching declaration is actually true.
    assert binding_audit(table, contract())["semantic_truth_verified"] is False


def test_direct_positive_transfer_is_audited_without_claiming_semantic_correctness():
    table = answer(binding={"request_id": "q1", "subject": "Noel", "relation": "team size",
                            "qualifier": "engineering manager", "match": "conflict",
                            "reason": "The requested role differs from the cited role."})
    # This is intentionally a wrong positive transfer from the manager role to
    # the senior role. An arbitrary answer string does not declare its polarity;
    # accepting its citation cannot guarantee semantic truth or repair it.
    assert check(table, request=contract()) == "4"
    audit = binding_audit(table, contract())
    assert audit["records"][0]["match"] == "conflict"
    assert audit["semantic_truth_verified"] is False


def test_direct_negative_answer_can_use_conflicting_source_as_counterevidence():
    question = "Did Riley attend a pottery class?"
    quote = "Riley donated pottery to the studio."
    request = AnswerContract(outputs=[AnswerOutput(
        output_id="q1", question_span=question, subjects=["Riley"], relation="class attendance")])
    table = answer(question=question, quote=quote,
                   value="Attendance at the class is not established by this donation.",
                   binding={"request_id": "q1", "subject": "Riley", "relation": "donation",
                            "match": "conflict", "reason": "Donation does not establish attendance."})
    assert check(table, request=request, text=quote) == "Attendance at the class is not established by this donation."
    assert validation_feedback(table, ValueError("Separate format error."), request) == "Separate format error."
    assert binding_audit(table, request)["records"][0]["match"] == "conflict"


@pytest.mark.parametrize("operation", ["count", "sum", "time_difference", "time_compare", "time_since"])
def test_executed_answer_cannot_use_its_own_explicit_role_conflict(operation):
    table = answer(binding={"request_id": "q1", "match": "conflict"}, value="", operation=operation,
                   inputs=[bound()])
    before = table.model_dump()
    with pytest.raises(ValueError, match="explicitly declares conflict") as error:
        check(table, request=contract())
    assert table.model_dump() == before
    feedback = validation_feedback(table, error.value, contract())
    assert "do not merely relabel" in feedback
    assert "Different wording alone" in feedback
    assert "Preserve independent" in feedback


def test_context_can_explain_a_nonqualifying_fact_without_publishing_it():
    table = answer(binding={"request_id": "q1", "match": "conflict"}, role="context")
    validate_table(table, [source()], [QUESTION], "text", request_contract=contract())
    assert "4" not in render_table(table)


def test_conflicting_unused_premise_can_support_a_negative_direct_answer():
    table = answer(binding={"request_id": "q1", "match": "unresolved"},
                   value="The senior engineer's team size is not established by the manager's count.",
                   inputs=[bound(qualification={"match": "conflict"})])
    assert "not established" in check(table, request=contract())
    records = binding_audit(table, contract())["records"]
    assert records[1]["request_id"] == "q1"
    assert records[1]["subject"] == ""  # Inherit only the request address.


@pytest.mark.parametrize("operation", ["count", "sum"])
def test_explicitly_conflicting_member_cannot_be_an_executed_operand(operation):
    table = answer(binding={"request_id": "q1", "match": "matched"}, value="", operation=operation,
                   inputs=[bound(qualification={"match": "conflict", "reason": "Different event."})])
    with pytest.raises(ValueError, match="input 'one'.*explicitly declares conflict"):
        check(table, request=contract())


@pytest.mark.parametrize("request_id", [None, "unknown", "source_1"])
def test_binding_request_addresses_must_resolve_when_contract_is_available(request_id):
    table = answer(binding={"request_id": request_id, "match": "matched"})
    with pytest.raises(ValueError, match="unknown request"):
        check(table, request=contract())


def test_binding_without_contract_is_recorded_without_inventing_missing_request():
    table = answer(binding={"request_id": "historical-request", "match": "unresolved"})
    assert check(table) == "4"


def test_input_can_inherit_request_address_but_not_factual_qualification():
    table = answer(binding={"request_id": "q1", "subject": "Noel", "match": "matched"},
                   operation="count", value="", inputs=[bound(qualification={"match": "matched"})])
    assert check(table, request=contract()) == "1"
    record = binding_audit(table, contract())["records"][1]
    assert record["request_id"] == "q1"
    assert record["subject"] == record["relation"] == record["qualifier"] == ""


def test_unknown_input_request_is_not_hidden_by_a_valid_parent_request():
    table = answer(binding={"request_id": "q1", "match": "matched"}, operation="count", value="",
                   inputs=[bound(qualification={"request_id": "q2", "match": "matched"})])
    with pytest.raises(ValueError, match="unknown request 'q2'"):
        check(table, request=contract())


def test_unresolved_exact_detail_allows_a_sourced_qualified_answer():
    question = "Roughly how heavy is the parcel?"
    quote = "The parcel weighs between 80 and 90 grams; its exact weight was not measured."
    table = answer(question=question, quote=quote, value="About 80 to 90 grams",
                   binding={"request_id": "q1", "match": "unresolved", "reason": "Exact mass is unknown."},
                   missing="Exact weight was not measured.", missing_kind="optional_detail")
    assert check(table, request=contract(question), text=quote) == "About 80 to 90 grams"


def test_unknown_stage_duration_can_remain_in_a_qualified_nonexecuted_answer():
    question = "How many years of education are documented?"
    quote = "Four years in school, four at university, and a college course of unknown duration."
    inputs = [bound(identity="school", amount="4", quote=quote),
              bound(identity="university", amount="4", quote=quote),
              bound(identity="college", amount=None, quote=quote,
                    qualification={"request_id": "q1", "match": "unresolved"})]
    table = answer(question=question, quote=quote,
                   value="Eight documented years plus a college course of unknown duration.", inputs=inputs,
                   binding={"request_id": "q1", "match": "unresolved"},
                   missing="The college duration is unknown.", missing_kind="requested_part", coverage="partial")
    assert "Eight documented years" in check(table, request=contract(question), text=quote)
    exact = deepcopy(table)
    exact.slots[0].values[0].operation = "sum"
    exact.slots[0].values[0].value = ""
    with pytest.raises(ValueError, match="unknown is not zero"):
        check(exact, request=contract(question), text=quote)


def test_existing_dependency_removal_preserves_independent_answers():
    table = answer(binding={"request_id": "q1", "match": "conflict"})
    dependent = table.slots[0].values[0]
    dependent.unresolved_premise = "The requested senior role's team size is absent."
    independent = deepcopy(dependent)
    independent.value = "The manager supervised four people."
    independent.request_binding = None
    independent.unresolved_premise = ""
    table.slots[0].values.append(independent)
    rendered = check(table, request=contract())
    assert "The manager supervised four people." in rendered
    assert len(table.slots[0].values) == 1
    assert table.slots[0].coverage == "partial"


def test_bindings_do_not_replace_provenance_checks():
    table = answer(binding={"request_id": "q1", "match": "matched"}, quote="Invented team size.")
    with pytest.raises(ValueError, match="does not contain exact quote"):
        check(table, request=contract())


def test_failure_feedback_collects_independent_binding_conflicts_without_mutation():
    table = answer(binding={"request_id": "q1", "match": "conflict"}, value="", operation="count",
                   inputs=[bound()])
    before = table.model_dump()
    feedback = validation_feedback(table, ValueError("A separate format error."), contract())
    assert "separate format error" in feedback
    assert "explicitly declares conflict" in feedback
    assert table.model_dump() == before


@pytest.mark.parametrize("source_id,quote,error", [
    (1, "Invented direct answer evidence.", "does not contain exact quote"),
    (2, QUOTE, "Unknown source number"),
])
def test_direct_conflict_audit_does_not_bypass_ordinary_source_validation(source_id, quote, error):
    table = answer(binding={"request_id": "q1", "match": "conflict"}, quote=quote)
    table.slots[0].values[0].supports[0].source_id = source_id
    with pytest.raises(ValueError, match=error):
        check(table, request=contract())


def test_direct_conflict_audit_still_requires_a_valid_request_address():
    table = answer(binding={"request_id": "unknown", "match": "conflict"})
    with pytest.raises(ValueError, match="unknown request"):
        check(table, request=contract())


def test_binding_audit_is_nonmutating_and_does_not_claim_global_set_completeness():
    table = answer(binding={"request_id": "q1", "match": "unresolved"},
                   inputs=[bound(qualification={"match": "unresolved"})])
    before = table.model_dump()
    audit = binding_audit(table, contract())
    assert table.model_dump() == before
    assert audit["requests_without_binding"] == []
    assert audit["semantic_truth_verified"] is False
    assert [record["match"] for record in audit["records"]] == ["unresolved", "unresolved"]

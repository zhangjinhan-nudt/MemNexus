"""Executed calendar reasoning uses selected source mentions, never guessed dates."""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from agent_memory.models import Hit, QueryPlan
from agent_memory.providers import CallableModel, ModelOutputError
from agent_memory.synthesis import (
    AnswerTable,
    binding_audit,
    render_table,
    validate_table,
    validation_feedback,
)
from agent_memory.temporal import resolve_time_references

QUESTION = "How long elapsed between the first and second recorded events?"


def source(text, stamp="2030-06-01T12:00:00+00:00", **event_fields):
    return Hit(content=text, score=1, source_file="fixture", chunk_id=text, metadata={
        "source_events": [{"text": text, "role": "user", "speaker": "Rhea",
                           "occurred_at": stamp, "evidence_grade": "primary", **event_fields}],
    })


def operand(sources, source_id, identity, *, source_event=1, reference=1, quote=None):
    event = sources[source_id - 1].metadata["source_events"][source_event - 1]
    return {"identity": identity, "statement": identity,
            "supports": [{"source_id": source_id, "quote": quote or event["text"]}],
            "time_ref": {"source_id": source_id, "source_event": source_event, "reference": reference}}


def table(sources, *, operation="time_difference", unit="days", value="", inputs=None):
    return AnswerTable.model_validate({"slots": [{"target": QUESTION,
        "derivation": "Select the two described events and their stated time mentions in the requested order.",
        "values": [{"basis": "inference", "operation": operation, "unit": unit, "value": value,
                    "inputs": inputs if inputs is not None else [
                        operand(sources, 1, "first event"), operand(sources, 2, "second event")]}]}]})


def execute(answer, sources, **kwargs):
    validate_table(answer, sources, [QUESTION], "duration", **kwargs)
    return render_table(answer)


@pytest.mark.parametrize("unit,expected", [("days", "14 days"), ("weeks", "2 weeks")])
def test_exact_difference_across_year_boundary(unit, expected):
    sources = [source("The first event happened on 2028-12-29."),
               source("The second event happened on 2029-01-12.")]
    assert execute(table(sources, unit=unit), sources) == expected


def test_reversed_input_order_retains_negative_difference():
    sources = [source("The first event happened on 2030-07-15."),
               source("The second event happened on 2030-07-01.")]
    assert execute(table(sources, unit="weeks"), sources) == "-2 weeks"


def test_nonintegral_week_difference_does_not_round_to_an_exact_week():
    sources = [source("The first event happened on 2030-07-01."),
               source("The second event happened on 2030-07-10.")]
    assert execute(table(sources, unit="weeks"), sources) == "9 days"


def test_month_precision_produces_a_duration_range_instead_of_a_fabricated_day():
    sources = [source("The first event happened in April 2030."),
               source("The second event happened on May 10, 2030.")]
    assert execute(table(sources, unit="weeks"), sources) == "10 to 39 days"


def test_week_precision_is_retained_in_the_bound_difference():
    sources = [source("The first event happened on 2031-03-01."),
               source("The second event happened last week.", "2031-03-19T12:00:00+00:00")]
    ref = resolve_time_references(sources[1].content, source_time="2031-03-19T12:00:00+00:00")[0]
    assert ref.precision == "week"
    assert execute(table(sources), sources) == "9 to 15 days"


def test_each_relative_reference_keeps_its_own_source_clock_not_question_clock():
    sources = [source("The first event happened exactly one week ago.", "2028-01-29T10:00:00+00:00"),
               source("The second event happened exactly one week ago.", "2028-02-12T10:00:00+00:00")]
    assert execute(table(sources), sources, as_of="2040-11-30") == "14 days"


def test_leap_day_is_computed_without_whole_year_approximations():
    sources = [source("The first event happened on 2032-02-28."),
               source("The second event happened on 2032-03-01.")]
    assert execute(table(sources), sources) == "2 days"


@pytest.mark.parametrize("dates,relation", [
    (("2030-01-02", "2030-01-03"), "first event occurred before second event"),
    (("2030-01-03", "2030-01-02"), "first event occurred after second event"),
])
def test_compare_reports_order_in_the_selected_input_order(dates, relation):
    sources = [source(f"The first event happened on {dates[0]}."),
               source(f"The second event happened on {dates[1]}.")]
    assert execute(table(sources, operation="time_compare"), sources) == relation


def test_distinct_events_on_same_day_are_not_merged_or_assigned_an_order():
    sources = [source("Delivery A arrived on 2030-08-15."),
               source("Delivery B arrived on 2030-08-15.")]
    result = execute(table(sources, operation="time_compare"), sources)
    assert result == "first event and second event occurred on the same day; their order within that day is unknown"


def test_overlapping_intervals_publish_the_supported_relation_without_refusal():
    sources = [source("The first event happened in April 2030."),
               source("The second event happened on April 10, 2030.")]
    answer = table(sources, operation="time_compare")
    assert execute(answer, sources) == (
        "The supported time ranges for first event and second event overlap; their order is not determined")
    assert answer.slots[0].coverage == "complete"
    assert len(answer.slots[0].values) == 1


def test_interval_difference_crossing_zero_does_not_invent_chronological_order():
    sources = [source("The first event happened in April 2030."),
               source("The second event happened on April 10, 2030.")]
    assert execute(table(sources), sources) == "-20 to 9 days"


def test_source_event_position_selects_clock_even_when_wording_is_repeated():
    sources = [source("The first event happened exactly one week ago.", "2030-01-08T12:00:00+00:00"),
               source("The second event happened on 2030-01-20.")]
    other = deepcopy(sources[0].metadata["source_events"][0])
    other["occurred_at"] = "2030-01-15T12:00:00+00:00"
    sources[0].metadata["source_events"].append(other)
    inputs = [operand(sources, 1, "first event", source_event=2), operand(sources, 2, "second event")]
    assert execute(table(sources, inputs=inputs), sources) == "12 days"


def test_reference_position_selects_the_quoted_time_mention_instead_of_first_date():
    sources = [source("The event was planned on 2030-01-02 and completed on 2030-01-09."),
               source("The second event happened on 2030-01-10.")]
    inputs = [operand(sources, 1, "completion", reference=2, quote="completed on 2030-01-09"),
              operand(sources, 2, "second event")]
    assert execute(table(sources, inputs=inputs), sources) == "1 day"


def test_selected_time_expression_must_be_in_this_inputs_assertion_quote():
    sources = [source("The event was planned on 2030-01-02 and completed on 2030-01-09."),
               source("The second event happened on 2030-01-10.")]
    inputs = [operand(sources, 1, "completion", reference=1, quote="completed on 2030-01-09"),
              operand(sources, 2, "second event")]
    with pytest.raises(ValueError, match="containing its time expression"):
        execute(table(sources, inputs=inputs), sources)


def test_source_timestamp_is_not_automatically_an_event_date():
    sources = [source("The first event happened recently."),
               source("The second event happened on 2030-07-01.")]
    with pytest.raises(ValueError, match="Unknown time_ref reference"):
        execute(table(sources), sources)


def test_missing_source_clock_does_not_fall_back_to_question_clock():
    sources = [source("The first event happened exactly one week ago.", None),
               source("The second event happened on 2030-07-01.")]
    with pytest.raises(ValueError, match="Unknown time_ref reference"):
        execute(table(sources), sources, as_of="2030-07-15")


def test_approximate_unbounded_reference_stays_answerable_with_none():
    sources = [source("The first event happened about two weeks ago.", "2030-07-01T12:00:00+00:00"),
               source("The second event happened on 2030-07-01.")]
    with pytest.raises(ValueError, match="no bounded interval"):
        execute(table(sources), sources)
    # No temporal guard is applied to the existing supported inference path.
    answer = table(sources, operation="none", value="About two weeks")
    assert execute(answer, sources) == "About two weeks"


@pytest.mark.parametrize("field,value,message", [
    ("source_id", 3, "Unknown time_ref source"),
    ("source_event", 2, "Unknown time_ref source_event"),
    ("reference", 2, "Unknown time_ref reference"),
])
def test_unknown_addresses_are_not_silently_substituted(field, value, message):
    sources = [source("The first event happened on 2030-07-01."),
               source("The second event happened on 2030-07-15.")]
    answer = table(sources)
    setattr(answer.slots[0].values[0].inputs[0].time_ref, field, value)
    with pytest.raises(ValueError, match=message):
        execute(answer, sources)


def test_computed_temporal_value_preserves_order_and_audits_replaced_model_display():
    sources = [source("The first event happened on 2030-07-15."),
               source("The second event happened on 2030-07-01.")]
    answer = table(sources, value="14 days")
    assert execute(answer, sources) == "-14 days"
    assert binding_audit(answer)["computed_temporal"] == [{
        "slot": 1, "value": 1, "operation": "time_difference", "declared": "14 days", "computed": "-14 days"}]


def test_explicitly_conflicting_temporal_input_uses_existing_repair_boundary():
    sources = [source("The first event happened on 2030-07-01."),
               source("The second event happened on 2030-07-15.")]
    answer = table(sources)
    raw = answer.model_dump()
    raw["slots"][0]["values"][0]["inputs"][0]["qualification"] = {"match": "conflict"}
    with pytest.raises(ValueError, match="explicitly declares conflict"):
        execute(AnswerTable.model_validate(raw), sources)


def test_one_input_cannot_make_unknown_second_endpoint_zero():
    sources = [source("The first event happened on 2030-07-01.")]
    answer = table(sources, inputs=[operand(sources, 1, "first event")])
    with pytest.raises(ValueError, match="exactly two selected inputs"):
        execute(answer, sources)


def test_assertion_in_other_event_does_not_upgrade_selected_question_to_fact():
    sources = [source("Did the first event happen on 2030-07-01?", evidence_grade="none"),
               source("The second event happened on 2030-07-15.")]
    sources[0].metadata["source_events"].append({
        "text": "The first event did happen.", "role": "user", "evidence_grade": "primary"})
    inputs = [operand(sources, 1, "first event"), operand(sources, 2, "second event")]
    inputs[0]["supports"].append({"source_id": 1, "quote": "The first event did happen."})
    with pytest.raises(ValueError, match="time_ref needs an assertion quote"):
        execute(table(sources, inputs=inputs), sources)


@pytest.mark.parametrize("source_date,question_date,unit,expected", [
    ("2029-11-09", "2030-02-09", "months", "3 months ago"),
    ("2029-02-09", "2031-02-09", "years", "2 years ago"),
    ("2030-07-01", "2030-07-15", "weeks", "2 weeks ago"),
    ("2030-07-01", "2030-07-02", "days", "1 day ago"),
    ("2030-07-01", "2030-07-01", "months", "0 months ago"),
])
def test_time_since_uses_one_source_and_supplied_question_date(source_date, question_date, unit, expected):
    sources = [source(f"The event happened on {source_date}.", "2040-01-01T12:00:00+00:00")]
    inputs = [operand(sources, 1, "recorded event")]
    answer = table(sources, operation="time_since", unit=unit, inputs=inputs)
    before = deepcopy(answer.slots[0].values[0].inputs)
    assert execute(answer, sources, as_of=question_date) == expected
    assert answer.slots[0].values[0].inputs == before
    assert len(answer.slots[0].values[0].supports) == 1  # No fabricated source anchor.


@pytest.mark.parametrize("operation", ["time_since", "time_difference"])
@pytest.mark.parametrize("start,end,unit,expected", [
    ("2030-01-31", "2030-02-28", "months", "28 days"),
    ("2030-07-01", "2031-07-02", "years", "366 days"),
    ("2032-02-29", "2033-02-28", "years", "365 days"),
])
def test_inexact_calendar_unit_falls_back_to_exact_days(operation, start, end, unit, expected):
    sources = [source(f"The first event happened on {start}.")]
    inputs = [operand(sources, 1, "first event")]
    if operation == "time_difference":
        sources.append(source(f"The second event happened on {end}."))
        inputs.append(operand(sources, 2, "second event"))
    answer = table(sources, operation=operation, unit=unit, inputs=inputs)
    assert execute(answer, sources, as_of=end) == expected + (" ago" if operation == "time_since" else "")


@pytest.mark.parametrize("unit,expected", [("months", "12 months"), ("years", "1 year")])
def test_two_source_difference_uses_existing_exact_calendar_units(unit, expected):
    sources = [source("The first event happened on 2030-07-01."),
               source("The second event happened on 2031-07-01.")]
    assert execute(table(sources, unit=unit), sources) == expected


def test_since_coarse_last_month_keeps_original_interval_width():
    sources = [source("The event happened last month.", "2031-06-10T12:00:00+00:00")]
    answer = table(sources, operation="time_since", unit="months", inputs=[operand(sources, 1, "event")])
    assert execute(answer, sources, as_of="2031-07-01") == "31 to 61 days ago"


def test_since_relative_source_date_and_question_date_keep_separate_clocks():
    sources = [source("The event happened exactly one week ago.", "2030-01-22T12:00:00+00:00")]
    answer = table(sources, operation="time_since", unit="weeks", inputs=[operand(sources, 1, "event")])
    assert execute(answer, sources, as_of="2030-02-05") == "3 weeks ago"


def test_since_does_not_use_question_clock_to_resolve_missing_source_clock():
    sources = [source("The event happened exactly one week ago.", None)]
    answer = table(sources, operation="time_since", inputs=[operand(sources, 1, "event")])
    with pytest.raises(ValueError, match="Unknown time_ref reference"):
        execute(answer, sources, as_of="2030-02-05")


@pytest.mark.parametrize("question_date", [None, "", "not-a-date", "2030-01-01T12:00:00"])
def test_since_requires_a_supplied_unambiguous_question_clock(question_date):
    sources = [source("The event happened on 2030-01-01.")]
    answer = table(sources, operation="time_since", inputs=[operand(sources, 1, "event")])
    with pytest.raises(ValueError, match="supplied question date"):
        execute(answer, sources, as_of=question_date)


def test_since_future_event_keeps_negative_calendar_difference():
    sources = [source("The event happened on 2030-04-09.")]
    answer = table(sources, operation="time_since", unit="months", inputs=[operand(sources, 1, "event")])
    assert execute(answer, sources, as_of="2030-03-09") == (
        "-1 month ago (the event is after the question date)")


def test_since_range_crossing_question_date_keeps_signed_uncertainty():
    sources = [source("The event happened in April 2030.")]
    answer = table(sources, operation="time_since", unit="months", inputs=[operand(sources, 1, "event")])
    assert execute(answer, sources, as_of="2030-04-10") == (
        "-20 to 9 days ago (the supported range crosses the question date)")


def test_two_source_calendar_difference_does_not_swallow_reversed_direction():
    sources = [source("The first event happened on 2031-07-01."),
               source("The second event happened on 2030-07-01.")]
    assert execute(table(sources, unit="years"), sources) == "-1 year"


def test_since_rejects_a_fabricated_second_source_anchor_with_existing_repair():
    sources = [source("The event happened on 2030-01-01.")]
    bound = operand(sources, 1, "event")
    answer = table(sources, operation="time_since", inputs=[bound, deepcopy(bound)])
    with pytest.raises(ValueError, match="exactly one source input"):
        execute(answer, sources, as_of="2030-04-01")


def test_since_redundant_model_display_is_replaced_using_same_question_clock():
    sources = [source("The event happened on 2030-01-01.")]
    answer = table(sources, operation="time_since", unit="months", value="4 months ago",
                   inputs=[operand(sources, 1, "event")])
    assert execute(answer, sources, as_of="2030-04-01") == "3 months ago"
    record = binding_audit(answer)["computed_temporal"][0]
    assert record["declared"] == "4 months ago" and record["computed"] == "3 months ago"


def test_none_approximate_month_answer_does_not_require_a_question_clock():
    sources = [source("The event happened about three months ago.")]
    answer = table(sources, operation="none", value="About three months ago", unit="months",
                   inputs=[operand(sources, 1, "event")])
    assert execute(answer, sources) == "About three months ago"


def test_approximate_month_display_does_not_reject_a_source_bound_day_difference():
    sources = [source("The event happened on 2030-05-03.")]
    answer = table(sources, operation="time_since", unit="months", value="About two months ago",
                   inputs=[operand(sources, 1, "event")])
    assert execute(answer, sources, as_of="2030-07-01") == "59 days ago"
    assert binding_audit(answer)["computed_temporal"][0] == {
        "slot": 1, "value": 1, "operation": "time_since",
        "declared": "About two months ago", "computed": "59 days ago"}


def test_reversed_order_prose_is_replaced_but_preserved_in_existing_audit():
    sources = [source("The first event happened on 2030-07-01."),
               source("The second event happened on 2030-07-15.")]
    answer = table(sources, operation="time_compare", value="The second event came first.")
    assert execute(answer, sources) == "first event occurred before second event"
    assert binding_audit(answer)["computed_temporal"][0]["declared"] == "The second event came first."


def test_failed_time_reference_still_uses_repair_without_recording_a_computed_answer():
    sources = [source("The event happened on 2030-05-03.")]
    inputs = [operand(sources, 1, "event", reference=2)]
    answer = table(sources, operation="time_since", value="", inputs=inputs)
    with pytest.raises(ValueError, match="Unknown time_ref reference") as error:
        execute(answer, sources, as_of="2030-07-01")
    assert "Unknown time_ref reference" in validation_feedback(
        answer, error.value, sources=sources, as_of="2030-07-01")
    assert answer.slots[0].values[0].value == ""
    assert binding_audit(answer)["computed_temporal"] == []


def test_original_model_display_survives_revalidation_without_wire_or_schema_fields():
    sources = [source("The first event happened on 2030-07-01."),
               source("The second event happened on 2030-07-15.")]
    answer = table(sources, value="About two weeks")
    assert execute(answer, sources) == execute(answer, sources) == "14 days"
    assert binding_audit(answer)["computed_temporal"][0]["declared"] == "About two weeks"
    assert "_declared_temporal_value" not in answer.model_dump_json()
    assert "_declared_temporal_value" not in str(AnswerTable.model_json_schema())


def test_time_display_does_not_add_irrelevant_feedback_to_other_errors():
    sources = [source("The first event happened on 2030-07-01."),
               source("The second event happened on 2030-07-15.")]
    answer = table(sources, value="About two weeks")
    assert validation_feedback(answer, ValueError("Separate source error"), sources=sources) == "Separate source error"


@pytest.mark.parametrize("selection", [None, {"source_id": 1, "source_event": 1, "reference": 1}])
def test_unexecutable_optional_time_annotation_preserves_ordinary_supported_inference(selection):
    sources = [source("I returned from the trip about three weeks ago.", "2030-07-01T12:00:00+00:00")]
    inputs = [operand(sources, 1, "return from trip")]
    inputs[0]["time_ref"] = selection
    answer = table(sources, operation="time_since", value="About three weeks ago", inputs=inputs)
    assert execute(answer, sources, as_of="2030-07-01") == "About three weeks ago"
    assert answer.slots[0].values[0].operation == "none"
    audit = binding_audit(answer)
    assert audit["computed_temporal"] == []
    assert audit["semantic_truth_verified"] is False
    failed = audit["not_executed_temporal"][0]
    assert failed["requested_operation"] == "time_since" and failed["not_executed"] is True
    assert failed["declared_value"] == "About three weeks ago" and failed["error"]


def test_approximate_unbounded_annotation_falls_back_without_discarding_its_model_precision():
    sources = [source("The event happened about two weeks ago.", "2030-07-01T12:00:00+00:00")]
    answer = table(sources, operation="time_since", value="About two weeks ago",
                   inputs=[operand(sources, 1, "event")])
    assert execute(answer, sources, as_of="2030-07-01") == "About two weeks ago"
    assert "no bounded interval" in binding_audit(answer)["not_executed_temporal"][0]["error"]


def test_missing_question_clock_can_preserve_an_existing_sourced_relative_answer():
    sources = [source("The event happened exactly one week ago.")]
    answer = table(sources, operation="time_since", value="One week before that conversation",
                   inputs=[operand(sources, 1, "event")])
    assert execute(answer, sources) == "One week before that conversation"
    assert "supplied question date" in binding_audit(answer)["not_executed_temporal"][0]["error"]


def test_fallback_reenters_identifier_validation_instead_of_computed_value_exemption():
    sources = [source("My account is @riverwalk30.")]
    inputs = [operand(sources, 1, "account")]
    inputs[0]["time_ref"] = None
    answer = table(sources, operation="time_since", value="@riverwalk31",
                   inputs=inputs)
    with pytest.raises(ValueError, match="original spelling exactly"):
        execute(answer, sources, as_of="2030-07-01")


def test_fallback_does_not_legalize_an_invalid_quote():
    sources = [source("I recently returned from the trip.")]
    inputs = [operand(sources, 1, "return", quote="I returned exactly three weeks ago.")]
    inputs[0]["time_ref"] = None
    answer = table(sources, operation="time_since", value="Three weeks ago", inputs=inputs)
    with pytest.raises(ValueError, match="does not contain exact quote"):
        execute(answer, sources, as_of="2030-07-01")


@pytest.mark.parametrize("bad_address", ["support", "time_ref"])
def test_fallback_cannot_bypass_unknown_source_numbers(bad_address):
    sources = [source("I recently returned from the trip.")]
    inputs = [operand(sources, 1, "return")]
    if bad_address == "support":
        inputs[0]["supports"][0]["source_id"] = 2
        inputs[0]["time_ref"] = None
    else:
        inputs[0]["time_ref"]["source_id"] = 2
    answer = table(sources, operation="time_since", value="About two weeks ago", inputs=inputs)
    with pytest.raises(ValueError, match="Unknown (time_ref )?source"):
        execute(answer, sources, as_of="2030-07-01")


def test_unknown_time_source_is_not_hidden_by_wrong_input_count():
    sources = [source("I recently returned from the trip.")]
    inputs = [operand(sources, 1, "return")]
    inputs[0]["time_ref"]["source_id"] = 3
    answer = table(sources, operation="time_difference", value="About two weeks", inputs=inputs)
    with pytest.raises(ValueError, match="Unknown time_ref source"):
        execute(answer, sources)


def test_fallback_does_not_bypass_declared_semantic_conflict():
    sources = [source("I recently returned from the trip.")]
    inputs = [operand(sources, 1, "return")]
    inputs[0]["time_ref"] = None
    inputs[0]["qualification"] = {"match": "conflict"}
    answer = table(sources, operation="time_since", value="About two weeks ago", inputs=inputs)
    with pytest.raises(ValueError, match="explicitly declares conflict"):
        execute(answer, sources, as_of="2030-07-01")


def test_fallback_does_not_bypass_personal_recommendation_premise_check():
    sources = [source("You could take another trip.", role="assistant", evidence_grade="secondary")]
    inputs = [operand(sources, 1, "trip")]
    inputs[0]["time_ref"] = None
    answer = table(sources, operation="time_since", value="Take a trip next week",
                   inputs=inputs)
    answer.slots[0].values[0].basis = "recommendation"
    with pytest.raises(ValueError, match="Personalized recommendations require"):
        execute(answer, sources, as_of="2030-07-01")


def test_successful_and_unexecuted_time_values_have_distinct_audit_records():
    sources = [source("The event happened on 2030-07-01."),
               source("I returned from the trip about two weeks ago.")]
    answer = table(sources, operation="time_since", value="Two weeks ago",
                   inputs=[operand(sources, 1, "dated event")])
    unresolved = table(sources, operation="time_since", value="About two weeks ago",
                       inputs=[operand(sources, 2, "return")])
    answer.slots[0].values.extend(unresolved.slots[0].values)
    assert execute(answer, sources, as_of="2030-07-15") == "14 days ago, About two weeks ago"
    audit = binding_audit(answer)
    assert len(audit["computed_temporal"]) == len(audit["not_executed_temporal"]) == 1
    assert audit["computed_temporal"][0]["computed"] == "14 days ago"
    assert audit["not_executed_temporal"][0]["declared_value"] == "About two weeks ago"
    assert "_unexecuted_temporal" not in answer.model_dump_json()
    assert "_unexecuted_temporal" not in str(AnswerTable.model_json_schema())


def test_existing_one_repair_can_switch_unexecutable_annotation_to_supported_none():
    sources = [source("I returned from the trip about two weeks ago.", "2030-07-01T12:00:00+00:00",
                      context_only=False, time_derivations=[])]
    initial = table(sources, operation="time_since", inputs=[operand(sources, 1, "return")])
    repaired = deepcopy(initial)
    repaired.slots[0].values[0].operation = "none"
    repaired.slots[0].values[0].value = "About two weeks ago"
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return initial.model_dump_json() if len(calls) == 1 else repaired.model_dump_json()

    answer = asyncio.run(CallableModel(chat).synthesize(
        QUESTION, QueryPlan(), sources, as_of=datetime(2030, 7, 15, tzinfo=UTC)))
    assert render_table(answer) == "About two weeks ago"
    assert len(calls) == 2
    assert "does not by itself mean the original evidence is insufficient" in calls[1]
    assert "use operation=none" in calls[1]
    assert "Do not invent a parser reference" in calls[1]


def test_existing_one_repair_does_not_fabricate_a_value_from_derivation():
    sources = [source("I returned from the trip about two weeks ago.", "2030-07-01T12:00:00+00:00",
                      context_only=False, time_derivations=[])]
    initial = table(sources, operation="time_since", inputs=[operand(sources, 1, "return")])
    initial.slots[0].derivation = "It seems about two weeks, but no explicit answer value was produced."
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return initial.model_dump_json()

    with pytest.raises(ModelOutputError, match="Time annotation cannot execute"):
        asyncio.run(CallableModel(chat).synthesize(
            QUESTION, QueryPlan(), sources, as_of=datetime(2030, 7, 15, tzinfo=UTC)))
    assert len(calls) == 2
    assert initial.slots[0].values[0].value == ""


def test_nonempty_inference_with_unexecutable_annotation_does_not_consume_repair():
    question = "Roughly how long ago did I return from the trip?"
    sources = [source("I returned from the trip about two weeks ago.", "2030-07-01T12:00:00+00:00",
                      context_only=False, time_derivations=[])]
    raw = table(sources, operation="time_since", value="About two weeks ago",
                inputs=[operand(sources, 1, "return")])
    raw.slots[0].target = question
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return raw.model_dump_json()

    answer = asyncio.run(CallableModel(chat).synthesize(
        question, QueryPlan(), sources, as_of=datetime(2030, 7, 15, tzinfo=UTC)))
    assert len(calls) == 1
    assert render_table(answer) == "About two weeks ago"
    assert binding_audit(answer)["not_executed_temporal"][0]["not_executed"] is True


@pytest.mark.parametrize("selection", [
    {"source_id": 1, "source_event": 7, "reference": 1},
    {"source_id": 1, "source_event": 1, "reference": 7},
])
def test_nonempty_answer_cannot_hide_an_invalid_concrete_time_address(selection):
    sources = [source("The event happened on 2030-07-01.")]
    inputs = [operand(sources, 1, "event")]
    inputs[0]["time_ref"] = selection
    answer = table(sources, operation="time_since", value="Two weeks ago", inputs=inputs)
    with pytest.raises(ValueError, match="Invalid temporal binding: Unknown time_ref"):
        execute(answer, sources, as_of="2030-07-15")
    assert answer.slots[0].values[0].operation == "time_since"
    assert binding_audit(answer)["not_executed_temporal"] == []


def test_first_explicit_reference_to_empty_parser_output_is_also_invalid():
    sources = [source("I recently returned from the trip.")]
    inputs = [operand(sources, 1, "return")]
    answer = table(sources, operation="time_since", value="Recently", inputs=inputs)
    with pytest.raises(ValueError, match="Unknown time_ref reference position"):
        execute(answer, sources, as_of="2030-07-15")
    # No fabricated concrete address: an absent optional annotation can retain
    # the source's actual precision, audited as ordinary unverified inference.
    inputs[0]["time_ref"] = None
    ordinary = table(sources, operation="time_since", value="Recently", inputs=inputs)
    assert execute(ordinary, sources, as_of="2030-07-15") == "Recently"
    assert binding_audit(ordinary)["semantic_truth_verified"] is False
    assert len(binding_audit(ordinary)["not_executed_temporal"]) == 1


@pytest.mark.parametrize("declared", ["second device", "unknown"])
def test_nonempty_answer_cannot_hide_extra_temporal_operands(declared):
    sources = [source("The first device arrived on 2030-02-20."),
               source("The second device was ordered on 2030-01-28."),
               source("The second device arrived on 2030-02-25.")]
    inputs = [operand(sources, 1, "first device"), operand(sources, 2, "second device order"),
              operand(sources, 3, "second device")]
    answer = table(sources, operation="time_earlier", value=declared, inputs=inputs)
    with pytest.raises(ValueError, match="exactly two selected inputs"):
        execute(answer, sources)
    assert answer.slots[0].values[0].operation == "time_earlier"
    assert binding_audit(answer)["not_executed_temporal"] == []


@pytest.mark.parametrize("annotation", ["missing", "unbounded"])
def test_representation_limit_in_first_input_cannot_hide_second_invalid_address(annotation):
    sources = [source("The first event happened about two weeks ago."),
               source("The second event happened on 2030-07-01.")]
    inputs = [operand(sources, 1, "first event"), operand(sources, 2, "second event", reference=7)]
    if annotation == "missing":
        inputs[0]["time_ref"] = None
    answer = table(sources, value="About two weeks", inputs=inputs)
    with pytest.raises(ValueError, match="Unknown time_ref reference position"):
        execute(answer, sources)


def test_nonempty_answer_cannot_hide_selected_time_outside_its_quote():
    sources = [source("The event was planned on 2030-01-02 and completed on 2030-01-09."),
               source("The second event happened on 2030-01-10.")]
    inputs = [operand(sources, 1, "completion", reference=1, quote="completed on 2030-01-09"),
              operand(sources, 2, "second event")]
    answer = table(sources, value="8 days", inputs=inputs)
    with pytest.raises(ValueError, match="containing its time expression"):
        execute(answer, sources)
    assert answer.slots[0].values[0].operation == "time_difference"


def test_unexpected_computation_errors_are_not_downgraded_to_ordinary_answers(monkeypatch):
    import agent_memory.synthesis as synthesis

    sources = [source("The first event happened on 2030-07-01."),
               source("The second event happened on 2030-07-15.")]
    answer = table(sources, value="Two weeks")

    def fail(*args, **kwargs):
        raise ValueError("Unexpected internal computation failure")

    monkeypatch.setattr(synthesis, "_computed_time_value", fail)
    with pytest.raises(ValueError, match="Unexpected internal computation failure"):
        execute(answer, sources)
    assert answer.slots[0].values[0].operation == "time_difference"
    assert binding_audit(answer)["not_executed_temporal"] == []


def test_existing_single_repair_can_select_matching_event_phases_and_compute():
    question = "Which device did I receive first?"
    sources = [source(text, context_only=False, time_derivations=[]) for text in (
        "The first device arrived on 2030-02-20.",
        "The second device was ordered on 2030-01-28.",
        "The second device arrived on 2030-02-25.")]
    inputs = [operand(sources, 1, "first device"), operand(sources, 2, "second device order"),
              operand(sources, 3, "second device")]
    initial = table(sources, operation="time_earlier", value="second device", inputs=inputs)
    initial.slots[0].target = question
    repaired = deepcopy(initial)
    repaired.slots[0].values[0].inputs = [repaired.slots[0].values[0].inputs[index] for index in (0, 2)]
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return initial.model_dump_json() if len(calls) == 1 else repaired.model_dump_json()

    answer = asyncio.run(CallableModel(chat).synthesize(question, QueryPlan(), sources))
    assert len(calls) == 2
    assert "ordering, receiving and completing are different events" in calls[1]
    assert "none merely to bypass a bad address or wrong input count" in calls[1]
    assert render_table(answer) == "first device"
    assert binding_audit(answer)["computed_temporal"][0]["declared"] == "second device"
    assert binding_audit(answer)["computed_temporal"][0]["computed"] == "first device"


@pytest.mark.parametrize("reference", [1, 7])
@pytest.mark.parametrize("declared", ["", "Recently"])
def test_absent_parser_reference_uses_one_repair_for_sourced_qualified_answer(reference, declared):
    question = "How long ago did I return from the inspection?"
    sources = [source("I returned from the inspection recently.",
                      context_only=False, time_derivations=[])]
    initial = table(sources, operation="time_since", value=declared,
                    inputs=[operand(sources, 1, "return from inspection", reference=reference)])
    initial.slots[0].target = question
    repaired = deepcopy(initial)
    repaired.slots[0].values[0].operation = "none"
    repaired.slots[0].values[0].inputs[0].time_ref = None
    repaired.slots[0].values[0].value = "Recently"
    repaired.slots[0].coverage = "partial"
    repaired.slots[0].missing = "The exact elapsed duration is not stated."
    repaired.slots[0].missing_kind = "requested_part"
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return initial.model_dump_json() if len(calls) == 1 else repaired.model_dump_json()

    answer = asyncio.run(CallableModel(chat).synthesize(question, QueryPlan(), sources))
    assert len(calls) == 2
    assert "no parsed time references" in calls[1]
    assert "parser representation gap is not itself a missing fact" in calls[1]
    assert "remove the unavailable time_ref" in calls[1]
    assert "coverage=partial" in calls[1] and "missing_kind=requested_part" in calls[1]
    assert "none merely to bypass" not in calls[1]
    assert render_table(answer).startswith("Recently")
    assert answer.slots[0].missing_kind == "requested_part"
    assert binding_audit(answer)["computed_temporal"] == []
    assert binding_audit(answer)["semantic_truth_verified"] is False


def test_parser_representation_feedback_does_not_accept_an_unrepaired_concrete_reference():
    question = "How long ago did I return from the inspection?"
    sources = [source("I returned from the inspection recently.",
                      context_only=False, time_derivations=[])]
    initial = table(sources, operation="time_since", value="Recently",
                    inputs=[operand(sources, 1, "return from inspection")])
    initial.slots[0].target = question
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return initial.model_dump_json()

    with pytest.raises(ModelOutputError, match="Unknown time_ref reference position"):
        asyncio.run(CallableModel(chat).synthesize(question, QueryPlan(), sources))
    assert len(calls) == 2


@pytest.mark.parametrize("bad_address", ["source_event", "reference"])
def test_real_address_errors_do_not_receive_missing_parser_guidance(bad_address):
    sources = [source("The event happened on 2030-07-01.")]
    answer = table(sources, operation="time_since", value="Two weeks ago",
                   inputs=[operand(sources, 1, "event")])
    setattr(answer.slots[0].values[0].inputs[0].time_ref, bad_address, 7)
    with pytest.raises(ValueError) as caught:
        execute(answer, sources, as_of="2030-07-15")
    assert "none merely to bypass a bad address or wrong input count" in str(caught.value)
    assert "parser representation gap" not in str(caught.value)

from datetime import UTC, datetime, timedelta
import json

import pytest

from agent_memory.grounding import _source_event, close_evidence_bundles, reread_evidence, retrieve_gaps, source_payload
from agent_memory.models import AnswerContract, AnswerOutput, CollectionQuery, Event, Hit, QueryPlan, Turn
from agent_memory.synthesis import AnswerCandidate, BindingRevision, EvidenceGap, RoleBinding, merge_incremental
from agent_memory.reasoning import apply_route, classify_question
from agent_memory.synthesis import (
    AnswerTable, BoundAnswer, BoundSlot, BoundValue, Support,
    answer_type, check_and_execute, execution_contract, render_table, validate_table,
)


def turn(n, text, *, session="s", minute=0):
    return Turn(turn_id=f"t{n}", request_id=f"r{n}", session_id=session, sequence=n,
                events=[Event(role="user", speaker="Inez", content=text,
                              occurred_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=minute))])


def hit(t):
    return Hit(content=t.events[0].content, score=.9, source_file="source", chunk_id=t.turn_id,
               metadata={"source_events": [_source_event(t, 0, t.events[0].content, 0)]})


def table(value="red", quote="red", source_id=1, target="colors"):
    return AnswerTable.model_validate({"slots": [{"target": target, "values": [
        {"value": value, "supports": [{"source_id": source_id, "quote": quote}]}]}]})


def test_all_conjuncts_rendered_and_quotes_checked():
    sources = [hit(turn(1, "I prefer red and blue."))]
    result = table()
    result.slots[0].values.append(table("blue", "blue").slots[0].values[0])
    validate_table(result, sources, ["colors"], "text")
    assert render_table(result) == "red; blue"


@pytest.mark.parametrize("result", [table(quote="green"), table(source_id=2), table(target="other")])
def test_invalid_provenance_or_lost_target_rejected(result):
    with pytest.raises(ValueError):
        validate_table(result, [hit(turn(1, "red"))], ["colors"], "text")


def test_duration_cannot_turn_into_event_enumeration():
    question = "After how many weeks did Inez visit again?"
    route = classify_question(question)
    plan = QueryPlan(required_info=["first visit date", "return date"],
                     collection_query=CollectionQuery(subjects=["Inez"], predicate="visits"))
    routed = apply_route(plan, route)
    assert route.kind == routed.query_type == "temporal"
    assert routed.required_info == plan.required_info
    assert answer_type(question) == "duration"
    with pytest.raises(ValueError, match="Duration"):
        validate_table(table("visit", "visit"), [hit(turn(1, "visit"))], ["colors"], "duration")


def test_bundle_closes_explanation_without_crossing_authorization_or_time():
    turns = [turn(1, "I finished a project."), turn(2, "What was it?"),
             turn(3, "A solar-powered pump."), turn(4, "Retracted secret."),
             turn(5, "Other session.", session="other"), turn(6, "Next day.", minute=1440)]
    sources = close_evidence_bundles([hit(turns[0])], turns, session_id="s",
                                    blocked_events={("t4", 0)}, budget=5000)
    events = sources[0].metadata["source_events"]
    assert {e["turn_id"] for e in events} == {"t1", "t2", "t3"}
    assert next(e for e in events if e["turn_id"] == "t2")["context_only"]
    assert not next(e for e in events if e["turn_id"] == "t3")["context_only"]


def test_date_difference_is_computed_and_cannot_invent_endpoints():
    source = hit(turn(1, "We met on 2024-03-01 and returned on 2024-03-15."))
    result = table("999 weeks", source.content)
    from agent_memory.synthesis import DateDifference
    result.slots[0].values[0].calculation = DateDifference(start="2024-03-01", end="2024-03-15", unit="weeks")
    validate_table(result, [source], ["colors"], "duration")
    assert render_table(result) == "2 weeks"
    result.slots[0].values[0].calculation = DateDifference(start="2024-02-23", end="2024-03-15", unit="weeks")
    with pytest.raises(ValueError, match="endpoints"):
        validate_table(result, [source], ["colors"], "duration")


def test_unresolved_slot_is_not_silently_treated_as_complete():
    result = AnswerTable.model_validate({"slots": [{"target": "colors", "missing": "No statement"}]})
    validate_table(result, [], ["colors"], "text")
    assert render_table(result) == "Not mentioned in the conversation"
    assert result.slots[0].missing


def test_context_premise_does_not_leak_into_final_answer():
    result = table("two years", "two years")
    context = table("2025-01-01", "two years", target="anchor").slots[0]
    context.role = "context"
    result.slots.append(context)
    validate_table(result, [hit(turn(1, "two years"))], ["colors", "anchor"], "duration")
    assert render_table(result) == "two years"


def test_question_can_link_context_but_cannot_establish_fact_alone():
    source = hit(turn(1, "Is it blue?"))
    source.metadata["source_events"].append(_source_event(turn(2, "It is red."), 0, "It is red.", 0))
    result = table("red", "Is it blue?")
    with pytest.raises(ValueError, match="alone"):
        validate_table(result, [source], ["colors"], "text")
    from agent_memory.synthesis import Support
    result.slots[0].values[0].supports.append(Support(source_id=1, quote="It is red."))
    validate_table(result, [source], ["colors"], "text")


def test_inverse_duration_only_establishes_start_year():
    from agent_memory.synthesis import DateDifference
    result = table("2020", "two years")
    result.slots[0].values[0].calculation = DateDifference(start="2023-01-01", end="2025-01-01", unit="years")
    validate_table(result, [hit(turn(1, "two years"))], ["colors"], "date")
    assert render_table(result) == "2023"
    assert result.slots[0].values[0].calculation is None
    result.slots[0].values[0].calculation = DateDifference(start="2022-01-01", end="2025-01-01", unit="years")
    with pytest.raises(ValueError, match="Start-year"):
        validate_table(result, [hit(turn(1, "two years"))], ["colors"], "date")


def test_execution_contract_keeps_mandatory_output_separate_from_premises():
    plan = QueryPlan(required_info=["event date", "reference date"], query_type="temporal")
    contract = execution_contract("How long ago was the birthday?", plan)
    assert len(contract.outputs) == 1
    assert contract.outputs[0].operator == "duration"
    assert contract.outputs[0].required_inputs == ["event date", "reference date"]


def test_count_uses_members_and_reports_open_scope():
    source = hit(turn(1, "I have Max and Daisy."))
    contract = AnswerContract(outputs=[AnswerOutput(
        output_id="pets", question_span="How many pets?", operator="count",
        subjects=["Inez"], relation="pets",
    )])
    answer = BoundAnswer(slots=[BoundSlot(output_id="pets", values=[
        BoundValue(value="Max", identity="max", subjects=["Inez"], supports=[Support(source_id=1, quote="Max")]),
        BoundValue(value="Daisy", identity="daisy", subjects=["Inez"], supports=[Support(source_id=1, quote="Daisy")]),
    ])])
    result = check_and_execute(contract, answer, [source])
    assert result.values["pets"] == ["at least 2"]
    assert any(g.code == "member_scope_unknown" for g in result.gaps)


def test_count_does_not_treat_my_dogs_as_exhaustive_inventory():
    source = hit(turn(1, "My dogs. Max and Daisy."))
    contract = AnswerContract(outputs=[AnswerOutput(
        output_id="pets", question_span="How many pets?", operator="count",
        subjects=["Inez"], relation="pets",
    )])
    answer = BoundAnswer(slots=[BoundSlot(output_id="pets", scope="closed", scope_support=[
        Support(source_id=1, quote="My dogs")], values=[
        BoundValue(value="Max", identity="max", subjects=["Inez"], supports=[Support(source_id=1, quote="Max")]),
        BoundValue(value="Daisy", identity="daisy", subjects=["Inez"], supports=[Support(source_id=1, quote="Daisy")]),
    ])])
    result = check_and_execute(contract, answer, [source])
    assert result.values["pets"] == ["at least 2"]
    assert any(g.code == "member_scope_unknown" for g in result.gaps)


def test_count_rejects_subject_transfer():
    source = hit(turn(1, "I have Max."))
    contract = AnswerContract(outputs=[AnswerOutput(
        output_id="pets", question_span="How many pets does Nora have?", operator="count",
        subjects=["Nora"], relation="pets",
    )])
    answer = BoundAnswer(slots=[BoundSlot(output_id="pets", values=[
        BoundValue(value="Max", identity="max", subjects=["Inez"], supports=[Support(source_id=1, quote="Max")]),
    ])])
    result = check_and_execute(contract, answer, [source])
    assert result.values["pets"] == []
    assert any(g.code == "subject_unresolved" for g in result.gaps)


def test_gap_retrieval_returns_only_new_authorized_sources_and_stops_without_gap():
    turns = [turn(1, "I bought the camera on Monday."),
             turn(2, "It arrived on Friday.", minute=10),
             turn(3, "I like hiking.", minute=20)]
    plan = QueryPlan(answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="duration", question_span="How long did delivery take?", operator="duration",
        subjects=["Inez"], relation="delivery", required_inputs=["order date", "arrival date"],
    )]), required_info=["order date", "arrival date"], query_entities=["Inez"])
    initial = [hit(turns[0])]
    gaps = [EvidenceGap(output_id="duration", code="time_operand_missing", cue="arrived delivery")]
    added, receipt = retrieve_gaps("How long did delivery take?", plan, gaps, initial,
                                  turns=turns, session_id="s", blocked_events=set(), budget=4000)
    assert added
    assert receipt["stop_reason"] == "one_round_completed"
    assert all(h.chunk_id != initial[0].chunk_id for h in added)
    # An already-complete answer does not trigger acquisition in the caller;
    # the helper itself has no gap and therefore returns no candidates.
    none, empty_receipt = retrieve_gaps("How long did delivery take?", plan, [], [*initial, *added],
                                       turns=turns, session_id="s", blocked_events=set(), budget=4000)
    assert none == [] and empty_receipt["gap_codes"] == []


def test_incremental_patch_preserves_old_binding_when_old_source_is_not_repeated():
    source = hit(turn(1, "I prefer red."))
    contract = AnswerContract(outputs=[AnswerOutput(output_id="color", question_span="What color?",
        operator="direct", subjects=["Inez"], relation="preference")])
    old = BoundAnswer(slots=[BoundSlot(output_id="color", values=[
        BoundValue(binding_id="b1", value="red", subjects=["Inez"],
                   supports=[Support(source_id=1, quote="I prefer red.")])])])
    new = BoundAnswer(slots=[BoundSlot(output_id="color", values=[], replaced_values={
        "red": "old source was not in the new prompt"})])
    merged = merge_incremental(old, new, {"color"}, sources=[source], contract=contract)
    assert [x.value for x in merged.slots[0].values] == ["red"]


def test_each_subject_mode_accepts_separate_bindings():
    source = hit(turn(1, "Deborah has a pendant. Jolene has an amulet."))
    contract = AnswerContract(outputs=[AnswerOutput(output_id="gifts", question_span="What gifts do they each have?",
        operator="set", subjects=["Deborah", "Jolene"], subject_mode="each", relation="gift")])
    answer = BoundAnswer(slots=[BoundSlot(output_id="gifts", scope="open", values=[
        BoundValue(value="pendant", subjects=["Deborah"], identity="pendant", supports=[Support(source_id=1, quote="Deborah has a pendant.")]),
        BoundValue(value="amulet", subjects=["Jolene"], identity="amulet", supports=[Support(source_id=1, quote="Jolene has an amulet.")]),
    ])])
    result = check_and_execute(contract, answer, [source])
    assert result.values["gifts"] == ["Deborah: pendant", "Jolene: amulet"]


def test_time_point_reuses_source_relative_derivation():
    source = hit(turn(1, "I started surfing five years ago."))
    contract = AnswerContract(outputs=[AnswerOutput(output_id="year", question_span="What year?",
        operator="time_point", subjects=["Inez"], relation="start surfing")])
    answer = BoundAnswer(slots=[BoundSlot(output_id="year", values=[
        BoundValue(value="2019", time_expression="five years ago", subjects=["Inez"],
                   supports=[Support(source_id=1, quote="I started surfing five years ago.")])])])
    result = check_and_execute(contract, answer, [source])
    assert result.values["year"] == ["2020"]


def test_compact_source_payload_preserves_provenance_with_less_text():
    source = hit(turn(1, "A long factual statement about a project and its date."))
    full = source_payload(source)
    compact = source_payload(source, compact=True, include_time=False)
    assert compact["claim"] == source.content
    assert compact["source_events"][0]["text"] == source.content
    assert len(json.dumps(compact, ensure_ascii=False)) <= len(json.dumps(full, ensure_ascii=False))


def test_accepted_candidate_is_materialized_and_not_lost_in_rendering():
    source = hit(turn(1, "I have Max, Daisy, and Ned."))
    contract = AnswerContract(outputs=[AnswerOutput(
        output_id="pets", question_span="What pets?", operator="set",
        subjects=["Inez"], relation="pets",
    )])
    answer = BoundAnswer(slots=[BoundSlot(output_id="pets", candidates=[
        AnswerCandidate(candidate_id="max", value="Max", identity="max", subjects=["Inez"],
                        supports=[Support(source_id=1, quote=source.content)], disposition="accepted"),
        AnswerCandidate(candidate_id="daisy", value="Daisy", identity="daisy", subjects=["Inez"],
                        supports=[Support(source_id=1, quote=source.content)], disposition="accepted"),
        AnswerCandidate(candidate_id="ned", value="Ned", identity="ned", subjects=["Inez"],
                        supports=[Support(source_id=1, quote=source.content)], disposition="accepted"),
    ])])
    result = check_and_execute(contract, answer, [source])
    assert result.values["pets"] == ["Max", "Daisy", "Ned"]


def test_owner_role_blocks_name_only_subject_transfer():
    source = hit(turn(1, "Gina said the store is doing great."))
    contract = AnswerContract(outputs=[AnswerOutput(
        output_id="store", question_span="What is Jon's store doing?", operator="direct",
        subjects=["Jon"], relation="store status",
    )])
    answer = BoundAnswer(slots=[BoundSlot(output_id="store", values=[
        BoundValue(value="doing great", subjects=["Jon"], identity="store",
                   role_bindings=[RoleBinding(role="owner", entity="Gina",
                                              supports=[Support(source_id=1, quote=source.content)])],
                   supports=[Support(source_id=1, quote=source.content)])
    ])])
    result = check_and_execute(contract, answer, [source])
    assert result.values["store"] == []
    assert any(g.code == "subject_unresolved" for g in result.gaps)


def test_reread_expands_local_context_without_scanning_new_sessions():
    turns = [turn(1, "I finished a project."), turn(2, "It was a solar pump.", minute=5),
             turn(3, "Other session answer.", session="other", minute=5)]
    reread_sources, receipt = reread_evidence([hit(turns[0])], turns=turns, session_id="s",
                                     blocked_events=set(), budget=5000)
    assert receipt["action"] == "reread"
    assert receipt["changed_source_ids"] == [1]
    assert {event["turn_id"] for source in reread_sources for event in source.metadata["source_events"]} == {"t1", "t2"}


def test_conflicting_incremental_values_are_retained_and_marked():
    source_old = hit(turn(1, "I prefer red."))
    source_new = hit(turn(2, "I prefer blue."))
    contract = AnswerContract(outputs=[AnswerOutput(
        output_id="color", question_span="What color?", operator="direct",
        subjects=["Inez"], relation="preference",
    )])
    old = BoundAnswer(slots=[BoundSlot(output_id="color", values=[
        BoundValue(binding_id="b1", value="red", identity="item", subjects=["Inez"],
                   supports=[Support(source_id=1, quote=source_old.content)])])])
    revised = BoundAnswer(slots=[BoundSlot(output_id="color", values=[
        BoundValue(binding_id="b2", value="blue", identity="item", subjects=["Inez"],
                   supports=[Support(source_id=2, quote=source_new.content)])])])
    merged = merge_incremental(old, revised, {"color"},
                               sources=[source_old, source_new], contract=contract)
    assert {value.value for value in merged.slots[0].values} == {"red", "blue"}
    assert merged.slots[0].conflict_records
    result = check_and_execute(contract, merged, [source_old, source_new])
    assert any(g.code == "conflict_unresolved" for g in result.gaps)


def test_explicit_retraction_can_supersede_an_old_binding():
    old_source = hit(turn(1, "I own Max."))
    retract_source = hit(turn(2, "I no longer own Max."))
    contract = AnswerContract(outputs=[AnswerOutput(
        output_id="pets", question_span="What do I own?", operator="set",
        subjects=["Inez"], relation="ownership",
    )])
    old = BoundAnswer(slots=[BoundSlot(output_id="pets", values=[
        BoundValue(binding_id="b1", value="Max", identity="max", subjects=["Inez"],
                   supports=[Support(source_id=1, quote=old_source.content)])])])
    revised = BoundAnswer(slots=[BoundSlot(output_id="pets", values=[], revisions=[
        BindingRevision(binding_id="b1", reason="contradicting_evidence",
                        explanation="The later statement explicitly retracts the old ownership.",
                        supports=[Support(source_id=2, quote=retract_source.content)])])])
    merged = merge_incremental(old, revised, {"pets"},
                               sources=[old_source, retract_source], contract=contract)
    assert len(merged.slots[0].values) == 1
    assert merged.slots[0].values[0].status == "retracted"
    assert check_and_execute(contract, merged, [old_source, retract_source]).values["pets"] == []

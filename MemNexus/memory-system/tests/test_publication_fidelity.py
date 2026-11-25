"""Publication preserves explicit quantities and deterministic winner identity.

These fixtures vary entities, languages, clocks and negative cases; none use
benchmark questions, answer strings, or additional generation/validation paths.
"""
from copy import deepcopy

import pytest

from agent_memory.models import AnswerContract, Hit
from agent_memory.synthesis import (
    AnswerTable,
    SupportedValue,
    _display_value,
    binding_audit,
    render_table,
    validate_table,
)


@pytest.mark.parametrize("value,unit,kind,expected", [
    ("7", "days", "duration", "7 days"),
    ("-7", "days", "duration", "-7 days"),
    ("+0.75", "weeks", "duration", "+0.75 weeks"),
    ("3", "records", "count", "3 records"),
    ("7", "天", "duration", "7 天"),
    ("2", "years", "duration", "2 years"),
    ("2017", "year", "date", "2017"),
    ("2017", "year", "text", "2017"),
    ("0007", "ID", "text", "0007"),
    ("3", "model", "text", "3"),
    ("7", "", "duration", "7"),
    ("7 days", "days", "duration", "7 days"),
    ("about 7", "days", "duration", "about 7"),
    ("7 to 9", "days", "duration", "7 to 9"),
    ("7 if delivery was on Monday", "days", "duration", "7 if delivery was on Monday"),
])
def test_scalar_display_preserves_precision_and_nonquantities(value, unit, kind, expected):
    item = SupportedValue(value=value, unit=unit)
    before = item.model_dump()
    assert _display_value(item, kind=kind) == expected
    assert item.model_dump() == before


def test_source_quantity_supports_unknown_request_kind_without_guessing_unit():
    item = SupportedValue(value="3.5", unit="kg", supports=[
        {"source_id": 1, "quote": "The sample weighs 3.5 kg."}])
    assert _display_value(item, kind="text") == "3.5 kg"
    item.unit = "lbs"
    assert _display_value(item, kind="text") == "3.5"


def comparison(left="Aurora", right="Borealis", dates=("2030-03-04", "2030-03-09")):
    question = f"Which arrived first, {left} or {right}?"
    texts = [f"{name} arrived on {stamp}." for name, stamp in zip((left, right), dates)]
    sources = [Hit(content=text, score=1, source_file="fixture", chunk_id=str(i), metadata={
        "source_events": [{"text": text, "role": "user", "evidence_grade": "primary",
                           "occurred_at": "2030-04-01T00:00:00Z"}]}) for i, text in enumerate(texts)]
    contract = AnswerContract.model_validate({"outputs": [{"output_id": "winner",
        "question_span": question, "operator": "comparison", "subjects": [left, right]}]})
    answer = AnswerTable.model_validate({"slots": [{"target": question,
        "derivation": "Compare the two stated arrival dates.", "values": [{
            "operation": "time_earlier", "value": "deliberately wrong declared answer",
            "request_binding": {"request_id": "winner"},
            "inputs": [{"identity": f"{name} arrival", "statement": text,
                        "supports": [{"source_id": i + 1, "quote": text}],
                        "time_ref": {"source_id": i + 1, "source_event": 1, "reference": 1}}
                       for i, (name, text) in enumerate(zip((left, right), texts))]}]}]})
    return answer, sources, contract


def execute(answer, sources, contract):
    validate_table(answer, sources, [answer.slots[0].target], "text", request_contract=contract)
    return render_table(answer)


@pytest.mark.parametrize("left,right", [("Aurora", "Borealis"), ("Box 12", "Box 14"), ("青鸟", "白鹭")])
@pytest.mark.parametrize("reverse", [False, True])
def test_computed_winner_display_uses_question_name_and_tracks_changed_chronology(left, right, reverse):
    dates = ("2030-03-09", "2030-03-04") if reverse else ("2030-03-04", "2030-03-09")
    answer, sources, contract = comparison(left, right, dates)
    expected = right if reverse else left
    assert execute(answer, sources, contract) == expected
    item = answer.slots[0].values[0]
    assert item.value == f"{expected} arrival"
    assert binding_audit(answer)["computed_temporal"][0]["declared"] == "deliberately wrong declared answer"
    assert binding_audit(answer)["computed_temporal"][0]["display"] == expected
    assert execute(answer, sources, contract) == expected
    assert "_temporal_display" not in answer.model_dump_json()
    assert "_temporal_winner_index" not in str(AnswerTable.model_json_schema())


def test_latest_winner_uses_same_source_bound_display():
    answer, sources, contract = comparison()
    answer.slots[0].values[0].operation = "time_later"
    assert execute(answer, sources, contract) == "Borealis"


@pytest.mark.parametrize("change", ["absent_contract", "absent_binding", "same_subject",
                                   "not_literal_subject", "missing_name_in_quote", "prefix_collision"])
def test_ambiguous_or_missing_display_binding_retains_full_computed_identity(change):
    answer, sources, contract = comparison()
    item = answer.slots[0].values[0]
    if change == "absent_contract":
        contract = None
    elif change == "absent_binding":
        item.request_binding = None
    elif change == "same_subject":
        item.inputs[1].identity = "Aurora dispatch"
    elif change == "not_literal_subject":
        contract.outputs[0].question_span = "Which one arrived first?"
    elif change == "missing_name_in_quote":
        quote = "It arrived on 2030-03-04."
        item.inputs[0].supports[0].quote = quote
        sources[0].metadata["source_events"][0]["text"] = quote
    elif change == "prefix_collision":
        contract.outputs[0].subjects[0] = "Aur"
    assert execute(answer, sources, contract) == "Aurora arrival"


def test_overlapping_times_keep_uncertainty_instead_of_displaying_a_winner():
    answer, sources, contract = comparison(dates=("2030-03-04", "2030-03-04"))
    assert "their order within that day is unknown" in execute(answer, sources, contract)


def partial(value, gap):
    return AnswerTable.model_validate({"slots": [{"target": "Known value and unknown result",
        "derivation": "Publish the known part and retain the unknown part.",
        "values": [{"value": value}], "missing": gap, "missing_kind": "requested_part",
        "coverage": "partial"}]})


def test_exact_gap_sentence_is_published_once_without_changing_receipt():
    answer = partial("The first item costs 12 credits. The total is unknown.", "The total is unknown.")
    before = deepcopy(answer.model_dump())
    assert render_table(answer) == "The first item costs 12 credits. The total is unknown."
    assert answer.model_dump() == before


def test_entire_duplicate_gap_normalizes_only_formatting():
    answer = partial("The total is unknown", "The total is unknown.")
    assert render_table(answer) == "The total is unknown"


@pytest.mark.parametrize("gap", ["The remaining cost is unknown.",
                                "The total was unknown last year.",
                                "The total is not unknown.",
                                "The total is unknown. The delivery charge is unknown."])
def test_related_but_distinct_gaps_and_added_conditions_remain(gap):
    answer = partial("The total is unknown.", gap)
    assert render_table(answer).endswith(gap)


def test_keyword_inside_a_longer_clause_is_not_a_duplicate_gap_sentence():
    answer = partial("If the total is unknown, ask the vendor.", "The total is unknown.")
    assert render_table(answer).endswith("The total is unknown.")

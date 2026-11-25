"""Equivalent exact scalar displays share the already-declared unit."""
import copy

import pytest

from agent_memory.models import Hit
from agent_memory.synthesis import AnswerTable, render_table, validate_table, validation_feedback


def case(value, unit="credits", amounts=("1.25", "-0.5"), operation="sum"):
    quote = "Recorded adjustments: " + ", ".join(amounts) + "."
    support = {"source_id": 1, "quote": quote}
    table = AnswerTable.model_validate({"slots": [{"target": "Result", "derivation": "Use the recorded operands.",
        "values": [{"operation": operation, "unit": unit, "value": value, "inputs": [
            {"identity": f"entry-{i}", "amount": amount, "statement": f"Recorded adjustment {i}",
             "supports": [support]} for i, amount in enumerate(amounts)]}]}]})
    hit = Hit(content=quote, score=1, source_file="source", chunk_id="one", metadata={"source_events": [
        {"text": quote, "role": "user", "speaker": "Quin", "context_only": False, "evidence_grade": "primary"}]})
    return table, [hit]


@pytest.mark.parametrize("value,unit,amounts,operation,expected", [
    ("2", "records", ("1", "1"), "count", "2 records"),
    ("0.75", "credits", ("1.25", "-0.5"), "sum", "0.75 credits"),
    ("-0.75", "credits", ("-1.25", "0.5"), "sum", "-0.75 credits"),
    ("0", "net credits", ("-1.25", "1.25"), "sum", "0 net credits"),
    (" 0.75 ", " credits ", ("1.25", "-0.5"), "sum", "0.75 credits"),
    ("0.75 credits", "credits", ("1.25", "-0.5"), "sum", "0.75 credits"),
    ("", "credits", ("1.25", "-0.5"), "sum", "0.75 credits"),
    ("0.75", "", ("1.25", "-0.5"), "sum", "0.75"),
    ("+0.75", "credits", ("1.25", "-0.5"), "sum", "0.75 credits"),
    ("00.75", "credits", ("1.25", "-0.5"), "sum", "0.75 credits"),
    ("0.750", "credits", ("1.25", "-0.5"), "sum", "0.75 credits"),
    ("0.750", "", ("1.25", "-0.5"), "sum", "0.75"),
    (".75", "credits", ("1.25", "-0.5"), "sum", "0.75 credits"),
])
def test_equivalent_exact_scalar_display_is_normalized(value, unit, amounts, operation, expected):
    table, sources = case(value, unit, amounts, operation)
    original_inputs = copy.deepcopy(table.slots[0].values[0].inputs)
    validate_table(table, sources, ["Result"], "text")
    assert render_table(table) == expected
    assert table.slots[0].values[0].inputs == original_inputs
    validate_table(table, sources, ["Result"], "text")
    assert render_table(table) == expected


@pytest.mark.parametrize("value,unit", [
    ("0.8", "credits"), ("0.75 dollars", "credits"), ("about 0.75", "credits"),
    ("at least 0.75 credits", "credits"), ("0.75 for known entries", "credits"),
    ("0.75 if the remaining entry is zero", "credits"),
    ("0.75 credits", ""), ("value: 0.75", "credits"),
])
def test_wrong_or_qualified_values_are_not_normalized(value, unit):
    table, sources = case(value, unit)
    with pytest.raises(ValueError, match="Declared operation"):
        validate_table(table, sources, ["Result"], "text")
    assert table.slots[0].values[0].value == value


def test_failed_response_feedback_uses_the_same_format_equivalence():
    table, _ = case("0.75")
    assert validation_feedback(table, ValueError("A separate existing failure")) == "A separate existing failure"
    table.slots[0].values[0].value = "0.75 for the known part"
    assert "Declared operation='sum'" in validation_feedback(table, ValueError("A separate existing failure"))

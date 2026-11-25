import pytest
from test_answer_synthesis import hit, table, turn

from agent_memory.synthesis import BoundInput, Support, SupportedValue, render_table, validate_table


def operand(identity, amount=None, index=1):
    return BoundInput(identity=identity, statement=f"A qualified input: {identity}",
                      supports=[{"source_id": index, "quote": "I attended A for 4 years and B for 2 years."}], amount=amount)


def computed(operation, inputs, quote="I attended A for 4 years and B for 2 years."):
    result = table("", quote)
    item = result.slots[0].values[0]
    for bound in inputs:
        for ref in bound.supports:
            ref.quote = quote
    item.operation, item.inputs, item.unit = operation, inputs, "years" if operation == "sum" else "films"
    return result, [hit(turn(1, quote))]


def check(result, sources):
    validate_table(result, sources, ["colors"], "text")
    return render_table(result)


def test_count_deduplicates_members_not_source_mentions():
    result, sources = computed("count", [operand("film A"), operand("FILM A"), operand("film B")])
    assert check(result, sources) == "2 films"
    assert check(result, sources) == "2 films"  # replay is idempotent


def test_distinct_events_same_place_are_distinct_inputs():
    result, sources = computed("count", [operand("Paris 2023"), operand("Paris 2024")])
    assert check(result, sources) == "2 films"


def test_sum_uses_selected_stages_and_exact_decimals():
    result, sources = computed("sum", [operand("A", "0.1"), operand("B", "0.2"), operand("A", "0.1")])
    assert check(result, sources) == "0.3 years"


@pytest.mark.parametrize("inputs", [[], [operand("A")], [operand("A", "2"), operand("A", "3")]])
def test_missing_or_conflicting_amounts_are_not_guessed(inputs):
    result, sources = computed("sum", inputs)
    with pytest.raises(ValueError):
        check(result, sources)


def test_unknown_input_reference_rejected():
    result, sources = computed("count", [operand("A", index=2)])
    with pytest.raises(ValueError, match="Unknown source"):
        check(result, sources)


def test_direct_total_needs_no_member_enumeration():
    result = table("12", "I watched 12 films.")
    assert check(result, [hit(turn(1, "I watched 12 films."))]) == "12"


def test_computation_cannot_silently_erase_approximation():
    result, sources = computed("sum", [operand("A", "4"), operand("B", "2")],
                               "I attended A for about four years, then B for two years.")
    item = result.slots[0].values[0]
    item.value = "about 6 years"
    with pytest.raises(ValueError, match="qualified"):
        check(result, sources)
    assert item.value == "about 6 years"
    item.operation = "none"
    assert check(result, sources) == "about 6 years"


def test_missing_dependency_removes_only_dependent_value_and_citations():
    result = table("5 tomatoes", "I planted 5 tomatoes.")
    blocked = SupportedValue(value="4 peppers", supports=[{"source_id": 99, "quote": "unavailable"}],
                             unresolved_premise="Pepper quantity was not provided")
    result.slots[0].values.append(blocked)
    answer = check(result, [hit(turn(1, "I planted 5 tomatoes."))])
    assert "5 tomatoes" in answer and "Pepper quantity was not provided" in answer
    assert "4 peppers" not in answer
    assert len(result.slots[0].values) == 1
    assert result.slots[0].coverage == "partial"
    assert check(result, [hit(turn(1, "I planted 5 tomatoes."))]) == answer


def test_all_values_dependent_on_missing_premise_become_unavailable_without_retry():
    result = table("4", "Senior")
    result.slots[0].values[0].unresolved_premise = "Manager count is unknown"
    check(result, [])
    assert result.slots[0].coverage == "unavailable"
    assert not result.slots[0].values


def test_necessary_gap_requires_explicit_dependency_but_independent_gap_keeps_value():
    result = table("5 tomatoes", "I planted 5 tomatoes.")
    slot = result.slots[0]
    slot.coverage = "partial"
    slot.missing = "Pepper quantity is unknown"
    slot.missing_kind = "necessary_premise"
    sources = [hit(turn(1, "I planted 5 tomatoes."))]
    with pytest.raises(ValueError, match="linked"):
        check(result, sources)
    assert slot.values[0].value == "5 tomatoes"  # Never infer which value to remove.
    slot.missing_kind = "requested_part"
    assert "5 tomatoes" in check(result, sources)


def test_third_party_statement_is_valid_support():
    result, sources = computed("count", [operand("Melanie event")], "Caroline: Melanie visited Paris.")
    assert check(result, sources) == "1 films"


def test_inputs_keep_direct_source_quotes_without_local_index_indirection():
    result, sources = computed("count", [operand("old car"), operand("new car")])
    item = result.slots[0].values[0]
    item.supports = []
    sources = [hit(turn(1, "I sold my old Prius.")), hit(turn(2, "I bought a new Prius."))]
    item.inputs[0].supports = [Support(source_id=1, quote="I sold my old Prius.")]
    item.inputs[1].supports = [Support(source_id=2, quote="I bought a new Prius.")]
    # Validate the externally supplied structure exactly as a provider would.
    from agent_memory.synthesis import AnswerTable
    result = AnswerTable.model_validate(result.model_dump())
    check(result, sources)
    assert [(s.source_id, s.quote) for s in result.slots[0].values[0].supports] == [
        (1, "I sold my old Prius."), (2, "I bought a new Prius.")]


def test_local_missing_reasons_are_not_silently_cut_off():
    result = table("5 tomatoes", "I planted 5 tomatoes.")
    reasons = ["A " * 240, "B " * 240]
    for reason in reasons:
        result.slots[0].values.append(SupportedValue(unresolved_premise=reason))
    check(result, [hit(turn(1, "I planted 5 tomatoes."))])
    assert all(reason.strip() in result.slots[0].missing for reason in reasons)

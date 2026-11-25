import pytest
from test_answer_synthesis import hit, table, turn

from agent_memory.synthesis import SupportedValue, validate_table


def test_general_suggestions_share_explicit_personal_premise_without_fabricated_citations():
    result = table("Use your power bank", "I own a power bank.")
    result.slots[0].values[0].basis = "recommendation"
    result.slots[0].values.append(SupportedValue(value="Enable battery saver", basis="recommendation"))
    validate_table(result, [hit(turn(1, "I own a power bank."))], ["colors"], "text")
    assert result.slots[0].values[1].supports == []


def test_unpersonalized_bundle_still_requires_a_user_premise():
    result = table("Enable battery saver")
    result.slots[0].values = [SupportedValue(value="Enable battery saver", basis="recommendation")]
    with pytest.raises(ValueError, match="quoted user assertion"):
        validate_table(result, [], ["colors"], "text")


def test_factual_claim_cannot_borrow_recommendation_exemption():
    result = table("Use your power bank", "I own a power bank.")
    result.slots[0].values[0].basis = "recommendation"
    result.slots[0].values.append(SupportedValue(value="Your battery is damaged", basis="memory"))
    with pytest.raises(ValueError, match="original source supports"):
        validate_table(result, [hit(turn(1, "I own a power bank."))], ["colors"], "text")

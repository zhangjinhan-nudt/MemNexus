"""Optional list-position semantics must not perturb unrelated model requests."""
import asyncio
import hashlib
import json

import pytest
from test_request_binding_integration import ScriptedChat, prompt_data

from agent_memory.models import AnswerContract, AnswerOutput, QueryPlan
from agent_memory.prompt_codec import synthesis_request_payload
from agent_memory.providers import CallableModel

CONTEXT = {"relation_catalog": [["person", "first draft", "one", None]], "query_time": "2025-02-10"}
# Captured with the actual pre-member_positions provider, using the same local
# scripted callback, context and output language. These are request bytes, not
# model predictions or benchmark answers; changing a shared prompt is visible.
PRE_POSITION_PROMPTS = [
    ("How many days passed between the workshop and the exhibition?",
     "174c008f7c897fd90bc99ac1ab001b91d618d044be1809902c2201ae129dfce6"),
    ("Which coastal dish did you recommend if fresh fruit was available?",
     "669f816591c80829d1db5370dd8ce1173c09c6810670e0a4a325f40369f92885"),
    ("What exercise would fit my preference for short outdoor sessions?",
     "624ab4f020b9bf2c56b91128489b68202ca0d40df365d190dc552e523a7e06ad"),
    ("Where does Mira live?",
     "d43326ff1ff0d0e22e7c06abda33f0373980671cc6b732c0114e52ca34c04a54"),
]


@pytest.mark.parametrize("question,expected_sha256", PRE_POSITION_PROMPTS)
def test_nonordinal_planner_request_is_byte_identical_to_original(question, expected_sha256):
    chat = ScriptedChat({"route": "both"})
    asyncio.run(CallableModel(chat, output_language="en").plan(question, CONTEXT))
    assert len(chat.prompts) == 1
    prompt = chat.prompts[0]
    assert "member_positions" not in prompt
    # The catalog contains 'first'; only the original query enables the field.
    assert "first draft" in prompt
    assert hashlib.sha256(prompt.encode()).hexdigest() == expected_sha256


@pytest.mark.parametrize("ordinal,position", [("first", 1), ("fifth", 5), ("27th", 27), ("twenty-seventh", 27)])
def test_literal_ordinal_keeps_semantic_binding_in_the_same_planner_call(ordinal, position):
    question = f"Which was the {ordinal} configuration?"
    chat = ScriptedChat({"route": "both", "answer_contract": {"outputs": [{
        "output_id": "configuration", "question_span": question, "member_positions": [position],
    }]}})
    plan = asyncio.run(CallableModel(chat, output_language="en").plan(question, CONTEXT))
    assert len(chat.prompts) == 1
    assert '"member_positions"' in chat.prompts[0]
    assert "Leave it empty for event chronology" in chat.prompts[0]
    assert plan.answer_contract.outputs[0].member_positions == [position]


def test_empty_member_positions_are_omitted_only_from_reader_payload():
    question = "Where does Mira live?"
    contract = AnswerContract(outputs=[AnswerOutput(output_id="city", question_span=question)])
    encoded = synthesis_request_payload(contract)
    assert "member_positions" not in encoded["outputs"][0]
    assert contract.outputs[0].model_dump()["member_positions"] == []
    assert json.loads(contract.model_dump_json())["outputs"][0]["member_positions"] == []
    chat = ScriptedChat({"slots": [{"target": question, "derivation": "No supplied sources.",
                                   "values": [], "missing": "No supplied sources."}]})
    plan = QueryPlan(answer_contract=contract)
    asyncio.run(CallableModel(chat, output_language="en").synthesize(question, plan, []))
    assert len(chat.prompts) == 1
    assert "member_positions" not in prompt_data(chat.prompts[0])["request_contract"]["outputs"][0]


def test_nonempty_member_position_is_not_compacted_away():
    contract = AnswerContract(outputs=[AnswerOutput(
        output_id="configuration", question_span="Which was the fifth configuration?", member_positions=[5])])
    assert synthesis_request_payload(contract)["outputs"][0]["member_positions"] == [5]

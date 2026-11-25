"""The existing answer wire distinguishes executable inputs from published results."""
import asyncio
import copy
import json

from agent_memory.models import Hit, QueryPlan
from agent_memory.providers import CallableModel
from agent_memory.synthesis import render_table


def source(text):
    return Hit(content=text, score=1, source_file="fixture", chunk_id="one", metadata={
        "source_events": [{"text": text, "role": "user", "speaker": "Noel",
                           "occurred_at": "2030-02-10T12:00:00+00:00", "context_only": False,
                           "evidence_grade": "primary"}]})


def test_selected_operands_are_executed_and_each_requested_result_is_published():
    question = "Which packets were delivered, and how many were there?"
    quote = "The amber and ivory packets arrived. The violet packet remains in transit."
    support = {"source_id": 1, "quote": quote}
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return json.dumps({"slots": [{"target": question,
            "derivation": "Only the amber and ivory packets arrived; violet is excluded as in transit.",
            "values": [{"value": "amber packet", "supports": [support]},
                       {"value": "ivory packet", "supports": [support]},
                       {"operation": "count", "unit": "packets", "inputs": [
                           {"identity": "amber packet", "statement": "The amber packet was delivered.", "supports": [support]},
                           {"identity": "ivory packet", "statement": "The ivory packet was delivered.", "supports": [support]}]}]}]})

    table = asyncio.run(CallableModel(chat).synthesize(question, QueryPlan(), [source(quote)]))
    assert render_table(table) == "amber packet, ivory packet, 2 packets"
    assert len(calls) == 1
    assert [i.identity for i in table.slots[0].values[-1].inputs] == ["amber packet", "ivory packet"]


def test_exact_requested_gap_can_coexist_with_a_supported_range_in_native_repair():
    question = "Exactly how many grams does the sealed parcel weigh?"
    quote = "The sealed parcel weighs somewhere from 80 to 90 grams; no exact measurement was taken."
    initial = {"slots": [{"target": question, "derivation": "An exact measurement is absent; a range is supported.",
        "values": [{"value": "80 to 90 grams", "basis": "memory", "supports": [{"source_id": 1, "quote": quote}]}],
        "missing": "The exact weight was not measured.", "missing_kind": "necessary_premise", "coverage": "partial"}]}
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(initial)
        assert "its requested precision" in prompt
        assert "keep the qualified answer and name that unmet request" in prompt
        corrected = copy.deepcopy(initial)
        corrected["slots"][0]["missing_kind"] = "requested_part"
        return json.dumps(corrected)

    table = asyncio.run(CallableModel(chat).synthesize(question, QueryPlan(), [source(quote)]))
    assert len(calls) == 2
    assert render_table(table) == "80 to 90 grams. The exact weight was not measured."
    assert table.slots[0].coverage == "partial"


def test_approximate_request_does_not_require_an_exact_measurement():
    question = "Roughly how heavy is the sealed parcel?"
    quote = "The sealed parcel weighs somewhere from 80 to 90 grams; no exact measurement was taken."
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return json.dumps({"slots": [{"target": question, "derivation": "The recorded range meets the approximate request.",
            "values": [{"value": "About 80 to 90 grams", "supports": [{"source_id": 1, "quote": quote}]}],
            "coverage": "complete", "missing": "Exact weight was not measured.", "missing_kind": "optional_detail"}]})

    table = asyncio.run(CallableModel(chat).synthesize(question, QueryPlan(), [source(quote)]))
    assert len(calls) == 1
    assert render_table(table) == "About 80 to 90 grams"

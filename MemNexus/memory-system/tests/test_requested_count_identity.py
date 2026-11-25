"""The existing reader/executor interface distinguishes events from categories.

Model responses are frozen fixtures: these tests check transport and computation,
not that a model can always enumerate all qualifying members in free text.
"""

import asyncio
import json

import pytest

from agent_memory.models import AnswerContract, AnswerOutput, Hit, QueryPlan
from agent_memory.providers import CallableModel
from agent_memory.synthesis import render_table

QUOTE = "I attended pottery on Monday and Wednesday, and swimming on Friday."
REPEAT = "Wednesday's pottery class was enjoyable."


def source():
    return Hit(content=QUOTE + " " + REPEAT, score=1, source_file="fixture", chunk_id="one",
               metadata={"source_events": [{
                   "text": QUOTE + " " + REPEAT, "speaker": "Inez", "role": "user",
                   "occurred_at": "2034-06-10T12:00:00+00:00", "context_only": False,
                   "evidence_grade": "primary", "time_derivations": [],
               }]})


@pytest.mark.parametrize("question,count_unit,identities,expected", [
    ("How many activity sessions did I attend?", "event",
     ["pottery Monday", "pottery Wednesday", "swimming Friday", "pottery Wednesday"], "3"),
    ("How many different activity types did I attend?", "value",
     ["pottery", "pottery", "swimming", "pottery"], "2"),
])
def test_same_sources_preserve_requested_unit_and_distinct_occurrence_bindings(question, count_unit, identities, expected):
    calls = []
    plan = QueryPlan(answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="q1", question_span=question, subjects=["I"], relation="activity attendance",
        operator="count", count_unit=count_unit,
    )]))

    async def chat(prompt):
        calls.append(prompt)
        data = json.JSONDecoder().raw_decode(prompt[prompt.index('{"question":'):])[0]
        assert data["question"] == question
        assert data["request_contract"]["outputs"][0]["count_unit"] == count_unit
        assert data["sources"][0]["source_events"][0]["text"] == QUOTE + " " + REPEAT
        return json.dumps({"slots": [{"target": question,
            "derivation": "Bind the identities at the unit requested by the question.", "values": [{
            "value": "", "operation": "count", "inputs": [
                {"identity": identity, "statement": "The stated activity and occurrence.",
                 "supports": [{"source_id": 1, "quote": QUOTE if index < 3 else REPEAT}]}
                for index, identity in enumerate(identities)
            ],
        }]}]})

    table = asyncio.run(CallableModel(chat).synthesize(question, plan, [source()]))
    assert len(calls) == 1  # No new enumeration, verification or repair call.
    assert render_table(table) == expected


def test_unenumerated_frequency_can_remain_a_qualified_answer_without_new_refusal():
    question = "How many activity sessions did I attend?"
    quote = "I do pottery regularly, and attended swimming on Friday."
    evidence = source()
    evidence.content = quote
    evidence.metadata["source_events"][0]["text"] = quote
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return json.dumps({"slots": [{"target": question, "coverage": "partial",
            "derivation": "Only one swimming occurrence is bounded; pottery frequency is unspecified.",
            "missing": "The number of pottery sessions is unspecified.", "missing_kind": "requested_part",
            "values": [{"value": "One swimming session plus an unspecified number of pottery sessions.",
                        "operation": "none", "basis": "inference",
                        "supports": [{"source_id": 1, "quote": quote}]}]}]})

    table = asyncio.run(CallableModel(chat).synthesize(question, QueryPlan(), [evidence]))
    assert len(calls) == 1
    assert "One swimming session" in render_table(table)
    assert "unspecified" in render_table(table)

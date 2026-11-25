"""A partly-supported conjunction must be answered *and* flagged, not one or the other.

LongMemEval's unanswerable items state the expectation directly: "The information
provided is not enough. You mentioned planting 5 plants for tomatoes but you did
not mention chili peppers." The answer layer currently does one half of that — it
returns the supported value and leaves `missing` empty — so a question whose
evidence covers one conjunct scores zero even though the covered part is right.

The prompt already requires every conjunct to be preserved; what it never said is
what to do when a conjunct has no support. This pins that clause, since it is the
only thing standing between "partial answer" and "wrong answer".
"""

import asyncio

from agent_memory import providers
from agent_memory.models import Hit, QueryPlan
from agent_memory.synthesis import render_table

QUESTION = "How many plants did I plant for tomatoes and chili peppers?"


def hit(value: str, quote: str) -> Hit:
    return Hit(
        content=quote,
        score=0.9,
        source_file="memory:test",
        chunk_id="test:1",
        metadata={
            "memory_id": "m1",
            "subject": "user",
            "predicate": "planted",
            "value": value,
            "evidence": [],
            "source_events": [
                {"turn_id": "turn-1", "event_index": 0, "role": "user", "speaker": "user",
                 "text": quote, "context_only": False,
                 "occurred_at": "2023-05-20T09:00:00+00:00"}
            ],
        },
    )


def test_answer_prompt_requires_naming_an_unmet_conjunct():
    captured = {}

    class _FakeModel:
        async def _structured(self, prompt, model, validate, data=None):
            captured["prompt"] = prompt
            captured["data"] = data
            table = model.model_validate({"slots": [{"target": QUESTION,
                 "derivation": "Tomato count is independently supported; chili pepper count is unknown.",
                 "values": [{"value": "5", "supports": [{"source_id": 1, "quote": "5 tomato plants"}]}],
                 "coverage": "partial",
                 "missing": "you did not mention chili peppers", "missing_kind": "requested_part"}]})
            validate(table)
            return table

    sources = [hit("5", "I planted 5 tomato plants.")]
    result = asyncio.run(
        providers.CallableModel.synthesize(_FakeModel(), QUESTION, QueryPlan(), sources)
    )

    prompt = captured["prompt"]
    assert "If only part is supported, keep" in prompt and "coverage=partial" in prompt
    assert captured["data"]["question"] == QUESTION
    assert render_table(result, question=QUESTION) == "5. you did not mention chili peppers"

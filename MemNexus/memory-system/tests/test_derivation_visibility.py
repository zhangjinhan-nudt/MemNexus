"""A derivation computed over the whole evidence set must reach the prompt.

`prompt_context` appends up to forty node lines before any derivation, then cuts
the result at `max_chars`. Measured on recorded runs, that cut removed *every*
non-order derivation: 103 of 103 LongMemEval rows and 426 of 426 LoCoMo rows. The
model was therefore told to compute nothing it had not already been given, while
the runtime paid for the computation.

These tests pin the two properties the fix must keep: a derivation survives a full
node list, and a time-only derivation is *not* rendered when temporal execution is
off (otherwise the flag's effect cannot be measured, and LoCoMo prompts would
change for no gain).
"""

from agent_memory.models import Hit
from agent_memory.reasoning import EventGraph, classify_question


def hit(content: str, *, subject: str | None = None) -> Hit:
    return Hit(
        content=content,
        score=0.9,
        source_file="memory:test",
        chunk_id=f"test:{abs(hash(content)) % 10000}",
        metadata={
            "memory_id": f"m{abs(hash(content)) % 10000}",
            "subject": subject,
            "predicate": None,
            "value": content,
            "evidence": [],
            "source_events": [
                {
                    "turn_id": "turn-1",
                    "event_index": 0,
                    "role": "user",
                    "text": content,
                    "occurred_at": "2023-05-20T09:00:00+00:00",
                }
            ],
        },
    )


def filler(count: int) -> list[Hit]:
    """Enough nodes that the node list alone exceeds the 2600-char budget."""
    return [hit(f"Note {index}: " + ("detail " * 40)) for index in range(count)]


def test_unbound_date_pairs_are_not_published_as_answers():
    # Two dates alone do not bind the requested event roles.
    hits = [
        hit("The user received feedback about the suspension on 2023-03-17."),
        hit("The user tested the new setup on 2023-04-24."),
    ] + filler(40)
    graph = EventGraph.from_hits(hits)
    route = classify_question("How many days passed between the two events?", [])
    context = graph.prompt_context(
        "How many days passed between the two events?", route, max_chars=2600, temporal_execution=True
    )
    assert "ELAPSED:" not in context
    # The node list still gets its share: the context is not just derivations.
    assert "Note 0" in context


def test_time_only_derivations_are_absent_when_temporal_execution_is_off():
    hits = [hit("I am planning a trip next month.")] + filler(40)
    graph = EventGraph.from_hits(hits)
    question = "When am I planning a trip?"
    route = classify_question(question, [])
    context = graph.prompt_context(question, route, max_chars=2600, temporal_execution=False)
    for marker in ("ELAPSED:", "DATED:", "LENGTH STATED IN SOURCE:", "ORDER IS DETERMINED"):
        assert marker not in context
    # And the derivations that would be rendered are exactly none.
    derivations = graph.deterministic_derivations(question, route, temporal_execution=False)
    assert graph._derivation_lines(derivations, 2600, question) == []


def test_geography_bridge_is_rendered_only_for_place_questions():
    hits = [hit("Jolene bought the pendant in Paris.")] + filler(5)
    graph = EventGraph.from_hits(hits)
    question = "In what country did Jolene buy the pendant?"
    route = classify_question(question, [])
    assert "INFERRED: Paris is in France." in graph.prompt_context(question, route, 2600)

    other = "When did Jolene buy the pendant?"
    assert "Paris is in France" not in graph.prompt_context(other, classify_question(other, []), 2600)


def test_derivation_budget_leaves_room_for_evidence():
    # A ledger of dated rows must not consume the whole context: the node list is
    # what the quotes are verified against.
    hits = [hit(f"Event on 2023-0{index + 3}-1{index % 9}: " + ("text " * 30)) for index in range(9)]
    hits += filler(30)
    graph = EventGraph.from_hits(hits)
    question = "How long had I been taking classes?"
    route = classify_question(question, [])
    context = graph.prompt_context(question, route, 2600, temporal_execution=True)
    assert "DATED:" in context
    # Node lines also start with "- ", so count only the derivation prefixes.
    prefixes = ("- ELAPSED:", "- DATED:", "- LENGTH STATED IN SOURCE:", "- INFERRED:",
                "ORDER IS DETERMINED")
    derivation_chars = sum(
        len(line) for line in context.splitlines() if line.startswith(prefixes)
    )
    assert derivation_chars <= 2600 * 0.35 + 200

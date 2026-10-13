"""Offline checks for query routing, event graph construction, and typed gates."""

from datetime import UTC, datetime

from agent_memory.grounding import ground_evidence
from agent_memory.models import Event, Evidence, Hit, Memory, QueryPlan, Turn
from agent_memory.reasoning import (
    EventGraph,
    apply_route,
    classify_question,
    indexed_memory_hits,
    typed_negative_gate,
)


def hit(content: str, *, subject: str | None = None, predicate: str | None = None) -> Hit:
    return Hit(
        content=content,
        score=0.9,
        source_file="memory:test",
        chunk_id="test:1",
        metadata={
            "memory_id": "m1",
            "subject": subject,
            "predicate": predicate,
            "value": content,
            "evidence": [],
            "source_events": [
                {
                    "turn_id": "turn-1",
                    "event_index": 0,
                    "speaker": subject or "Caroline",
                    "role": "user",
                    "occurred_at": "2023-05-08T13:00:00+00:00",
                    "text": content,
                    "context_only": False,
                    "time_derivations": [],
                }
            ],
        },
    )


def test_question_router_selects_structured_policies():
    assert classify_question("How many road trips did Evan take?").kind == "count"
    assert classify_question("Why did Jon choose the studio?").kind == "causal"
    assert classify_question("What did Jon and Gina both have in common?").kind == "relation"
    route = classify_question("Which DJ was Dave's favorite in April 2023? Choose exactly one option.")
    assert route.kind == "adversarial"
    assert route.negative_gate and "object_type=dj" in route.constraints
    assert "month=april" in route.constraints
    assert classify_question("What are John and James' favorite games?", ["John", "James"]).kind == "relation"
    assert classify_question("What are Dave's dreams?", ["Dave"]).kind == "set"
    assert classify_question("What sports does John like besides basketball?", ["John"]).kind == "set"
    assert classify_question("What is John's favorite game?", ["John"]).kind == "fact"


def test_event_graph_keeps_typed_sources_and_derives_geography():
    graph = EventGraph.from_hits([hit("Jolene bought the pendant in Paris.", subject="Jolene")])
    route = classify_question("In what country did Jolene buy the pendant?")
    derivations = graph.deterministic_derivations("In what country did Jolene buy the pendant?", route)
    assert any(
        item["kind"] == "bounded_geography" and item["text"] == "Paris is in France." for item in derivations
    )
    context = graph.prompt_context("In what country did Jolene buy the pendant?", route)
    assert "event_type=purchase" in context
    assert "France" in context


def test_typed_negative_gate_records_missing_support_without_proving_absence():
    route = classify_question(
        "Which DJ was Dave's favorite at the music festival in April 2023? Choose exactly one option."
    )
    graph = EventGraph.from_hits(
        [hit("Dave said Aerosmith was his favorite band last weekend.", subject="Dave")]
    )
    gate = typed_negative_gate(
        route=route, query="Which DJ was Dave's favorite at the music festival in April 2023?", graph=graph
    )
    assert not gate.blocked
    assert gate.reason.startswith("not_found_in_examined_evidence")
    assert "subject/type/time" in gate.reason

    wrong_subject = EventGraph.from_hits(
        [hit("Caroline chose an LGBTQ adoption agency.", subject="Caroline")]
    )
    melanie = classify_question(
        "What type of individuals does the adoption agency Melanie is considering support? "
        "Not mentioned in the conversation."
    )
    assert typed_negative_gate(
        "What type of individuals does the adoption agency Melanie is considering support?",
        melanie,
        wrong_subject,
    ).reason.startswith("not_found_in_examined_evidence")


def test_route_is_recorded_without_overwriting_model_semantics():
    plan = QueryPlan(required_info=["items"], depth=3)
    routed = apply_route(plan, classify_question("What items did Calvin buy?"))
    assert routed.query_type == "set"
    assert routed.recall_policy == "recall_first"
    assert routed.required_info == ["items"]
    assert routed.depth >= 8


def test_indexed_memory_retrieval_restores_the_second_subject_and_source():
    query = "What type of volunteering have John and Maria both done?"
    route = classify_question(query, ["John", "Maria"])
    source_texts = {
        "john": "John's service group went to a homeless shelter to give out food and supplies.",
        "maria": "Maria volunteers at a homeless shelter and donated her old car there.",
    }
    turns = {
        name: Turn(
            turn_id=name,
            session_id="session",
            sequence=index,
            request_id=name,
            events=[
                Event(
                    role="user",
                    speaker=name.title(),
                    content=text,
                    occurred_at=datetime(2023, 1, 1, tzinfo=UTC),
                )
            ],
        )
        for index, (name, text) in enumerate(source_texts.items(), 1)
    }
    memories = [
        Memory(
            memory_id=name,
            session_id="session",
            scope_id="session",
            subject="John's service group" if name == "john" else "Maria",
            predicate="event" if name == "john" else "volunteering",
            value=text,
            content=text,
            evidence=[Evidence(turn_id=name, event_index=0, quote=text)],
        )
        for name, text in source_texts.items()
    ]
    first = Hit(
        content=memories[1].content,
        score=0.8,
        source_file="[memory:maria]",
        chunk_id="maria:1",
        metadata={
            "memory_id": "maria",
            "subject": "Maria",
            "predicate": "volunteering",
            "evidence": [item.model_dump(mode="json") for item in memories[1].evidence],
        },
    )
    extra = indexed_memory_hits(query, route, memories, [first])
    assert len(extra) == 1 and extra[0].metadata["memory_id"] == "john"
    selected = ground_evidence(
        query,
        apply_route(QueryPlan(required_info=["John", "Maria", "homeless shelter"]), route),
        [first, *extra],
        session_id="session",
        current_turns=list(turns.values()),
        resolved_turns=turns,
        blocked_events=set(),
        max_sources=8,
        raw_limit=0,
        budget=5000,
    )
    selected_turns = {
        source["turn_id"] for hit in selected.sources for source in hit.metadata["source_events"]
    }
    assert selected_turns == {"john", "maria"}
    assert selected.trace.indexed_sources_selected == 1

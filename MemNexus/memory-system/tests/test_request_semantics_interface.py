"""Request transport and repair contracts; all model responses are local stubs."""

import asyncio
import copy
import json
from datetime import UTC, datetime

import pytest

from agent_memory.models import CollectionQuery, Event, ExtractionWindow, Hit, QueryPlan, Session, Turn
from agent_memory.providers import CallableModel, ModelOutputError
from agent_memory.synthesis import (
    AnswerTable,
    BoundInput,
    SupportedValue,
    answer_type,
    render_table,
    validate_table,
)


def source(text, stamp="2025-05-05T12:00:00+00:00"):
    return Hit(content=text, score=.8, source_file="source", chunk_id=text, metadata={
        "source_events": [{"text": text, "role": "user", "speaker": "Inez", "occurred_at": stamp,
                           "context_only": False, "evidence_grade": "primary", "time_derivations": []}],
    })


def answer(question, text, quote, **slot_fields):
    return {"slots": [{"target": question, "derivation": "Bind the requested relation to the source.",
                       "values": [{"value": text, "basis": "inference", "supports": [
                           {"source_id": 1, "quote": quote}]}], **slot_fields}]}


def payload(prompt):
    return json.JSONDecoder().raw_decode(prompt[prompt.index('{"question":'):])[0]


def test_retrieval_strategy_variants_leave_the_synthesis_request_identical():
    question, quote = "Which color does Inez prefer?", "I prefer blue."
    plans = [
        QueryPlan(),
        QueryPlan(subject="Inez", predicate="prefers", required_info=["preferred color"],
                  query_entities=["Inez"], query_constraints=["current preference"]),
        QueryPlan(subject="Kai", predicate="borrowed", query_type="count",
                  semantic_queries=["a retrieval strategy"], keywords=["retrieval keyword"],
                  required_info=["another retrieval goal"], depth=9),
    ]
    snapshots, calls = [plan.model_dump() for plan in plans], []

    async def chat(prompt):
        calls.append(prompt)
        return json.dumps(answer(question, "blue", quote))

    for plan in plans:
        table = asyncio.run(CallableModel(chat).synthesize(question, plan, [source(quote)]))
        assert render_table(table) == "blue"
    assert len(calls) == len(plans) and len(set(calls)) == 1
    assert [plan.model_dump() for plan in plans] == snapshots


def test_collection_retrieval_strategy_does_not_add_an_answer_requirement():
    question, quote = "Which color does Inez prefer?", "I prefer blue."
    plans = [QueryPlan(), QueryPlan(
        subject="Inez", predicate="read", query_type="count", query_entities=["Inez", "Kai"],
        collection_query=CollectionQuery(subjects=["Inez"], predicate="read", operation="count",
                                         qualifier="during the trip", exclude_values=["draft"]))]
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        data = payload(prompt)
        assert data["question"] == question and data["targets"] == [question]
        assert "plan" not in data and "collection_query" not in data
        return json.dumps(answer(question, "blue", quote))

    for plan in plans:
        asyncio.run(CallableModel(chat).synthesize(question, plan, [source(quote)]))
    assert len(calls) == 2 and calls[0] == calls[1]


def test_original_question_sources_and_clock_reach_the_same_synthesis_call():
    question, quote = "Which color does Inez prefer?", "I prefer blue."
    plan = QueryPlan(subject="Inez", predicate="prefers", required_info=["preferred color"],
                     query_constraints=["current preference"])
    original_sources = [source(quote)]
    before = [hit.model_dump() for hit in original_sources]
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        data = payload(prompt)
        assert data["question"] == question and data["targets"] == [question]
        assert "plan" not in data and "required_info" not in data
        event = data["sources"][0]["source_events"][0]
        assert event["text"] == quote and event["role"] == "user" and event["speaker"] == "Inez"
        assert event["occurred_at"] == "2025-05-05T12:00:00+00:00"
        assert data["as_of"] == "2025-05-20"
        assert "The supplied plan" not in prompt
        return json.dumps(answer(question, "blue", quote))

    table = asyncio.run(CallableModel(chat).synthesize(
        question, plan, original_sources, as_of=datetime(2025, 5, 20, tzinfo=UTC)))
    assert render_table(table) == "blue" and len(calls) == 1
    assert [hit.model_dump() for hit in original_sources] == before


def test_plan_alone_cannot_supply_an_original_quote():
    question = "Which color does Inez prefer?"
    plan = QueryPlan(required_info=["Inez prefers orange."])
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return json.dumps(answer(question, "orange", "Inez prefers orange."))

    with pytest.raises(ModelOutputError, match="does not contain exact quote"):
        asyncio.run(CallableModel(chat).synthesize(question, plan, [source("I prefer blue.")]))
    assert len(calls) == 2  # Existing bounded repair only, not a new checking call.


@pytest.mark.parametrize("question,value", [
    ("How many days ago did I visit the gallery?", "15 days ago"),
    ("How many days ago had I visited the gallery when I started the course?", "5 days"),
    ("How long since my gallery visit?", "About two weeks"),
])
def test_legacy_offset_metadata_does_not_rewrite_a_published_value(question, value):
    sources = [source("I visited the gallery on 2025-05-05."),
               source("I started the course on 2025-05-10.", "2025-05-10T12:00:00+00:00")]
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        data = payload(prompt)
        assert data["answer_type"] == "offset"
        assert data["question"] == question and data["as_of"] == "2025-05-20"
        assert [s["source_events"][0]["occurred_at"] for s in data["sources"]] == [
            "2025-05-05T12:00:00+00:00", "2025-05-10T12:00:00+00:00"]
        return json.dumps(answer(question, value, sources[0].content))

    table = asyncio.run(CallableModel(chat).synthesize(
        question, QueryPlan(), sources, as_of=datetime(2025, 5, 20, tzinfo=UTC)))
    assert render_table(table, kind="offset") == value and len(calls) == 1


@pytest.mark.parametrize("question,expected", [
    ("How many books did I read in the last month?", "count"),
    ("When did I start the course?", "date"),
    ("How old was I when I started the course?", "text"),
])
def test_unrelated_question_kinds_are_unchanged(question, expected):
    assert answer_type(question) == expected


@pytest.mark.parametrize("keep_approximate", [True, False])
def test_native_repair_reassesses_necessity_without_changing_acceptance_rules(keep_approximate):
    question = "Roughly how long ago did I start using the app?"
    quote = "I just downloaded the app."
    initial = answer(question, "About three weeks ago", quote, missing="An exact start day is not stated.",
                     missing_kind="necessary_premise", coverage="partial")
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(initial)
        assert "the label itself does not establish necessity" in prompt
        assert "source-supported approximate or conditional answer" in prompt
        assert "Do not substitute a different person, event, relation" in prompt
        repaired = copy.deepcopy(initial)
        if keep_approximate:
            repaired["slots"][0].update(missing_kind="optional_detail", coverage="complete")
        else:
            repaired["slots"][0]["values"][0]["unresolved_premise"] = "The requested start event is unsupported."
        return json.dumps(repaired)

    table = asyncio.run(CallableModel(chat).synthesize(question, QueryPlan(), [source(quote)],
                                                     as_of=datetime(2025, 5, 26, tzinfo=UTC)))
    assert len(calls) == 2
    assert render_table(table) == ("About three weeks ago" if keep_approximate else "Not mentioned in the conversation")


def test_true_missing_premise_still_blocks_its_value_but_not_an_independent_answer():
    quote, question = "Parcel A arrived Tuesday; parcel B is only planned.", "Which arrived first, and when did A arrive?"
    data = answer(question, "Parcel A arrived first", quote,
                  missing="Parcel B arrival is unknown.", missing_kind="necessary_premise", coverage="partial")
    table = AnswerTable.model_validate(data)
    with pytest.raises(ValueError, match="necessary missing premise"):
        validate_table(table, [source(quote)], [question], "text")
    data["slots"][0]["values"][0]["unresolved_premise"] = "Parcel B arrival is unknown."
    data["slots"][0]["values"].append(answer(question, "Tuesday", quote)["slots"][0]["values"][0])
    table = AnswerTable.model_validate(data)
    validate_table(table, [source(quote)], [question], "text")
    assert table.slots[0].values[0].value == "Tuesday"
    assert len(table.slots[0].values) == 1 and table.slots[0].coverage == "partial"
    assert "Parcel B arrival is unknown" in render_table(table)


def test_optional_semantic_and_time_bindings_preserve_legacy_none():
    assert set(BoundInput.model_fields) == {
        "identity", "statement", "supports", "amount", "qualification", "time_ref"}
    # Backward compatibility requires the old operators to remain supported;
    # it does not prohibit additional, independently tested operators.
    assert {"none", "count", "sum", "time_difference", "time_compare", "time_since"} <= set(
        SupportedValue.model_json_schema()["properties"]["operation"]["enum"])
    assert not SupportedValue.model_fields["request_binding"].is_required()
    assert not BoundInput.model_fields["qualification"].is_required()
    assert not BoundInput.model_fields["time_ref"].is_required()

    question, quote = "Which color does Inez prefer?", "I prefer blue."
    raw = answer(question, "blue", quote)
    raw_value = raw["slots"][0]["values"][0]
    raw_value["inputs"] = [{"identity": "preference", "statement": "Inez prefers blue.",
                            "supports": copy.deepcopy(raw_value["supports"])}]
    table = AnswerTable.model_validate(raw)
    validate_table(table, [source(quote)], [question], "text")
    value = table.slots[0].values[0]
    assert value.operation == "none"
    assert value.request_binding is None and value.inputs[0].qualification is None
    assert value.inputs[0].time_ref is None
    compact = value.model_dump(exclude_none=True)
    assert "request_binding" not in compact and "qualification" not in compact["inputs"][0]
    assert "time_ref" not in compact["inputs"][0]
    assert render_table(table) == "blue"


def test_extraction_context_keeps_original_sequences_and_clocks():
    def turn(number):
        return Turn(turn_id=f"t{number}", request_id=f"r{number}", session_id="s", sequence=number,
                    events=[Event(role="user", content=f"Statement {number}.", occurred_at=datetime(2025, 1, number, tzinfo=UTC))])

    window = ExtractionWindow(session=Session(session_id="s"), new_turns=[turn(9)], context_turns=[turn(1), turn(7)])
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        data = json.JSONDecoder().raw_decode(prompt[prompt.index('{"session":'):])[0]
        assert [t["sequence"] for t in data["context_turns"]] == [1, 7]
        assert data["context_turns"][1]["events"][0]["occurred_at"].startswith("2025-01-07")
        # Only the optional sparse-context instruction was withdrawn; explicit
        # input sequence numbers and clocks must still pass through unchanged.
        assert "context_turns 最多 15 轮" in prompt
        return '{"candidates":[],"summary":"","state_patch":{}}'

    asyncio.run(CallableModel(chat).extract(window))
    assert len(calls) == 1

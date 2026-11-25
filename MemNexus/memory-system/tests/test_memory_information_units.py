"""One extraction contract carries complete units and their literal sources.

The model is frozen locally: these tests check the prompt, transport and source
integrity, not whether a live model will always obey the extraction instruction.
"""

import asyncio
import json

import pytest

from agent_memory.grounding import _source_event, source_payload
from agent_memory.models import Candidate, Event, ExtractionWindow, Hit, Scope, TurnInput
from agent_memory.providers import CallableModel, ModelOutputError
from agent_memory.store import SQLiteStore

OWNER = Scope(tenant_id="units", owner_id="reader")
SOURCE = (
    "For the Vale drill, use this sequence only for the small pump:\n"
    "1. Close the inlet valve.\n"
    "2. Wait 2–3 seconds; do not open the drain.\n"
    "3. Turn the blue handle twice."
)
STEPS = SOURCE.splitlines()[1:]


@pytest.fixture
def storage(tmp_path):
    store = SQLiteStore(tmp_path / "units.db")
    session = store.create_session(OWNER)
    yield store, session
    store.close()


def window_for(store, session, *, role="user", content=SOURCE, summary=""):
    turn = store.append_turn(OWNER, session.session_id, TurnInput(
        request_id="source", events=[Event(role=role, speaker="Nora" if role == "user" else "Assistant",
                                           content=content)],
    ))
    return ExtractionWindow(session=session.model_copy(update={"summary": summary}), new_turns=[turn],
                            context_turns=[])


def unit_response(turn, *, role):
    attribution = "Nora described" if role == "user" else "The assistant recommended"
    return {
        "candidates": [{
            "content": f"{attribution} this small-pump-only Vale drill sequence: " + " ".join(STEPS),
            "kind": "procedure", "subject": "small pump at the Vale drill", "predicate": "sequence",
            "value": "Close the inlet, wait without opening the drain, then turn the blue handle.",
            "scope_type": "session", "durable": False,
            "evidence": [{"turn_id": turn.turn_id, "event_index": 0, "quote": SOURCE}],
            "semantics": {
                "cardinality": "many", "event_key": "Vale drill",
                "members": [{"value": step, "evidence": [
                    {"turn_id": turn.turn_id, "event_index": 0, "quote": step},
                ]} for step in STEPS],
            },
        }],
        "summary": "A small-pump procedure was described.",
    }


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_complete_unit_reaches_source_reader_without_an_extra_model_call(storage, monkeypatch, role):
    monkeypatch.setenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "true")
    store, session = storage
    window = window_for(store, session, role=role)
    turn = window.new_turns[0]
    response = unit_response(turn, role=role)
    calls = []

    def chat(prompt):
        calls.append(prompt)
        return json.dumps(response, ensure_ascii=False)

    extracted = asyncio.run(CallableModel(chat).extract(window))
    assert len(calls) == 1
    assert extracted.candidates[0].content == response["candidates"][0]["content"]
    # The same schema carries both readable relations and atomic members. No
    # second restatement field or character-counting task is sent to the model.
    schema_text = calls[0].split("support:\n", 1)[1].split("\n输入数据：\n", 1)[0]
    schema = json.loads(schema_text)
    assert set(schema["$defs"]["Candidate"]["properties"]) == set(Candidate.model_fields)
    assert "start" not in schema["$defs"]["Evidence"]["properties"]
    assert "end" not in schema["$defs"]["Evidence"]["properties"]
    assert "content 用独立可理解的自然语言完整重述" in calls[0]
    assert "members 按原文逐项列出" in calls[0]
    assert "不按重要性重排或只保留首项" in calls[0]
    if role == "assistant":
        assert "not a user action" in calls[0]
        assert "session-scoped and non-durable" in calls[0]

    [saved] = store.add_candidates(OWNER, session.session_id, extracted.candidates)
    loaded = store.get_memory(OWNER, saved.memory_id)
    assert loaded.content == extracted.candidates[0].content
    assert loaded.tier == "short" and loaded.scope_type == "session" and not loaded.durable
    assert [m.value for m in loaded.semantics.members] == STEPS
    member_offsets = []
    for member in loaded.semantics.members:
        [ref] = member.evidence
        assert SOURCE[ref.start:ref.end] == ref.quote == member.value
        member_offsets.append(ref.start)
    assert member_offsets == sorted(member_offsets)

    hit = Hit(content=loaded.content, source_file="memory", chunk_id=loaded.memory_id, score=1,
              metadata={**loaded.source_metadata(),
                        "source_events": [_source_event(turn, 0, SOURCE, 0)]})
    [visible] = source_payload(hit)["source_events"]
    assert visible["text"] == SOURCE
    assert visible["role"] == role
    assert [m["value"] for m in visible["extraction_hint"]["members"]] == STEPS
    for member in visible["extraction_hint"]["members"]:
        assert all(visible["text"][left:right] == member["value"] for left, right in member["spans"])


def test_a_summary_cannot_supply_missing_original_evidence(storage):
    store, session = storage
    fabricated = "The drain must be opened."
    window = window_for(store, session, summary=fabricated)
    response = unit_response(window.new_turns[0], role="user")
    response["candidates"][0]["evidence"][0]["quote"] = fabricated
    calls = []

    def chat(prompt):
        calls.append(prompt)
        return json.dumps(response)

    with pytest.raises(ModelOutputError, match="exact substring"):
        asyncio.run(CallableModel(chat).extract(window))
    assert len(calls) == 2  # Existing one-repair allowance; no extra rescue path.
    assert store.memories(OWNER) == []


def test_new_prompt_does_not_add_a_sentence_shape_rejection(storage):
    store, session = storage
    window = window_for(store, session, content="Nora prefers blue.")
    response = {"candidates": [{
        "content": "blue", "subject": "Nora", "predicate": "preference", "value": "blue",
        "evidence": [{"turn_id": window.new_turns[0].turn_id, "event_index": 0,
                      "quote": "Nora prefers blue."}],
    }]}
    calls = []

    def chat(prompt):
        calls.append(prompt)
        return json.dumps(response)

    extracted = asyncio.run(CallableModel(chat).extract(window))
    assert len(calls) == 1
    [saved] = store.add_candidates(OWNER, session.session_id, extracted.candidates)
    assert saved.content == "blue"  # Legacy/terse valid records remain readable.

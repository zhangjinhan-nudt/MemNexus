"""Offline interface checks; no benchmark labels or semantic accuracy claims."""

from datetime import UTC, datetime

import pytest

from agent_memory.grounding import _cost, ground_evidence, source_payload
from agent_memory.models import (
    Event,
    Evidence,
    FactMember,
    FactSemantics,
    Memory,
    QueryPlan,
    Turn,
)
from agent_memory.prompt_codec import evidence_catalog
from agent_memory.reasoning import indexed_memory_hits
from agent_memory.retrieval import Retriever, _Match


def source(identifier, text, *, speaker="Iris", session_id="elsewhere"):
    return Turn(turn_id=identifier, request_id=identifier, session_id=session_id,
                sequence=1, events=[Event(role="user", speaker=speaker, content=text,
                                         occurred_at=datetime(2037, 1, 1, tzinfo=UTC))])


def evidence(turn_id, quote, event_index=0):
    return Evidence(turn_id=turn_id, event_index=event_index, quote=quote)


def memory(semantics=None, *, refs=None, **changes):
    fields = dict(memory_id="m", session_id="s", scope_id="s", subject="Iris",
                  predicate="owns", value="red and blue", content="Iris owns red and blue.",
                  evidence=refs or [evidence("a", "I own red.")],
                  semantics=semantics or FactSemantics())
    fields.update(changes)
    return Memory(**fields)


def indexed(item):
    hits = indexed_memory_hits("What does Iris own?", QueryPlan(query_entities=["Iris"]),
                               [item], [], limit=1)
    assert len(hits) == 1
    return hits[0]


def payload(hit, turns, *, current=None):
    result = ground_evidence("What does Iris own?", QueryPlan(), [hit], session_id="s",
                             current_turns=current or [],
                             resolved_turns={turn.turn_id: turn for turn in turns},
                             blocked_events=set(), max_sources=1, budget=10000)
    assert len(result.sources) == 1
    return source_payload(result.sources[0]), result


@pytest.mark.parametrize("operation", ["add", "remove", "replace_set"])
def test_same_memory_keeps_collection_semantics_across_recall_sources(operation):
    item = memory(FactSemantics(cardinality="many", operation=operation, members=[
        FactMember(value="red", identity="object:red", evidence=[evidence("a", "I own red.")])]))
    actual = indexed(item)
    main = Retriever._hit(_Match(item, score=actual.score), set())
    assert actual.metadata["semantics"] == main.metadata["semantics"]
    assert actual.metadata["evidence"] == main.metadata["evidence"]
    text = source("a", "I own red. Unrelated private detail.")
    actual_payload, _ = payload(actual, [text])
    main_payload, _ = payload(main, [text])
    assert actual_payload["source_events"] == main_payload["source_events"]
    hint = actual_payload["source_events"][0]["extraction_hint"]
    assert hint["operation"] == operation
    assert hint["members"] == [{"value": "red", "identity": "object:red", "spans": [[0, 10]]}]
    assert "private" not in str(actual_payload)


def test_serialized_semantics_is_not_a_mutable_alias_of_memory():
    item = memory(FactSemantics(cardinality="many", operation="add", members=[
        FactMember(value="red", identity="red-id", evidence=[evidence("a", "I own red.")])]))
    actual = indexed(item)
    actual.metadata["semantics"]["members"][0]["identity"] = "modified"
    assert item.semantics.members[0].identity == "red-id"


def test_only_visible_member_quote_gets_identity_not_whole_collection_value():
    item = memory(FactSemantics(cardinality="many", operation="add", members=[
        FactMember(value="red", identity="red-id", evidence=[evidence("a", "I own red.")]),
        FactMember(value="blue", identity="blue-id", evidence=[evidence("b", "I own blue.")]),
    ]))
    actual = indexed(item)
    turns = [source("a", "I own red."), source("b", "I own blue.")]
    current, grounded = payload(actual, turns)
    hint = current["source_events"][0]["extraction_hint"]
    assert [member["identity"] for member in hint["members"]] == ["red-id"]
    assert "value" not in hint
    assert "blue-id" not in str(current["source_events"])
    catalog = evidence_catalog(grounded.sources)["sources"]
    assert catalog[0]["source_id"] == 1
    assert catalog[0]["source_events"] == current["source_events"]
    # The compressed claim is still explicitly a fallible hint. The metadata
    # repair alone is not a proof that the model ignores its unsupported part.
    assert catalog[0]["claim_hint"] == item.content
    # The existing downstream consumer explains why carrying semantics matters:
    # absent it, the same visible red quote is labelled with the whole value.
    old = actual.model_copy(deep=True)
    old.metadata.pop("semantics", None)
    legacy, _ = payload(old, turns)
    assert legacy["source_events"][0]["extraction_hint"]["value"] == "red and blue"


@pytest.mark.parametrize("bad_ref", [evidence("other", "I own red."),
                                     evidence("a", "I own red.", event_index=1),
                                     evidence("a", "I own blue.")])
def test_matching_words_cannot_bind_a_member_from_wrong_source_identity(bad_ref):
    item = memory(FactSemantics(cardinality="many", operation="add", members=[
        FactMember(value="blue", identity="unseen-blue", evidence=[bad_ref])]))
    result, _ = payload(indexed(item), [source("a", "I own red.")])
    assert result["source_events"][0]["text"] == "I own red."
    assert "extraction_hint" not in result["source_events"][0]


def test_unresolvable_parent_quote_cannot_expose_collection_identity():
    item = memory(FactSemantics(cardinality="many", operation="add", members=[
        FactMember(value="red", identity="red-id", evidence=[evidence("a", "I own red.")])]),
        refs=[evidence("a", "I own blue.")])
    result, grounded = payload(indexed(item), [source("a", "I own red.")])
    assert result["source_events"] == []
    assert result["provenance"] == "unresolved"
    assert grounded.trace.rejected_source_refs == 1


@pytest.mark.parametrize("assertion", ["stated", "planned"])
def test_event_identity_is_preserved_as_hint_without_state_promotion(assertion):
    item = memory(FactSemantics(cardinality="one", event_key="season-series",
                                event_identity="workshop-instance-a"), assertion=assertion)
    actual = indexed(item)
    result, _ = payload(actual, [source("a", "I own red.")])
    assert result["attributes"]["assertion"] == assertion
    hint = result["source_events"][0]["extraction_hint"]
    assert hint["event_identity"] == "workshop-instance-a"
    assert hint["event_key"] == "season-series"
    assert "count" not in hint


def test_attribution_conflict_suppresses_hint_but_preserves_original_quote():
    item = memory(FactSemantics(cardinality="many", operation="add", members=[
        FactMember(value="red", identity="red-id", evidence=[evidence("a", "I own red.")])]))
    text = source("a", "I own red.", speaker="Nora", session_id="s")
    iris = source("b", "Iris is here.", session_id="s")
    result, grounded = payload(indexed(item), [text, iris], current=[text, iris])
    assert grounded.trace.attribution_conflicts == 1
    assert result["source_events"][0]["text"] == "I own red."
    assert "extraction_hint" not in result["source_events"][0]


def test_third_party_report_keeps_explicit_subject_and_member_identity():
    quote = "Iris owns red."
    item = memory(FactSemantics(cardinality="many", operation="add", members=[
        FactMember(value="red", identity="red-id", evidence=[evidence("a", quote)])]),
        refs=[evidence("a", quote)])
    text = source("a", quote, speaker="Nora", session_id="s")
    iris = source("b", "Iris is here.", session_id="s")
    result, grounded = payload(indexed(item), [text, iris], current=[text, iris])
    assert grounded.trace.attribution_conflicts == 0
    hint = result["source_events"][0]["extraction_hint"]
    assert hint["subject"] == "Iris"
    assert hint["members"][0]["identity"] == "red-id"


def test_default_unknown_semantics_does_not_change_plain_scalar_payload():
    actual = indexed(memory())
    turns = [source("a", "I own red.")]
    current, _ = payload(actual, turns)
    old = actual.model_copy(deep=True)
    old.metadata.pop("semantics", None)
    legacy, _ = payload(old, turns)
    assert current == legacy


def test_ranking_limit_prioritizes_subject_without_hiding_other_relevant_facts():
    records = [memory(FactSemantics(event_identity="a"), memory_id="a"),
               memory(FactSemantics(event_identity="b"), memory_id="b"),
               memory(memory_id="other", subject="Nora")]
    plan = QueryPlan(query_entities=["Iris"])
    hits = indexed_memory_hits("What does Iris own?", plan, records, [], limit=1)
    assert [hit.metadata["memory_id"] for hit in hits] == ["a"]
    next_hits = indexed_memory_hits("What does Iris own?", plan, records, hits, limit=10)
    assert [hit.metadata["memory_id"] for hit in next_hits] == ["b", "other"]
    assert hits[0].metadata["semantics"]["event_identity"] == "a"


def test_added_identity_is_charged_to_existing_budget_and_can_change_selection():
    item = memory(FactSemantics(event_key="series", event_identity="specific-instance"))
    text = source("a", "I own red.")
    actual = indexed(item)
    _, hydrated = payload(actual, [text])
    old = hydrated.sources[0].model_copy(deep=True)
    old.metadata.pop("semantics", None)
    old_cost = _cost(old)
    assert _cost(hydrated.sources[0]) > old_cost
    common = dict(session_id="s", current_turns=[], resolved_turns={"a": text},
                  blocked_events=set(), max_sources=1, budget=old_cost)
    legacy = actual.model_copy(deep=True)
    legacy.metadata.pop("semantics", None)
    before = ground_evidence("What does Iris own?", QueryPlan(), [legacy], **common)
    after = ground_evidence("What does Iris own?", QueryPlan(), [actual], **common)
    assert len(before.sources) == 1
    assert after.sources == []
    assert after.trace.context_tokens_estimate <= old_cost

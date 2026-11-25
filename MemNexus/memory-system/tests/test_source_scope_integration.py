"""Source-scope integration must preserve valid QA and serialization fences."""
import json

import pytest
from test_answer_synthesis import table
from test_reading_units import make_turn

from agent_memory.grounding import _cost, _source_event, ground_evidence, source_payload
from agent_memory.models import AnswerResponse, Hit, QueryPlan
from agent_memory.synthesis import _support_is_assertion, validate_table


def source(text, *, quote=None, role="assistant", session="s"):
    turn = make_turn("scope", text, role=role, session=session)
    quote = text if quote is None else quote
    start = text.index(quote)
    event = _source_event(turn, 0, quote, start, assistant_is_primary=True)
    hit = Hit(content=quote, score=.8, source_file="scope", chunk_id="scope", metadata={
        "evidence": [{"turn_id": turn.turn_id, "event_index": 0, "quote": quote,
                      "start": start, "end": start + len(quote)}],
        "source_events": [event], "source_support_status": "authorized_session_quote",
    })
    return turn, hit


@pytest.mark.parametrize(("text", "quote"), [
    ("The service supports Linux and does not support Windows.", "supports Linux"),
    ("I bought a coat because it was not expensive.", "I bought a coat"),
    ("I own two cats and no dogs.", "two cats"),
])
def test_unrelated_qualification_cannot_veto_an_independent_supported_detail(text, quote):
    _, hit = source(text)
    assert _support_is_assertion(hit.metadata["source_events"][0], quote)
    validate_table(table(quote, quote), [hit], ["colors"], "text")


@pytest.mark.parametrize("tail", ["I do not cycle.", "Where does Alex live?"])
def test_format_normalization_does_not_merge_line_scope(tail):
    _, hit = source("I enjoy chess\n" + tail)
    event = hit.metadata["source_events"][0]
    assert _support_is_assertion(event, "I enjoy chess")
    assert _support_is_assertion(event, "i enjoy chess")


@pytest.mark.parametrize(("text", "quote"), [
    ("Use **Straße** for the field.", "use STRASSE for the field."),
    ("Ｔｈｅ device is red.", "The device is red."),
    ("Caroline: A past note. Now I enjoy chess.", "Caroline: Now I enjoy chess."),
    ("Use **the device** carefully.", "Use the device carefully."),
])
def test_preexisting_harmless_quote_variants_remain_accepted(text, quote):
    _, hit = source(text)
    assert _support_is_assertion(hit.metadata["source_events"][0], quote)


@pytest.mark.parametrize(("text", "quote"), [
    ("I live in Paris. Does Emily live in Denver?", "Denver"),
    ("I enjoy chess. Do you cycle?", "Do you cycle?"),
    ("I bought a coat. If it rains, I will wear it.", "I will wear it"),
])
def test_question_or_cropped_condition_cannot_borrow_another_sentence(text, quote):
    _, hit = source(text)
    with pytest.raises(ValueError):
        validate_table(table(quote, quote), [hit], ["colors"], "text")


def test_historical_assistant_reply_before_followup_remains_usable():
    quote = "I recommend the coast road"
    _, hit = source(quote + ", can I help with anything else?")
    validate_table(table("coast road", quote), [hit], ["colors"], "text")


def test_question_shaped_list_label_is_answerable_as_a_label():
    label = "What is rust?"
    _, hit = source("Suggested topics:\n- " + label)
    query = "Which topic did the assistant list?"
    validate_table(table(label, label, target=query), [hit], [query], "text")


def test_question_shaped_label_does_not_establish_its_proposed_factual_answer():
    _, hit = source("Suggested topics:\n- Does Emily live in Denver?")
    query = "Where does Emily live?"
    with pytest.raises(ValueError):
        validate_table(table("Denver", "Denver", target=query), [hit], [query], "text")


def test_internal_scope_does_not_expand_provider_context_or_cost():
    _, hit = source("My private code is SECRET near Paris.", quote="Paris")
    assert "SECRET" not in json.dumps(source_payload(hit))
    stripped = hit.model_copy(deep=True)
    for record in stripped.metadata["source_events"]:
        for key in list(record):
            if key.startswith("assertion_"):
                del record[key]
    assert _cost(stripped) == _cost(hit)


def test_internal_scope_does_not_expose_ungranted_text_in_answer_response():
    _, hit = source("My private code is SECRET near Paris.", quote="Paris")
    response = AnswerResponse(generated_text="Paris", citations=[1], sources=[hit],
                              retrieval_count=1, elapsed_ms=0)
    assert "SECRET" not in response.model_dump_json()


def test_scope_metadata_never_duplicates_the_full_original_text():
    text = "My private code is SECRET near Paris."
    _, hit = source(text, quote="Paris")
    record = hit.metadata["source_events"][0]
    assert record["text"] == "Paris"
    # Check nested metadata as well as the provider's fixed-field projection.
    assert "SECRET" not in json.dumps(hit.model_dump(mode="json"))


def test_hydration_rebuilds_old_negative_grade_from_original_source():
    turn, hit = source("I recommend the coast road, can I help with anything else?")
    hit.metadata["source_events"][0].update(evidence_grade="none", context_only=True, is_question=True)
    result = ground_evidence("What road was recommended?", QueryPlan(route="none"), [hit],
        session_id="s", current_turns=[turn], resolved_turns={turn.turn_id: turn},
        blocked_events=set(), budget=3000)
    assert any(_support_is_assertion(event, "I recommend the coast road")
               for source in result.sources for event in source.metadata["source_events"])


def test_hydration_does_not_reuse_old_positive_grade_for_a_cropped_question():
    turn, hit = source("Does Emily live in Denver?", quote="Denver", session="old")
    hit.metadata["source_events"][0].update(evidence_grade="primary", context_only=False, is_question=False)
    result = ground_evidence("Where does Emily live?", QueryPlan(route="none"), [hit],
        session_id="new", current_turns=[], resolved_turns={turn.turn_id: turn},
        blocked_events={(turn.turn_id, 0)}, budget=3000,
        authorized_quotes={(turn.turn_id, 0): hit.metadata["evidence"]})
    assert result.sources
    assert all(not _support_is_assertion(event, "Denver")
               for source in result.sources for event in source.metadata["source_events"])

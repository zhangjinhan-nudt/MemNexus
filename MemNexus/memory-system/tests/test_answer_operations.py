"""The single answer table audits premises without a second publication executor."""
import asyncio
import json

import pytest
from pydantic import ValidationError

from agent_memory.models import Hit, QueryPlan
from agent_memory.providers import CallableModel
from agent_memory.synthesis import AnswerSlot, AnswerTable, answer_type, render_table, validate_table


def source(text, role="user", grade=None):
    event = {"text": text, "role": role, "speaker": "A",
             "occurred_at": "2024-05-20T12:00:00+00:00", "context_only": False,
             "time_derivations": []}
    if grade:
        event["evidence_grade"] = grade
    return Hit(content=text, score=1, source_file="source", chunk_id=text,
               metadata={"source_events": [event]})


def value(text, source_id=1, quote="I prefer quiet spaces.", basis="memory"):
    return {"value": text, "basis": basis, "supports": [{"source_id": source_id, "quote": quote}]}


def table(question, values, *, derivation="The cited assertions supply the requested premises.", **extra):
    return AnswerTable.model_validate({"slots": [{"target": question, "derivation": derivation,
                                                  "values": values, **extra}]})


@pytest.mark.parametrize("derivation", [None, ""])
def test_derivation_is_required_before_values(derivation):
    slot = {"target": "q", "values": [], "missing": "No evidence."}
    if derivation is not None:
        slot["derivation"] = derivation
    with pytest.raises(ValidationError, match="derivation"):
        AnswerSlot.model_validate(slot)


def test_derivation_has_a_bounded_modeling_space():
    assert AnswerSlot(target="q", derivation="x" * 1000).derivation == "x" * 1000
    with pytest.raises(ValidationError):
        AnswerSlot(target="q", derivation="x" * 1001)


def test_single_answer_table_keeps_direct_source_bound_inputs():
    schema = AnswerTable.model_json_schema()
    assert list(schema["properties"]) == ["slots"]
    assert "derivation" in schema["$defs"]["AnswerSlot"]["required"]
    assert "supports" in schema["$defs"]["SupportedValue"]["properties"]
    assert {"statement", "supports"} <= schema["$defs"]["BoundInput"]["properties"].keys()
    with pytest.raises(ValidationError, match="Extra inputs"):
        AnswerTable.model_validate({"notes": [], "slots": [{"target": "q", "derivation": "Bind facts."}]})


@pytest.mark.parametrize("field", ["calculation", "computation", "time_expression"])
def test_retired_publication_dsl_is_not_an_answer_field(field):
    item = value("A conclusion")
    item[field] = {} if field != "time_expression" else "yesterday"
    with pytest.raises(ValidationError, match="Extra inputs"):
        table("q", [item])


def test_derivation_is_auditable_but_never_rendered():
    summary = "Compare the two quoted quantities; keep the stated units."
    result = table("q", [value("27 years")], derivation=summary)
    assert result.model_dump()["slots"][0]["derivation"] == summary
    assert render_table(result) == "27 years"
    assert summary not in render_table(result)


@pytest.mark.parametrize("answer,quote,kind", [
    ("999", "The two amounts are 2 and 3.", "count"),
    ("59 days ago", "I moved on March 15, 2024.", "offset"),
    ("last month", "I bought a bike last month.", "date"),
])
def test_validation_preserves_conclusions_instead_of_claiming_to_prove_them(answer, quote, kind):
    # These inputs deliberately separate provenance from semantic correctness.
    result = table("q", [value(answer, quote=quote, basis="inference")])
    before = result.model_dump()
    validate_table(result, [source(quote)], ["q"], kind, as_of="2024-05-20",
                   derivations=[{"kind": "offset_order", "first": "another answer"}])
    assert result.model_dump() == before
    assert render_table(result, [source(quote)], kind=kind) == answer


def test_new_recommendation_is_supported_by_personal_constraint_not_historical_advice():
    quote = "I prefer quiet spaces and minimal furniture."
    result = table("Suggest a room layout.", [value("Keep a quiet reading corner.", quote=quote,
                                                   basis="recommendation")])
    validate_table(result, [source(quote)], [result.slots[0].target], "text")
    with pytest.raises(ValueError, match="user assertion"):
        validate_table(result, [source(quote, "assistant")], [result.slots[0].target], "text")


def recommendation_table():
    return table("Suggest a quiet room layout.", [
        value("A quiet reading corner", quote="I prefer quiet spaces.", basis="recommendation"),
        value("Add a low bookshelf", source_id=2, quote="A low bookshelf can divide a room.", basis="recommendation"),
    ])


def test_recommendations_can_share_user_context_inside_one_slot():
    result = recommendation_table()
    validate_table(result, [source("I prefer quiet spaces."),
                           source("A low bookshelf can divide a room.", "assistant")],
                   [result.slots[0].target], "text")
    assert render_table(result) == "A quiet reading corner, Add a low bookshelf"


@pytest.mark.parametrize("bad", ["missing_quote", "unknown_source"])
def test_shared_context_does_not_relax_each_values_provenance(bad):
    result = recommendation_table()
    support = result.slots[0].values[1].supports[0]
    if bad == "missing_quote":
        support.quote = "Invented evidence"
    else:
        support.source_id = 3
    with pytest.raises(ValueError, match="exact quote|Unknown source"):
        validate_table(result, [source("I prefer quiet spaces."),
                               source("A low bookshelf can divide a room.", "assistant")],
                       [result.slots[0].target], "text")


@pytest.mark.parametrize("role,grade", [("assistant", "secondary"), ("user", "none")])
def test_assistant_only_or_user_question_context_cannot_license_recommendation(role, grade):
    result = recommendation_table()
    with pytest.raises(ValueError, match="user assertion|question alone"):
        validate_table(result, [source("I prefer quiet spaces.", role, grade),
                               source("A low bookshelf can divide a room.", "assistant")],
                       [result.slots[0].target], "text")


def test_recommendations_cannot_borrow_user_context_from_another_slot():
    result = recommendation_table()
    first = result.slots[0].model_copy(deep=True)
    first.target = "Preference"
    first.values = first.values[:1]
    first.values[0].basis = "memory"
    result.slots[0].values = result.slots[0].values[1:]
    result.slots.insert(0, first)
    with pytest.raises(ValueError, match="same target"):
        validate_table(result, [source("I prefer quiet spaces."),
                               source("A low bookshelf can divide a room.", "assistant")],
                       [slot.target for slot in result.slots], "text")


def test_explicit_partial_coverage_is_not_gated_by_question_keyword():
    result = table("What was the manager's team size?", [value("4 engineers", quote="4 engineers")],
                   coverage="partial", missing="Only engineer role is supported; manager role is unknown.")
    assert "manager role is unknown" in render_table(result)


def test_age_is_a_quantity_even_when_question_contains_when():
    assert answer_type("How old was I when I moved?") == "text"


def test_time_references_and_derivation_share_one_provider_request_without_publication_rewrite():
    question = "When did I meet Lee?"
    prompts = []
    def chat(prompt):
        prompts.append(prompt)
        return json.dumps({"slots": [{"target": question,
            "derivation": "The user explicitly dated the requested meeting.",
            "values": [{"value": "yesterday", "supports": [{"source_id": 1, "quote": "I met Lee yesterday."}]}]}]})
    result = asyncio.run(CallableModel(chat).synthesize(question, QueryPlan(), [source("I met Lee yesterday.")]))
    assert len(prompts) == 1
    assert '"time_references"' in prompts[0] and "2024-05-19" in prompts[0]
    assert '"derivation"' in prompts[0] and '"notes"' not in prompts[0]
    assert "qualification" in prompts[0]
    assert '"computation"' not in prompts[0] and '"time_expression"' not in prompts[0]
    assert result.slots[0].values[0].supports[0].quote == "I met Lee yesterday."
    assert render_table(result) == "yesterday"


@pytest.mark.parametrize("answer,kind", [("Eleven books", "count"), ("1.5 years ago", "offset"),
                                          ("a fortnight", "duration"), ("the festival opening day", "date")])
def test_source_bound_natural_values_do_not_require_a_small_shape_grammar(answer, kind):
    result = table("q", [value(answer, quote=answer)])
    validate_table(result, [source(answer)], ["q"], kind)
    assert render_table(result) == answer

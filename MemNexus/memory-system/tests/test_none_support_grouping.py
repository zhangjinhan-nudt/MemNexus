"""Non-executable input grouping preserves the existing flat-support contract."""

import copy

import pytest

from agent_memory.models import Hit
from agent_memory.synthesis import AnswerTable, render_table, validate_table


def source(text, role="user", grade="primary"):
    return Hit(content=text, score=1, source_file="fixture", chunk_id=text, metadata={
        "source_events": [{"text": text, "role": role, "speaker": role,
                           "occurred_at": "2034-06-18T12:00:00+00:00",
                           "context_only": grade == "none", "evidence_grade": grade}]})


def table(references, grouped, *, value="About five years", basis="inference", operation="none"):
    item = {"value": value, "basis": basis, "operation": operation}
    if grouped:
        item["inputs"] = [{"identity": f"premise-{i}", "statement": "Selected contextual premise",
                           "amount": "1", "supports": [reference]}
                          for i, reference in enumerate(references)]
    else:
        item["supports"] = references
    return AnswerTable.model_validate({"slots": [{"target": "Original request",
        "derivation": "The model selects a contextual inference; provenance is not entailment.",
        "values": [item]}]})


def check(result, sources):
    validate_table(result, sources, ["Original request"], "text")
    return render_table(result)


QUESTION = "Could my cousin be nineteen?"
ASSERTION = "I am twenty-four."
REFERENCES = [{"source_id": 1, "quote": QUESTION}, {"source_id": 2, "quote": ASSERTION}]


@pytest.mark.parametrize("basis", ["memory", "inference", "recommendation"])
def test_none_grouped_and_flat_quotes_have_the_same_acceptance(basis):
    sources = [source(QUESTION, grade="none"), source(ASSERTION)]
    flat = table(REFERENCES, False, basis=basis)
    grouped = table(REFERENCES, True, basis=basis)
    assert check(flat, sources) == check(grouped, sources) == "About five years"
    assert flat.slots[0].values[0].supports == grouped.slots[0].values[0].supports
    # Acceptance only establishes the unchanged provenance boundary. It does
    # not establish that the cousin is nineteen or that this inference is true.
    assert sources[0].metadata["source_events"][0]["evidence_grade"] == "none"


@pytest.mark.parametrize("grouped", [False, True])
def test_question_only_still_cannot_establish_a_fact(grouped):
    with pytest.raises(ValueError, match="question alone"):
        check(table(REFERENCES[:1], grouped), [source(QUESTION, grade="none")])


@pytest.mark.parametrize("operation", ["count", "sum"])
def test_executable_operands_each_still_require_assertion(operation):
    sources = [source(QUESTION, grade="none"), source(ASSERTION)]
    with pytest.raises(ValueError, match="Each bound input needs an assertion"):
        check(table(REFERENCES, True, value="", operation=operation), sources)


@pytest.mark.parametrize("grouped", [False, True])
def test_advice_still_requires_a_user_assertion(grouped):
    sources = [source(QUESTION, grade="none"), source(ASSERTION, role="assistant", grade="secondary")]
    with pytest.raises(ValueError, match="Personalized recommendations require a quoted user assertion"):
        check(table(REFERENCES, grouped, basis="recommendation"), sources)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("bad_ref,error", [
    ({"source_id": 1, "quote": "An invented statement."}, "does not contain exact quote"),
    ({"source_id": 3, "quote": QUESTION}, "Unknown source number"),
])
def test_all_citations_still_validate_before_grouping_is_considered(grouped, bad_ref, error):
    sources = [source(QUESTION, grade="none"), source(ASSERTION)]
    with pytest.raises(ValueError, match=error):
        check(table([bad_ref, REFERENCES[1]], grouped), sources)


@pytest.mark.parametrize("grouped", [False, True])
def test_question_only_identifier_remains_unsupported(grouped):
    question = "Is @new_alias your profile?"
    sources = [source(question, grade="none"), source("My profile is @old_alias.")]
    refs = [{"source_id": 1, "quote": question}, {"source_id": 2, "quote": "My profile is @old_alias."}]
    with pytest.raises(ValueError, match="Recalled @ identifiers must match"):
        check(table(refs, grouped, value="@new_alias"), sources)


@pytest.mark.parametrize("grouped", [False, True])
def test_explicit_unresolved_premise_still_removes_only_its_value(grouped):
    sources = [source(QUESTION, grade="none"), source(ASSERTION)]
    result = table(REFERENCES, grouped)
    independent = copy.deepcopy(result.slots[0].values[0])
    independent.value = "twenty-four"
    independent.inputs = []
    independent.supports = table(REFERENCES[1:], False).slots[0].values[0].supports
    result.slots[0].values[0].unresolved_premise = "The cousin's actual age is unknown."
    result.slots[0].values.append(independent)
    rendered = check(result, sources)
    assert "About five years" not in rendered and "twenty-four" in rendered
    assert result.slots[0].coverage == "partial"
    assert len(result.slots[0].values) == 1


@pytest.mark.parametrize("grouped", [False, True])
def test_declared_necessary_gap_still_requires_dependency_resolution(grouped):
    sources = [source(QUESTION, grade="none"), source(ASSERTION)]
    result = table(REFERENCES, grouped)
    result.slots[0].missing = "The cousin's actual age is unknown."
    result.slots[0].missing_kind = "necessary_premise"
    result.slots[0].coverage = "partial"
    with pytest.raises(ValueError, match="A necessary missing premise must be linked"):
        check(result, sources)

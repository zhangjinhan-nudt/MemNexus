"""Source-local speech-act scope, independently of topic, benchmark or model."""
import pytest

from agent_memory.composition import collection_sources
from agent_memory.evidence_policy import (
    assertion_span_supported,
    assertion_supported,
    listed_label_supported,
    source_assertion_scopes,
    source_assertion_spans,
)
from agent_memory.models import (
    Candidate,
    CollectionQuery,
    Event,
    Evidence,
    ExtractionResult,
    ExtractionWindow,
    Scope,
    Session,
    Turn,
    TurnInput,
)
from agent_memory.ports import NotFoundError
from agent_memory.providers import ModelOutputError, _validate_evidence
from agent_memory.store import SQLiteStore

CASES = [
    ("I'm visiting my sister in Denver, and I was wondering if you knew any attractions?",
     "I'm visiting my sister in Denver", True),
    ("I bought a bicycle, can you suggest a helmet?", "I bought a bicycle", True),
    ("I live in Paris. Does Emily live in Denver?", "Does Emily live in Denver?", False),
    ("I live in Paris. Does Emily live in Denver?", "Denver", False),
    ("Is it true that Emily lives in Denver?", "Emily lives in Denver", False),
    ("If Emily lives in Denver, which airport should I use?", "Emily lives in Denver", False),
    ("I don't think Emily lives in Denver, could you check?", "Emily lives in Denver", False),
    ("Emily does not live in Denver.", "live in Denver", False),
    ("Emily does not live in Denver.", "does not live in Denver", True),
    ("Emily does not live in Denver.", "Emily does not live in Denver.", True),
    ("If the road closes, we will take a train.", "we will take a train", False),
    ("If the road closes, we will take a train.", "If the road closes, we will take a train.", True),
    ("I don't drive, but Emily owns a bicycle.", "Emily owns a bicycle", True),
    ("When I arrived, I called Alex.", "I called Alex", True),
    ("When I arrived, who called Alex?", "Alex", False),
    ("When I arrived, I called Alex?", "I called Alex", False),
    ("Where Rachel works, she leads a small team.", "she leads a small team", True),
    ("Dr. Chen lives near St. James Park.", "St. James Park", True),
    ("The rate is 3.5 percent.", "3.5 percent", True),
    ("I like not only chess but also music.", "chess", True),
    ("I did not order pasta and enjoyed pizza.", "I did not order pasta and enjoyed pizza.", True),
    ("I did not buy flowers and potted plants.", "potted plants", False),
    ("I did not order pasta and pizza.", "pizza", False),
    ("Emily lives in Denver without a car.", "Emily lives in Denver", True),
    ("The service supports Linux and does not support Windows.", "supports Linux", True),
    ("The service supports Linux and does not support Windows.", "Windows", False),
    ("The service supports Linux and does not support Windows.", "does not support Windows", True),
    ("I bought a coat because it was not expensive.", "I bought a coat", True),
    ("I own two cats and no dogs.", "two cats", True),
    ("I will drive if the road opens.", "I will drive", False),
    ("I live in Paris. Does Emily live in Denver?", "I live in Paris. Does Emily live in Denver?", True),
]


@pytest.mark.parametrize('text,quote,expected', CASES)
def test_scope_is_preserved_at_the_actual_literal_location(text, quote, expected):
    assert assertion_supported(text, quote) is expected
    start = text.index(quote)
    assert assertion_supported(text, quote, start=start, end=start + len(quote)) is expected


def test_repeated_quote_does_not_guess_a_favorable_source_context():
    text = "I do not live in Denver. Emily lives in Denver."
    assert not assertion_supported(text, 'Denver')
    start = text.rindex('Denver')
    assert assertion_supported(text, 'Denver', start=start, end=start + len('Denver'))
    assert not assertion_supported(text, 'Denver', start=0, end=6)
    assert not assertion_supported(text, 'Denver', start=start)


def test_named_label_exception_respects_exact_source_positions():
    text, quote = 'Someone asked What is rust?\n- What is rust?', 'What is rust?'
    assert not listed_label_supported(text, quote, 'topics', quote)
    last = text.rindex(quote)
    assert listed_label_supported(text, quote, 'topics', quote, start=last, end=last + len(quote))
    first = text.index(quote)
    assert not listed_label_supported(text, quote, 'topics', quote, start=first, end=first + len(quote))


def test_spans_preserve_original_coordinates_and_keep_qualifiers():
    text = "  I don't drive, but Emily owns a bicycle. Can I borrow it?"
    spans = source_assertion_spans(text)
    assert [text[a:b].strip() for a, b in spans] == ["I don't drive", "Emily owns a bicycle."]


@pytest.mark.parametrize('text,quote,expected', CASES)
def test_numeric_provenance_keeps_scope_without_any_neighbor_text(text, quote, expected):
    start = text.index(quote)
    translated = [
        {'start': scope['start'] - start, 'end': scope['end'] - start,
         'operators': [{**operator, 'start': operator['start'] - start, 'end': operator['end'] - start}
                       for operator in scope['operators']]}
        for scope in source_assertion_scopes(text)
        if scope['start'] < start + len(quote) and scope['end'] > start
    ]
    assert assertion_span_supported(translated, 0, len(quote)) is expected
    assert all(set(scope) == {'start', 'end', 'operators'} for scope in translated)


def fixture(text, quote, *, role='user'):
    turn = Turn(turn_id='t', request_id='r', session_id='s', sequence=1,
                events=[Event(role=role, content=text)])
    candidate = Candidate(content='A source-backed fixture.', subject='person', predicate='detail',
                          value=quote, evidence=[Evidence(turn_id='t', event_index=0, quote=quote)])
    window = ExtractionWindow(session=Session(session_id='s'), new_turns=[turn], context_turns=[])
    return window, candidate


@pytest.mark.parametrize('text,quote,expected', CASES)
def test_write_validator_uses_the_same_source_scope(text, quote, expected, monkeypatch):
    monkeypatch.setenv('MEMORY_ALLOW_ASSISTANT_EVIDENCE', 'false')
    window, candidate = fixture(text, quote)
    result = ExtractionResult(candidates=[candidate])
    if expected:
        _validate_evidence(window, result)
    else:
        with pytest.raises(ModelOutputError):
            _validate_evidence(window, result)


@pytest.mark.parametrize('text,quote,expected', CASES)
def test_persistent_write_uses_source_scope_without_changing_owner_fence(text, quote, expected,
                                                                      tmp_path, monkeypatch):
    monkeypatch.setenv('MEMORY_ALLOW_ASSISTANT_EVIDENCE', 'false')
    store = SQLiteStore(tmp_path / 'memory.db')
    scope = Scope(tenant_id='t', owner_id='owner')
    session = store.create_session(scope)
    turn = store.append_turn(scope, session.session_id, TurnInput(request_id='r', events=[Event(role='user', content=text)]))
    _, candidate = fixture(text, quote)
    candidate.evidence[0].turn_id = turn.turn_id
    try:
        if expected:
            memories = store.add_candidates(scope, session.session_id, [candidate])
            assert len(memories) == 1
            query = CollectionQuery(subjects=['person'], predicate='detail')
            assert collection_sources(query, memories, {turn.turn_id: turn})
        else:
            with pytest.raises(ValueError):
                store.add_candidates(scope, session.session_id, [candidate])
        with pytest.raises(NotFoundError):
            store.add_candidates(Scope(tenant_id='t', owner_id='stranger'), session.session_id, [candidate])
    finally:
        store.close()


def test_assistant_history_permission_does_not_become_user_fact(monkeypatch):
    monkeypatch.setenv('MEMORY_ALLOW_ASSISTANT_EVIDENCE', 'false')
    text = 'I recommended the coast road, can I help with anything else?'
    window, candidate = fixture(text, 'I recommended the coast road', role='assistant')
    assert assertion_supported(text, candidate.evidence[0].quote)
    with pytest.raises(ModelOutputError, match='Assistant'):
        _validate_evidence(window, ExtractionResult(candidates=[candidate]))
    candidate.assertion = 'planned'
    _validate_evidence(window, ExtractionResult(candidates=[candidate]))


def test_question_shaped_named_member_survives_storage_and_collection(tmp_path, monkeypatch):
    monkeypatch.setenv('MEMORY_ALLOW_ASSISTANT_EVIDENCE', 'false')
    text, label = 'Suggested topics:\n- What is rust?', 'What is rust?'
    store = SQLiteStore(tmp_path / 'memory.db')
    scope = Scope(tenant_id='t', owner_id='owner')
    session = store.create_session(scope)
    turn = store.append_turn(scope, session.session_id, TurnInput(request_id='r', events=[Event(role='user', content=text)]))
    candidate = Candidate.model_validate({'content': 'The selected outline.', 'subject': 'person',
        'predicate': 'topics', 'value': 'outline', 'evidence': [{'turn_id': turn.turn_id, 'event_index': 0, 'quote': text}],
        'semantics': {'cardinality': 'many', 'members': [{'value': label, 'evidence': [
            {'turn_id': turn.turn_id, 'event_index': 0, 'quote': label}]}]}})
    try:
        memories = store.add_candidates(scope, session.session_id, [candidate])
        assert collection_sources(CollectionQuery(subjects=['person'], predicate='topics'), memories, {turn.turn_id: turn})
    finally:
        store.close()

def test_source_expansion_closes_quoted_sentence_before_other_topical_text():
    from agent_memory.models import Event, Hit, QueryPlan, Turn
    from agent_memory.grounding import _source_event, close_evidence_bundles

    premise = 'Which helmet fits my red bicycle?'
    original = premise + '\n\n' + ('I am looking at helmets and bicycle accessories. ' * 30)
    turn = Turn(turn_id='t', request_id='r', session_id='s', sequence=1,
                events=[Event(role='user', speaker='A', content=original)])
    quote = 'my red bicycle'
    hit = Hit(content=quote, score=1, source_file='s', chunk_id='c',
              metadata={'source_events': [_source_event(turn, 0, quote, original.index(quote))]})
    result = close_evidence_bundles([hit], [turn], session_id='s', blocked_events=set(),
        budget=2000, query='What color is my bicycle?', plan=QueryPlan(), include_neighbors=False)
    events = result[0].metadata['source_events']
    assert any(event['text'].strip() == premise for event in events)
    # Seeing the full question is not automatically permission to assert it.
    assert events[0]['role'] == 'user'
    assert events[0]['text'] in original


def test_sentence_closure_preserves_an_existing_coherent_short_utterance():
    from agent_memory.models import Event, Hit, QueryPlan, Turn
    from agent_memory.grounding import _source_event, close_evidence_bundles

    original = 'I attended a painting class. I also visited a ceramics exhibition.'
    turn = Turn(turn_id='t', request_id='r', session_id='s', sequence=1,
                events=[Event(role='user', speaker='A', content=original)])
    quote = 'a painting class'
    hit = Hit(content=quote, score=1, source_file='s', chunk_id='c',
              metadata={'source_events': [_source_event(turn, 0, quote, original.index(quote))]})
    result = close_evidence_bundles([hit], [turn], session_id='s', blocked_events=set(),
        budget=2000, query='Which art activities did I attend?', plan=QueryPlan(), include_neighbors=False)
    assert result[0].metadata['source_events'][0]['text'] == original

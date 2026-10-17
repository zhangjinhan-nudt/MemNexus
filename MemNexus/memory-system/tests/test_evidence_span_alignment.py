"""Literal source anchoring, never fuzzy quotation repair or benchmark answers."""
import asyncio

import pytest

from agent_memory.models import ExtractionResult, ExtractionWindow, Session, Turn, Event
from agent_memory.providers import CallableModel, ModelOutputError, _anchor_literal_quotes, _validate_evidence


def fixture(text, quote, turn_id='t'):
    window=ExtractionWindow(session=Session(session_id='s'),context_turns=[],new_turns=[
        Turn(turn_id='t',session_id='s',sequence=1,request_id='r',events=[Event(role='user',content=text)])])
    result=ExtractionResult.model_validate({'candidates':[{
        'content':text,'subject':'Inez','predicate':'read','value':'The River; Quiet Harbor',
        'evidence':[{'turn_id':'t','event_index':0,'quote':text}],
        'semantics':{'cardinality':'many','members':[
            {'value':'The River','evidence':[{'turn_id':'t','event_index':0,'quote':'The River'}]},
            {'value':'Quiet Harbor','evidence':[{'turn_id':turn_id,'event_index':0,'quote':quote}]}]}}]})
    return window,result


@pytest.mark.parametrize('quote', ['"Quiet Harbor"','“Quiet Harbor”',"'Quiet Harbor'",'`Quiet Harbor`'])
def test_presentation_quotes_resolve_to_unique_literal_without_dropping_members(quote):
    text='I read "The River" and "Quiet Harbor."'
    window,result=fixture(text,quote)
    receipts=_anchor_literal_quotes(window,result)
    assert len(receipts)==1
    record=receipts[0]
    assert text[record['start']:record['end']]==record['resolved_quote']=='Quiet Harbor'
    assert [m.value for m in result.candidates[0].semantics.members]==['The River','Quiet Harbor']
    assert result.candidates[0].value=='The River; Quiet Harbor'
    for member in result.candidates[0].semantics.members:
        _validate_evidence(window,result.model_copy(update={'candidates':[
            result.candidates[0].model_copy(update={'evidence':member.evidence})]}))


@pytest.mark.parametrize('text,quote,source_id', [
    ('I read The River. Quiet Harbor was mentioned, then Quiet Harbor again.','"Quiet Harbor"','t'),
    ('I read The River. I do not read Quiet Harbor.','"I read Quiet Harbor"','t'),
    ('I read The River and Quiet Harbor.','"Quiet Harbor"','another-turn'),
])
def test_ambiguous_paraphrased_or_wrong_source_quotes_remain_invalid(text,quote,source_id):
    window,result=fixture(text,quote,source_id)
    assert _anchor_literal_quotes(window,result)==[]
    member=result.candidates[0].semantics.members[1]
    assert member.evidence[0].quote==quote
    with pytest.raises(ModelOutputError):
        _validate_evidence(window,result.model_copy(update={'candidates':[
            result.candidates[0].model_copy(update={'evidence':member.evidence})]}))


def test_already_exact_quote_is_untouched():
    window,result=fixture('I read The River and "Quiet Harbor".','"Quiet Harbor"')
    assert _anchor_literal_quotes(window,result)==[]
    assert result.candidates[0].semantics.members[1].evidence[0].quote=='"Quiet Harbor"'


def test_overlapping_literal_occurrences_are_ambiguous():
    window,result=fixture('I read The River and banana.','"ana"')
    assert _anchor_literal_quotes(window,result)==[]


def test_extractor_anchors_before_validation_without_an_extra_model_call():
    async def scenario():
        window,result=fixture('I read "The River" and "Quiet Harbor."','"Quiet Harbor"')
        calls=[]
        def chat(prompt):
            calls.append(prompt)
            return result.model_dump_json()
        parsed=await CallableModel(chat).extract(window)
        assert len(calls)==1
        assert len(parsed.candidates[0].semantics.members)==2
        assert parsed.candidates[0].semantics.members[1].evidence[0].quote=='Quiet Harbor'
    asyncio.run(scenario())


def test_source_anchoring_does_not_turn_questions_into_facts():
    window,result=fixture('Did I read The River and Quiet Harbor?','"Quiet Harbor"')
    assert _anchor_literal_quotes(window,result)
    with pytest.raises(ModelOutputError,match='question'):
        _validate_evidence(window,result)

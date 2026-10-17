"""Named labels are data; questions must never establish their proposed answers."""
import pytest

from agent_memory.models import Event,ExtractionResult,ExtractionWindow,Session,Turn
from agent_memory.providers import ModelOutputError,_drop_invalid_candidates,_validate_evidence


def case(text,quote,value,predicate='content plan topic'):
    window=ExtractionWindow(session=Session(session_id='s'),context_turns=[],new_turns=[
        Turn(turn_id='t',session_id='s',sequence=1,request_id='r',events=[Event(role='user',content=text)])])
    result=ExtractionResult.model_validate({'candidates':[{'subject':'Inez','predicate':predicate,
        'value':'outline','content':'Inez has an outline','evidence':[{'turn_id':'t','event_index':0,'quote':text}],
        'semantics':{'cardinality':'many','members':[{'value':value,
          'evidence':[{'turn_id':'t','event_index':0,'quote':quote}]}]}}]})
    return window,result


@pytest.mark.parametrize('text,title,predicate',[
    ('Outline:\nHow to paint a wall\nChoosing supplies','How to paint a wall','content plan topic'),
    ('Suggested topics:\n- What is rust?','What is rust?','topics'),
    ('以下是标题：\n- 如何选择颜色','如何选择颜色','标题'),
])
def test_verbatim_labels_in_an_outline_are_not_answers_to_questions(text,title,predicate):
    window,result=case(text,title,title,predicate)
    _validate_evidence(window,result)
    assert not _drop_invalid_candidates(window,result)
    assert len(result.candidates[0].semantics.members)==1


@pytest.mark.parametrize('text,quote,value,predicate',[
    ('How to paint a wall?','How to paint a wall?','How to paint a wall?','topic'),
    ('Someone asked:\n- Did Inez adopt a child?','Did Inez adopt a child?','a child','adopted'),
    ('Outline:\nHow to paint a wall','How to paint a wall','Inez painted a wall','topic'),
])
def test_titles_do_not_license_world_facts_or_standalone_questions(text,quote,value,predicate):
    window,result=case(text,quote,value,predicate)
    with pytest.raises(ModelOutputError,match='question'):
        _validate_evidence(window,result)


def test_invalid_member_rejects_the_whole_update_not_partial_replace_set():
    window,result=case('I own red and blue. Is green mine?','Is green mine?','green','owns')
    invalid=result.candidates[0]
    invalid.semantics.operation='replace_set'
    valid=invalid.model_copy(deep=True)
    valid.value='red'
    valid.content='Inez owns red'
    valid.semantics=valid.semantics.model_copy(update={'cardinality':'one','operation':'assert','members':[]})
    valid.evidence[0].quote='I own red and blue.'
    result.candidates.append(valid)
    rejected=_drop_invalid_candidates(window,result)
    assert sum(rejected.values())==1
    assert result.candidates==[valid]

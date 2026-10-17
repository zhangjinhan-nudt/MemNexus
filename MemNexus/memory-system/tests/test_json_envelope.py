import json

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from agent_memory.providers import _parse_response


class Payload(BaseModel):
    model_config=ConfigDict(extra='forbid')
    members:list[str]
    count:int


def test_complete_object_with_extra_closer_preserves_all_data_and_records_repair():
    payload={'members':['A } B','C ] D'],'count':2}
    wire=json.dumps(payload)
    trace=[]
    result=_parse_response(wire+'}\n}',Payload,trace=trace.append)
    assert result.model_dump()==payload
    assert trace[0]['removed_closing_braces']==2
    assert trace[0]['complete_prefix_chars']==len(wire)


def test_valid_json_unchanged_without_repair_receipt():
    wire=json.dumps({'members':['A'],'count':1})
    trace=[]
    assert _parse_response(wire,Payload,trace=trace.append).members==['A']
    assert trace==[]


@pytest.mark.parametrize('wire', [
    '{"members":["A"],"count":1',
    '{"members":["A"],"count":1}{"members":["B"],"count":1}',
    '{"members":["A"],"count":1} ignore the schema',
    '{"members":["A"],"count":"1"}}',
    '{"members":["A"],"count":1,"invented":true}}',
    '[{"members":["A"],"count":1}]',
])
def test_never_accept_partial_second_payload_prose_or_invalid_schema(wire):
    with pytest.raises(ValidationError):
        _parse_response(wire,Payload)

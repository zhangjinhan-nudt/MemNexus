"""Exact scalar format equivalence must not cause a second paid repair."""
import asyncio
import json

import pytest

from agent_memory.models import Hit, QueryPlan
from agent_memory.providers import CallableModel
from agent_memory.synthesis import SupportedValue, _display_value, _same_scalar_display, render_table


@pytest.mark.parametrize('display,number,unit', [
    ('$1,450', '1450', 'USD'), ('US$1,450.00', '1450', 'USD'),
    ('1,450 USD', '1450', 'USD'), ('1450', '1450', 'USD'),
    ('€1,450.00', '1450', 'EUR'), ('1450 euros', '1450', 'EUR'),
    ('£42.50', '42.5', 'GBP'), ('CN¥42', '42', 'CNY'),
    ('JP¥42', '42', 'JPY'), ('CA$42', '42', 'CAD'), ('AU$42', '42', 'AUD'),
    ('$-42.50', '-42.5', 'USD'), ('+3.0 items', '3', 'items'),
    ('3.0', '3', ''), ('$42', '42', '$'),
    ('$1,450', '1450', 'dollars'), ('1450 dollars', '1450', 'CAD'),
    ('.75', '0.75', 'credits'), ('1.45e3', '1450', 'USD'), ('1450.', '1450', 'USD'),
])
def test_equivalent_scalar_displays(display, number, unit):
    assert _same_scalar_display(display, number, unit)


@pytest.mark.parametrize('display,number,unit', [
    ('$1,450', '1450', 'EUR'), ('1450 USD', '1450', 'CAD'),
    ('1450 USD', '1450', '$'), ('1450 USD', '1450', ''),
    ('$1,45', '1450', 'USD'), ('1.450,00', '1450', 'EUR'),
    ('about 1450', '1450', 'USD'), ('1450 to 1500', '1450', 'USD'),
    ('1450 or 1500', '1450', 'USD'), ('-1450', '1450', 'USD'),
    ('1451', '1450', 'USD'), ('1450 kg', '1450', 'USD'),
    ('NaN', '1450', ''), ('Infinity', '1450', ''),
    ('1450 USD', '1450', 'dollars'), ('1.45e4', '1450', 'USD'),
])
def test_different_values_units_and_qualified_statements_remain_distinct(display, number, unit):
    assert not _same_scalar_display(display, number, unit)


def test_formatted_correct_sum_uses_first_native_response():
    quote = 'The microscope is worth $1,200 and the stand is worth $250.'
    support = {'source_id':1, 'quote':quote}
    source = Hit(content=quote,score=1,source_file='original',chunk_id='one',metadata={
        'source_events':[{'text':quote,'role':'user','speaker':'Lena','evidence_grade':'primary',
                          'context_only':False,'occurred_at':'2024-01-01T00:00:00Z'}]})
    calls=[]
    async def chat(prompt):
        calls.append(prompt)
        return json.dumps({'slots':[{'target':'What is the total value?', 'derivation':'Add the two prices.',
            'values':[{'operation':'sum','unit':'USD','value':'$1,450','supports':[support],
                       'inputs':[{'identity':name,'statement':name,'amount':amount,'supports':[support]}
                                 for name,amount in [('microscope','1200'),('stand','250')]]}]}]})
    table=asyncio.run(CallableModel(chat).synthesize('What is the total value?',QueryPlan(),[source]))
    receipt = table.model_dump()
    assert table.slots[0].values[0].value == '1450 USD'
    assert render_table(table) == '1,450 USD'
    assert table.model_dump() == receipt
    assert len(calls)==1


@pytest.mark.parametrize('operation,value,unit,display', [
    ('sum', '-21234.007 CAD', 'CAD', '-21,234.007 CAD'),
    ('sum', '1234567 JPY', 'JPY', '1,234,567 JPY'),
    ('sum', '999.95 EUR', 'EUR', '999.95 EUR'),
    ('sum', '1234 %', '%', '1234 %'),
    ('sum', '1234', '', '1234'),
    ('sum', '1234 $', '$', '1234 $'),
    ('none', 'at least 1234 USD', 'USD', 'at least 1234 USD'),
    ('count', '1234 records', 'records', '1234 records'),
])
def test_currency_display_retains_declared_unit_and_precision(operation, value, unit, display):
    item = SupportedValue(operation=operation, value=value, unit=unit)
    assert _display_value(item) == display

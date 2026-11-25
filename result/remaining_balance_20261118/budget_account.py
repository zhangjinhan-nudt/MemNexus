"""Forward allowance from the user's current100RMB balance report.

Historical ledgers stay frozen and are not refunded or reset. This replaces
the previous estimated remaining allowance, rather than adding that leftover.
One coordinator admits components; active components reserve their full caps.
"""
import json
import math
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def reconcile():
    auth=read(HERE/'authorization.json')
    admissions=read(HERE/'admissions.json');spent=reserved=0.
    for a in admissions:
        p=HERE/'components'/a['arm']/a['sample_id']
        if not (p/'completion.json').exists():reserved+=a['cap'];continue
        if (p/'usage.json').exists():
            u=read(p/'usage.json');charge=float(u.get('cost_rmb',0))+float(u.get('reserved_rmb',0))
        else:
            # A lost usage ledger must never make a recorded request free.
            trace=p/'usage.calls.jsonl'
            attempted=trace.exists() and any(json.loads(l).get('event')=='request' for l in trace.read_text(encoding='utf-8').splitlines())
            charge=a['cap'] if attempted else 0.
        assert math.isfinite(charge) and charge>=0
        spent+=charge
    limit=auth['forward_allowance_rmb']
    assert spent+reserved<=limit+1e-7,'Forward100 budget exceeded'
    save(HERE/'budget.json',{'cap_rmb':limit,'spent_rmb':round(spent,8),'reserved_rmb':round(reserved,8),
        'remaining_rmb':round(limit-spent-reserved,8),'authorization':'authorization.json',
        'historical_cost_accounting':'unchanged; forward allowance replaces old estimated remainder, not additive',
        'platform_balance_verified_by_tool':False})
    return admissions,spent,reserved

def admit(arm,sid,cap):
    assert cap>0 and math.isfinite(cap)
    assert Path(arm).name==arm and Path(sid).name==sid
    admissions,spent,reserved=reconcile();p=HERE/'components'/arm/sid
    if (p/'started.json').exists():raise RuntimeError('Already started; inspect original component, never rerun blindly')
    if spent+reserved+cap>read(HERE/'authorization.json')['forward_allowance_rmb']:
        save(HERE/'budget_stop.json',{'arm':arm,'id':sid,'cap':cap,'spent':spent,'reserved':reserved})
        return None
    admissions.append({'arm':arm,'sample_id':sid,'cap':cap,'time':time.time()})
    save(HERE/'admissions.json',admissions)
    save(p/'started.json',{'cap':cap,'time':time.time(),'authorization':'current_remaining100_user_report'})
    reconcile();return p

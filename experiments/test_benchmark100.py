"""Offline runner checks. Test fixtures are never published as benchmark results."""

import asyncio
import hashlib
import json

import pytest

import benchmark100 as runner


def test_resume_compatible_only_and_completed_run_not_reused(tmp_path):
    signature={"data":"a","source":"b"}
    folder, report=runner.choose_run(tmp_path,signature,force_new=True)
    assert report is None
    runner.save(folder/"run.json",{"status":"interrupted","signature":signature})
    selected,report=runner.choose_run(tmp_path,signature)
    assert selected==folder
    with pytest.raises(ValueError,match="配置已变化"):
        runner.choose_run(tmp_path,{"data":"a","source":"c"},explicit=folder)
    runner.save(folder/"run.json",{"status":"completed","signature":signature})
    selected,_=runner.choose_run(tmp_path,signature)
    assert selected!=folder


def test_failed_attempts_and_reservations_count_once(tmp_path):
    a=runner.new_attempt(tmp_path,"conv_a_memnexus")
    runner.save(a/"usage.json",{"cost_rmb":1.2,"reserved_rmb":.3})
    b=runner.new_attempt(tmp_path,"conv_a_memnexus")
    runner.save(b/"usage.json",{"cost_rmb":.5,"reserved_rmb":0})
    assert a!=b
    assert runner.all_spend(tmp_path)==pytest.approx(2)
    assert runner.remaining(tmp_path,3)==pytest.approx(1)
    with pytest.raises(RuntimeError,match="预算已用完"):
        runner.remaining(tmp_path,2)


@pytest.mark.parametrize("raw",[
    '{"label":"CORRECT","reason":"It should be WRONG"}',
    'preface {"label":"CORRECT","reason":"ok"}',
    '{"label":"MAYBE","reason":"unclear"}',
    '{"label":"CORRECT","reason":""}',
])
def test_judge_rejects_invalid_or_contradictory_output(raw):
    with pytest.raises(ValueError):runner.parse_judgment(raw)


def test_stage_and_judge_resume_do_not_repeat_completed_calls(tmp_path,monkeypatch):
    import run_locomo_pilot
    def forbidden(*args,**kwargs):raise AssertionError("paid runner must not be called")
    monkeypatch.setattr(run_locomo_pilot,"run_simplemem",forbidden)
    monkeypatch.setattr(run_locomo_pilot,"run_memnexus",forbidden)
    monkeypatch.setattr(runner,"judge_one",forbidden)
    cohort={"max_session":5,"questions":[{"sample_id":"fixture","sample_index":0,"qa_index":0,
                                         "question":"fixture?","evidence":[]}]}
    dataset=[{"sample_id":"fixture","qa":[{"question":"fixture?"}]}]
    report={"stages":{},"judgments":{},"judge_seed":1}
    for system in runner.SYSTEMS:
        row={"system":system,"sample_id":"fixture","qa_index":0,"question":"fixture?","prediction":"fixture answer"}
        a=runner.new_attempt(tmp_path,"fixture_"+system.lower())
        runner.save(a/"stage_result.json",{"stage":{"memory_count":1},"rows":[row]})
        j=runner.new_attempt(tmp_path,"judge_fixture_0_"+system.lower())
        runner.save(j/"judgment.json",{"judge_result":"CORRECT"})
    asyncio.run(runner.execute(tmp_path,report,cohort,dataset,1))
    assert len(report["stages"])==len(report["judgments"])==2
    first=json.loads(json.dumps(report))
    asyncio.run(runner.execute(tmp_path,report,cohort,dataset,1))
    assert report==first


def test_ledger_token_tampering_fails(tmp_path):
    from benchmark100_report import ledger_audit
    path=tmp_path/"usage.json"
    ledger={"provider_attempts":1,"input_tokens":10,"output_tokens":2,"cost_rmb":.0000432}
    runner.save(path,ledger)
    events=[{"event":"request","body":{"model":"deepseek-v4-flash","temperature":0}},
            {"event":"response","body":{"usage":{"prompt_tokens":10,"completion_tokens":2}}}]
    path.with_suffix(".calls.jsonl").write_text("\n".join(json.dumps(x) for x in events),encoding="utf-8")
    ledger_audit(path)
    ledger["input_tokens"]=11
    runner.save(path,ledger)
    with pytest.raises(ValueError,match="token账本不一致"):ledger_audit(path)


def test_render_only_uses_historical_real_answers(tmp_path,monkeypatch):
    """Exercise all renderers on actual archived data; temporary files only."""
    import benchmark100_report as report_module
    original=runner.read(runner.ROOT/"result/locomo_100_collection_20261013_v3.json")
    rows,stages=original["rows"],original["stages"]
    assert len(rows)==200
    report={"signature":{"settings":{"semantic_execution":False}},"cohort":original["cohort"]}
    # Isolate rendering, not audit: these metadata fields are explicitly test-only.
    checked={"all_attempts_cost_upper_rmb":0,"effective_completed_cost_rmb":0,"judge_cost_rmb":0,
             "abandoned_attempts":[],"unknown_usage_upper_rmb":0,"answers":200,"judgments":200,
             "independent_memory_builds":20}
    monkeypatch.setattr(report_module,"audit",lambda folder:(report,rows,stages,checked))
    hashes=[hashlib.sha256(r["prediction"].encode()).hexdigest() for r in rows]
    summary=report_module.generate(tmp_path)
    assert summary["stats"]["SimpleMem"]["correct"]==73
    assert summary["stats"]["MemNexus"]["correct"]==68
    assert [hashlib.sha256(r["prediction"].encode()).hexdigest() for r in rows]==hashes
    for name in ("report.html","report.md","comparison.png","comparison.svg","summary.json","results.json"):
        assert (tmp_path/name).stat().st_size>100
    assert (tmp_path/"report.html").read_text(encoding="utf-8").count("<details>")==100


def test_second_process_lock_is_refused(tmp_path):
    with runner.run_lock(tmp_path):
        with pytest.raises(RuntimeError,match="已有100题测试"):
            with runner.run_lock(tmp_path):pass

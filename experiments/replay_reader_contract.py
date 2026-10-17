"""Real-provider fixed-evidence reader diagnostic; NOT an end-to-end benchmark.

Uses every archived final source, without gold-dependent selection. Gold is used
only after generation by the same blinded judge. Retrieval, planner and builds
are not rerun, so costs are reader-only and cannot replace system costs.
"""

import argparse
import asyncio
from copy import deepcopy
import os
from pathlib import Path
import random
import sys
import time
import zipfile

from benchmark100 import configure
from benchmark_mixed200 import ROOT, configure_semantic, dataset_settings, digest, read, save, source_files, token_f1, bleu1
from run_locomo_pilot import judge_rows

sys.path.insert(0, str(ROOT / "MemNexus/memory-system/src"))
from agent_memory.models import Hit, QueryPlan
from agent_memory.providers import answer_call_budget, build_model, ModelOutputError
from agent_memory.settings import Settings
from agent_memory.synthesis import check_and_execute, execution_contract, reader_profile, render_table


async def run(args):
    if read(args.source / "run.json")["status"] != "completed":
        raise ValueError("Wait for the complete frozen source run")
    args.output.mkdir(parents=True, exist_ok=False)
    original = read(args.source / "results.json")
    rows = [r for r in original["rows"] if r["system"] == "MemNexus"]
    configure_semantic()
    configure({"variant": "semantic"})
    hashes = source_files()
    hashes[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__))
    report = {"status": "running", "kind": "fixed_final_evidence_reader_diagnostic_not_benchmark",
              "source": str(args.source.resolve()), "source_sha256": digest(args.source / "results.json"),
              "code_sha256": hashes, "rows": [], "limitations": [
                  "All final supplied sources reused, including previous supplemental retrieval",
                  "No new build, planner or retrieval; reader-only costs are not full QA costs",
                  "Development cohort, not independent held-out evidence"]}
    with zipfile.ZipFile(args.output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in hashes:
            archive.write(ROOT / path, path)
    save(args.output / "results.json", report)
    try:
        for old in rows:
            item = old["item_id"]
            ledger = args.output / "attempts" / item / "usage.json"
            os.environ["EXPERIMENT_USAGE_LEDGER_PATH"] = str(ledger)
            os.environ["EXPERIMENT_COMPONENT_BUDGET_RMB"] = "10000"
            with dataset_settings(old["dataset"]):
                model = build_model(Settings(model_timeout=180, model_max_tokens=int(os.environ["EXPERIMENT_MEMNEXUS_MAX_TOKENS"]),
                                             output_language="en"))
                if model is None or model.model != "deepseek-v4-flash":
                    raise ValueError("Configured DeepSeek model is required")
                sources = [Hit.model_validate(s) for s in old["retrieved_sources"]]
                plan = QueryPlan.model_validate(old["plan"])
                query = old["evaluation_query"]
                prior_path = old.get("reasoning", {}).get("answer_path", "")
                semantic = prior_path.startswith("semantic") and reader_profile(query) not in {"entity_selection", "recommendation"}
                trace = {"original_path": prior_path, "reader_profile": reader_profile(query)}
                started = time.perf_counter()
                try:
                    with answer_call_budget(True, max_calls=2, prompt_limit=50000) as budget:
                        budget["active"] = True
                        if semantic:
                            contract = execution_contract(query, plan)
                            try:
                                bound = await model.bind_answer(query, contract, sources)
                                execution = check_and_execute(contract, bound, sources)
                                prediction = execution.render(choice_question="Choose exactly one" in query)
                                trace.update(path="semantic_execution", bound=bound.model_dump(mode="json"),
                                             execution=execution.model_dump(mode="json"))
                            except (ModelOutputError, ValueError) as error:
                                trace["binding_error"] = str(error)
                                table = await model.synthesize(query, plan, sources)
                                prediction = render_table(table)
                                trace.update(path="fallback_synthesis", table=table.model_dump(mode="json"))
                        else:
                            table = await model.synthesize(query, plan, sources)
                            prediction = render_table(table)
                            trace.update(path="source_bound_synthesis", table=table.model_dump(mode="json"))
                        trace["call_budget"] = dict(budget)
                except (ModelOutputError, ValueError) as error:
                    # Retain failed generation as a failure, never replace it with an old/gold answer.
                    prediction = "Insufficient evidence to determine the answer."
                    trace.update(path="reader_failed", error=str(error))
                finally:
                    await model.aclose()
                result = deepcopy(old)
                result.update(prediction=prediction, original_prediction=old["prediction"],
                              original_judge_result=old["judge_result"], judge_result=None,
                              reasoning=trace, model_calls=dict(model.usage),
                              latency_ms=(time.perf_counter()-started)*1000,
                              f1=token_f1(prediction, old["gold_answer"]), bleu1=bleu1(prediction, old["gold_answer"]))
                report["rows"].append(result)
                save(args.output / "results.json", report)
                print(f"{len(report['rows'])}/{len(rows)} {item}: {prediction}", flush=True)
        simple = [deepcopy(r) for r in original["rows"] if r["system"] == "SimpleMem"]
        for row in simple:
            row["original_judge_result"] = row["judge_result"]
            row["judge_result"] = None
        report["rows"].extend(simple)
        shuffled = list(report["rows"])
        random.Random(20261017).shuffle(shuffled)
        report["judge_usage"] = await asyncio.to_thread(judge_rows, shuffled, 10000,
                                                       usage_ledger=args.output / "judge_usage.json")
        if any(digest(ROOT / p) != sha for p, sha in hashes.items()):
            raise ValueError("Reader source changed during diagnostic")
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="aborted", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        save(args.output / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    asyncio.run(run(parser.parse_args()))

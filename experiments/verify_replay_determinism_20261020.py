"""Precondition for every A/B in this project: is the QA stage reproducible?

Freezing the build stage removes the largest noise source (rebuilding changed
the evidence set on 150/150 questions). It does not remove the answering stage's
own nondeterminism. If two runs on the *same* frozen base disagree on any
prediction, then a before/after delta of 2-4 questions is still unreadable and
the honest move is N repeats, not a bigger claim.

Two questions, in order:

1. Did each run actually replay the frozen base? Read from
   ``attempts/*/usage.stage.json`` -> ``memory_base_replayed`` / ``memory_snapshot_sha256``.
2. Do two replays of the same base produce identical predictions?

Usage::

    python experiments/verify_replay_determinism_20261020.py \
        --run-a result/benchmark150_paper/<run1> \
        --run-b result/benchmark150_paper/<run2>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_rows(run: Path) -> dict:
    payload = json.loads((run / "results.json").read_text(encoding="utf-8"))
    return {
        (row["sample_id"], row["question"]): row
        for row in payload["rows"]
        if row.get("system") == "MemNexus"
    }


def replay_flags(run: Path) -> dict:
    flags = {}
    for path in sorted(run.glob("attempts/*_memnexus__*/usage.stage.json")):
        try:
            stage = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if stage.get("system") != "MemNexus":
            continue
        digest = stage.get("memory_snapshot_sha256") or (stage.get("memory_snapshot") or {}).get("sha256")
        flags[stage.get("sample_id") or path.parent.name] = {
            "replayed": bool(stage.get("memory_base_replayed")),
            "sha256": digest,
        }
    return flags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    args = parser.parse_args()

    run_a, run_b = args.run_a.resolve(), args.run_b.resolve()
    flags_a, flags_b = replay_flags(run_a), replay_flags(run_b)
    for label, flags in (("run-a", flags_a), ("run-b", flags_b)):
        total = len(flags)
        replayed = sum(1 for item in flags.values() if item["replayed"])
        print(f"{label}: {replayed}/{total} 段对话回放了冻结基座")
        if total and replayed != total:
            print("   注意：该运行至少重建了一份基座，不构成回放条件")

    shared_samples = sorted(set(flags_a) & set(flags_b))
    mismatched = [s for s in shared_samples
                  if flags_a[s]["sha256"] and flags_a[s]["sha256"] != flags_b[s]["sha256"]]
    print(f"两侧共享对话 {len(shared_samples)} 段；sha256 不一致 {len(mismatched)} 段")
    for sample in mismatched[:5]:
        print(f"   {sample}: {(flags_a[sample]['sha256'] or '')[:12]} != "
              f"{(flags_b[sample]['sha256'] or '')[:12]}")

    rows_a, rows_b = load_rows(run_a), load_rows(run_b)
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        print("没有可比较的题目")
        return 2

    same_text = sum(1 for key in shared if rows_a[key]["prediction"] == rows_b[key]["prediction"])
    same_verdict = sum(1 for key in shared
                       if rows_a[key].get("judge_result") == rows_b[key].get("judge_result"))
    print()
    print(f"共享题目            : {len(shared)}")
    print(f"预测完全一致        : {same_text}/{len(shared)} ({same_text/len(shared)*100:.1f}%)")
    print(f"裁判判定一致        : {same_verdict}/{len(shared)} ({same_verdict/len(shared)*100:.1f}%)")

    if same_text < len(shared):
        by_category = Counter()
        for key in shared:
            if rows_a[key]["prediction"] != rows_b[key]["prediction"]:
                by_category[rows_a[key]["category"]] += 1
        print("  不一致题型分布:", dict(by_category))
        for key in [k for k in shared if rows_a[k]["prediction"] != rows_b[k]["prediction"]][:5]:
            print("   Q:", key[1][:80])
            print("     A:", str(rows_a[key]["prediction"])[:100])
            print("     B:", str(rows_b[key]["prediction"])[:100])

    no_replay = not any(item["replayed"] for item in list(flags_a.values()) + list(flags_b.values()))
    deterministic = same_text == len(shared) and not mismatched and not no_replay
    print()
    if no_replay:
        print("结论：两侧都没有回放冻结基座，这个比较无法回答可复现性。"
              "请用 --memory-snapshots 跑两次同快照再判。")
    else:
        print("结论：" + (
            "QA 在冻结基座上可复现，before/after 差值可判读。"
            if deterministic
            else "QA 不可复现，或两侧基座不同：冻结只消掉了建库噪声，仍需多次重复取均值。"
        ))
    return 0 if deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())

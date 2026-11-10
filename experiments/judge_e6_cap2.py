"""对 E6 cap=2 全量运行的 MemNexus 行补跑 Judge（与基线 rows_judged.json 同口径）。

口径唯一来源：run_locomo_pilot.judge_rows → benchmark100.JUDGE_INSTRUCTION
（SimpleMem 原论文 Appendix A.4 逐字移植，generous grader）；
判分模型与基线一致（SimpleMem config.JUDGE_*），保证两轮 Judge 可直接对比。

断点：judge ledger（.rows.jsonl）里已判的行跳过，只补缺。
产物：result/e6_full_cap2_20261110/rows_judged.json（结构与母运行 rows_judged.json 一致）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

SRC = ROOT / "result/e6_full_cap2_20261110/rows.json"
OUT = ROOT / "result/e6_full_cap2_20261110/rows_judged.json"
LEDGER = ROOT / "result/e6_full_cap2_20261110/judge_ledger.json"
LEDGER_ROWS = ROOT / "result/e6_full_cap2_20261110/judge_ledger.rows.jsonl"


def key_of(row: dict) -> tuple:
    return (row.get("sample_id"), row.get("qa_index"), row.get("system"))


def main() -> int:
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    mn = [r for r in rows if r.get("system") == "MemNexus"]
    print(f"MemNexus 行 {len(mn)}；判分口径 = SimpleMem Appendix A.4（与基线同源）")

    done: dict[tuple, dict] = {}
    if LEDGER_ROWS.exists():
        for line in LEDGER_ROWS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("judge_result"):
                done[key_of(row)] = row
        print(f"断点：ledger 已判 {len(done)} 条")

    todo = [r for r in mn if key_of(r) not in done]
    if todo:
        from run_locomo_pilot import judge_rows
        usage = judge_rows(todo, 15.0, usage_ledger=LEDGER)
        print(f"本次判分 {len(todo)} 条；judge usage: {usage}")

    merged = [done.get(key_of(r)) or r for r in mn]
    missing = [r for r in merged if not r.get("judge_result")]
    if missing:
        print(f"[FAIL] {len(missing)} 条缺 judge_result，不写出产物", file=sys.stderr)
        return 3

    OUT.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(r["judge_result"] == "CORRECT" for r in merged)
    adv = [r for r in merged if r.get("category_id") == 5]
    nonadv = [r for r in merged if r.get("category_id") != 5]
    ok_nonadv = sum(r["judge_result"] == "CORRECT" for r in nonadv)
    print(f"[OK] 写出 {OUT}")
    print(f"     全部 {len(merged)} 条 CORRECT {ok/len(merged):.1%}")
    if nonadv:
        print(f"     非对抗 {len(nonadv)} 条 CORRECT {ok_nonadv/len(nonadv):.1%}"
              f"（基线 MN 非对抗 77.5% / SM 82.7%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

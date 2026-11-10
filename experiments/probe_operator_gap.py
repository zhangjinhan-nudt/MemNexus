"""Quantify the gap between the declared operator and the answer's real shape.

Two questions:

  Q1. For items whose GOLD answer is a multi-item list, how often does the
      contract declare ``operator=set``?  A miss means the values are joined
      with "; " and the evaluator sees ONE long item instead of N.

  Q2. What is the F1 difference between the two groups?

This isolates the connector decision from retrieval quality: both groups come
from the same run, same model, same evidence.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paper_protocol as pp  # noqa: E402

ROOT = HERE.parent
ROWS = ROOT / "result" / "bench260" / "rows.json"
ADVERSARIAL = 5


def split_items(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;]", str(text or "")) if p.strip()]


def outputs_of(row: dict) -> list[dict]:
    plan = row.get("plan") or {}
    contract = plan.get("answer_contract") or {}
    return list(contract.get("outputs") or [])


def operators_of(row: dict) -> set[str]:
    return {str(o.get("operator") or "") for o in outputs_of(row)}


def main() -> int:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    mn = [r for r in rows if r.get("system") == "MemNexus"
          and str(r.get("prediction") or "").strip()
          and r.get("category_id") != ADVERSARIAL]

    print("=== Q1. gold is a multi-item list -> does the contract say `set`? ===")
    tbl = defaultdict(lambda: [0, 0])
    for row in mn:
        g = len(split_items(row.get("gold_answer")))
        if g < 2:
            continue
        is_set = "set" in operators_of(row)
        e = tbl[(row.get("category"), "set" if is_set else "NOT set")]
        e[0] += 1
        e[1] += 100 * pp.f1_of(row)
    print(f"{'category':<14}{'contract':<10}{'n':>5}{'mean F1':>10}")
    for (cat, kind) in sorted(tbl):
        n, s = tbl[(cat, kind)]
        print(f"{cat:<14}{kind:<10}{n:>5}{s/n:>10.2f}")
    print()

    print("=== Q2. connector actually emitted ===")
    conn = Counter()
    for row in mn:
        pred = str(row.get("prediction") or "")
        if "," in pred and ";" in pred:
            conn["both"] += 1
        elif "," in pred:
            conn["comma only"] += 1
        elif ";" in pred:
            conn["semicolon only"] += 1
        else:
            conn["neither"] += 1
    for k, v in conn.most_common():
        print(f"  {k:<16}{v:>5}")
    print()

    print("=== Q3. operator x connector cross-tab (the invariant to enforce) ===")
    cross = defaultdict(lambda: [0, 0.0])
    for row in mn:
        pred = str(row.get("prediction") or "")
        ops = operators_of(row)
        op = "set" if "set" in ops else ("multi" if len(ops) > 1 else (next(iter(ops)) or "?"))
        if "," in pred:
            c = "comma"
        elif ";" in pred:
            c = "semicolon"
        else:
            c = "none"
        e = cross[(op, c)]
        e[0] += 1
        e[1] += 100 * pp.f1_of(row)
    print(f"{'operator':<12}{'connector':<12}{'n':>5}{'mean F1':>10}")
    for (op, c) in sorted(cross, key=lambda k: -cross[k][0]):
        n, s = cross[(op, c)]
        print(f"{op:<12}{c:<12}{n:>5}{s/n:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

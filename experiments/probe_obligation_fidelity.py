"""How faithful is the answer to the obligation contract that was set upstream?

The architecture claim is: obligation (L5) constrains the answer (L7).
This probe measures whether it actually does, item by item:

  1. operator distribution in ``plan.answer_contract.outputs[*].operator``
  2. the unit of the obligation (single vs set) vs the unit actually rendered
  3. for over-long predictions: are the extra tokens grounded in retrieved
     evidence (retrieved-but-irrelevant) or unsupported (generation drift)?
  4. the F1 cost of getting the enumeration unit wrong, per category
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

STOP = {
    "the", "is", "a", "an", "what", "was", "were", "of", "my", "me", "please",
    "and", "to", "in", "on", "at", "for", "did", "does", "do", "are", "he", "she",
    "they", "it", "his", "her", "their", "that", "this", "with", "how", "many",
    "when", "which", "who", "whose", "had", "has", "have", "been", "be",
}

# operators whose answer unit is a SET (several items) rather than one value
SET_OPERATORS = {"set", "list", "enumerate", "count", "union", "intersection"}


def toks(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text).casefold())
            if t not in STOP and len(t) > 1}


def ev_tokens(row: dict) -> set[str]:
    return toks(" ".join(json.dumps(s, ensure_ascii=False)
                         for s in (row.get("retrieved_sources") or [])))


def split_items(text: str) -> list[str]:
    """Split on the connectors LoCoMo's official scorer honours."""
    return [p.strip() for p in re.split(r"[,;]", str(text or "")) if p.strip()]


def outputs_of(row: dict) -> list[dict]:
    plan = row.get("plan") or {}
    contract = plan.get("answer_contract") or {}
    return list(contract.get("outputs") or [])


def main() -> int:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    mn_rows = [r for r in rows if r.get("system") == "MemNexus"
               and str(r.get("prediction") or "").strip()]

    ops = Counter()
    op_f1 = defaultdict(list)
    for row in mn_rows:
        f1 = 100 * pp.f1_of(row)
        for out in outputs_of(row):
            op = str(out.get("operator") or "(none)")
            ops[op] += 1
            op_f1[op].append(f1)

    print("=== 1. operator distribution (plan.answer_contract.outputs[*]) ===")
    print(f"{'operator':<16}{'n':>5}{'mean F1':>10}")
    for op, n in ops.most_common():
        print(f"{op:<16}{n:>5}{statistics.mean(op_f1[op]):>10.2f}")
    print()

    # ---- 2. is the contract's unit honoured? -------------------------------
    print("=== 2. obligation unit vs rendered unit ===")
    unit_stats = defaultdict(lambda: {"n": 0, "f1": [], "pred": [], "gold": []})
    for row in mn_rows:
        f1 = 100 * pp.f1_of(row)
        outs = outputs_of(row)
        set_like = any(str(o.get("operator")) in SET_OPERATORS for o in outs)
        multi_subject = any(str(o.get("subject_mode")) not in ("single", "None")
                            for o in outs)
        key = ("contract=set" if set_like
               else "contract=multi-subject" if multi_subject
               else "contract=single")
        pred_n = len(split_items(row.get("prediction")))
        gold_n = len(split_items(row.get("gold_answer")))
        s = unit_stats[key]
        s["n"] += 1
        s["f1"].append(f1)
        s["pred"].append(pred_n)
        s["gold"].append(gold_n)
    print(f"{'contract unit':<26}{'n':>5}{'F1':>8}{'pred_items':>12}{'gold_items':>12}")
    for key in sorted(unit_stats):
        s = unit_stats[key]
        print(f"{key:<26}{s['n']:>5}{statistics.mean(s['f1']):>8.2f}"
              f"{statistics.mean(s['pred']):>12.2f}{statistics.mean(s['gold']):>12.2f}")
    print()

    # ---- 3. grounded extra vs drift ---------------------------------------
    grounded, drifted = 0, 0
    offenders = []
    for row in mn_rows:
        gold_tok = toks(row.get("gold_answer"))
        pred_tok = toks(row.get("prediction"))
        extra = pred_tok - gold_tok
        if not extra:
            continue
        ev = ev_tokens(row)
        hit = len(extra & ev) / len(extra)
        if hit >= 0.8:
            grounded += 1
        else:
            drifted += 1
        offenders.append((hit, len(extra), 100 * pp.f1_of(row),
                          str(row.get("sample_id")), row.get("qa_index"),
                          str(row.get("prediction"))[:64]))
    tot = grounded + drifted
    print("=== 3. over-generation: grounded in evidence or drift? ===")
    print(f"items whose prediction carries extra tokens: {tot}")
    print(f"  >=80% of extra tokens present in retrieved evidence: {grounded} "
          f"({100*grounded/max(tot,1):.1f}%)  <- said too much of what WAS retrieved")
    print(f"  <80%  (unsupported)                                : {drifted} "
          f"({100*drifted/max(tot,1):.1f}%)  <- generation drift")
    print()
    print("  least-grounded offenders:")
    offenders.sort(key=lambda x: (x[0], -x[1]))
    for hit, n_extra, f1, sid, qi, pred in offenders[:10]:
        print(f"    {sid}/{qi} F1={f1:5.1f} extra={n_extra:<3} grounded={100*hit:5.1f}%  {pred}")
    print()

    # ---- 4. cost of getting the unit wrong, per category -------------------
    print("=== 4. enumeration length mismatch vs F1, per category ===")
    by_cat = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for row in mn_rows:
        if row.get("category_id") == ADVERSARIAL:
            continue
        f1 = 100 * pp.f1_of(row)
        p = len(split_items(row.get("prediction")))
        g = len(split_items(row.get("gold_answer")))
        bucket = "longer" if p > g else ("shorter" if p < g else "equal")
        e = by_cat[row.get("category")][bucket]
        e[0] += 1
        e[1] += f1
    print(f"{'category':<14}{'bucket':<10}{'n':>5}{'mean F1':>10}")
    for cat in ("multi-hop", "temporal", "open-domain", "single-hop"):
        for bucket in ("equal", "longer", "shorter"):
            n, s = by_cat[cat][bucket]
            if n:
                print(f"{cat:<14}{bucket:<10}{n:>5}{s/n:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Print full pipeline traces for the items that matter most.

Usage: probe_traces.py <mode>
  longer   - predictions that enumerate MORE items than the gold answer
  drift    - predictions whose extra tokens are not supported by evidence
  contract - items whose contract operator is `set`
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys
from collections import defaultdict

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
SET_OPERATORS = {"set", "list", "enumerate", "count", "union", "intersection"}


def toks(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text).casefold())
            if t not in STOP and len(t) > 1}


def ev_tokens(row: dict) -> set[str]:
    return toks(" ".join(json.dumps(s, ensure_ascii=False)
                         for s in (row.get("retrieved_sources") or [])))


def split_items(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;]", str(text or "")) if p.strip()]


def outputs_of(row: dict) -> list[dict]:
    plan = row.get("plan") or {}
    contract = plan.get("answer_contract") or {}
    return list(contract.get("outputs") or [])


def one(text, limit=110):
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


def trace(row: dict, label: str) -> None:
    r = row.get("reasoning") or {}
    plan = row.get("plan") or {}
    table = r.get("answer_table") or {}
    slots = table.get("slots") or []
    calls = row.get("model_calls") or {}
    gate = r.get("typed_negative_gate") or {}
    print(f"--- {row.get('sample_id')}/{row.get('qa_index')} · {row.get('category')} · "
          f"F1={100*pp.f1_of(row):.1f} · [{label}]")
    print(f"    Q     : {one(row.get('question'))}")
    print(f"    GOLD  : {one(row.get('gold_answer'))}")
    print(f"    MN    : {one(row.get('prediction'))}")
    print(f"    kind  : {r.get('question_type')} / recall={r.get('recall_policy')} "
          f"/ route={plan.get('route')} / depth={plan.get('depth')}")
    print(f"    graph : nodes={r.get('graph_nodes')} edges={r.get('graph_edges')} "
          f"deriv={r.get('deterministic_derivations')} path={r.get('answer_path')}")
    print(f"    gate  : blocked={gate.get('blocked')} reason={gate.get('reason')}")
    print(f"    contract: " + " | ".join(
        f"[{o.get('operator')}] subj={o.get('subjects')} mode={o.get('subject_mode')} "
        f"rel={o.get('relation')} q={o.get('qualifier')}" for o in outputs_of(row)))
    print(f"    slots : {len(slots)}")
    for s in slots:
        vals = ", ".join(str(v.get("value", "")) for v in (s.get("values") or []))
        print(f"        - [{s.get('role')}] {one(vals or (s.get('missing') or '∅'), 90)}")
    print(f"    evid : retrieved={row.get('retrieval_count')} citations={row.get('citations')}")
    print(f"    cost : in={calls.get('input_tokens')} out={calls.get('output_tokens')} "
          f"¥{calls.get('cost_rmb')} · {row.get('latency_ms')} ms")
    print()


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "longer"
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    mn = [r for r in rows if r.get("system") == "MemNexus"
          and str(r.get("prediction") or "").strip()]

    picked = []
    for row in mn:
        p = len(split_items(row.get("prediction")))
        g = len(split_items(row.get("gold_answer")))
        extra = toks(row.get("prediction")) - toks(row.get("gold_answer"))
        hit = (len(extra & ev_tokens(row)) / len(extra)) if extra else 1.0
        if mode == "longer" and p > g:
            picked.append((p - g, row))
        elif mode == "drift" and extra and hit < 0.8:
            picked.append((-len(extra), row))
        elif mode == "contract" and any(
                str(o.get("operator")) in SET_OPERATORS for o in outputs_of(row)):
            picked.append((0, row))
    picked.sort(key=lambda t: t[0], reverse=True)
    print(f"mode={mode} picked={len(picked)}\n")
    for _, row in picked[:14]:
        trace(row, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

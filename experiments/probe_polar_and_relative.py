"""Two rendering defects that are cheap to fix and expensive to ignore.

Defect 1 -- POLAR ELABORATION.
    The question asks a yes/no question.  The contract therefore expects a
    single polar token.  The synthesis step answers correctly and then appends
    a justification.  The official F1 is token-overlap based, so the extra
    tokens are pure precision loss: the answer is *right*, the score is *low*.

Defect 2 -- UNRESOLVED RELATIVE TIME.
    ``render_table`` promises to convert a relative time phrase into the
    calendar form the question asked for.  When it does not fire, the answer
    states the correct fact in the wrong surface form and scores ~0 against a
    gold answer written as a year or a date.

Both are measured against the official scorer, on the same run, with no
re-running of the model.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paper_protocol as pp  # noqa: E402

ROOT = HERE.parent
ROWS = ROOT / "result" / "bench260" / "rows.json"
ADVERSARIAL = 5
CATS = ("multi-hop", "temporal", "open-domain", "single-hop")

POLAR = re.compile(r"^(yes|no|yeah|nope|true|false)\b", re.I)
RELATIVE_TIME = re.compile(
    r"\b(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(year|years|month|months|week|weeks|day|days)\s+(before|after|prior to|earlier|later)\b",
    re.I,
)
CALENDAR = re.compile(r"\b(19|20)\d{2}\b")


def norm(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def main() -> int:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key.setdefault((r.get("sample_id"), r.get("qa_index")), {})[r.get("system")] = r
    pairs = [(v["MemNexus"], v["SimpleMem"]) for v in by_key.values()
             if v.get("MemNexus") and v.get("SimpleMem")
             and norm(v["MemNexus"].get("prediction"))
             and norm(v["SimpleMem"].get("prediction"))]

    print("=== Defect 1: polar (yes/no) questions, MN elaborated ===")
    n_polar = 0
    n_elab = 0
    losses = []
    for mn, sm in pairs:
        gold = norm(mn.get("gold_answer"))
        if not POLAR.match(gold):
            continue
        n_polar += 1
        pred = norm(mn.get("prediction"))
        gold_tok = gold.casefold().split()
        pred_tok = pred.casefold().split()
        # "correct" = the polar token is present
        correct = any(t.strip(".,!") in gold_tok for t in pred_tok)
        if correct and len(pred_tok) > 1:
            n_elab += 1
            f1 = 100 * pp.f1_of(mn)
            losses.append(f1)
    print(f"  polar-gold items                      : {n_polar}")
    print(f"  MN got the polarity right but elaborated: {n_elab}")
    if losses:
        print(f"  mean F1 on those                       : {statistics.mean(losses):.2f}")
    print()

    print("=== Defect 2: relative time phrase left unresolved ===")
    hits = []
    for mn, sm in pairs:
        pred = norm(mn.get("prediction"))
        gold = norm(mn.get("gold_answer"))
        # Gold is a bare calendar value (a year, or a year plus a month/day);
        # the prediction answers in relative form ("3 years before <date>").
        # The fact is the same, the surface form the scorer compares is not.
        if RELATIVE_TIME.search(pred) and CALENDAR.search(gold) and len(gold.split()) <= 4:
            hits.append((100 * pp.f1_of(mn), mn.get("sample_id"), mn.get("qa_index"),
                         mn.get("category"), gold[:40], pred[:80]))
    print(f"  items: {len(hits)}")
    for f1, sid, qi, cat, gold, pred in sorted(hits):
        print(f"    {sid}/{qi} [{cat}] F1={f1:5.1f}  gold='{gold}'  mn='{pred}'")
    print()

    # what a minimal normalisation would buy
    print("=== Combined: ceiling if both defects were normalised ===")
    by_cat = {c: [] for c in CATS}
    n_items = {c: 0 for c in CATS}
    for mn, sm in pairs:
        if mn.get("category_id") == ADVERSARIAL:
            continue
        cat = mn.get("category")
        n_items[cat] = n_items.get(cat, 0) + 1
    print("  (see probe_length_ceiling.py for the enumeration-length bound)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

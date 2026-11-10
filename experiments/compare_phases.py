"""Compare any number of phase folders under the official LoCoMo protocol.

Every number here goes through ``paper_protocol``: official F1/BLEU-1 per
category, four-category equal-weight average, adversarial success as a separate
rate, and paired subsets only. Category labels come from ``category_id``
because the stored label in older phase files is not trustworthy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from paper_protocol import adversarial_success, bleu1_of, f1_of  # noqa: E402

CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ABSTAIN = ("not mentioned", "insufficient evidence", "context does not", "unable to", "cannot determine")


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def category_of(row) -> str:
    return CATEGORY.get(int(row.get("category_id") or 0), row.get("category", "?"))


def load(folder: Path) -> dict[tuple, dict]:
    rows: dict[tuple, dict] = {}
    for path in sorted(Path(folder).glob("conv-*.json")):
        if "_usage" in path.name:
            continue
        payload = read(path)
        questions = payload.get("questions") or []
        sample_id = (payload.get("stage") or {}).get("sample_id")
        for index, row in enumerate(payload["rows"]):
            reference = questions[index] if index < len(questions) else {}
            key = (row.get("sample_id") or reference.get("sample_id") or sample_id,
                   row.get("qa_index", reference.get("qa_index")))
            rows[key] = row
    if not rows:
        for name in ("rows.json", "rows_judged.json"):
            path = Path(folder) / name
            if path.exists():
                for row in read(path):
                    rows[(row["sample_id"], row["qa_index"])] = row
                break
    return rows


def abstained(row: dict) -> bool:
    text = (row.get("prediction") or "").strip().casefold()
    return any(text.startswith(marker) for marker in ABSTAIN)


def table(phases: dict[str, dict[tuple, dict]]) -> list[str]:
    keys = sorted(set.intersection(*[set(rows) for rows in phases.values()]))
    lines = [
        "| category | n | " + " | ".join(phases) + " |",
        "|---|---:|" + "---:|" * len(phases),
    ]
    for category in ("multi-hop", "temporal", "open-domain", "single-hop", "adversarial"):
        ids = [key for key in keys if category_of(phases[next(iter(phases))][key]) == category]
        if not ids:
            continue
        cells = []
        for rows in phases.values():
            scores = [f1_of(rows[key]) for key in ids]
            cells.append(f"{statistics.mean(scores):.3f}")
        lines.append(f"| {category} | {len(ids)} | " + " | ".join(cells) + " |")
    lines.append("| **mean (paired)** | " + str(len(keys)) + " | " +
                 " | ".join(f"**{statistics.mean(f1_of(rows[key]) for key in keys):.3f}**"
                            for rows in phases.values()) + " |")
    answered = [key for key in keys if category_of(phases[next(iter(phases))][key]) != "adversarial"]
    lines.append("| mean (non-adversarial) | " + str(len(answered)) + " | " +
                 " | ".join(f"{statistics.mean(f1_of(rows[key]) for key in answered):.3f}"
                            for rows in phases.values()) + " |")
    lines.append("| adversarial success | " +
                 str(sum(1 for key in keys if category_of(phases[next(iter(phases))][key]) == "adversarial")) +
                 " | " + " | ".join(
                     str(sum(adversarial_success(rows[key]) for key in keys
                             if category_of(rows[key]) == "adversarial")) for rows in phases.values()) + " |")
    lines.append("| abstentions | " + str(len(keys)) + " | " +
                 " | ".join(str(sum(abstained(rows[key]) for key in keys)) for rows in phases.values()) + " |")
    lines.append("| input tokens/QA | " + str(len(keys)) + " | " +
                 " | ".join(f"{statistics.mean((rows[key].get('model_calls') or {}).get('input_tokens', 0) for key in keys):.0f}"
                            for rows in phases.values()) + " |")
    lines.append("| output tokens/QA | " + str(len(keys)) + " | " +
                 " | ".join(f"{statistics.mean((rows[key].get('model_calls') or {}).get('output_tokens', 0) for key in keys):.0f}"
                            for rows in phases.values()) + " |")
    return lines


def flips(phases: dict[str, dict[tuple, dict]], reference: str) -> list[str]:
    keys = sorted(set.intersection(*[set(rows) for rows in phases.values()]))
    base = phases[reference]
    out = []
    for key in keys:
        scores = {name: f1_of(rows[key]) for name, rows in phases.items()}
        spread = max(scores.values()) - min(scores.values())
        if spread > 1e-9:
            out.append((spread, key, scores, base[key]))
    out.sort(reverse=True)
    lines = ["| question | category | " + " | ".join(phases) + " | prediction (" + reference + ") |",
             "|---|---|" + "---:|" * len(phases) + "---|"]
    for spread, key, scores, row in out:
        cells = " | ".join(f"{scores[name]:.2f}" for name in phases)
        lines.append(f"| {key[0]}/{key[1]} | {category_of(row)} | {cells} | "
                     f"{(row.get('prediction') or '')[:70]!r} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", action="append", required=True, help="name=path")
    parser.add_argument("--reference", default="")
    parser.add_argument("--markdown")
    args = parser.parse_args()

    phases = {}
    for entry in args.phase:
        name, _, path = entry.partition("=")
        phases[name] = load(ROOT / path)
    body = ["### Official protocol", *table(phases)]
    reference = args.reference or next(iter(phases))
    body += ["", f"### Per-question movement (reference: {reference})", *flips(phases, reference)]
    markdown = "\n".join(body)
    print(markdown)
    if args.markdown:
        Path(args.markdown).write_text(markdown + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

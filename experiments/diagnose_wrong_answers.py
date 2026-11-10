"""逐题错因诊断：MemNexus 错在哪一段，SimpleMem 同题怎么做的。

只读分析，不 import 被测源码，运行期间可反复调用。

错因分类（按信息链路的断裂位置，互斥）：
    对抗误答      gold 是否定型（Not mentioned），模型却给了具体答案
    过度拒答      模型拒答了，但 gold 不是否定型
    建库丢失      gold 实词在记忆库里的覆盖率 < 0.5
    检索未召回    库里有，但检索到的证据覆盖率 < 0.5
    生成未用上    证据够了，答案覆盖率 < 0.5，且没有拒答
    部分漏项      0.5 <= 答案覆盖 < 0.9（多值题只答了一部分）
    答案冗余      答案覆盖 >= 0.9 但本地 F1 < 0.7（多说了不该说的）
    未归类        其余

判定顺序很重要：先判拒答/否定型，再看漏斗，最后看完整性。

用法：
    python experiments/diagnose_wrong_answers.py
    python experiments/diagnose_wrong_answers.py --run result/benchmark100/20261020_161439_654687
    python experiments/diagnose_wrong_answers.py --cause 检索未召回 --detail 15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("memnexus", "simplemem")
LABEL = {"memnexus": "MN", "simplemem": "SM"}
CATS = ("multi-hop", "temporal", "open-domain", "single-hop", "adversarial")
CAUSES = ("对抗误答", "过度拒答", "建库丢失", "检索未召回", "生成未用上",
          "部分漏项", "答案冗余", "未归类")

STOP = set("""
a an the is are was were be been being am do does did doing done has have had having
of in on at to for with by from and or but if then than that this these those it its
as not no so such what when where who whom which whose how why i you he she they we
me my your his her their our them us can could would should may might must will shall
about into over under more most other some any all each both few many much very also
just only own same too s t now there here out up down off again further once
""".split())

ABSTAIN = ("not mentioned", "no information", "not stated", "cannot be determined",
           "unable to", "no mention", "does not contain", "insufficient",
           "not provided", "not specified", "unknown", "n't mentioned")
NEGATIVE_GOLD = ("not mentioned", "no information", "not stated", "no mention",
                 "does not", "did not", "never", "none of")


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [w for w in words if w not in STOP and len(w) >= 3]


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def cov(text: str, hay: str) -> float:
    words = content_words(text)
    if not words:
        return 0.0
    hit = sum(1 for w in words if w[:5] in hay)
    return hit / len(words)


def missing_words(text: str, hay: str) -> list[str]:
    return [w for w in content_words(text) if w[:5] not in hay]


def is_abstain(pred: str) -> bool:
    p = (pred or "").lower()
    return (not p.strip()) or any(m in p for m in ABSTAIN)


def is_negative_gold(gold: str) -> bool:
    g = (gold or "").lower()
    return any(m in g for m in NEGATIVE_GOLD)


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[row.get("question")] = row   # 续跑时后者覆盖前者
    return out


def sources_text(row: dict) -> str:
    parts: list[str] = []
    for src in row.get("retrieved_sources") or []:
        if isinstance(src, str):
            parts.append(src)
        elif isinstance(src, dict):
            parts.append(str(src.get("content") or ""))
            meta = src.get("metadata") or {}
            for key in ("subject", "predicate", "value"):
                if meta.get(key):
                    parts.append(str(meta[key]))
            for w in meta.get("keywords") or []:
                parts.append(str(w))
    return norm(" ".join(parts))


def memory_text(unit: Path) -> str:
    data = read_json(unit / "usage.memory.json")
    parts: list[str] = []
    for item in data.get("memories") or []:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("content", "lossless_restatement", "subject", "predicate", "value", "summary"):
            if isinstance(item.get(key), str):
                parts.append(item[key])
        for w in item.get("keywords") or []:
            parts.append(str(w))
    return norm(" ".join(parts))


def local_correct(row: dict) -> bool | None:
    """只用官方判分。本地 F1 不能代替对错判断——
    LoCoMo 官方 F1 按词重叠算，短答案会被严重惩罚（答 "5 years" 完全正确但 F1 只有 0.33），
    拿 F1 当阈值会把大量正确题误判成错题。没有判分就返回 None。"""
    verdict = row.get("judge_result")
    if verdict == "CORRECT":
        return True
    if verdict == "INCORRECT":
        return False
    return None


def classify(row: dict, lib: str, gold: str, judged: bool) -> tuple[str, dict]:
    pred = str(row.get("prediction") or "")
    src = sources_text(row)
    lib_c = cov(gold, lib)
    src_c = cov(gold, src)
    pred_c = cov(gold, norm(pred))   # 必须归一化：'Ned' 与 'ned' 不做小写化会假性 0 覆盖
    f1 = row.get("f1") if isinstance(row.get("f1"), (int, float)) else 0.0
    info = {"lib": lib_c, "src": src_c, "pred": pred_c,
            "missing": missing_words(gold, norm(pred))[:6]}

    if is_negative_gold(gold):
        return ("对抗误答" if not is_abstain(pred) else "未归类"), info
    if is_abstain(pred):
        return "过度拒答", info
    if lib_c < 0.5:
        return "建库丢失", info
    if src_c < 0.5:
        return "检索未召回", info
    # 阈值 0.3 而非 0.5：短答案（"5 years" vs "Mel and her husband have been married for 5 years"）
    # 天然覆盖率低，用 0.5 会把正确答案误判成"没答出"。
    if pred_c < 0.3:
        return "生成未用上", info
    if pred_c < 0.9:
        return "部分漏项", info
    # 「答案冗余」依赖 F1，只有拿到官方判分才有意义
    if judged and f1 < 0.7:
        return "答案冗余", info
    return "未归类", info


def latest_run() -> Path | None:
    roots = (ROOT / "result" / "benchmark100", ROOT / "result" / "benchmark100_paper")
    runs: list[Path] = []
    for root in roots:
        if root.is_dir():
            runs.extend(p.parent for p in root.glob("*/run.json"))
    return max(runs, key=lambda p: p.name) if runs else None


def collect(folder: Path) -> tuple[dict, list[dict], dict]:
    report = read_json(folder / "run.json")
    # 判分结果在 attempts/judge_*/judgment.json，不在 usage.rows.jsonl 里。
    # 用 (sample_id, system, prediction_sha256) 关联，避免依赖 qa_index 字段是否存在。
    verdicts: dict[tuple, str] = {}
    for path in (folder / "attempts").glob("judge_*/judgment.json"):
        verdict = read_json(path)
        key = (verdict.get("sample_id"), verdict.get("system"), verdict.get("prediction_sha256"))
        if all(key):
            verdicts[key] = str(verdict.get("judge_result") or "")

    units: dict[tuple[str, str], Path] = {}
    for d in sorted((folder / "attempts").iterdir()):
        if not d.is_dir() or d.name.startswith("judge_"):
            continue
        m = re.match(r"(.+)_(memnexus|simplemem)__\d+$", d.name)
        if m and (d / "usage.rows.jsonl").exists():
            key = (m.group(1), m.group(2))
            # 续跑会产生 __002，取最后完成的那个
            if key not in units or (d / "stage_result.json").exists():
                units[key] = d
    records: list[dict] = []
    lib_cache: dict[tuple[str, str], str] = {}
    for (session, system), d in units.items():
        lib = lib_cache.setdefault((session, system), memory_text(d))
        for q, row in read_rows(d / "usage.rows.jsonl").items():
            gold = str(row.get("gold_answer") or "")
            prediction = str(row.get("prediction") or "")
            digest = hashlib.sha256(prediction.encode()).hexdigest()
            label = verdicts.get((row.get("sample_id"), row.get("system"), digest))
            judged = label in ("CORRECT", "INCORRECT", "WRONG")
            cause, info = classify(row, lib, gold, judged)
            records.append({
                "system": system, "session": session,
                "category": row.get("category") or "other",
                "question": q, "gold": gold,
                "prediction": prediction,
                "f1": row.get("f1"),
                "correct": (None if label is None
                            else label in ("CORRECT", "correct", "True")),
                "cause": cause, **info,
                "n_sources": len(row.get("retrieved_sources") or []),
                "cost": float((row.get("model_calls") or {}).get("cost_rmb") or 0),
            })
    return report, records, lib_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--cause", default=None, help="只看某一类错因")
    parser.add_argument("--detail", type=int, default=10)
    args = parser.parse_args()

    folder = args.run or latest_run()
    if folder is None:
        print("找不到运行目录")
        return 2
    report, records, _ = collect(folder)
    by_sys = {s: {r["question"]: r for r in records if r["system"] == s} for s in SYSTEMS}
    common = sorted(set(by_sys["memnexus"]) & set(by_sys["simplemem"]))
    total = len(records)
    has_official = any(r.get("correct") is not None for r in records)

    print(f"运行 {folder.name}   记录 {total} 条   两系统共同题 {len(common)}")
    if has_official:
        print("判分：官方 judge_result（逐题 LLM 判定）")
    else:
        print("判分：尚未产出。正确率与交叉表不可用；")
        print("      下面只给【结构性诊断】——哪些题在 库/检索/答案 任一段断了，与对错无关。")
        print("      ⚠ 本地 F1 不能代替判分：官方 F1 按词重叠算，答 '5 years' 完全正确但 F1 仅 0.33。")
    print("=" * 92)

    # 1. 总体
    print("\n【1】总体")
    header = f"{'系统':6s}{'题数':>5s}{'已判':>5s}{'判对':>5s}{'正确率':>8s}{'均F1':>8s}{'均证据':>7s}{'均成本¥':>9s}"
    print(header)
    for sysname in SYSTEMS:
        rows = [r for r in records if r["system"] == sysname]
        judged_rows = [r for r in rows if r["correct"] is not None]
        f1s = [r["f1"] for r in rows if isinstance(r["f1"], (int, float))]
        ok = sum(1 for r in judged_rows if r["correct"])
        rate = (ok / len(judged_rows)) if judged_rows else float("nan")
        print(f"{LABEL[sysname]:6s}{len(rows):5d}{len(judged_rows):5d}{ok:5d}{rate:8.1%}"
              f"{statistics.mean(f1s):8.3f}{statistics.mean(r['n_sources'] for r in rows):7.1f}"
              f"{sum(r['cost'] for r in rows)/len(rows):9.4f}")

    # 2. 交叉表（需要判分）
    if has_official:
        print("\n【2】逐题交叉（共同题）")
        cross = Counter()
        paired = 0
        for q in common:
            m, n = by_sys["memnexus"][q]["correct"], by_sys["simplemem"][q]["correct"]
            if m is None or n is None:
                continue
            cross[(bool(m), bool(n))] += 1
            paired += 1
        both_ok, mn_only = cross[(True, True)], cross[(True, False)]
        sm_only, both_bad = cross[(False, True)], cross[(False, False)]
        print(f"  双方都已判 {paired} 题：都对 {both_ok}   MN 独对 {mn_only}   SM 独对 {sm_only}   都错 {both_bad}")
        print(f"  → MN 净胜 {mn_only - sm_only:+d}   一致率 {(both_ok + both_bad) / max(paired, 1):.1%}")
    else:
        print("\n【2】逐题交叉：待官方判分")

    # 3. 错因/结构诊断分布
    title = "【3】MemNexus 错因分布（只统计判错的题）" if has_official else "【3】MemNexus 结构诊断（全部题，与对错无关）"
    print(f"\n{title}")
    pool = ([r for r in records if r["system"] == "memnexus" and r["correct"] is False]
            if has_official else [r for r in records if r["system"] == "memnexus"])
    dist = Counter(r["cause"] for r in pool)
    print(f"{'类别':12s}{'题数':>5s}{'占比':>7s}  说明")
    notes = {
        "对抗误答": "该拒没拒，捏造答案",
        "过度拒答": "证据不足时拒答，但 gold 有答案",
        "建库丢失": "记忆库里就没有关键实体",
        "检索未召回": "库里有，检索没取到",
        "生成未用上": "证据在手，答案没用上",
        "部分漏项": "多值题只答了一部分",
        "答案冗余": "覆盖够了但多说了，被 F1 罚",
        "未归类": "其余",
    }
    for cause in CAUSES:
        n = dist.get(cause, 0)
        if not n:
            continue
        print(f"{cause:12s}{n:5d}{n/max(len(pool),1):7.1%}  {notes[cause]}")

    # 4. 错因 × 题型
    print("\n【4】错因 × 题型（MN）")
    print(f"{'错因':12s}" + "".join(f"{c:>13s}" for c in CATS))
    for cause in CAUSES:
        cells = []
        for cat in CATS:
            n = sum(1 for r in pool if r["cause"] == cause and r["category"] == cat)
            cells.append(f"{(n if n else '-'):>13}")
        if any(c.strip() for c in cells):
            print(f"{cause:12s}" + "".join(cells))

    # 5. MN 错 & SM 对：机制差异
    print("\n【5】MN 错 / SM 对 的题：两边差在哪")
    pairs = ([(by_sys["memnexus"][q], by_sys["simplemem"][q]) for q in common
              if not by_sys["memnexus"][q]["correct"] and by_sys["simplemem"][q]["correct"]]
             if has_official else [])
    if not pairs:
        print("  （无）")
    else:
        print(f"{'题型':12s}{'错因':11s}{'MN检索':>7s}{'SM检索':>7s}{'MN答':>6s}{'SM答':>6s}")
        for m, s in sorted(pairs, key=lambda x: x[0]["category"]):
            print(f"{m['category']:12s}{m['cause']:11s}{m['src']:7.0%}{s['src']:7.0%}"
                  f"{m['pred']:6.0%}{s['pred']:6.0%}")
        print("\n  逐题：")
        for m, s in pairs[: args.detail]:
            print(f"\n  [{m['category']}/{m['cause']}] {m['question'][:82]}")
            print(f"    金: {m['gold'][:88]}")
            print(f"    MN: {m['prediction'][:88]}   (检索{m['src']:.0%} 答{m['pred']:.0%} F1={m['f1']:.2f})")
            print(f"    SM: {s['prediction'][:88]}   (检索{s['src']:.0%} 答{s['pred']:.0%} F1={s['f1']:.2f})")
            if m["missing"]:
                print(f"    MN 漏: {', '.join(m['missing'])}")

    # 6. 指定错因明细
    if args.cause:
        print(f"\n【6】错因「{args.cause}」明细")
        subset = [r for r in pool if r["cause"] == args.cause][: args.detail]
        for r in subset:
            print(f"\n  [{r['category']}] {r['question'][:86]}")
            print(f"    金: {r['gold'][:90]}")
            print(f"    答: {r['prediction'][:90]}")
            print(f"    库{r['lib']:.0%} → 检索{r['src']:.0%} → 答{r['pred']:.0%}  F1={r['f1']:.2f}  证据{r['n_sources']}条")
            if r["missing"]:
                print(f"    漏: {', '.join(r['missing'])}")

    # 7. SM 错 MN 对：MN 的优势题型
    print("\n【7】SM 错 / MN 对（MN 的优势）")
    adv = ([(by_sys["memnexus"][q], by_sys["simplemem"][q]) for q in common
            if by_sys["memnexus"][q]["correct"] and not by_sys["simplemem"][q]["correct"]]
           if has_official else [])
    if not adv:
        print("  （无）")
    else:
        print(f"  {len(adv)} 题，按题型：{dict(Counter(m['category'] for m, _ in adv))}")
        for m, s in adv[: min(5, args.detail)]:
            print(f"\n  [{m['category']}] {m['question'][:82]}")
            print(f"    MN: {m['prediction'][:80]}")
            print(f"    SM: {s['prediction'][:80]}  (SM 错因: {s['cause']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

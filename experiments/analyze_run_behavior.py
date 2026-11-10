"""观测运行中的行为：信息在哪一段丢失，钱花在哪一段。

只读分析，不 import 被测源码，运行期间可反复调用。

三段漏斗（对每道题，用 gold answer 的实词覆盖衡量）：
    库覆盖   gold 实词出现在该系统建库产出的记忆里
    检索覆盖 gold 实词出现在该题实际检索到的证据里
    答案覆盖 gold 实词出现在最终答案里
落差的含义：
    库→检索 掉 = 检索问题（信息在库里，但没取出来）
    检索→答案 掉 = 生成/策略问题（证据在手，没用上）

用法：
    python experiments/analyze_run_behavior.py
    python experiments/analyze_run_behavior.py --run result/benchmark100/20261020_161439_654687
    python experiments/analyze_run_behavior.py --detail 20   # 打印前 20 道诊断题
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("memnexus", "simplemem")
LABEL = {"memnexus": "MemNexus", "simplemem": "SimpleMem"}
CATS = ("multi-hop", "temporal", "open-domain", "single-hop", "adversarial")

STOP = set("""
a an the is are was were be been being am do does did doing done has have had having
of in on at to for with by from and or but if then than that this these those it its
as not no so such what when where who whom which whose how why i you he she they we
me my your his her their our them us can could would should may might must will shall
about into over under more most other some any all each both few many much very also
just only own same too s t don now there here out up down off again further once
""".split())


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [w for w in words if w not in STOP and len(w) >= 3]


def covered(word: str, haystack: str) -> bool:
    """前 5 字符前缀匹配，容忍 kickboxing/kickbox、volunteers/volunteered 这类变形。"""
    return word[:5] in haystack


def coverage(text: str, haystack: str) -> tuple[float, list[str]]:
    words = content_words(text)
    if not words:
        return (0.0, [])
    missing = [w for w in words if not covered(w, haystack)]
    return ((len(words) - len(missing)) / len(words), missing)


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: dict[tuple, dict] = {}
    order: list[tuple] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("question")
        if key not in out:
            order.append(key)
        out[key] = row
    return [out[k] for k in order]


def latest_run() -> Path | None:
    roots = (ROOT / "result" / "benchmark100", ROOT / "result" / "benchmark100_paper")
    runs: list[Path] = []
    for root in roots:
        if root.is_dir():
            runs.extend(p.parent for p in root.glob("*/run.json"))
    return max(runs, key=lambda p: p.name) if runs else None


def sources_text(row: dict) -> str:
    """把检索到的证据拼成一整段可做前缀匹配的文本。"""
    parts: list[str] = []
    for src in row.get("retrieved_sources") or []:
        if isinstance(src, str):
            parts.append(src)
        elif isinstance(src, dict):
            parts.append(str(src.get("content") or ""))
            meta = src.get("metadata") or {}
            for key in ("subject", "predicate", "value"):
                value = meta.get(key)
                if value:
                    parts.append(str(value))
            for word in meta.get("keywords") or []:
                parts.append(str(word))
    return norm(" ".join(parts))


def memory_text(unit_dir: Path) -> str:
    data = read_json(unit_dir / "usage.memory.json")
    memories = data.get("memories") or []
    parts: list[str] = []
    for item in memories:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("content", "lossless_restatement", "subject", "predicate", "value", "summary"):
            value = item.get(key)
            if isinstance(value, str):
                parts.append(value)
        for word in item.get("keywords") or []:
            parts.append(str(word))
    return norm(" ".join(parts))


def is_abstain(prediction: str) -> bool:
    p = (prediction or "").lower()
    if not p.strip():
        return True
    markers = ("not mentioned", "no information", "not stated", "cannot be determined",
               "unable to", "no mention", "does not contain", "insufficient",
               "not provided", "not specified", "unknown")
    return any(m in p for m in markers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--detail", type=int, default=12, help="打印多少道诊断题")
    args = parser.parse_args()

    folder = args.run or latest_run()
    if folder is None:
        print("找不到运行目录")
        return 2
    report = read_json(folder / "run.json")
    gold = {q["question"]: q.get("gold_answer") for q in (report.get("cohort") or {}).get("questions") or []}

    records: list[dict] = []
    cost = defaultdict(lambda: defaultdict(float))
    for unit_dir in sorted((folder / "attempts").glob("*")):
        if not unit_dir.is_dir() or unit_dir.name.startswith("judge_"):
            continue
        stem = unit_dir.name.split("__")[0]
        system = next((s for s in SYSTEMS if stem.endswith("_" + s)), None)
        if system is None:
            continue
        session = stem[: -(len(system) + 1)]
        rows = read_rows(unit_dir / "usage.rows.jsonl")
        if not rows:
            continue
        lib = memory_text(unit_dir)
        stage = read_json(unit_dir / "usage.stage.json")
        build = stage.get("build_usage") or {}
        total = stage.get("total_usage") or {}
        cost[system]["build"] += float(build.get("cost_rmb") or 0)
        cost[system]["total"] += float(total.get("cost_rmb") or 0)
        cost[system]["build_in"] += float(build.get("input_tokens") or 0)
        cost[system]["build_out"] += float(build.get("output_tokens") or 0)
        cost[system]["total_in"] += float(total.get("input_tokens") or 0)
        cost[system]["total_out"] += float(total.get("output_tokens") or 0)

        for row in rows:
            answer = str(row.get("gold_answer") or gold.get(row.get("question")) or "")
            prediction = str(row.get("prediction") or "")
            src = sources_text(row)
            lib_cov, _ = coverage(answer, lib)
            src_cov, _ = coverage(answer, src)
            pred_cov, missing = coverage(answer, norm(prediction))
            calls = row.get("model_calls") or {}
            records.append({
                "system": system,
                "session": session,
                "category": row.get("category") or "other",
                "question": row.get("question"),
                "gold": answer,
                "prediction": prediction,
                "lib_cov": lib_cov,
                "src_cov": src_cov,
                "pred_cov": pred_cov,
                "missing": missing,
                "f1": row.get("f1"),
                "abstain": is_abstain(prediction),
                "n_sources": len(row.get("retrieved_sources") or []),
                "pred_len": len(prediction.split()),
                "tokens_in": float(calls.get("input_tokens") or 0),
                "tokens_out": float(calls.get("output_tokens") or 0),
                "cost": float(calls.get("cost_rmb") or 0),
                "latency": float(row.get("latency_ms") or 0) / 1000,
                "plan": row.get("plan") or {},
                "warnings": row.get("warnings") or [],
            })

    if not records:
        print("没有可用记录")
        return 2

    print(f"运行 {folder.name}   已完成 {len(records)} 条（{len(records)//len(SYSTEMS)} 题 × {len(SYSTEMS)} 系统）")
    print("=" * 88)

    # 1. 三段漏斗
    print("\n【1】信息漏斗：gold 答案实词在各段的覆盖率（越高越好）")
    print(f"{'系统':11s}{'条目':>5s}{'库覆盖':>8s}{'检索覆盖':>9s}{'答案覆盖':>9s}"
          f"{'库→检索损失':>12s}{'检索→答案损失':>13s}")
    funnel = {}
    for system in SYSTEMS:
        sub = [r for r in records if r["system"] == system]
        if not sub:
            continue
        lib = statistics.mean(r["lib_cov"] for r in sub)
        src = statistics.mean(r["src_cov"] for r in sub)
        pred = statistics.mean(r["pred_cov"] for r in sub)
        funnel[system] = (lib, src, pred)
        print(f"{LABEL[system]:11s}{len(sub):5d}{lib:8.1%}{src:9.1%}{pred:9.1%}"
              f"{lib-src:+12.1%}{src-pred:+13.1%}")

    # 2. 按题型
    print("\n【2】按题型：检索覆盖 → 答案覆盖（差值大 = 证据在手却没用上）")
    print(f"{'题型':12s}" + "".join(f"{LABEL[s]:>24s}" for s in SYSTEMS))
    for cat in CATS:
        cells = [f"{cat:12s}"]
        for system in SYSTEMS:
            sub = [r for r in records if r["system"] == system and r["category"] == cat]
            if not sub:
                cells.append(f"{'-':>24s}")
                continue
            src = statistics.mean(r["src_cov"] for r in sub)
            pred = statistics.mean(r["pred_cov"] for r in sub)
            f1 = statistics.mean([r["f1"] for r in sub if isinstance(r["f1"], (int, float))] or [0])
            cells.append(f"n={len(sub):2d} 检索{src:5.1%}→答{pred:5.1%} F1{f1:.2f}")
        print("".join(cells))

    # 3. 拒答与答案形态
    print("\n【3】答案形态（拒答是对抗题的策略，也是开放域的主要失分来源）")
    print(f"{'系统':11s}{'拒答率':>8s}{'平均词数':>9s}{'平均证据数':>10s}{'空答案':>7s}")
    for system in SYSTEMS:
        sub = [r for r in records if r["system"] == system]
        if not sub:
            continue
        abstain = sum(1 for r in sub if r["abstain"]) / len(sub)
        words = statistics.mean(r["pred_len"] for r in sub)
        srcs = statistics.mean(r["n_sources"] for r in sub)
        empty = sum(1 for r in sub if not r["prediction"].strip())
        print(f"{LABEL[system]:11s}{abstain:8.1%}{words:9.1f}{srcs:10.1f}{empty:7d}")

    print("\n    拒答 × 题型：")
    for system in SYSTEMS:
        row = []
        for cat in CATS:
            sub = [r for r in records if r["system"] == system and r["category"] == cat]
            if sub:
                row.append(f"{cat} {sum(1 for r in sub if r['abstain'])}/{len(sub)}")
        print(f"      {LABEL[system]:10s}" + "   ".join(row))

    # 4. 成本结构
    print("\n【4】成本结构：建库 vs 答题")
    print(f"{'系统':11s}{'建库¥':>8s}{'答题¥':>8s}{'合计¥':>8s}{'建库占比':>9s}"
          f"{'建库token':>18s}{'答题token':>18s}")
    for system in SYSTEMS:
        c = cost[system]
        if not c["total"]:
            continue
        build = c["build"]
        ask = max(c["total"] - build, 0)
        share = build / c["total"] if c["total"] else 0
        print(f"{LABEL[system]:11s}{build:8.3f}{ask:8.3f}{c['total']:8.3f}{share:9.0%}"
              f"{int(c['build_in']):9d}/{int(c['build_out']):<8d}{int(c['total_in']-c['build_in']):9d}/{int(c['total_out']-c['build_out']):<8d}")

    # 5. 每题 token 与延迟
    print("\n【5】每题成本与延迟（答题阶段，不含建库）")
    print(f"{'系统':11s}{'输入tok':>9s}{'输出tok':>9s}{'¥/题':>8s}{'延迟s':>7s}{'每题调用':>8s}")
    for system in SYSTEMS:
        sub = [r for r in records if r["system"] == system]
        if not sub:
            continue
        print(f"{LABEL[system]:11s}{int(statistics.mean(r['tokens_in'] for r in sub)):9d}"
              f"{int(statistics.mean(r['tokens_out'] for r in sub)):9d}"
              f"{statistics.mean(r['cost'] for r in sub):8.4f}"
              f"{statistics.mean(r['latency'] for r in sub):7.1f}"
              f"{'':8s}")

    # 6. MemNexus 检索侧：证据条数 vs 答案覆盖
    print("\n【6】MemNexus 检索条数与效果（条数多不等于覆盖高）")
    mn = [r for r in records if r["system"] == "memnexus"]
    if mn:
        buckets = defaultdict(list)
        for r in mn:
            n = r["n_sources"]
            key = "≤8" if n <= 8 else "9-16" if n <= 16 else "17-24" if n <= 24 else ">24"
            buckets[key].append(r)
        print(f"{'证据条数':10s}{'条目':>5s}{'检索覆盖':>10s}{'答案覆盖':>10s}{'F1':>7s}")
        for key in ("≤8", "9-16", "17-24", ">24"):
            sub = buckets.get(key) or []
            if not sub:
                continue
            print(f"{key:10s}{len(sub):5d}{statistics.mean(r['src_cov'] for r in sub):10.1%}"
                  f"{statistics.mean(r['pred_cov'] for r in sub):10.1%}"
                  f"{statistics.mean([r['f1'] for r in sub if isinstance(r['f1'], (int, float))] or [0]):7.2f}")

    # 7. 诊断题：检索覆盖高但答案覆盖低
    print(f"\n【7】诊断：证据里有关键信息但没答出来（前 {args.detail} 条，按落差排序）")
    gap = [r for r in records if r["src_cov"] - r["pred_cov"] > 0.3]
    gap.sort(key=lambda r: r["src_cov"] - r["pred_cov"], reverse=True)
    for r in gap[: args.detail]:
        print(f"  [{LABEL[r['system']]}/{r['category']}] 检索{r['src_cov']:.0%}→答{r['pred_cov']:.0%} F1={r['f1'] if r['f1'] is None else round(r['f1'],2)}")
        print(f"    问: {r['question'][:88]}")
        print(f"    金: {r['gold'][:80]}")
        print(f"    答: {r['prediction'][:80]}")
        if r["missing"]:
            print(f"    漏: {', '.join(r['missing'][:8])}")

    # 8. 诊断题：检索就没覆盖
    print(f"\n【8】诊断：检索阶段就没拿到关键信息（前 {args.detail} 条）")
    miss = [r for r in records if r["src_cov"] < 0.4]
    miss.sort(key=lambda r: r["src_cov"])
    for r in miss[: args.detail]:
        print(f"  [{LABEL[r['system']]}/{r['category']}] 库{r['lib_cov']:.0%} 检索{r['src_cov']:.0%} 答{r['pred_cov']:.0%} 证据{r['n_sources']}条")
        print(f"    问: {r['question'][:88]}")
        print(f"    金: {r['gold'][:80]}")
        print(f"    答: {r['prediction'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

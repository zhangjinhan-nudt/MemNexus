"""只读进度看板：LoCoMo 基准跑到第几题、各会话进度、实时 F1 与成本结构。

设计约束（重要）：
- 纯只读。不修改任何文件，不 import 被测源码，因此可以在运行期间反复调用，
  也不会触发 benchmark100.py 里的 provenance 守卫（该守卫只盯 src 与指定的几个脚本）。
- 数据源全部是运行器自己落盘的产物：
    run.json                     队列、active_unit、预算、签名
    attempts/*/usage.rows.jsonl   逐题：category / f1 / prediction / latency
    attempts/*/usage.stage.json   分期账本：build_usage（建库）/ total_usage（建库+答题）
- 判分（judge_result）是整轮结束后统一做的，运行中一律为 null，
  所以这里只显示本地可算的 F1，不显示正确率。

用法：
    python experiments/watch_progress.py                  # 看最新一轮，输出一次
    python experiments/watch_progress.py --watch          # 每 15 秒刷新
    python experiments/watch_progress.py --run result/benchmark100/20261020_161439_654687
    python experiments/watch_progress.py --interval 5 --watch
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = (ROOT / "result" / "benchmark100", ROOT / "result" / "benchmark100_paper")
SYSTEMS = ("memnexus", "simplemem")
SYSTEM_LABEL = {"memnexus": "MemNexus", "simplemem": "SimpleMem"}
CATEGORY_ORDER = ("multi-hop", "temporal", "open-domain", "single-hop", "adversarial")

# 唯一评测标准是 SimpleMem 论文口径（experiments/paper_protocol.py）。
# 早期 runner 写入的 row["f1"] 只是简单分词重叠，不是官方 F1，看板不得使用。
try:
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paper_protocol import f1_of as _official_f1
except Exception:  # pragma: no cover - 仅在 paper_protocol 缺失时降级
    _official_f1 = None


def score_of(row: dict) -> float | None:
    """该题的官方 LoCoMo F1；无法计算时返回 None（不回退到诊断字段）。"""
    if _official_f1 is None or row.get("category_id") is None:
        return None
    try:
        return _official_f1(row)
    except Exception:
        return None


def latest_run() -> Path | None:
    """按目录名（时间戳）挑最新一轮。"""
    candidates: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        candidates.extend(p.parent for p in root.glob("*/run.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_rows(path: Path) -> list[dict]:
    """逐题记录。续跑会在同一文件追加，按 (system, sample_id, question) 去重保留最后一条。"""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows: dict[tuple, dict] = {}
    order: list[tuple] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (row.get("system"), row.get("sample_id"), row.get("question"))
        if key not in rows:
            order.append(key)
        rows[key] = row
    return [rows[k] for k in order]


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def elapsed_text(started: datetime | None) -> str:
    if started is None:
        return "未知"
    now = datetime.now(started.tzinfo) if started.tzinfo else datetime.now()
    delta = (now - started).total_seconds()
    if delta < 0:
        return "未知"
    minutes, seconds = divmod(int(delta), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "░" * width
    filled = max(0, min(width, int(round(width * done / total))))
    return "█" * filled + "░" * (width - filled)


def num(block: dict, key: str) -> float:
    value = block.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def collect(folder: Path) -> dict:
    report = read_json(folder / "run.json")
    cohort = (report.get("cohort") or {}).get("questions") or []
    per_session: Counter[str] = Counter(q.get("sample_id") for q in cohort)
    sessions = list(dict.fromkeys(q.get("sample_id") for q in cohort))
    total_questions = sum(per_session.values())

    per_unit: dict[tuple[str, str], dict] = {}
    for attempt_dir in sorted((folder / "attempts").glob("*")):
        if not attempt_dir.is_dir():
            continue
        stem = attempt_dir.name.split("__")[0]
        system = next((s for s in SYSTEMS if stem.endswith("_" + s)), None)
        if system is None:
            continue
        session = stem[: -(len(system) + 1)]
        rows = read_rows(attempt_dir / "usage.rows.jsonl")
        stage = read_json(attempt_dir / "usage.stage.json")
        build = stage.get("build_usage") or {}
        total = stage.get("total_usage") or {}
        f1_values = [score_of(r) for r in rows if score_of(r) is not None]
        per_unit[(session, system)] = {
            "rows": rows,
            "answered": len(rows),
            "f1_mean": (sum(f1_values) / len(f1_values)) if f1_values else None,
            "build_cost": num(build, "cost_rmb"),
            "total_cost": num(total, "cost_rmb"),
            "build_in": num(build, "input_tokens"),
            "build_out": num(build, "output_tokens"),
            "total_in": num(total, "input_tokens"),
            "total_out": num(total, "output_tokens"),
            "build_ms": stage.get("build_ms") or 0,
            "memory_count": stage.get("memory_count"),
            "replayed": bool(stage.get("memory_base_replayed")),
            "provider_errors": int(num(total, "provider_errors")),
        }

    answered_total = sum(u["answered"] for u in per_unit.values())
    f1_all = [s for u in per_unit.values() for s in (score_of(r) for r in u["rows"]) if s is not None]

    active = report.get("active_unit") or ""
    active_session = active_system = ""
    for system in SYSTEMS:
        if active.endswith("_" + system):
            active_session = active[: -(len(system) + 1)]
            active_system = system
            break
    active_done = len((per_unit.get((active_session, active_system)) or {}).get("rows") or [])
    active_total = per_session.get(active_session, 0) if active_session else 0

    return {
        "folder": folder,
        "report": report,
        "per_session": per_session,
        "sessions": sessions,
        "total_questions": total_questions,
        "per_unit": per_unit,
        "answered_total": answered_total,
        "expected_total": total_questions * len(SYSTEMS),
        "f1_all": f1_all,
        "active": active,
        "active_session": active_session,
        "active_system": active_system,
        "active_done": active_done,
        "active_total": active_total,
        "judged": len(report.get("judgments") or {}),
    }


def cost_by_system(state: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for system in SYSTEMS:
        units = [u for (_, s), u in state["per_unit"].items() if s == system]
        out[system] = {
            "build": sum(u["build_cost"] for u in units),
            "total": sum(u["total_cost"] for u in units),
            "build_in": sum(u["build_in"] for u in units),
            "build_out": sum(u["build_out"] for u in units),
            "total_in": sum(u["total_in"] for u in units),
            "total_out": sum(u["total_out"] for u in units),
            "asked": sum(u["answered"] for u in units),
        }
    return out


def render(state: dict, width: int = 78) -> str:
    folder = state["folder"]
    report = state["report"]
    out: list[str] = []
    rule = "─" * width

    started = parse_iso(report.get("started_at"))
    settings = ((report.get("signature") or {}).get("settings")) or {}
    out.append(f"运行 {folder.name}   状态={report.get('status') or '?'}   已用时 {elapsed_text(started)}")
    out.append(f"配置 模型={settings.get('model') or '?'}   语义执行={settings.get('semantic_execution')}"
               f"   预算 ¥{report.get('budget_rmb')}")
    out.append(rule)

    answered, expected = state["answered_total"], state["expected_total"]
    pct = (answered / expected * 100) if expected else 0.0
    out.append(f"总进度  {bar(answered, expected)}  {answered}/{expected} 题  {pct:5.1f}%")

    if state["active_session"]:
        out.append(f"当前    {state['active']}  第 {state['active_done'] + 1}/{state['active_total']} 题"
                   f"  （已完成 {state['active_done']} 题，正在答下一题）")
    else:
        out.append("当前    无活动单元（可能已结束）")

    f1 = state["f1_all"]
    f1_text = f"{sum(f1) / len(f1):.3f}" if f1 else "—"
    out.append(f"实时官方F1 {f1_text}（论文口径逐题均值，{len(f1)} 题；最终以四类等权 Average 为准）"
               + (f"   已判分 {state['judged']} 题" if state["judged"] else "   正确率待整轮结束统一判分"))
    out.append("")

    header = f"{'会话':9s}{'题数':>4s} │ {'MemNexus':22s}│ {'SimpleMem':22s}│"
    out.append(header)
    out.append(rule)
    for session in state["sessions"]:
        total = state["per_session"][session]
        cells = [f"{session:9s}{total:4d} │"]
        for system in SYSTEMS:
            unit = state["per_unit"].get((session, system))
            if not unit:
                cells.append(f"{'':<24s}│")
                continue
            done = unit["answered"]
            mark = "●" if (session == state["active_session"] and system == state["active_system"]) else " "
            f1_cell = f"{unit['f1_mean']:.2f}" if unit["f1_mean"] is not None else "  - "
            left = f"{bar(done, total, 8)} {done:2d}/{total:<2d} {f1_cell:>4s}{mark}"
            cells.append(f"{left}{' ' * max(24 - len(left), 0)}│")
        out.append(" ".join(cells))
    out.append(rule)

    # 成本结构：建库 vs 答题
    costs = cost_by_system(state)
    out.append(f"{'成本结构':10s}{'建库':>9s}{'答题':>9s}{'合计':>9s}{'建库占比':>9s}   {'每题均摊':>9s}")
    for system in SYSTEMS:
        c = costs[system]
        ask = max(c["total"] - c["build"], 0.0)
        share = (c["build"] / c["total"] * 100) if c["total"] else 0.0
        per_q = (c["total"] / c["asked"]) if c["asked"] else 0.0
        out.append(f"{SYSTEM_LABEL[system]:10s}{c['build']:9.3f}{ask:9.3f}{c['total']:9.3f}"
                   f"{share:8.0f}%   {per_q:9.3f}")
    both = sum(c["total"] for c in costs.values())
    out.append(f"{'合计':10s}{sum(c['build'] for c in costs.values()):9.3f}"
               f"{both - sum(c['build'] for c in costs.values()):9.3f}{both:9.3f}")
    out.append("")

    # 题型进度
    cat_done: Counter[str] = Counter()
    cat_f1: dict[str, list[float]] = {}
    for unit in state["per_unit"].values():
        for row in unit["rows"]:
            cat = row.get("category") or "other"
            cat_done[cat] += 1
            score = score_of(row)
            if score is not None:
                cat_f1.setdefault(cat, []).append(score)
    if cat_done:
        parts = []
        for cat in CATEGORY_ORDER:
            if cat_done.get(cat):
                mean = sum(cat_f1[cat]) / len(cat_f1[cat]) if cat_f1.get(cat) else None
                tail = f"(F1 {mean:.2f})" if mean is not None else ""
                parts.append(f"{cat} {cat_done[cat]}{tail}")
        out.append("已完成题型（两系统合计）  " + "   ".join(parts))

    errors = sum(u["provider_errors"] for u in state["per_unit"].values())
    empty = sum(1 for u in state["per_unit"].values() for r in u["rows"]
                if not str(r.get("prediction") or "").strip())
    health = "✓ 无 provider 错误、无空预测" if not (errors or empty) else f"⚠ provider 错误 {errors} 次、空预测 {empty} 条"
    out.append(health)

    # 每会话建库概况
    built = [(sess, sys, u) for (sess, sys), u in state["per_unit"].items() if u["memory_count"]]
    if built:
        out.append("")
        out.append("建库概况")
        for sess, sys, u in sorted(built, key=lambda x: (x[0], x[1])):
            replay = "（回放，¥0）" if u["replayed"] else ""
            out.append(f"  {sess:9s}{SYSTEM_LABEL[sys]:10s} 记忆 {str(u['memory_count']):>4s} 条"
                       f"   建库 {u['build_ms'] / 1000:6.1f}s   ¥{u['build_cost']:.3f}{replay}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="只读进度看板")
    parser.add_argument("--run", type=Path, default=None, help="运行目录；默认取最新")
    parser.add_argument("--watch", action="store_true", help="持续刷新")
    parser.add_argument("--interval", type=float, default=15.0, help="刷新间隔秒，默认 15")
    args = parser.parse_args()

    folder = args.run or latest_run()
    if folder is None or not (folder / "run.json").exists():
        print("找不到运行目录。请用 --run 指定。", file=sys.stderr)
        return 2

    if not args.watch:
        print(render(collect(folder)))
        return 0

    try:
        while True:
            columns = shutil.get_terminal_size((90, 24)).columns
            frame = render(collect(folder), width=min(78, max(64, columns - 2)))
            if sys.platform == "win32":
                print("\033[H\033[J" + frame, flush=True)
            else:
                print("\033[H\033[2J" + frame, flush=True)
            print(f"\n（每 {args.interval:g} 秒刷新，Ctrl+C 退出）", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止刷新。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

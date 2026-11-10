"""子集 A/B 比较器：把 MemNexus 变体结果与母轮 SimpleMem 基线离线配对比较。

为什么需要它
------------
阶段 B 用 `run_parallel_bench.py --only-system MemNexus` 只重跑 MN 变体，
SimpleMem 行为完全不变，因此 SM 基线可直接复用母轮 `LOCOMO1000_FINAL` 的
已跑结果——每轮省一半成本与时间，并消除网关漂移带来的混杂。

但"复用 SM 基线"只有在**配对正确且基线未漂移**时才成立。本脚本负责：
  1. 从母轮 rows.json 抽取 SM 基线行，按 (sample_id, qa_index) 与变体 MN 行配对；
  2. 用与主表完全相同的 paper_protocol 口径重算 F1/BLEU-1/AdvSR；
  3. 复用 finalize_locomo1000 的 conversation-cluster bootstrap 出 CI；
  4. **基线漂移检测**：若传入 baseline MN 结果，逐题比对其官方 F1 与母轮是否一致，
     不一致就说明环境/快照/代码漂了，本轮变体结果不可信，直接 FAIL；
  5. 守护指标：AdvSR、accepted tokens/题，任一越界即标记。

口径纪律（与项目长期约束一致）
------------------------------
- 主指标 = 四类等权 Average F1（不含 adversarial），官方 F1 由 paper_protocol 提供。
- AdvSR 单独报告，是 P2 的一票否决项。
- 子集簇数远小于 10，cluster bootstrap 的 CI 会很宽甚至退化；本脚本**只把 CI 当
  方向性参考**，并在 n_clusters 过小时显式告警。子集阶段的目标是"筛掉无效变体"，
  不是"证明显著"；显著性只能由阶段 C 的全量 1000 题给出。
- pooled F1 只作诊断，不替代 Average。
- gross 与 accepted 永不相加。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import paper_protocol as pp  # noqa: E402
import finalize_locomo1000 as fin  # noqa: E402

TOLERANCE = fin.TOLERANCE
CATEGORIES = fin.CATEGORIES  # {1: MultiHop, 2: Temporal, 3: OpenDomain, 4: SingleHop, 5: adversarial}
ADVERSARIAL_ID = 5
# 子集阶段簇数下限：低于此值 cluster bootstrap 基本无意义，只报点估计。
MIN_CLUSTERS_FOR_CI = 6


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "rows" in payload:
        payload = payload["rows"]
    if not isinstance(payload, list):
        raise ValueError(f"{path} 应为行列表或含 rows 的对象")
    return [row for row in payload if isinstance(row, dict)]


def index_by_system(rows: list[dict[str, Any]], system: str) -> dict[tuple[Any, Any], dict[str, Any]]:
    index: dict[tuple[Any, Any], dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        if row.get("system") != system:
            continue
        if not fin.valid_answer(row):
            continue  # 空预测或失败答案不参与配对
        key = fin.item_key(row)
        if key in index:
            duplicates.append(fin.key_text(key))
        index[key] = row
    if duplicates:
        raise ValueError(f"{system} 存在重复有效行，配对不可信：{duplicates[:5]}")
    return index


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    """用官方口径重算单行分数与成本，避免信任行内可能过期的 f1 字段。"""
    calls = row.get("model_calls") or {}
    return {
        "f1": pp.f1_of(row),
        "bleu1": pp.bleu1_of(row),
        "category": row.get("category"),
        "category_id": row.get("category_id"),
        "sample_id": row.get("sample_id"),
        "qa_index": row.get("qa_index"),
        "input_tokens": int(calls.get("input_tokens") or 0),
        "output_tokens": int(calls.get("output_tokens") or 0),
        "latency_ms": row.get("latency_ms"),
        "prediction": row.get("prediction"),
    }


def build_paired_items(variant_mn: list[dict[str, Any]],
                       baseline_sm: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把变体 MN 与母轮 SM 基线配对，返回 (逐题项, 配对诊断)。"""
    mn_index = index_by_system(variant_mn, "MemNexus")
    sm_index = index_by_system(baseline_sm, "SimpleMem")

    common = sorted(set(mn_index) & set(sm_index),
                    key=lambda key: (str(key[0]), key[1] if isinstance(key[1], int) else 0))
    only_mn = sorted(set(mn_index) - set(sm_index))
    only_sm = sorted(set(sm_index) - set(mn_index))

    items: list[dict[str, Any]] = []
    for key in common:
        mn_row = mn_index[key]
        sm_row = sm_index[key]
        mn = score_row(mn_row)
        sm = score_row(sm_row)
        # 配对必须同题同类，否则 category_id 错位会污染分类统计。
        if mn["category_id"] != sm["category_id"]:
            raise ValueError(f"{fin.key_text(key)} 两侧 category_id 不一致，配对无效")
        delta = mn["f1"] - sm["f1"]
        outcome = "win" if delta > TOLERANCE else "loss" if delta < -TOLERANCE else "tie"
        is_adv = mn["category_id"] == ADVERSARIAL_ID
        items.append({
            "key": fin.key_text(key), "sample_id": mn["sample_id"], "qa_index": mn["qa_index"],
            "category": mn["category"], "category_id": mn["category_id"],
            "mn_f1": mn["f1"], "sm_f1": sm["f1"], "f1_delta": delta,
            "mn_bleu1": mn["bleu1"], "sm_bleu1": sm["bleu1"],
            "outcome": outcome,
            # AdvSR 用官方拒答判定，只在对抗题有意义；配对存两侧，保证同题比较。
            "mn_adv_success": bool(pp.adversarial_success(mn_row)) if is_adv else None,
            "sm_adv_success": bool(pp.adversarial_success(sm_row)) if is_adv else None,
            "mn_input_tokens": mn["input_tokens"], "mn_output_tokens": mn["output_tokens"],
            "mn_latency_ms": mn["latency_ms"],
        })

    diagnostics = {
        "paired": len(items),
        "variant_mn_valid": len(mn_index),
        "baseline_sm_valid": len(sm_index),
        "unpaired_variant_only": len(only_mn),
        "unpaired_baseline_only": len(only_sm),
        "unpaired_examples": {"variant_only": [fin.key_text(k) for k in only_mn[:5]],
                              "baseline_only": [fin.key_text(k) for k in only_sm[:5]]},
    }
    return items, diagnostics


def category_means(items: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    """按四类分别求均值并算等权 Average（不含 adversarial），与主表同口径。"""
    by_cat: dict[int, list[float]] = {cid: [] for cid in range(1, 5)}
    for item in items:
        cid = item["category_id"]
        if cid in by_cat:
            by_cat[cid].append(item[field])
    means = {CATEGORIES[cid]: (statistics.fmean(values) if values else None)
             for cid, values in by_cat.items()}
    present = [value for value in means.values() if value is not None]
    means["Average"] = statistics.fmean(present) if present else None
    return means


def adversarial_sr(items: list[dict[str, Any]]) -> dict[str, Any]:
    """AdvSR 在配对对抗题上算，保证 MN 与 SM 是**同一批题**，基数一致。

    此前用原始行分别算，会出现"变体 73 题 vs 母轮 225 题"的基数不一致，
    delta 会误导。AdvSR 是 P2 的一票否决项，必须同题配对。
    """
    adv = [item for item in items if item["category_id"] == ADVERSARIAL_ID
           and item.get("mn_adv_success") is not None]
    if not adv:
        return {"n": 0, "variant_mn": {"adv_sr": None, "n": 0},
                "baseline_sm": {"adv_sr": None, "n": 0}, "delta_pp": None,
                "paired": True}
    mn_hits = sum(1 for item in adv if item["mn_adv_success"])
    sm_hits = sum(1 for item in adv if item["sm_adv_success"])
    mn_sr, sm_sr = mn_hits / len(adv), sm_hits / len(adv)
    return {
        "n": len(adv),
        "variant_mn": {"adv_sr": mn_sr, "n": len(adv)},
        "baseline_sm": {"adv_sr": sm_sr, "n": len(adv)},
        "delta_pp": (mn_sr - sm_sr) * 100,
        "paired": True,
    }


def cost_per_question(items: list[dict[str, Any]]) -> dict[str, Any]:
    mn_in = [item["mn_input_tokens"] for item in items if item["mn_input_tokens"]]
    mn_out = [item["mn_output_tokens"] for item in items if item["mn_output_tokens"] is not None]
    lat = [item["mn_latency_ms"] for item in items if item["mn_latency_ms"]]
    return {
        "mn_input_tokens_per_q": statistics.fmean(mn_in) if mn_in else None,
        "mn_output_tokens_per_q": statistics.fmean(mn_out) if mn_out else None,
        "mn_latency_ms_per_q": statistics.fmean(lat) if lat else None,
    }


def detect_drift(variant_rows: list[dict[str, Any]], parent_rows: list[dict[str, Any]],
                 tolerance: float = 1e-9) -> dict[str, Any]:
    """基线漂移检测：variant 若含 baseline 配置的 MN 行，其官方 F1 必须与母轮一致。

    用途：跑变体时通常会带一个 baseline 臂。若 baseline 臂的 MN F1 与母轮对不上，
    说明快照/代码/网关漂了，本轮所有变体结果都不可信。
    """
    parent_mn = {fin.item_key(row): row for row in parent_rows
                 if row.get("system") == "MemNexus" and fin.valid_answer(row)}
    variant_mn = {fin.item_key(row): row for row in variant_rows
                  if row.get("system") == "MemNexus" and fin.valid_answer(row)}
    shared = sorted(set(parent_mn) & set(variant_mn), key=lambda k: (str(k[0]), k[1] or 0))
    drifted: list[dict[str, Any]] = []
    for key in shared:
        parent_f1 = pp.f1_of(parent_mn[key])
        variant_f1 = pp.f1_of(variant_mn[key])
        if abs(parent_f1 - variant_f1) > tolerance:
            drifted.append({"key": fin.key_text(key), "parent_f1": parent_f1, "variant_f1": variant_f1})
    return {
        "shared_keys": len(shared),
        "drifted": len(drifted),
        "drift_examples": drifted[:10],
        "drift_detected": bool(drifted),
    }


def analyze(variant_path: Path, parent_rows_path: Path, sm_source: str,
            bootstrap: int, seed: int, adv_floor: float | None,
            token_ceiling_pct: float | None, drift_path: Path | None) -> dict[str, Any]:
    variant_rows = load_rows(variant_path)
    parent_rows = load_rows(parent_rows_path)

    # SM 基线：默认从母轮 rows.json 抽（--only-system 复用场景）；
    # 也允许从变体 run 自己的 rows.json 抽（双系统同轮场景）。
    sm_rows_source = parent_rows if sm_source == "parent" else variant_rows
    items, pair_diag = build_paired_items(variant_rows, sm_rows_source)

    if not items:
        raise ValueError("没有可配对的有效行；检查变体与母轮的键是否对齐")

    nonadv = [item for item in items if item["category_id"] != ADVERSARIAL_ID]
    mn_means = category_means(nonadv, "mn_f1")
    sm_means = category_means(nonadv, "sm_f1")
    adv = adversarial_sr(items)
    cost = cost_per_question(nonadv)

    conversations = sorted({str(item["sample_id"]) for item in nonadv})
    ci_enabled = len(conversations) >= MIN_CLUSTERS_FOR_CI
    bootstrap_result = fin.cluster_bootstrap(nonadv, replicates=bootstrap, seed=seed) \
        if ci_enabled else None

    # 守护指标判定
    guards: dict[str, Any] = {}
    if adv_floor is not None and adv["variant_mn"]["adv_sr"] is not None:
        guards["adv_sr_floor"] = {
            "floor": adv_floor, "actual": adv["variant_mn"]["adv_sr"],
            "passed": adv["variant_mn"]["adv_sr"] >= adv_floor - 1e-12,
        }
    if token_ceiling_pct is not None and cost["mn_input_tokens_per_q"] is not None:
        # 母轮 MN accepted 输入 token/题作为基准（8190532/1000）
        parent_mn_in = [int((row.get("model_calls") or {}).get("input_tokens") or 0)
                        for row in parent_rows
                        if row.get("system") == "MemNexus" and fin.valid_answer(row)]
        if parent_mn_in:
            parent_mean = statistics.fmean(parent_mn_in)
            ceiling = parent_mean * (1 + token_ceiling_pct / 100)
            guards["token_ceiling"] = {
                "parent_mean": parent_mean, "ceiling": ceiling,
                "actual": cost["mn_input_tokens_per_q"], "pct_ceiling": token_ceiling_pct,
                "passed": cost["mn_input_tokens_per_q"] <= ceiling + 1e-9,
            }

    drift = None
    if drift_path is not None:
        drift = detect_drift(load_rows(drift_path), parent_rows)

    outcomes = Counter(item["outcome"] for item in nonadv)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"variant": str(variant_path), "parent_rows": str(parent_rows_path),
                   "sm_source": sm_source},
        "pairing": pair_diag,
        "primary": {
            "mn_average_f1": mn_means["Average"],
            "sm_average_f1": sm_means["Average"],
            "delta_pp": ((mn_means["Average"] - sm_means["Average"]) * 100)
            if (mn_means["Average"] is not None and sm_means["Average"] is not None) else None,
            "mn_by_category": mn_means,
            "sm_by_category": sm_means,
        },
        "outcomes_nonadversarial": {"win": outcomes["win"], "tie": outcomes["tie"],
                                    "loss": outcomes["loss"]},
        "adversarial": adv,
        "cost_per_question": cost,
        "cluster_bootstrap": bootstrap_result,
        "ci_note": (None if ci_enabled else
                    f"对话簇仅 {len(conversations)} 个（< {MIN_CLUSTERS_FOR_CI}），"
                    "cluster bootstrap 退化，仅报点估计；子集阶段不用于宣称显著性。"),
        "guards": guards,
        "drift_check": drift,
        "pooled_diagnostic": {
            "mn": statistics.fmean(item["mn_f1"] for item in nonadv),
            "sm": statistics.fmean(item["sm_f1"] for item in nonadv),
            "is_not_average": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="子集 A/B 离线比较（复用母轮 SM 基线）")
    parser.add_argument("--variant", type=Path, required=True,
                        help="变体 run 的 rows.json（含 MN 变体行；若双系统同轮也含 SM）")
    parser.add_argument("--parent-rows", type=Path,
                        default=ROOT / "result" / "bench1000_locomo_20261023" / "rows.json",
                        help="母轮 rows.json，提供 SM 基线（默认 --only-system 复用场景）")
    parser.add_argument("--sm-source", choices=["parent", "variant"], default="parent",
                        help="SM 基线取自母轮(parent)还是变体run自己(variant)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--adv-floor", type=float, default=None,
                        help="AdvSR 下限（0-1）；低于则守护失败。P2 实验必须设置")
    parser.add_argument("--token-ceiling-pct", type=float, default=None,
                        help="accepted 输入 token/题 相对母轮的最大增幅百分比")
    parser.add_argument("--drift-baseline", type=Path, default=None,
                        help="baseline 臂的 rows.json；提供则做漂移检测")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.variant.exists():
        print(f"[FAIL] variant 不存在：{args.variant}", file=sys.stderr)
        return 3
    if not args.parent_rows.exists():
        print(f"[FAIL] parent-rows 不存在：{args.parent_rows}", file=sys.stderr)
        return 3

    report = analyze(args.variant, args.parent_rows, args.sm_source, args.bootstrap,
                     args.seed, args.adv_floor, args.token_ceiling_pct, args.drift_baseline)

    # 漂移或守护失败 → 非零退出，阻止把坏结果当成功
    blockers: list[str] = []
    if report["drift_check"] and report["drift_check"]["drift_detected"]:
        blockers.append(f"基线漂移：{report['drift_check']['drifted']} 题 MN F1 与母轮不一致")
    for name, guard in report["guards"].items():
        if not guard.get("passed", True):
            blockers.append(f"守护未过：{name} = {guard}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "subset_ab.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    primary = report["primary"]
    adv = report["adversarial"]
    print(f"[配对] {report['pairing']['paired']} 对；变体MN {report['pairing']['variant_mn_valid']}、"
          f"基线SM {report['pairing']['baseline_sm_valid']}；"
          f"未配对 变体{report['pairing']['unpaired_variant_only']}/基线{report['pairing']['unpaired_baseline_only']}")
    print(f"[主指标] MN Average F1 {primary['mn_average_f1']:.4f} vs SM {primary['sm_average_f1']:.4f} "
          f"→ Δ {primary['delta_pp']:+.2f} pp")
    if report["cluster_bootstrap"]:
        ci = report["cluster_bootstrap"]["ci95"]["Average"]
        print(f"[CI] 簇数 {report['cluster_bootstrap']['n_clusters']}；Average 95%CI "
              f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp（方向性参考）")
    else:
        print(f"[CI] {report['ci_note']}")
    if adv["variant_mn"]["adv_sr"] is not None:
        print(f"[AdvSR] 变体MN {adv['variant_mn']['adv_sr']:.3f} (n={adv['variant_mn']['n']}) vs "
              f"基线SM {adv['baseline_sm']['adv_sr']:.3f} → Δ {adv['delta_pp']:+.2f} pp")
    cost = report["cost_per_question"]
    if cost["mn_input_tokens_per_q"]:
        print(f"[成本] MN accepted 输入 {cost['mn_input_tokens_per_q']:.0f} tok/题；"
              f"输出 {cost['mn_output_tokens_per_q']:.0f}；延迟 {cost['mn_latency_ms_per_q']:.0f} ms")
    print(f"[OK] 写出 {json_path}")

    if blockers:
        print("[FAIL] 存在阻塞项，本轮变体结果不可采信：", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 6
    print("[OK] 无漂移、守护全过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

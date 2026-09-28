# -*- coding: utf-8 -*-
"""
template_rule_completion.py
===========================
模板驱动规则补全器（Template-guided Rule Completion）。

它替代“完全自由的决策树挖规则”：
- 规则语义结构来自 template_rules.py；
- 连续阈值 value 由轨迹/候选样本的分位数搜索得到；
- 加分/扣分 adjust 由候选网格生成，后续交给验证集 greedy_select_rules 筛选；
- support/confidence/coverage 用采集样本计算，用于初筛候选模板规则。

注意：这里不直接在本模块里跑验证集，因为原 v14 已经有 greedy_select_rules / 
greedy_select_rules_by_scenario。也就是说，本模块负责：
    样本 + 模板 -> candidate_guidance_rules
后续 v14 负责：
    candidate_guidance_rules -> selected_guidance_rules -> 反馈训练
"""
from __future__ import annotations

import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.innovation_A.feature_extractor import decode_label
from src.innovation_A.template_rules import build_template_rules


TASK_ALIAS = {
    "n_remaining_load_yard_ratio": "remaining_ratio_load_yard",
    "n_remaining_load_truck_ratio": "remaining_ratio_load_truck",
    "n_remaining_unload_yard_ratio": "remaining_ratio_unload_yard",
    "n_remaining_unload_truck_ratio": "remaining_ratio_unload_truck",
}

TASK_KIND_GROUPS = {
    "load": {"load_yard", "load_truck"},
    "unload": {"unload_yard", "unload_truck"},
    "truck": {"load_truck", "unload_truck"},
}

PRIORITY_FIXED_KEEP = {"safety", "feasibility"}


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return float(int(x))
        return float(x)
    except Exception:
        return default


def _is_number(x: Any) -> bool:
    return _safe_float(x, None) is not None and not isinstance(x, bool)


def _compare(actual: Any, op: str, expected: Any) -> bool:
    op = str(op or "eq").lower()
    if expected == "current_time":
        # 采样行中 current_time/t_now 都可能存在；真正实时执行由 RuleGuidanceEngine 处理。
        return True
    if op in ("eq", "=="):
        return actual == expected
    if op in ("ne", "!="):
        return actual != expected
    if op == "in":
        return actual in (expected or [])
    if op in ("not_in", "nin"):
        return actual not in (expected or [])
    af = _safe_float(actual, None)
    ef = _safe_float(expected, None)
    if af is None or ef is None:
        return False
    if op in ("gt", ">"):
        return af > ef
    if op in ("ge", ">=", "gte"):
        return af >= ef
    if op in ("lt", "<"):
        return af < ef
    if op in ("le", "<=", "lte"):
        return af <= ef
    if op == "abs_le":
        return abs(af) <= ef
    if op == "abs_ge":
        return abs(af) >= ef
    return False


def _rule_id(rule: Dict[str, Any]) -> str:
    return str(rule.get("id", rule.get("rule_id", "rule")))


def _action(rule: Dict[str, Any]) -> Dict[str, Any]:
    return dict(rule.get("action") or {})


def _effect(rule: Dict[str, Any]) -> str:
    act = _action(rule)
    return str(rule.get("effect", act.get("effect", "prefer"))).lower()


def _adjust(rule: Dict[str, Any]) -> float:
    act = _action(rule)
    return float(_safe_float(rule.get("adjust", act.get("adjust", 0.0)), 0.0) or 0.0)


def _set_adjust(rule: Dict[str, Any], adjust: float) -> Dict[str, Any]:
    r = deepcopy(rule)
    r["adjust"] = float(adjust)
    r.setdefault("action", {})["adjust"] = float(adjust)
    return r


def _set_rule_id(rule: Dict[str, Any], rid: str) -> Dict[str, Any]:
    rule["id"] = rid
    rule["rule_id"] = rid
    return rule


def _set_condition_value(rule: Dict[str, Any], cond_index: int, value: Any) -> None:
    rule["conditions"][cond_index]["value"] = value


def _is_truck_wait_template(rule: Dict[str, Any]) -> bool:
    """Identify templates related to load-truck waiting or load-truck destination waiting.

    These rules are rare by nature, especially destination candidates where one
    task may have hundreds of possible destinations and only one is selected.
    Therefore they need lower candidate-generation thresholds than ordinary
    task/load-pressure rules.
    """
    rid = _rule_id(rule).lower()
    txt = json.dumps(rule, ensure_ascii=False, default=str).lower()
    return (
        "truck_wait" in rid
        or "load_truck_wait" in rid
        or "dest_load_truck" in rid
        or "dest_load_truck_fast_cross" in rid
        or "load_truck_wait_norm" in txt
    )


def _is_destination_truck_wait_template(rule: Dict[str, Any]) -> bool:
    return str(rule.get("stage", "")).lower() == "destination" and _is_truck_wait_template(rule)


def _parse_float_list(text: Any, default: Sequence[float]) -> List[float]:
    if text is None:
        return [float(x) for x in default]
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    out: List[float] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            pass
    return out or [float(x) for x in default]


def _parse_adjust_candidates(effect: str, prefer: Any = None, avoid: Any = None) -> List[float]:
    effect = str(effect).lower()
    if effect in ("mask", "block"):
        return [-1_000_000.0]
    if effect in ("avoid", "penalty", "negative"):
        return _parse_float_list(avoid, [-30, -50, -70, -90, -120])
    return _parse_float_list(prefer, [30, 50, 70, 90, 120])


def _rows_from_trajectory_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    features = np.asarray(data.get("features", []))
    labels = np.asarray(data.get("labels", []))
    names = [str(x) for x in list(data.get("feature_names", []))]
    scenarios = list(data.get("scenarios", []))
    rows: List[Dict[str, Any]] = []
    if features.size == 0 or not names:
        return rows
    for i in range(features.shape[0]):
        row = {names[j]: float(features[i, j]) for j in range(min(len(names), features.shape[1]))}
        row["scenario"] = str(scenarios[i]) if i < len(scenarios) else "unknown"
        selected_kind = decode_label(int(labels[i])) if i < len(labels) else "unknown"
        row["selected_task_kind"] = selected_kind
        # 不把 task_kind 设置成 selected_task_kind 用于条件匹配，避免 task 规则 confidence 被人为做成 1。
        # 但为了导出诊断，也保留 selected_task_kind。
        for old, new in TASK_ALIAS.items():
            if old in row and new not in row:
                row[new] = row[old]
        lt_wait = float(row.get("truck_total_wait_norm", 0.0) or 0.0)
        row["load_truck_wait_norm"] = lt_wait
        row["unload_truck_wait_norm"] = 0.0
        load_pressure = float(row.get("remaining_ratio_load_yard", 0.0)) + float(row.get("remaining_ratio_load_truck", 0.0)) + 0.5 * lt_wait
        unload_pressure = float(row.get("remaining_ratio_unload_yard", 0.0)) + float(row.get("remaining_ratio_unload_truck", 0.0))
        row["load_pressure"] = load_pressure
        row["unload_pressure"] = unload_pressure
        row["load_minus_unload_pressure"] = load_pressure - unload_pressure
        row["unload_minus_load_pressure"] = unload_pressure - load_pressure
        rows.append(row)
    return rows


def _canonical_sample_rows(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in samples or []:
        row = dict(r)
        # CSV 读回或 env info 里 bool 可能是字符串；这里兼容一下。
        for k, v in list(row.items()):
            if isinstance(v, str):
                lv = v.strip().lower()
                if lv in ("true", "false"):
                    row[k] = (lv == "true")
        if "selected" in row:
            v = row["selected"]
            if isinstance(v, str):
                row["selected"] = v.strip().lower() in ("true", "1", "yes")
            else:
                row["selected"] = bool(v)

        # v2 truck-wait fix: destination/AQC candidate rows usually have
        # truck_total_wait but not load_truck_wait_norm.  The template library
        # uses load_truck_wait_norm, so create a stable fallback feature here.
        if "truck_total_wait_norm" not in row:
            row["truck_total_wait_norm"] = float(_safe_float(row.get("truck_total_wait", 0.0), 0.0) or 0.0) / 10000.0
        if "load_truck_wait_norm" not in row:
            row["load_truck_wait_norm"] = float(_safe_float(row.get("truck_total_wait_norm", 0.0), 0.0) or 0.0)
        if "unload_truck_wait_norm" not in row:
            row["unload_truck_wait_norm"] = 0.0
        rows.append(row)
    return rows


def _stage_rows(stage: str, task_rows: Sequence[Dict[str, Any]], dest_rows: Sequence[Dict[str, Any]], aqc_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if stage == "task":
        return list(task_rows)
    if stage == "destination":
        return list(dest_rows)
    if stage == "aqc":
        return list(aqc_rows)
    return []


def _condition_matches(row: Dict[str, Any], cond: Dict[str, Any], stage: str, *, ignore_task_kind_for_task_stage: bool = False) -> bool:
    feat = str(cond.get("feature", ""))
    if ignore_task_kind_for_task_stage and stage == "task" and feat == "task_kind":
        return True
    if feat not in row:
        return False
    expected = cond.get("value")
    if expected == "current_time":
        expected = row.get("current_time", row.get("t_now", 0.0))
    return _compare(row.get(feat), cond.get("op", "eq"), expected)


def _rule_matches_row(row: Dict[str, Any], rule: Dict[str, Any], stage: str, *, ignore_task_kind_for_task_stage: bool = False) -> bool:
    scenario = str(rule.get("scenario", "all"))
    if scenario not in ("all", "*", "") and scenario != str(row.get("scenario", "unknown")):
        return False
    for cond in rule.get("conditions", []) or []:
        if not _condition_matches(row, cond, stage, ignore_task_kind_for_task_stage=ignore_task_kind_for_task_stage):
            return False
    return True


def _target_match_task(row: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    kind = str(row.get("selected_task_kind", "unknown"))
    target = dict(rule.get("target") or (rule.get("action") or {}).get("target") or {})
    if "task_kind" in target:
        return kind == str(target.get("task_kind"))
    if "task_kind_group" in target:
        return kind in TASK_KIND_GROUPS.get(str(target.get("task_kind_group")), set())
    # train_policy / urgent 等 task 规则没有办法只靠 selected kind 判定，视为语义规则，后续交给 VAL。
    return True


def _target_match_candidate(row: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    return bool(row.get("selected", False))


def _support_confidence(rows: Sequence[Dict[str, Any]], rule: Dict[str, Any], stage: str) -> Tuple[int, float, float, int]:
    ignore_task_kind = stage == "task"
    matched = [r for r in rows if _rule_matches_row(r, rule, stage, ignore_task_kind_for_task_stage=ignore_task_kind)]
    support = int(len(matched))
    denom = int(len(rows))
    coverage = float(support / max(1, denom))
    if support <= 0:
        return 0, 0.0, coverage, denom
    if stage == "task":
        hits = sum(1 for r in matched if _target_match_task(r, rule))
    else:
        hits = sum(1 for r in matched if _target_match_candidate(r, rule))
    conf = float(hits / max(1, support))
    return support, conf, coverage, denom


def _candidate_thresholds(rows: Sequence[Dict[str, Any]], feature: str, op: str,
                          quantiles: Sequence[float], value_range: Optional[Sequence[float]] = None) -> List[float]:
    vals: List[float] = []
    for r in rows:
        if feature in r and _is_number(r.get(feature)):
            v = float(r.get(feature))
            if np.isfinite(v) and abs(v) < 1e12:
                vals.append(v)
    if not vals:
        return []
    arr = np.asarray(vals, dtype=float)
    qs = []
    for q in quantiles:
        try:
            qf = float(q)
            if qf > 1.0:
                qf /= 100.0
            qf = min(max(qf, 0.0), 1.0)
            qs.append(float(np.quantile(arr, qf)))
        except Exception:
            pass
    lo, hi = None, None
    if value_range and len(value_range) >= 2:
        lo, hi = float(value_range[0]), float(value_range[1])
    out: List[float] = []
    for v in qs:
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        # 避免过长小数，JSON 更好读。
        if abs(v) < 10:
            v = round(v, 4)
        else:
            v = round(v, 2)
        if v not in out:
            out.append(float(v))
    return out


def _threshold_condition_indices(rule: Dict[str, Any], rows: Sequence[Dict[str, Any]], quantiles: Sequence[float]) -> List[Tuple[int, List[float]]]:
    slots = dict(rule.get("slots") or {})
    out: List[Tuple[int, List[float]]] = []
    for i, cond in enumerate(rule.get("conditions", []) or []):
        feat = str(cond.get("feature", ""))
        val = cond.get("value")
        if isinstance(val, bool) or val == "current_time":
            continue
        if not _is_number(val):
            continue
        # 只对样本中存在的数值特征补全阈值。
        if not any(feat in r and _is_number(r.get(feat)) for r in rows):
            continue
        v_range = slots.get(feat)
        cands = _candidate_thresholds(rows, feat, str(cond.get("op", "eq")), quantiles, v_range)
        if cands:
            out.append((i, cands))
    return out


def _limited_product(options: Sequence[Tuple[int, List[float]]], max_variants: int) -> List[List[Tuple[int, float]]]:
    if not options:
        return [[]]
    keys = [x[0] for x in options]
    vals = [x[1] for x in options]
    combos: List[List[Tuple[int, float]]] = []
    for prod in itertools.product(*vals):
        combos.append(list(zip(keys, prod)))
        if len(combos) >= int(max_variants):
            break
    return combos


def _fixed_rule_should_keep(rule: Dict[str, Any]) -> bool:
    """Return True for rules that should be kept even when sample support is low.

    Important: this function should be conservative.  The old version kept every
    rule with priority in {safety, feasibility}; that unintentionally allowed many
    soft feasibility templates (for example unload_yard/unload_truck availability
    rules) to enter the candidate pool with support=0 and their large hard-coded
    adjust values.  Those rules then ignored the command-line weak-rule settings.

    Here we only force-keep hard safety/disturbance rules.  Ordinary soft
    prefer/avoid templates must pass support/confidence/coverage like other rules.
    """
    rid = _rule_id(rule).lower()
    effect = _effect(rule)
    if effect in ("mask", "block"):
        return True
    if "broken" in rid or "planning_blocked" in rid:
        return True
    # Disturbance rules may be rare in collected samples but are semantically valid.
    if "urgent" in rid or "not_arrived" in rid:
        return True
    # AQC conflict is a safety avoid rule; keep it, but its adjust is still allowed
    # to follow command-line avoid candidates below.
    if "aqc_conflict" in rid:
        return True
    return False


def complete_template_guidance_rules(
    trajectory_data: Dict[str, Any],
    destination_samples: Sequence[Dict[str, Any]],
    aqc_samples: Sequence[Dict[str, Any]],
    *,
    quantiles: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    prefer_adjust_candidates: Any = None,
    avoid_adjust_candidates: Any = None,
    min_support: int = 20,
    min_confidence: float = 0.75,
    min_coverage: float = 0.002,
    truck_wait_task_min_support: int = 2,
    truck_wait_task_min_confidence: float = 0.05,
    truck_wait_task_min_coverage: float = 0.00005,
    truck_wait_dest_min_support: int = 2,
    truck_wait_dest_min_confidence: float = 0.001,
    truck_wait_dest_min_coverage: float = 0.000001,
    truck_wait_candidate_quota: int = 2,
    max_rules: int = 120,
    max_variants_per_template: int = 80,
    include_fixed_rules: bool = True,
    scenarios: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """从模板库和采样数据中生成候选模板规则。"""
    task_rows = _rows_from_trajectory_data(trajectory_data)
    dest_rows = _canonical_sample_rows(destination_samples)
    aqc_rows = _canonical_sample_rows(aqc_samples)
    templates = build_template_rules(scenarios=scenarios, include_global=True)

    candidates: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for tmpl_idx, tmpl in enumerate(templates):
        stage = str(tmpl.get("stage", "task"))
        rows = _stage_rows(stage, task_rows, dest_rows, aqc_rows)
        rid0 = _rule_id(tmpl)
        if not rows:
            if include_fixed_rules and _fixed_rule_should_keep(tmpl):
                r = deepcopy(tmpl)
                r["support"] = 0
                r["confidence"] = 1.0
                r["coverage"] = 0.0
                r["completion_method"] = "fixed_no_samples"
                candidates.append(r)
            diagnostics.append({"template_id": rid0, "stage": stage, "status": "no_samples"})
            continue

        # 先用场景和非数值条件缩小样本，再从这些样本上取分位数。
        pre_rows = []
        for row in rows:
            scenario = str(tmpl.get("scenario", "all"))
            if scenario not in ("all", "*", "") and scenario != str(row.get("scenario", "unknown")):
                continue
            ok = True
            for c in tmpl.get("conditions", []) or []:
                # 数值阈值条件先不用于预过滤；分类/布尔条件用于预过滤。
                val = c.get("value")
                if _is_number(val) and not isinstance(val, bool):
                    continue
                if stage == "task" and c.get("feature") == "task_kind":
                    continue
                if not _condition_matches(row, c, stage, ignore_task_kind_for_task_stage=(stage == "task")):
                    ok = False
                    break
            if ok:
                pre_rows.append(row)
        rows_for_quantile = pre_rows or rows

        threshold_options = _threshold_condition_indices(tmpl, rows_for_quantile, quantiles)
        threshold_combos = _limited_product(threshold_options, max_variants=max_variants_per_template)
        if not threshold_combos:
            threshold_combos = [[]]

        n_added = 0
        n_tested = 0
        for combo_idx, combo in enumerate(threshold_combos):
            base_rule = deepcopy(tmpl)
            for cond_i, value in combo:
                _set_condition_value(base_rule, int(cond_i), value)
            support, conf, coverage, denom = _support_confidence(rows, base_rule, stage)
            n_tested += 1

            local_min_support = int(min_support)
            local_min_confidence = float(min_confidence)
            local_min_coverage = float(min_coverage)
            if _is_destination_truck_wait_template(base_rule):
                local_min_support = int(truck_wait_dest_min_support)
                local_min_confidence = float(truck_wait_dest_min_confidence)
                local_min_coverage = float(truck_wait_dest_min_coverage)
            elif _is_truck_wait_template(base_rule):
                local_min_support = int(truck_wait_task_min_support)
                local_min_confidence = float(truck_wait_task_min_confidence)
                local_min_coverage = float(truck_wait_task_min_coverage)

            keep_by_data = (
                support >= local_min_support
                and conf >= local_min_confidence
                and coverage >= local_min_coverage
            )
            keep_fixed = bool(include_fixed_rules and _fixed_rule_should_keep(base_rule))

            # Detailed diagnostics for truck-wait templates.  This is intentionally
            # kept small: only rare templates are expanded, so JSON stays readable.
            if _is_truck_wait_template(base_rule):
                try:
                    diagnostics.append({
                        "template_id": rid0,
                        "stage": stage,
                        "variant_index": int(combo_idx),
                        "variant_conditions": deepcopy(base_rule.get("conditions", [])),
                        "support": int(support),
                        "confidence": float(conf),
                        "coverage": float(coverage),
                        "denominator": int(denom),
                        "local_min_support": int(local_min_support),
                        "local_min_confidence": float(local_min_confidence),
                        "local_min_coverage": float(local_min_coverage),
                        "keep_by_data": bool(keep_by_data),
                        "reason": "kept" if keep_by_data else "filtered_by_support_confidence_or_coverage",
                    })
                except Exception:
                    pass

            if not (keep_by_data or keep_fixed):
                continue

            effect = _effect(base_rule)
            adjusts = _parse_adjust_candidates(effect, prefer_adjust_candidates, avoid_adjust_candidates)
            # Hard mask rules must remain hard masks.  For fixed prefer/avoid rules
            # (urgent, not_arrived, aqc_conflict, etc.), DO NOT keep the template's
            # hard-coded +120/-150/-90 values; use the command-line weak-rule grid
            # so --template_prefer_adjust_candidates / --template_avoid_adjust_candidates
            # actually control their scores.
            if keep_fixed and not keep_by_data and effect in ("mask", "block"):
                adjusts = [_adjust(base_rule)]
            for adj in adjusts:
                r = _set_adjust(base_rule, float(adj))
                rid = f"{rid0}__q{combo_idx:03d}__b{str(adj).replace('-', 'm').replace('.', 'p')}"
                _set_rule_id(r, rid)
                r["support"] = int(support)
                r["confidence"] = float(conf if support > 0 else 1.0)
                r["coverage"] = float(coverage)
                r["denominator"] = int(denom)
                r["template_id"] = rid0
                r["is_truck_wait_template"] = bool(_is_truck_wait_template(r))
                r["is_destination_truck_wait_template"] = bool(_is_destination_truck_wait_template(r))
                r["completion_method"] = "quantile_threshold_grid_beta" if keep_by_data else "fixed_rule"
                r["source"] = "template_rule_completion"
                candidates.append(r)
                n_added += 1
        diagnostics.append({
            "template_id": rid0,
            "stage": stage,
            "n_rows": len(rows),
            "n_prefilter_rows": len(pre_rows),
            "n_threshold_conditions": len(threshold_options),
            "n_variants_tested": n_tested,
            "n_rules_added": n_added,
        })

    # 去重：同 id 不会重复，但相同 conditions/action 可能重复。
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in candidates:
        sig = json.dumps({
            "scenario": r.get("scenario"),
            "stage": r.get("stage"),
            "conditions": r.get("conditions"),
            "effect": r.get("effect"),
            "target": r.get("target"),
            "adjust": r.get("adjust"),
            "priority": r.get("priority"),
        }, ensure_ascii=False, sort_keys=True, default=str)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(r)

    # 排序：先保留固定/安全规则，再按 confidence、support、coverage。
    def _score(r: Dict[str, Any]) -> Tuple[int, float, int, float]:
        fixed = 1 if str(r.get("completion_method")) == "fixed_rule" or _fixed_rule_should_keep(r) else 0
        return (fixed, float(r.get("confidence", 0.0)), int(r.get("support", 0)), float(r.get("coverage", 0.0)))

    uniq.sort(key=_score, reverse=True)
    if max_rules and int(max_rules) > 0:
        limit = int(max_rules)
        quota = max(0, int(truck_wait_candidate_quota))
        if quota > 0:
            truck_keep = [r for r in uniq if _is_truck_wait_template(r)][:quota * 3]
            truck_ids = {str(r.get("id", r.get("rule_id", ""))) for r in truck_keep}
            rest = [r for r in uniq if str(r.get("id", r.get("rule_id", ""))) not in truck_ids]
            uniq = (truck_keep + rest)[:limit]
        else:
            uniq = uniq[:limit]

    by_stage = {"task": 0, "destination": 0, "aqc": 0}
    by_scenario: Dict[str, int] = {}
    for r in uniq:
        by_stage[str(r.get("stage", "task"))] = by_stage.get(str(r.get("stage", "task")), 0) + 1
        by_scenario[str(r.get("scenario", "all"))] = by_scenario.get(str(r.get("scenario", "all")), 0) + 1

    return {
        "method": "template_rule_completion_quantile_support_confidence",
        "description": "Template-guided rule completion: quantile threshold search + beta grid; validation greedy selection is performed by run_v14.",
        "n_rules": int(len(uniq)),
        "rules": uniq,
        "meta": {
            "n_task_rows": int(len(task_rows)),
            "n_destination_rows": int(len(dest_rows)),
            "n_aqc_rows": int(len(aqc_rows)),
            "quantiles": [float(x) for x in quantiles],
            "min_support": int(min_support),
            "min_confidence": float(min_confidence),
            "min_coverage": float(min_coverage),
            "truck_wait_task_min_support": int(truck_wait_task_min_support),
            "truck_wait_task_min_confidence": float(truck_wait_task_min_confidence),
            "truck_wait_task_min_coverage": float(truck_wait_task_min_coverage),
            "truck_wait_dest_min_support": int(truck_wait_dest_min_support),
            "truck_wait_dest_min_confidence": float(truck_wait_dest_min_confidence),
            "truck_wait_dest_min_coverage": float(truck_wait_dest_min_coverage),
            "truck_wait_candidate_quota": int(truck_wait_candidate_quota),
            "max_rules": int(max_rules),
            "max_variants_per_template": int(max_variants_per_template),
            "by_stage": by_stage,
            "by_scenario": by_scenario,
            "diagnostics": diagnostics,
        },
    }


def save_template_completion_json(data: Dict[str, Any], path: str | Path, label: str = "save_template_completion") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[{label}] {data.get('n_rules', 0)} rules → {p}")
    return str(p)

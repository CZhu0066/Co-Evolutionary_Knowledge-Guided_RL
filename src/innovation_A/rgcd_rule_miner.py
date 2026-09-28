# -*- coding: utf-8 -*-
"""
rgcd_rule_miner.py
==================
把 Innovation A 原有“决策树蒸馏规则”转换为 v14-RGCD 可执行的
rule-guidance JSON。

核心思想：
- 原蒸馏规则：IF 25维状态条件 THEN predicted_kind
- RGCD规则：在 task 解码阶段，如果候选 task.kind == predicted_kind 且状态条件满足，
  则对该 candidate task 的 score 加分（prefer）。

这不是 MR 泛化；它是“从RL轨迹中挖知识 → 转成执行前规则引导”的基础闭环。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_OP_MAP = {
    "<=": "le",
    "<": "lt",
    ">": "gt",
    ">=": "ge",
    "==": "eq",
    "eq": "eq",
}


def _cond_to_guidance(cond: Dict[str, Any]) -> Dict[str, Any]:
    """distillation.RuleCondition dict -> rule_guidance condition dict"""
    op = str(cond.get("op", "eq"))
    return {
        "feature": str(cond.get("feature", "")),
        "op": _OP_MAP.get(op, op),
        "value": float(cond.get("threshold", cond.get("value", 0.0))),
    }


def _iter_distilled_rules(rules_data: Dict[str, Any]):
    """兼容普通 rules 与 rules_by_scenario 两种 distillation 输出。

    yield: (scenario, rule, scenario_n_samples, total_n_samples)
    scenario_n_samples 用来计算分场景 coverage；total_n_samples 用来兜底。
    """
    total_n = int(rules_data.get("n_samples", 0) or 0)
    if "rules_by_scenario" in rules_data:
        for scenario, pack in (rules_data.get("rules_by_scenario") or {}).items():
            sc_n = int((pack or {}).get("n_samples", 0) or 0)
            for r in (pack or {}).get("rules", []) or []:
                yield str(scenario), r, sc_n, total_n
    else:
        for r in rules_data.get("rules", []) or []:
            yield None, r, total_n, total_n


def convert_distilled_to_guidance(
    rules_data: Dict[str, Any],
    adjust: float = 100.0,
    min_confidence: float = 0.90,
    min_support: int = 30,
    min_coverage: float = 0.003,
    min_rule_conditions: int = 2,
    max_rule_conditions: int = 8,
    max_rules: int = 80,
    include_scenario_condition: bool = True,
) -> Dict[str, Any]:
    """把蒸馏规则转换为 rule_guidance JSON 数据结构。

    新增三类过滤，避免“粗规则/偶然规则”直接进入执行评价：
      - min_confidence：叶子节点多数类占比，近似规则准确率；
      - min_support：规则覆盖到的绝对样本数；
      - min_coverage：support / 当前场景样本数；
      - min_rule_conditions：要求规则至少有若干个状态条件，避免 root/过粗规则；
      - max_rule_conditions：限制规则过长，避免过拟合且不可读。
    """
    out_rules: List[Dict[str, Any]] = []
    seen = set()
    rejected = {
        "low_confidence": 0,
        "low_support": 0,
        "low_coverage": 0,
        "too_few_conditions": 0,
        "too_many_conditions": 0,
        "unknown_kind": 0,
        "duplicate": 0,
    }

    candidates: List[Dict[str, Any]] = []

    for scenario, r, scenario_n_samples, total_n_samples in _iter_distilled_rules(rules_data):
        conf = float(r.get("confidence", 0.0))
        support = int(r.get("support", 0) or 0)
        denom = max(1, int(scenario_n_samples or total_n_samples or rules_data.get("n_samples", 1)))
        coverage = float(support) / float(denom)
        kind = str(r.get("predicted_kind", ""))
        raw_conditions = list(r.get("conditions", []) or [])
        n_state_conditions = len(raw_conditions)
        if not kind or kind == "unknown":
            rejected["unknown_kind"] += 1
            continue
        if conf < float(min_confidence):
            rejected["low_confidence"] += 1
            continue
        if support < int(min_support):
            rejected["low_support"] += 1
            continue
        if coverage < float(min_coverage):
            rejected["low_coverage"] += 1
            continue
        if n_state_conditions < int(min_rule_conditions):
            rejected["too_few_conditions"] += 1
            continue
        if int(max_rule_conditions) > 0 and n_state_conditions > int(max_rule_conditions):
            rejected["too_many_conditions"] += 1
            continue

        conditions: List[Dict[str, Any]] = []
        if scenario and include_scenario_condition:
            conditions.append({"feature": "scenario", "op": "eq", "value": str(scenario)})
        # candidate task 必须是该规则预测的任务类型。
        conditions.append({"feature": "task_kind", "op": "eq", "value": kind})
        for c in raw_conditions:
            gc = _cond_to_guidance(c)
            if gc["feature"]:
                conditions.append(gc)

        key = (scenario, kind, tuple((c["feature"], c["op"], str(c["value"])) for c in conditions))
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)

        candidates.append({
            "scenario": scenario,
            "kind": kind,
            "conditions": conditions,
            "confidence": conf,
            "support": support,
            "coverage": coverage,
            "n_state_conditions": n_state_conditions,
            "human_readable": r.get("human_readable", ""),
        })

    # 更可靠、更常见、更细一点的规则优先进入候选集。
    candidates.sort(key=lambda x: (x["confidence"], x["support"], x["n_state_conditions"]), reverse=True)

    for item in candidates[: int(max_rules)]:
        scenario = item["scenario"]
        kind = item["kind"]
        rid = f"rgcd_task_prefer_{scenario or 'all'}_{kind}_{len(out_rules):03d}"
        out_rules.append({
            "id": rid,
            "stage": "task",
            "effect": "prefer",
            "adjust": float(adjust),
            "confidence": float(item["confidence"]),
            "support": int(item["support"]),
            "coverage": float(item["coverage"]),
            "n_state_conditions": int(item["n_state_conditions"]),
            "predicted_kind": kind,
            "source": "distilled_decision_tree_filtered",
            "human_readable": item.get("human_readable", ""),
            "conditions": item["conditions"],
        })

    return {
        "method": "distilled_to_rgcd_task_guidance_filtered",
        "description": "Rules mined from RL trajectories, filtered by confidence/support/coverage/condition count, then converted to task-score guidance.",
        "filter": {
            "min_confidence": float(min_confidence),
            "min_support": int(min_support),
            "min_coverage": float(min_coverage),
            "min_rule_conditions": int(min_rule_conditions),
            "max_rule_conditions": int(max_rule_conditions),
            "adjust": float(adjust),
            "max_rules": int(max_rules),
        },
        "rejected": rejected,
        "n_rules_before_limit": len(candidates),
        "n_rules": len(out_rules),
        "rules": out_rules,
    }


def save_guidance_rules(data: Dict[str, Any], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[save_guidance_rules] {data.get('n_rules', len(data.get('rules', [])))} 条 → {p}")
    return str(p)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

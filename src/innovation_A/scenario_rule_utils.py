# -*- coding: utf-8 -*-
"""
scenario_rule_utils.py
======================
Scenario-aware utilities for CoEvo-RGCD.

This module keeps one shared RL policy, but organizes rules by scenario
(1load_1unload / 2load / 2unload) so that mining, validation, testing and
feedback training can use a scenario-specific rule library.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.innovation_A.rule_guidance import RuleGuidanceEngine

SCENARIO_ORDER = ["1load_1unload", "2load", "2unload"]
STAGE_ORDER = ["task", "destination", "aqc"]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def infer_rule_scenario(rule: Dict[str, Any]) -> str:
    """Infer the target scenario of a rule from explicit fields, conditions, or id."""
    for key in ("scenario", "target_scenario"):
        v = rule.get(key)
        if v:
            return str(v)
    for c in rule.get("conditions", []) or []:
        if str(c.get("feature", "")) == "scenario" and str(c.get("op", "eq")) in ("eq", "=="):
            return str(c.get("value", "unknown"))
    rid = str(rule.get("id", rule.get("name", "")))
    for s in SCENARIO_ORDER:
        if s in rid:
            return s
    return "unknown"


def infer_rule_stage(rule: Dict[str, Any]) -> str:
    """Infer rule stage: task / destination / aqc."""
    stage = str(rule.get("stage", "") or "").lower()
    if stage in STAGE_ORDER:
        return stage
    rid = str(rule.get("id", rule.get("name", ""))).lower()
    if "destination" in rid or "dest" in rid:
        return "destination"
    if "aqc" in rid:
        return "aqc"
    if "task" in rid:
        return "task"
    return "unknown"


def is_global_rule(rule: Dict[str, Any]) -> bool:
    s = str(infer_rule_scenario(rule)).lower()
    return s in ("all", "*", "global")


def annotate_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copied rule with target_scenario and rule_stage metadata."""
    r = dict(rule)
    r.setdefault("target_scenario", infer_rule_scenario(r))
    r.setdefault("rule_stage", infer_rule_stage(r))
    return r


def rule_quality_key(rule: Dict[str, Any]):
    """Quality sort key: high confidence/support/coverage, fewer conditions preferred as tie-breaker."""
    nconds = len(rule.get("conditions", []) or [])
    return (
        _safe_float(rule.get("confidence", 0.0)),
        _safe_float(rule.get("support", 0.0)),
        _safe_float(rule.get("coverage", 0.0)),
        -float(nconds),
    )


def _is_truck_wait_rule(rule: Dict[str, Any]) -> bool:
    rid = str(rule.get("id", rule.get("rule_id", ""))).lower()
    tid = str(rule.get("template_id", "")).lower()
    txt = json.dumps(rule, ensure_ascii=False, default=str).lower()
    return (
        bool(rule.get("is_truck_wait_template", False))
        or "truck_wait" in rid
        or "truck_wait" in tid
        or "dest_load_truck" in rid
        or "dest_load_truck" in tid
        or "load_truck_wait_norm" in txt
    )


def infer_rule_family(rule: Dict[str, Any]) -> str:
    """Infer a coarse rule family using only rule id/template id and stage.

    This is intentionally conservative: it avoids inspecting the full rule JSON,
    because meta fields such as truck_wait_template_min_confidence can otherwise
    make unrelated rules look like truck-wait rules.
    """
    rid = str(rule.get("id", rule.get("rule_id", rule.get("name", ""))) or "").lower()
    tid = str(rule.get("template_id", "") or "").lower()
    st = str(rule.get("stage", rule.get("rule_stage", "")) or "").lower()
    key = f"{rid} {tid}"

    if "load_truck_wait" in key or "truck_wait" in key or "dest_load_truck" in key:
        return "truck_wait"
    if "unload_pressure" in key:
        return "unload_pressure"
    if "load_pressure" in key:
        return "load_pressure"
    if st == "destination" or "_dest_" in key or "destination" in key:
        return "destination"
    if st == "aqc" or "aqc_" in key or "_aqc_" in key:
        return "aqc"
    if "tpl_global" in key or str(rule.get("target_scenario", "")).lower() in ("all", "global", "*"):
        return "global"
    return "other"


def _rule_id(rule: Dict[str, Any]) -> str:
    return str(rule.get("id", rule.get("rule_id", rule.get("name", ""))) or "")


def _append_unique(out: List[Dict[str, Any]], seen: set, rule: Dict[str, Any]) -> bool:
    rid = _rule_id(rule)
    key = rid or json.dumps(rule, ensure_ascii=False, sort_keys=True, default=str)
    if key in seen:
        return False
    seen.add(key)
    out.append(rule)
    return True


def _family_quota_select(
    group: Sequence[Dict[str, Any]],
    stage_limit: int,
    family_quotas: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Select a stage bucket with family-level reserved seats, then fill by quality."""
    if stage_limit <= 0:
        return []
    sorted_group = sorted(group, key=rule_quality_key, reverse=True)
    selected: List[Dict[str, Any]] = []
    seen: set = set()
    family_order = ["load_pressure", "unload_pressure", "truck_wait", "destination", "aqc", "global", "other"]

    # First pass: keep the best rules from each requested family.
    for fam in family_order:
        quota = int(family_quotas.get(fam, 0) or 0)
        if quota <= 0 or len(selected) >= stage_limit:
            continue
        fam_rules = [r for r in sorted_group if infer_rule_family(r) == fam]
        taken = 0
        for r in fam_rules:
            if taken >= quota or len(selected) >= stage_limit:
                break
            if _append_unique(selected, seen, r):
                taken += 1

    # Second pass: fill remaining seats with the best remaining rules, regardless of family.
    for r in sorted_group:
        if len(selected) >= stage_limit:
            break
        _append_unique(selected, seen, r)
    return selected


def split_rules_by_scenario(rules: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SCENARIO_ORDER}
    out["unknown"] = []
    out["all"] = []
    for rule in rules:
        r = annotate_rule(rule)
        s = str(r.get("target_scenario", "unknown"))
        if s in ("*", "global"):
            s = "all"
            r["target_scenario"] = "all"
        if s not in out:
            out[s] = []
        out[s].append(r)
    return out


def apply_scenario_stage_quota(
    rules: Sequence[Dict[str, Any]],
    max_rules_per_scenario: int = 30,
    max_rules_per_stage_scenario: int = 10,
    keep_unknown: bool = True,
    truck_wait_candidate_quota: int = 0,
    family_candidate_quota: bool = False,
    family_quota_load_pressure: int = 2,
    family_quota_unload_pressure: int = 2,
    family_quota_truck_wait: int = 2,
    family_quota_destination: int = 2,
    family_quota_aqc: int = 2,
    family_quota_global: int = 2,
    family_quota_other: int = 0,
) -> List[Dict[str, Any]]:
    """Keep a balanced candidate pool for each scenario and each decision stage.

    Original quota logic kept the best rules in each scenario-stage bucket and only
    reserved seats for truck-wait rules. With ``family_candidate_quota=True``,
    every major rule family receives reserved seats first. This prevents a family
    such as unload_pressure from being generated in candidate_template but removed
    before VAL selection simply because another family has many high-ranked variants.
    """
    buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        s: {st: [] for st in STAGE_ORDER + ["unknown"]} for s in SCENARIO_ORDER + ["unknown"]
    }
    for rule in rules:
        r = annotate_rule(rule)
        s = str(r.get("target_scenario", "unknown"))
        st = str(r.get("rule_stage", "unknown"))
        if s not in buckets:
            s = "unknown"
        if st not in buckets[s]:
            st = "unknown"
        buckets[s][st].append(r)

    selected: List[Dict[str, Any]] = []
    family_quotas = {
        "load_pressure": int(family_quota_load_pressure),
        "unload_pressure": int(family_quota_unload_pressure),
        "truck_wait": int(family_quota_truck_wait),
        "destination": int(family_quota_destination),
        "aqc": int(family_quota_aqc),
        "global": int(family_quota_global),
        "other": int(family_quota_other),
    }
    # Keep backward compatibility: if users only set the old truck_wait_candidate_quota,
    # it still controls the truck-wait reserved seats when family quota is enabled.
    if int(truck_wait_candidate_quota or 0) > 0:
        family_quotas["truck_wait"] = int(truck_wait_candidate_quota)

    for s in SCENARIO_ORDER:
        scenario_rules: List[Dict[str, Any]] = []
        for st in STAGE_ORDER:
            group = sorted(buckets[s][st], key=rule_quality_key, reverse=True)
            stage_limit = max(0, int(max_rules_per_stage_scenario))
            if bool(family_candidate_quota):
                scenario_rules.extend(_family_quota_select(group, stage_limit, family_quotas))
            else:
                quota = max(0, int(truck_wait_candidate_quota))
                if quota > 0:
                    truck_group = [r for r in group if _is_truck_wait_rule(r)][:quota]
                    truck_ids = {_rule_id(r) for r in truck_group}
                    rest = [r for r in group if _rule_id(r) not in truck_ids]
                    scenario_rules.extend((truck_group + rest)[:stage_limit])
                else:
                    scenario_rules.extend(group[:stage_limit])
        scenario_rules = sorted(scenario_rules, key=rule_quality_key, reverse=True)
        selected.extend(scenario_rules[: max(0, int(max_rules_per_scenario))])

    if keep_unknown:
        unknown_rules: List[Dict[str, Any]] = []
        for st in STAGE_ORDER + ["unknown"]:
            unknown_rules.extend(buckets["unknown"][st])
        unknown_sorted = sorted(unknown_rules, key=rule_quality_key, reverse=True)
        if bool(family_candidate_quota):
            selected.extend(_family_quota_select(unknown_sorted, max(0, int(max_rules_per_scenario)), family_quotas))
        else:
            selected.extend(unknown_sorted[: max(0, int(max_rules_per_scenario))])

    # deduplicate by rule id, preserving order
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in selected:
        _append_unique(out, seen, r)
    return out


def save_rules_by_scenario_json(
    rules: Sequence[Dict[str, Any]],
    path: str | Path,
    method: str = "scenario_wise_rule_library",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    by = split_rules_by_scenario(rules)
    obj: Dict[str, Any] = {
        "method": method,
        "n_rules": int(len(rules)),
        "scenario_order": SCENARIO_ORDER,
        "global_rule_group": "all",
        "n_rules_by_scenario": {s: len(rs) for s, rs in by.items()},
        "rules_by_scenario": by,
        "rules": [annotate_rule(r) for r in rules],
    }
    if extra:
        obj.update(extra)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[save_rules_by_scenario_json] {len(rules)} rules → {p}")
    return str(p)


class ScenarioAwareRuleGuidanceEngine:
    """Callable rule engine that activates only the rule subset for env.scenario.

    It keeps one shared policy unchanged, while switching the rule library by
    scenario at decoding time. It can be passed directly to KGUnifiedYardEnv as
    rule_guidance_fn.
    """

    def __init__(self, rules: Sequence[Dict[str, Any]], selected_ids_by_scenario: Dict[str, Iterable[str]],
                 min_confidence: float = 0.0, engine_kwargs: Optional[Dict[str, Any]] = None):
        self.rules = [annotate_rule(r) for r in rules]
        self.engine_kwargs = dict(engine_kwargs or {})
        self.selected_ids_by_scenario: Dict[str, set] = {
            str(s): set(str(x) for x in ids) for s, ids in (selected_ids_by_scenario or {}).items()
        }
        # Trend-4: rules stored in the global "all" memory are active for every
        # concrete scenario, but are stored only once to avoid repeated global
        # rule injection/serialization.
        self.global_rule_ids: set = set(self.selected_ids_by_scenario.get("all", set()))

        all_ids = set(self.global_rule_ids)
        for ids in self.selected_ids_by_scenario.values():
            all_ids.update(ids)
        self.enabled_rule_ids = all_ids
        self._engine_all = RuleGuidanceEngine(
            self.rules,
            enabled_rule_ids=all_ids,
            min_confidence=min_confidence,
            **self.engine_kwargs,
        )
        self._engines: Dict[str, RuleGuidanceEngine] = {}
        for s in SCENARIO_ORDER + ["unknown"]:
            ids_set = set(self.selected_ids_by_scenario.get(s, set())) | set(self.global_rule_ids)
            # 空集合表示该场景暂未筛出规则且没有全局规则，应当完全不触发规则。
            if ids_set:
                self._engines[s] = RuleGuidanceEngine(
                    self.rules,
                    enabled_rule_ids=ids_set,
                    min_confidence=min_confidence,
                    **self.engine_kwargs,
                )

    def active_rule_ids(self) -> List[str]:
        return sorted(self.enabled_rule_ids)

    def subset(self, rule_ids: Iterable[str]) -> RuleGuidanceEngine:
        # Compatibility with code expecting RuleGuidanceEngine-like subset.
        return RuleGuidanceEngine(self.rules, enabled_rule_ids=set(rule_ids), **self.engine_kwargs)

    def __call__(self, env, stage: str, candidate: Any, task: Any = None, base_score: float = 0.0, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        scenario = str(getattr(env, "scenario", "unknown"))
        engine = self._engines.get(scenario)
        if engine is None:
            return {"adjust": 0.0, "mask": False, "matched_rules": [], "features": {}}
        return engine(env=env, stage=stage, candidate=candidate, task=task, base_score=base_score, extra=extra or {})


def make_scenario_aware_engine(
    rules: Sequence[Dict[str, Any]],
    selected_ids_by_scenario: Dict[str, Iterable[str]],
    min_confidence: float = 0.0,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[ScenarioAwareRuleGuidanceEngine]:
    if not selected_ids_by_scenario:
        return None
    if not any(list(ids) for ids in selected_ids_by_scenario.values()):
        return None
    return ScenarioAwareRuleGuidanceEngine(
        rules,
        selected_ids_by_scenario,
        min_confidence=min_confidence,
        engine_kwargs=engine_kwargs,
    )


def load_instance_scenario(path: str | Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "raw" in data and isinstance(data["raw"], dict) and data["raw"].get("scenario"):
                return str(data["raw"].get("scenario"))
            if data.get("scenario"):
                return str(data.get("scenario"))
    except Exception:
        pass
    # fallback from filename
    name = Path(path).name
    for s in SCENARIO_ORDER:
        if s in name:
            return s
    return "unknown"


def group_files_by_scenario(files: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {s: [] for s in SCENARIO_ORDER}
    out["unknown"] = []
    for f in files:
        s = load_instance_scenario(f)
        out.setdefault(s, []).append(str(f))
    return out

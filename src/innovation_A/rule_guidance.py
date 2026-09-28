# -*- coding: utf-8 -*-
"""
rule_guidance.py
================
v14-RGCD: Rule-Guided Continuous-action Decoding.

本模块把 LEGIBLE 的“规则引导执行”从离散 action 层迁移到本项目的
连续动作解码层：SAC/PPO 仍输出 9 维连续动作，规则不直接改 action，
而是在 task / destination / AQC 三个候选对象评分阶段做加分、扣分或屏蔽。

规则 JSON 示例：
{
  "rules": [
    {
      "id": "prefer_load_truck_when_truck_wait_high",
      "stage": "task",
      "effect": "prefer",
      "adjust": 500.0,
      "conditions": [
        {"feature": "task_kind", "op": "eq", "value": "load_truck"},
        {"feature": "truck_total_wait", "op": "ge", "value": 500.0}
      ]
    }
  ]
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


DEFAULT_PREFER_ADJUST = 500.0
DEFAULT_AVOID_ADJUST = -500.0

# 模板规则冲突优先级：安全 > 可行性 > 截止时间/紧急 > 等待 > 效率 > 距离/均衡。
# 数字越小，优先级越高。mask 规则永远覆盖 prefer/avoid。
RULE_PRIORITY_RANK = {
    "safety": 0,
    "feasibility": 1,
    "deadline": 2,
    "urgency": 2,
    "waiting": 3,
    "efficiency": 4,
    "distance_balance": 5,
    "balance": 5,
    "default": 6,
}


def _priority_rank(priority: Any) -> int:
    return int(RULE_PRIORITY_RANK.get(str(priority or "default").lower(), RULE_PRIORITY_RANK["default"]))


def normalize_guidance_rule(rule: Dict[str, Any], idx: int = 0) -> Dict[str, Any]:
    """兼容两种规则格式。

    老格式：
        {"id": ..., "stage": ..., "effect": ..., "adjust": ..., "conditions": [...]}

    新模板格式：
        {"rule_id": ..., "scenario": ..., "stage": ...,
         "conditions": [...],
         "action": {"effect": ..., "target": {...}, "adjust": ...},
         "priority": "safety|feasibility|deadline|waiting|efficiency|distance_balance"}
    """
    r = dict(rule or {})
    action = r.get("action") or {}
    if isinstance(action, dict):
        if "effect" not in r and "effect" in action:
            r["effect"] = action.get("effect")
        if "adjust" not in r and "adjust" in action:
            r["adjust"] = action.get("adjust")
        if "target" not in r and "target" in action:
            r["target"] = action.get("target")
    if "id" not in r:
        r["id"] = r.get("rule_id", f"rule_{idx}")
    if "priority" not in r:
        r["priority"] = "default"
    return r


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _get_nested(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = d
    for part in str(key).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _compare(actual: Any, op: str, expected: Any) -> bool:
    op = str(op or "eq").lower()
    if op in ("eq", "=="):
        return actual == expected
    if op in ("ne", "!="):
        return actual != expected
    if op in ("in",):
        return actual in (expected or [])
    if op in ("not_in", "nin"):
        return actual not in (expected or [])
    if op in ("contains",):
        try:
            return expected in actual
        except Exception:
            return False

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
    if op in ("abs_le",):
        return abs(af) <= ef
    if op in ("abs_ge",):
        return abs(af) >= ef
    return False


_NUMERIC_COMPARE_OPS = {"gt", ">", "ge", ">=", "gte", "lt", "<", "le", "<=", "lte", "abs_le", "abs_ge"}
_HARD_FEATURE_KEYWORDS = (
    "scenario", "stage", "kind", "type", "subtype", "id",
    "broken", "blocked", "occupied", "unavailable", "not_arrived",
    "is_", "train_not_arrived", "planning_blocked",
)


def _as_bool_like(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, (int, float)) and float(x) in (0.0, 1.0):
        return bool(int(x))
    if isinstance(x, str):
        xs = x.strip().lower()
        if xs in ("true", "false", "yes", "no", "0", "1"):
            return xs in ("true", "yes", "1")
    return None


def _is_soft_numeric_condition(cond: Dict[str, Any], actual: Any, expected: Any) -> bool:
    """Return True when a rule condition is safe to soften.

    Only continuous numeric threshold conditions should be softened. Categorical
    conditions and hard feasibility/safety flags remain exact 0/1 checks.
    """
    if bool(cond.get("hard", False)) or bool(cond.get("force_hard", False)):
        return False
    if bool(cond.get("soft", True)) is False:
        return False

    op = str(cond.get("op", "eq") or "eq").lower()
    if op not in _NUMERIC_COMPARE_OPS:
        return False

    feat = str(cond.get("feature", "")).lower()
    if any(k in feat for k in _HARD_FEATURE_KEYWORDS):
        return False

    if _as_bool_like(actual) is not None or _as_bool_like(expected) is not None:
        return False

    return (_safe_float(actual, None) is not None) and (_safe_float(expected, None) is not None)


def _sigmoid_stable(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 50.0:
        return 1.0
    if x <= -50.0:
        return 0.0
    return float(1.0 / (1.0 + np.exp(-x)))


def _condition_match_strength(
    actual: Any,
    op: str,
    expected: Any,
    cond: Dict[str, Any],
    *,
    soft_enabled: bool,
    temperature: float,
) -> float:
    """Return a continuous matching strength in [0, 1].

    When soft_enabled=False, this is exactly the original hard comparison.
    When soft_enabled=True, only numeric threshold comparisons are softened;
    categorical/safety conditions remain hard.
    """
    if not soft_enabled or not _is_soft_numeric_condition(cond, actual, expected):
        return 1.0 if _compare(actual, op, expected) else 0.0

    op = str(op or "eq").lower()
    af = _safe_float(actual, None)
    ef = _safe_float(expected, None)
    if af is None or ef is None:
        return 0.0

    # The global temperature is interpreted as a relative temperature. For
    # normalized thresholds, 0.05 means a smooth band around the threshold; for
    # large-scale variables (time/distance), it scales with |threshold|.
    local_temp = _safe_float(cond.get("temperature", cond.get("soft_temperature", None)), None)
    if local_temp is None or local_temp <= 0:
        local_temp = float(temperature) * max(1.0, abs(float(ef)))
    local_temp = max(float(local_temp), 1e-8)

    if op in ("gt", ">", "ge", ">=", "gte"):
        return _sigmoid_stable((float(af) - float(ef)) / local_temp)
    if op in ("lt", "<", "le", "<=", "lte"):
        return _sigmoid_stable((float(ef) - float(af)) / local_temp)
    if op == "abs_le":
        return _sigmoid_stable((float(ef) - abs(float(af))) / local_temp)
    if op == "abs_ge":
        return _sigmoid_stable((abs(float(af)) - float(ef)) / local_temp)

    return 1.0 if _compare(actual, op, expected) else 0.0


def _aggregate_strength(strengths: Sequence[float], mode: str = "min") -> float:
    vals = [float(np.clip(x, 0.0, 1.0)) for x in strengths]
    if not vals:
        return 1.0
    mode = str(mode or "min").lower()
    if mode in ("prod", "product", "mul"):
        return float(np.prod(vals))
    if mode in ("mean", "avg"):
        return float(np.mean(vals))
    # default: fuzzy AND by minimum, stable and easy to interpret.
    return float(min(vals))


def _available_yard_slot_ratio(env) -> float:
    slots = list(getattr(env, "slots", []) or [])
    if not slots:
        return 0.0
    current_height = getattr(env, "current_height", {}) or {}
    ok = 0
    for s in slots:
        if bool(getattr(s, "occupied", False)):
            continue
        h0 = current_height.get((getattr(s, "row", 0.0), getattr(s, "bay", 0.0)), -1)
        try:
            if int(getattr(s, "tier", 0)) == int(h0):
                ok += 1
        except Exception:
            ok += 1
    return float(ok) / float(max(1, len(slots)))


def _available_truck_slot_ratio(env) -> float:
    slots = list(getattr(env, "unload_truck_slots", []) or [])
    if not slots:
        return 0.0
    ok = sum(1 for s in slots if not bool(getattr(s, "occupied", False)))
    return float(ok) / float(max(1, len(slots)))


def _train_completion_features(env, task_obj: Any) -> Dict[str, float]:
    tasks = [t for t in list(getattr(env, "tasks", []) or []) if not getattr(t, "canceled", False)]
    tid = int(getattr(task_obj, "train_id", -1) or -1) if task_obj is not None else -1
    trains = list(getattr(env, "trains_involved", []) or [])
    if not trains:
        trains = sorted({int(getattr(t, "train_id", -1) or -1) for t in tasks if int(getattr(t, "train_id", -1) or -1) >= 0})

    ratios = []
    unload_ratios = []
    ratio_by_train: Dict[int, float] = {}
    unload_ratio_by_train: Dict[int, float] = {}
    for u in trains:
        ts = [t for t in tasks if int(getattr(t, "train_id", -1) or -1) == int(u)]
        done = [t for t in ts if bool(getattr(t, "done", False))]
        ratio = float(len(done)) / float(max(1, len(ts)))
        ratio_by_train[int(u)] = ratio
        ratios.append(ratio)

        uts = [t for t in ts if str(getattr(t, "kind", "")).startswith("unload_")]
        udone = [t for t in uts if bool(getattr(t, "done", False))]
        uratio = float(len(udone)) / float(max(1, len(uts))) if uts else ratio
        unload_ratio_by_train[int(u)] = uratio
        unload_ratios.append(uratio)

    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    mean_unload_ratio = float(np.mean(unload_ratios)) if unload_ratios else 0.0
    cur_ratio = float(ratio_by_train.get(tid, 0.0))
    cur_unload_ratio = float(unload_ratio_by_train.get(tid, cur_ratio))
    return {
        "task_train_completion_ratio": cur_ratio,
        "mean_train_completion_ratio": mean_ratio,
        "task_train_completion_gap": mean_ratio - cur_ratio,
        "task_train_unload_completion_ratio": cur_unload_ratio,
        "mean_train_unload_completion_ratio": mean_unload_ratio,
        "task_train_unload_completion_gap": mean_unload_ratio - cur_unload_ratio,
    }


def _deadline_pressure(env, task_obj: Any) -> float:
    if task_obj is None:
        return 0.0
    tid = int(getattr(task_obj, "train_id", -1) or -1)
    A2 = getattr(env, "A2", {}) or {}
    if tid not in A2:
        return 0.0
    tasks = [t for t in list(getattr(env, "tasks", []) or [])
             if int(getattr(t, "train_id", -1) or -1) == tid
             and not getattr(t, "done", False)
             and not getattr(t, "canceled", False)]
    done_durations = []
    for t in list(getattr(env, "tasks", []) or []):
        if getattr(t, "done", False):
            st = _safe_float(getattr(t, "start_time", 0.0))
            ft = _safe_float(getattr(t, "finish_time", 0.0))
            if ft > st:
                done_durations.append(ft - st)
    avg_task_time = float(np.mean(done_durations)) if done_durations else 120.0
    est_remaining = float(len(tasks)) * avg_task_time
    t_now = _safe_float(getattr(env, "t_now", 0.0))
    slack = _safe_float(A2.get(tid, 0.0)) - t_now - est_remaining
    # 归一化尺度：至少 1000，避免早期压力过高；slack<=0 时压力=1。
    horizon = max(1000.0, abs(_safe_float(A2.get(tid, 0.0))) + 1.0)
    return float(1.0 - np.clip(slack / horizon, 0.0, 1.0))


def _load_truck_wait_norm(env) -> float:
    """只计算 load_truck 等待压力；unload_truck 按用户假设不建模卡车等待。"""
    tasks = list(getattr(env, "tasks", []) or [])
    t_now = _safe_float(getattr(env, "t_now", 0.0))
    wait_done = sum(_safe_float(getattr(t, "total_wait", 0.0)) for t in tasks
                    if getattr(t, "done", False) and str(getattr(t, "kind", "")) == "load_truck")
    # 未完成 load_truck 的等待年龄只作为压力信号，不作为 objective 的 truck wait。
    wait_age = sum(max(0.0, t_now - _safe_float(getattr(t, "arrival_time", 0.0))) for t in tasks
                   if (not getattr(t, "done", False))
                   and (not getattr(t, "canceled", False))
                   and str(getattr(t, "kind", "")) == "load_truck")
    H = 10000.0
    return float(np.clip((wait_done + wait_age) / H, 0.0, 1.0))


def _task_best_destination_distance(env, task_obj: Any) -> float:
    if task_obj is None:
        return 1e9
    kind = str(getattr(task_obj, "kind", ""))
    init_bay = _safe_float(getattr(task_obj, "init_bay", 0.0))
    if kind in ("load_yard", "load_truck"):
        cars = list((getattr(env, "cars_by_train", {}) or {}).get(int(getattr(task_obj, "train_id", -1) or -1), []) or [])
        vals = [abs(_safe_float(getattr(c, "bay", 0.0)) - init_bay) for c in cars if not getattr(c, "occupied", False)]
        return float(min(vals)) if vals else 1e9
    if kind == "unload_yard":
        vals = []
        current_height = getattr(env, "current_height", {}) or {}
        for s in list(getattr(env, "slots", []) or []):
            if getattr(s, "occupied", False):
                continue
            h0 = current_height.get((getattr(s, "row", 0.0), getattr(s, "bay", 0.0)), -1)
            try:
                if int(getattr(s, "tier", 0)) != int(h0):
                    continue
            except Exception:
                pass
            vals.append(abs(_safe_float(getattr(s, "bay", 0.0)) - init_bay))
        return float(min(vals)) if vals else 1e9
    if kind == "unload_truck":
        vals = [abs(_safe_float(getattr(s, "bay", 0.0)) - init_bay)
                for s in list(getattr(env, "unload_truck_slots", []) or [])
                if not getattr(s, "occupied", False)]
        return float(min(vals)) if vals else 1e9
    return 1e9


def _current_yard_stack_height(env, dest: Any) -> float:
    if dest is None:
        return 0.0
    ch = getattr(env, "current_height", {}) or {}
    return _safe_float(ch.get((getattr(dest, "row", 0.0), getattr(dest, "bay", 0.0)), getattr(dest, "tier", 0.0)))


def extract_guidance_features(env, stage: str, candidate: Any, task: Any = None,
                              base_score: float = 0.0,
                              extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """提取 template/RGCD rule guidance 可用的状态特征 + 候选对象特征。

    重要假设：unload_truck 不计算卡车等待压力；它只受 TruckSlot 可用性、位置距离、
    AQC 可用性/安全性约束。load_truck 才使用 wait_cross/wait_aqc/total_wait。
    """
    extra = extra or {}
    task_obj = task if task is not None else candidate
    aqcs = list(getattr(env, "aqcs", []) or [])
    tasks = list(getattr(env, "tasks", []) or [])
    remaining = [t for t in tasks if (not getattr(t, "done", False)) and (not getattr(t, "canceled", False))]

    # 任务类型剩余数量
    kind_counts: Dict[str, int] = {}
    for t in remaining:
        k = str(getattr(t, "kind", "unknown"))
        kind_counts[k] = kind_counts.get(k, 0) + 1
    n_remaining = max(1, len(remaining))

    aqc_times = [_safe_float(getattr(a, "available_time", 0.0)) for a in aqcs]
    mean_aqc_t = float(np.mean(aqc_times)) if aqc_times else 0.0
    max_aqc_t = float(np.max(aqc_times)) if aqc_times else 0.0
    min_aqc_t = float(np.min(aqc_times)) if aqc_times else 0.0

    load_truck_wait_norm = _load_truck_wait_norm(env)
    f: Dict[str, Any] = {
        "stage": str(stage),
        "scenario": str(getattr(env, "scenario", "unknown")),
        "current_step": int(getattr(env, "current_step", 0)),
        "t_now": _safe_float(getattr(env, "t_now", 0.0)),
        "current_time": _safe_float(getattr(env, "t_now", 0.0)),
        "base_score": float(base_score),
        "obj": _safe_float(getattr(env, "obj", 0.0)),
        "train_obj": _safe_float(getattr(env, "train_obj", 0.0)),
        # 兼容老字段；这里仍只统计 objective 中 load_truck 产生的等待。
        "truck_total_wait": _safe_float(getattr(env, "truck_total_wait", 0.0)),
        "truck_total_wait_norm": load_truck_wait_norm,
        "load_truck_wait_norm": load_truck_wait_norm,
        "unload_truck_wait_norm": 0.0,
        "n_remaining_tasks": len(remaining),
        "mean_aqc_available_time": mean_aqc_t,
        "max_aqc_available_time": max_aqc_t,
        "min_aqc_available_time": min_aqc_t,
        "aqc_time_spread": max_aqc_t - min_aqc_t,
        "available_yard_slot_ratio": _available_yard_slot_ratio(env),
        "available_truck_slot_ratio": _available_truck_slot_ratio(env),
    }
    for k in ["load_yard", "load_truck", "unload_yard", "unload_truck"]:
        f[f"remaining_{k}"] = int(kind_counts.get(k, 0))
        f[f"remaining_ratio_{k}"] = float(kind_counts.get(k, 0)) / float(n_remaining)

    # 场景方向压力：unload_truck 不加入等待压力。
    load_pressure = f["remaining_ratio_load_yard"] + f["remaining_ratio_load_truck"] + 0.5 * load_truck_wait_norm
    unload_pressure = f["remaining_ratio_unload_yard"] + f["remaining_ratio_unload_truck"]
    f.update({
        "load_pressure": float(load_pressure),
        "unload_pressure": float(unload_pressure),
        "load_minus_unload_pressure": float(load_pressure - unload_pressure),
        "unload_minus_load_pressure": float(unload_pressure - load_pressure),
    })

    # task 相关特征
    if task_obj is not None:
        arrival = _safe_float(getattr(task_obj, "arrival_time", 0.0))
        tid = int(getattr(task_obj, "train_id", -1) or -1)
        A1 = getattr(env, "A1", {}) or {}
        train_arrived = getattr(env, "train_arrived", {}) or {}
        planning_blocked = getattr(env, "train_planning_blocked", set()) or set()
        task_train_not_arrived = bool(f["t_now"] < _safe_float(A1.get(tid, 0.0))) or (train_arrived.get(tid) is False)
        task_train_planning_blocked = bool(tid in planning_blocked)
        f.update({
            "task_id": int(getattr(task_obj, "id", -1) or -1),
            "task_kind": str(getattr(task_obj, "kind", "")),
            "task_type": str(getattr(task_obj, "type", "")),
            "task_subtype": str(getattr(task_obj, "subtype", "")),
            "task_train_id": tid,
            "task_arrival_time": arrival,
            "task_age": max(0.0, f["t_now"] - arrival),
            "task_init_row": _safe_float(getattr(task_obj, "init_row", 0.0)),
            "task_init_bay": _safe_float(getattr(task_obj, "init_bay", 0.0)),
            "task_urgency_boost": _safe_float(getattr(task_obj, "urgency_boost", 0.0)),
            "task_train_not_arrived": task_train_not_arrived,
            "task_train_planning_blocked": task_train_planning_blocked,
            "task_train_unavailable": bool(task_train_not_arrived or task_train_planning_blocked),
            "task_train_deadline_pressure": _deadline_pressure(env, task_obj),
            "task_best_destination_distance": _task_best_destination_distance(env, task_obj),
        })
        f.update(_train_completion_features(env, task_obj))

    # destination / slot / car / cross 相关特征
    if stage == "destination":
        dest = candidate
        dest_bay = _safe_float(getattr(dest, "bay", 0.0))
        dest_occupied = bool(getattr(dest, "occupied", False))
        f.update({
            "dest_kind": str(extra.get("dest_kind", type(dest).__name__)),
            "dest_row": _safe_float(getattr(dest, "row", 0.0)),
            "dest_bay": dest_bay,
            "dest_tier": _safe_float(getattr(dest, "tier", 0.0)),
            "dest_occupied": dest_occupied,
            # TruckSlot 没有 available_time；未占用则认为当前可用，占用则给极大时间。
            "dest_available_time": _safe_float(getattr(dest, "available_time", f["t_now"] if not dest_occupied else 1e18)),
            "yard_stack_height": _current_yard_stack_height(env, dest),
        })
        if task_obj is not None:
            f["dest_distance_to_task"] = abs(f["dest_bay"] - _safe_float(getattr(task_obj, "init_bay", 0.0)))
        cross = extra.get("cross")
        if cross is not None:
            f.update({
                "cross_id": int(getattr(cross, "id", -1) or -1),
                "cross_bay": _safe_float(getattr(cross, "bay", 0.0)),
                "cross_available_time": _safe_float(getattr(cross, "available_time", 0.0)),
                "cross_distance_to_task": abs(_safe_float(getattr(cross, "bay", 0.0)) - _safe_float(getattr(task_obj, "init_bay", 0.0))),
                "cross_distance_to_dest": abs(_safe_float(getattr(cross, "bay", 0.0)) - dest_bay),
            })
        aqc = extra.get("preferred_aqc")
        if aqc is not None:
            f.update({
                "preferred_aqc_id": int(getattr(aqc, "id", -1) or -1),
                "preferred_aqc_available_time": _safe_float(getattr(aqc, "available_time", 0.0)),
                "preferred_aqc_cur_bay": _safe_float(getattr(aqc, "cur_bay", 0.0)),
                "preferred_aqc_gap_to_task": abs(_safe_float(getattr(aqc, "cur_bay", 0.0)) - _safe_float(getattr(task_obj, "init_bay", 0.0))),
                "preferred_aqc_gap_to_dest": abs(_safe_float(getattr(aqc, "cur_bay", 0.0)) - dest_bay),
            })
        elif aqcs:
            f["min_aqc_gap_to_dest"] = min(abs(_safe_float(getattr(a, "cur_bay", 0.0)) - dest_bay) for a in aqcs)
        tf = extra.get("tf") or {}
        if isinstance(tf, dict):
            for k in ["finish", "start", "wait_aqc", "wait_cross", "total_wait", "h2", "o2"]:
                if k in tf:
                    f[f"tf_{k}"] = _safe_float(tf[k])

    # AQC 相关特征
    if stage == "aqc":
        aqc = candidate
        aqc_available_time = _safe_float(getattr(aqc, "available_time", 0.0))
        f.update({
            "aqc_id": int(getattr(aqc, "id", -1) or -1),
            "aqc_cur_bay": _safe_float(getattr(aqc, "cur_bay", 0.0)),
            "aqc_available_time": aqc_available_time,
            "aqc_available_delta": aqc_available_time - mean_aqc_t,
            "aqc_underload": max(0.0, mean_aqc_t - aqc_available_time),
            "aqc_imbalance": abs(aqc_available_time - mean_aqc_t),
            "aqc_overload": max(0.0, aqc_available_time - mean_aqc_t),
            "aqc_task_count": len(getattr(aqc, "tasks", []) or []),
            "aqc_is_broken": bool(getattr(aqc, "is_broken", False)),
        })
        task_counts = [len(getattr(a, "tasks", []) or []) for a in aqcs] or [0]
        f["mean_aqc_task_count"] = float(np.mean(task_counts))
        f["aqc_task_overload"] = float(f["aqc_task_count"] - f["mean_aqc_task_count"])
        if task_obj is not None:
            f["aqc_gap_to_task"] = abs(f["aqc_cur_bay"] - _safe_float(getattr(task_obj, "init_bay", 0.0)))
        dest = extra.get("chosen_dest")
        if dest is not None:
            f.update({
                "dest_row": _safe_float(getattr(dest, "row", 0.0)),
                "dest_bay": _safe_float(getattr(dest, "bay", 0.0)),
                "dest_tier": _safe_float(getattr(dest, "tier", 0.0)),
                "aqc_gap_to_dest": abs(f["aqc_cur_bay"] - _safe_float(getattr(dest, "bay", 0.0))),
            })
        cross = extra.get("chosen_cross")
        if cross is not None:
            f.update({
                "cross_id": int(getattr(cross, "id", -1) or -1),
                "cross_bay": _safe_float(getattr(cross, "bay", 0.0)),
                "cross_available_time": _safe_float(getattr(cross, "available_time", 0.0)),
                "cross_distance_to_task": abs(_safe_float(getattr(cross, "bay", 0.0)) - _safe_float(getattr(task_obj, "init_bay", 0.0))),
            })
        tf = extra.get("tf") or {}
        if isinstance(tf, dict):
            for k in ["finish", "start", "wait_aqc", "wait_cross", "total_wait", "h2", "o2"]:
                if k in tf:
                    f[f"tf_{k}"] = _safe_float(tf[k])
        # 候选在进入 _apply_rule_guidance 前通常已通过硬安全检查；这里保留可扩展风险特征。
        if "aqc_conflict_risk" in extra:
            f["aqc_conflict_risk"] = _safe_float(extra.get("aqc_conflict_risk"))
        else:
            others = [a for a in aqcs if int(getattr(a, "id", -1)) != int(getattr(aqc, "id", -2))]
            min_gap = min([abs(_safe_float(getattr(a, "cur_bay", 0.0)) - f["aqc_cur_bay"]) for a in others] or [9999.0])
            f["aqc_min_gap_to_other"] = min_gap
            f["aqc_conflict_risk"] = float(1.0 / (min_gap + 1.0))

    # 允许 extra 显式覆盖/补充特征，方便后续扩展
    for k, v in extra.items():
        if k not in ("done_tasks", "cross", "preferred_aqc", "chosen_dest", "chosen_cross", "tf"):
            f[str(k)] = v
    return f


def _rule_template_group_key(rule: Dict[str, Any]) -> str:
    """Return a stable family key for retrieval-time de-duplication."""
    tid = str(rule.get("template_id", "") or "").strip()
    if tid:
        return tid
    rid = str(rule.get("id", rule.get("rule_id", "")) or "")
    # Remove common variant suffixes such as __q000__b10p0 to group variants.
    if "__q" in rid:
        return rid.split("__q", 1)[0]
    return rid


def _rule_memory_dict(rule: Dict[str, Any]) -> Dict[str, Any]:
    mem = rule.get("memory", {})
    return mem if isinstance(mem, dict) else {}


def _rule_relevance_score(rule: Dict[str, Any], match_strength: float) -> float:
    """Trend-4 contextual rule-memory relevance score.

    The score only decides which matched rules are retrieved when Top-K retrieval
    is enabled. It does not directly change rule adjust values.
    """
    strength = float(np.clip(match_strength, 0.0, 1.0))
    mem = _rule_memory_dict(rule)

    # Good historical evidence: negative deltas mean improvement.
    target_delta = _safe_float(mem.get("target_delta", mem.get("mean_delta_obj", 0.0)), 0.0)
    score_delta = _safe_float(mem.get("target_score_delta", mem.get("score_delta", 0.0)), 0.0)
    gain = max(0.0, -float(target_delta))
    score_gain = max(0.0, -float(score_delta))

    # Risk terms. Keep scales small because retrieval is a ranker.
    p_w = max(0.0, _safe_float(mem.get("p_worsen", mem.get("local_p_worsen", 0.0)), 0.0))
    cvar = max(0.0, _safe_float(mem.get("cvar_worsen", mem.get("local_cvar_worsen", 0.0)), 0.0))
    high = max(0.0, _safe_float(mem.get("avg_high_strength_triggered", mem.get("local_avg_high_strength_triggered", 0.0)), 0.0))

    confidence = _safe_float(rule.get("confidence", 0.0), 0.0)
    coverage = _safe_float(rule.get("coverage", 0.0), 0.0)
    priority_bonus = max(0.0, 6.0 - float(_priority_rank(rule.get("priority", "default")))) * 0.03

    benefit = 0.02 * gain + 0.02 * score_gain + 0.10 * confidence + 0.50 * coverage + priority_bonus
    risk = 0.30 * p_w + 0.0005 * cvar + 0.001 * high
    return float(strength * (1.0 + benefit) - risk)


class RuleGuidanceEngine:
    """JSON 规则引擎：对 task/destination/AQC 候选评分做 prefer/avoid/mask。"""

    def __init__(self, rules: Sequence[Dict[str, Any]],
                 enabled_rule_ids: Optional[Iterable[str]] = None,
                 default_prefer_adjust: float = DEFAULT_PREFER_ADJUST,
                 default_avoid_adjust: float = DEFAULT_AVOID_ADJUST,
                 min_confidence: float = 0.0,
                 soft_rule_guidance: bool = False,
                 soft_rule_temperature: float = 0.05,
                 soft_rule_min_strength: float = 0.05,
                 soft_rule_aggregation: str = "min",
                 rule_memory_retrieval: bool = False,
                 rule_retrieval_top_k: int = 0,
                 rule_retrieval_dedup_templates: bool = False):
        self.rules = [normalize_guidance_rule(r, i) for i, r in enumerate(rules) if r]
        # IMPORTANT:
        #   enabled_rule_ids=None  => 启用全部规则（用于候选规则库整体加载）
        #   enabled_rule_ids=set() => 不启用任何规则（用于 scenario-wise 中某场景暂无已选规则）
        # 旧写法 `set(enabled_rule_ids) if enabled_rule_ids else None` 会把空集合误转成 None，
        # 导致“空规则集合”反而启用全部候选规则，进而在全局安全检查/反馈训练时造成跨场景误触发。
        self.enabled_rule_ids: Optional[Set[str]] = (set(enabled_rule_ids) if enabled_rule_ids is not None else None)
        self.default_prefer_adjust = float(default_prefer_adjust)
        self.default_avoid_adjust = float(default_avoid_adjust)
        self.min_confidence = float(min_confidence)
        self.soft_rule_guidance = bool(soft_rule_guidance)
        self.soft_rule_temperature = float(soft_rule_temperature)
        self.soft_rule_min_strength = float(soft_rule_min_strength)
        self.soft_rule_aggregation = str(soft_rule_aggregation or "min").lower()
        self.rule_memory_retrieval = bool(rule_memory_retrieval)
        self.rule_retrieval_top_k = int(rule_retrieval_top_k or 0)
        self.rule_retrieval_dedup_templates = bool(rule_retrieval_dedup_templates)

    @classmethod
    def from_json(cls, path: str, enabled_rule_ids: Optional[Iterable[str]] = None,
                  **kwargs) -> "RuleGuidanceEngine":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rules = data.get("rules", data if isinstance(data, list) else [])
        return cls(rules=rules, enabled_rule_ids=enabled_rule_ids, **kwargs)

    def subset(self, rule_ids: Iterable[str]) -> "RuleGuidanceEngine":
        return RuleGuidanceEngine(
            rules=self.rules,
            enabled_rule_ids=set(rule_ids),
            default_prefer_adjust=self.default_prefer_adjust,
            default_avoid_adjust=self.default_avoid_adjust,
            min_confidence=self.min_confidence,
            soft_rule_guidance=self.soft_rule_guidance,
            soft_rule_temperature=self.soft_rule_temperature,
            soft_rule_min_strength=self.soft_rule_min_strength,
            soft_rule_aggregation=self.soft_rule_aggregation,
            rule_memory_retrieval=self.rule_memory_retrieval,
            rule_retrieval_top_k=self.rule_retrieval_top_k,
            rule_retrieval_dedup_templates=self.rule_retrieval_dedup_templates,
        )

    def active_rule_ids(self) -> List[str]:
        ids = []
        for i, r in enumerate(self.rules):
            rid = str(r.get("id", f"rule_{i}"))
            if self._is_enabled(rid, r):
                ids.append(rid)
        return ids

    def _is_enabled(self, rid: str, rule: Dict[str, Any]) -> bool:
        if not bool(rule.get("enabled", True)):
            return False
        if self.enabled_rule_ids is not None and rid not in self.enabled_rule_ids:
            return False
        if _safe_float(rule.get("confidence", 1.0), 1.0) < self.min_confidence:
            return False
        return True

    def _rule_match_strength(self, rule: Dict[str, Any], features: Dict[str, Any], stage: str) -> float:
        """Return rule matching strength in [0, 1].

        Trend-6 soft rule guidance: numeric threshold conditions can partially
        match, so the final adjust becomes `base_adjust * strength`. Hard
        categorical/safety conditions remain exact. Mask/block rules are kept
        hard to avoid weakening feasibility constraints.
        """
        if str(rule.get("stage", "task")) != str(stage):
            return 0.0
        # 新模板规则允许把 scenario 写在顶层；scenario="all" 表示全场景。
        scenario = str(rule.get("scenario", "all"))
        if scenario not in ("all", "*", "") and scenario != str(features.get("scenario", "unknown")):
            return 0.0

        effect = str(rule.get("effect", "prefer")).lower()
        priority = str(rule.get("priority", "default")).lower()
        # Hard safety/feasibility masks should remain exact 0/1 shields.
        soft_enabled = bool(self.soft_rule_guidance) and effect not in ("mask", "block")
        if effect in ("mask", "block") or priority in ("safety", "feasibility"):
            soft_enabled = False

        strengths: List[float] = []
        for cond in rule.get("conditions", []) or []:
            feat = str(cond.get("feature", ""))
            expected = cond.get("value")
            # 允许 JSON 中写 value="current_time" 代表当前仿真时刻。
            if expected == "current_time":
                expected = features.get("current_time", features.get("t_now", 0.0))
            actual = _get_nested(features, feat, None)
            s = _condition_match_strength(
                actual,
                cond.get("op", "eq"),
                expected,
                cond,
                soft_enabled=soft_enabled,
                temperature=self.soft_rule_temperature,
            )
            if s <= 0.0:
                return 0.0
            strengths.append(float(s))

        strength = _aggregate_strength(strengths, self.soft_rule_aggregation)
        if strength < float(self.soft_rule_min_strength):
            return 0.0
        return float(np.clip(strength, 0.0, 1.0))

    def _rule_matches(self, rule: Dict[str, Any], features: Dict[str, Any], stage: str) -> bool:
        return self._rule_match_strength(rule, features, stage) > 0.0

    def __call__(self, env, stage: str, candidate: Any, task: Any = None,
                 base_score: float = 0.0, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        features = extract_guidance_features(env, stage, candidate, task, base_score, extra)
        mask = False
        priority_adjusts: Dict[int, float] = {}

        # First collect all matched rules, then optionally retrieve a Top-K subset
        # from the rule memory bank. This is Trend-4: rules can be retained in a
        # larger memory, while only the most relevant matched rules are injected
        # for the current candidate/state.
        candidates: List[Dict[str, Any]] = []
        for i, rule in enumerate(self.rules):
            rid = str(rule.get("id", f"rule_{i}"))
            if not self._is_enabled(rid, rule):
                continue
            match_strength = self._rule_match_strength(rule, features, stage)
            if match_strength <= 0.0:
                continue
            candidates.append({
                "rid": rid,
                "rule": rule,
                "match_strength": float(match_strength),
                "retrieval_score": _rule_relevance_score(rule, match_strength),
                "template_group": _rule_template_group_key(rule),
            })

        # Retrieval-time template de-dup: if several variants from the same
        # template match simultaneously, keep the most relevant one before Top-K.
        if self.rule_memory_retrieval and self.rule_retrieval_dedup_templates and candidates:
            grouped: Dict[str, Dict[str, Any]] = {}
            for c in candidates:
                key = str(c.get("template_group", c.get("rid", "")))
                old = grouped.get(key)
                if old is None or float(c.get("retrieval_score", -1e18)) > float(old.get("retrieval_score", -1e18)):
                    grouped[key] = c
            candidates = list(grouped.values())

        if self.rule_memory_retrieval and candidates:
            candidates.sort(key=lambda x: float(x.get("retrieval_score", -1e18)), reverse=True)
            if self.rule_retrieval_top_k > 0:
                candidates = candidates[: int(self.rule_retrieval_top_k)]

        matched: List[str] = []
        matched_details: List[Dict[str, Any]] = []
        for rank_idx, c in enumerate(candidates, 1):
            rid = str(c["rid"])
            rule = c["rule"]
            match_strength = float(c.get("match_strength", 1.0))
            retrieval_score = float(c.get("retrieval_score", 0.0))

            priority = str(rule.get("priority", "default"))
            rank = _priority_rank(priority)
            effect = str(rule.get("effect", "prefer")).lower()
            if effect in ("mask", "block"):
                raw_adjust = float(rule.get("adjust", -1e18))
                adjust = raw_adjust
                mask = True
            elif effect in ("avoid", "penalty", "negative"):
                raw_adjust = float(rule.get("adjust", self.default_avoid_adjust))
                adjust = raw_adjust * float(match_strength)
            else:
                raw_adjust = float(rule.get("adjust", self.default_prefer_adjust))
                adjust = raw_adjust * float(match_strength)

            matched.append(rid)
            matched_details.append({
                "id": rid,
                "priority": priority,
                "priority_rank": rank,
                "effect": effect,
                "match_strength": float(match_strength),
                "raw_adjust": float(raw_adjust),
                "adjust": float(adjust),
                "retrieval_score": float(retrieval_score),
                "retrieval_rank": int(rank_idx),
                "template_group": str(c.get("template_group", "")),
            })
            if effect not in ("mask", "block"):
                priority_adjusts[rank] = priority_adjusts.get(rank, 0.0) + adjust

        # 冲突处理：mask 永远覆盖；否则只采用最高优先级命中的加减分，
        # 同一优先级内累加。这对应“安全>可行性>截止/紧急>等待>效率>距离/均衡”。
        if mask:
            total_adjust = 0.0
        elif priority_adjusts:
            best_rank = min(priority_adjusts.keys())
            total_adjust = float(priority_adjusts.get(best_rank, 0.0))
        else:
            total_adjust = 0.0

        return {
            "adjust": float(total_adjust),
            "mask": bool(mask),
            "matched_rules": matched,
            "matched_rule_details": matched_details,
            "retrieval_enabled": bool(self.rule_memory_retrieval),
            "retrieved_rule_count": int(len(matched)),
            "features": features,
        }


def load_rule_guidance(path: Optional[str], enabled_rule_ids: Optional[Iterable[str]] = None,
                       **kwargs) -> Optional[RuleGuidanceEngine]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"rule guidance file not found: {path}")
    return RuleGuidanceEngine.from_json(str(p), enabled_rule_ids=enabled_rule_ids, **kwargs)

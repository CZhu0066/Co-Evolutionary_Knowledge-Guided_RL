# -*- coding: utf-8 -*-
"""
metrics.py
==========
Shared evaluation metrics for the dynamic AQC scheduling experiments.

The main-comparison protocol reports seven metrics:
    obj, train_obj, truck_total_wait, makespan, aqc_balance, runtime, completion_rate.
Here aqc_balance denotes AQC workload CV: std(B_k) / mean(B_k).

This module keeps those definitions in one place so that normal evaluation,
rule-guided evaluation, CoEvo-RGCD evaluation, and plotting scripts use the same
metric names and aggregation rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def compute_makespan_from_info(info: Dict[str, Any], env: Any = None) -> float:
    """Return the episode makespan.

    Priority:
      1) explicit info['makespan'] if already available;
      2) max over train finish-time dict info['F'];
      3) max completed task finish_time from env.tasks;
      4) env.t_now as a last resort.
    """
    if "makespan" in info:
        return safe_float(info.get("makespan"), 0.0)

    F = info.get("F", {}) or {}
    if isinstance(F, dict) and F:
        vals = [safe_float(v, 0.0) for v in F.values()]
        if vals:
            return float(max(vals))

    tasks = list(getattr(env, "tasks", []) or [])
    finish_times = [
        safe_float(getattr(t, "finish_time", 0.0), 0.0)
        for t in tasks
        if bool(getattr(t, "done", False))
    ]
    if finish_times:
        return float(max(finish_times))

    return safe_float(getattr(env, "t_now", 0.0), 0.0)


def _workload_cv(workloads: Sequence[Any]) -> float:
    vals = np.asarray([safe_float(x, 0.0) for x in workloads], dtype=float)
    if vals.size <= 1:
        return 0.0
    mean_val = float(np.mean(vals))
    if mean_val <= 1e-9:
        return 0.0
    return float(np.std(vals) / mean_val)


def compute_aqc_balance(env: Any = None, info: Dict[str, Any] | None = None) -> float:
    """Return AQC workload balance as a coefficient of variation.

    LoadBalance = std(B_1, ..., B_K) / mean(B_1, ..., B_K), where B_k is
    the cumulative processing workload assigned to AQC k. Lower is better.

    Priority:
      1) info['aqc_balance'] if already computed by the environment;
      2) info['aqc_workloads'] if present;
      3) env._current_aqc_balance() / env.aqc_workloads.

    Old files without workload information still remain readable.
    """
    info = info or {}
    if "aqc_balance" in info:
        return safe_float(info.get("aqc_balance"), 0.0)

    workloads_info = info.get("aqc_workloads", None)
    if isinstance(workloads_info, dict):
        return _workload_cv(list(workloads_info.values()))
    if isinstance(workloads_info, (list, tuple)):
        return _workload_cv(workloads_info)

    if env is not None:
        if hasattr(env, "_current_aqc_balance"):
            try:
                return safe_float(env._current_aqc_balance(), 0.0)
            except Exception:
                pass
        workloads = getattr(env, "aqc_workloads", None)
        if isinstance(workloads, dict):
            return _workload_cv(list(workloads.values()))
        if isinstance(workloads, (list, tuple)):
            return _workload_cv(workloads)

    return 0.0


def compute_completion(info: Dict[str, Any], env: Any = None) -> float:
    """Return 1.0 if all non-canceled tasks are complete, else 0.0."""
    if "completion" in info:
        return 1.0 if bool(info.get("completion")) else 0.0
    if "all_done" in info:
        return 1.0 if bool(info.get("all_done")) else 0.0

    tasks = list(getattr(env, "tasks", []) or [])
    if not tasks:
        return 0.0
    unfinished = [
        t for t in tasks
        if (not bool(getattr(t, "done", False))) and (not bool(getattr(t, "canceled", False)))
    ]
    return 1.0 if not unfinished else 0.0


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-instance evaluation rows into report-ready metrics."""
    if not rows:
        return {}

    def arr(key: str) -> np.ndarray:
        return np.asarray([safe_float(r.get(key, 0.0), 0.0) for r in rows], dtype=float)

    obj = arr("obj")
    train_obj = arr("train_obj")
    truck_wait = arr("truck_total_wait")
    makespan = arr("makespan")
    aqc_balance = arr("aqc_balance")
    runtime = arr("runtime_s")
    reward = arr("reward")
    length = arr("length")
    completion = arr("completion")
    triggered = arr("guidance_triggered")
    masked = arr("guidance_masked")
    mean_rule_strength = arr("guidance_mean_rule_strength")
    mean_trigger_strength = arr("guidance_mean_trigger_strength")
    matched_rule_count = arr("guidance_matched_rule_count")
    trigger_strength_sum = arr("guidance_trigger_strength_sum")
    high_strength_triggered = arr("guidance_high_strength_triggered")

    return {
        "n": int(len(rows)),
        "mean_obj": float(np.mean(obj)),
        "median_obj": float(np.median(obj)),
        "std_obj": float(np.std(obj)),
        "p90_obj": float(np.percentile(obj, 90)),
        "max_obj": float(np.max(obj)),
        "mean_train_obj": float(np.mean(train_obj)),
        "median_train_obj": float(np.median(train_obj)),
        "std_train_obj": float(np.std(train_obj)),
        "mean_truck_wait": float(np.mean(truck_wait)),
        "mean_truck_total_wait": float(np.mean(truck_wait)),
        "median_truck_total_wait": float(np.median(truck_wait)),
        "std_truck_total_wait": float(np.std(truck_wait)),
        "mean_makespan": float(np.mean(makespan)),
        "median_makespan": float(np.median(makespan)),
        "std_makespan": float(np.std(makespan)),
        "mean_aqc_balance": float(np.mean(aqc_balance)),
        "median_aqc_balance": float(np.median(aqc_balance)),
        "std_aqc_balance": float(np.std(aqc_balance)),
        "mean_runtime_s": float(np.mean(runtime)),
        "median_runtime_s": float(np.median(runtime)),
        "std_runtime_s": float(np.std(runtime)),
        "completion_rate": float(np.mean(completion)),
        "mean_reward": float(np.mean(reward)),
        "std_reward": float(np.std(reward)),
        "mean_length": float(np.mean(length)),
        "mean_guidance_triggered": float(np.mean(triggered)),
        "mean_guidance_masked": float(np.mean(masked)),
        "mean_guidance_matched_rule_count": float(np.mean(matched_rule_count)),
        "mean_guidance_trigger_strength_sum": float(np.mean(trigger_strength_sum)),
        "mean_guidance_high_strength_triggered": float(np.mean(high_strength_triggered)),
        "mean_guidance_mean_rule_strength": float(np.mean(mean_rule_strength)),
        "mean_guidance_mean_trigger_strength": float(np.mean(mean_trigger_strength)),
    }


def normalize_method_label(label: str) -> str:
    """Canonical method names used in figures."""
    s = str(label).strip()
    aliases = {
        "greedy": "Greedy-MinObj",
        "greedy-minobj": "Greedy-MinObj",
        "ppo": "PPO-150k",
        "rppo": "RPPO-150k",
        "sac": "SAC-150k",
        "rgcd-infer": "SAC + RGCD-Infer",
        "coevo": "SAC + CoEvo-RGCD-R2",
    }
    return aliases.get(s.lower(), s)


MAIN_METRICS = [
    "obj",
    "train_obj",
    "truck_total_wait",
    "makespan",
    "aqc_balance",
    "runtime_s",
    "completion",
]

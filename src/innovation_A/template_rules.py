# -*- coding: utf-8 -*-
"""
template_rules.py
=================
场景感知模板规则库（Scenario-aware Template-guided Rules）。

本文件不再依赖“纯决策树蒸馏规则”，而是把人工预设的规则模板写成可执行
RuleGuidanceEngine 规则。阈值 value 和加/扣分 adjust 当前给出一套安全默认值，
后续可以由模板补全算法基于轨迹/验证集自动替换。

重要建模假设：
- load_truck 显式计算 cross 等待、AQC 等待、total_wait；
- unload_truck 假设车辆供给充足，不计算卡车等待时间，只受 TruckSlot 可用性、
  位置占用、AQC 可达性/安全性约束。
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# 冲突优先级：安全 > 可行性 > 截止/紧急 > 等待 > 效率 > 距离/均衡
PRIORITY_ORDER = [
    "safety",
    "feasibility",
    "deadline",
    "waiting",
    "efficiency",
    "distance_balance",
]


def _r(rule_id: str, scenario: str, stage: str, conditions: List[Dict[str, Any]],
       effect: str, target: Dict[str, Any], adjust: float, priority: str,
       description: str, slots: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构造统一模板规则。slots 仅作为后续算法补全的元信息，不参与实时匹配。"""
    return {
        "rule_id": rule_id,
        "id": rule_id,
        "scenario": scenario,
        "stage": stage,
        "description": description,
        "conditions": conditions,
        "action": {
            "effect": effect,
            "target": target,
            "adjust": adjust,
        },
        # 兼容老 RuleGuidanceEngine 字段
        "effect": effect,
        "target": target,
        "adjust": adjust,
        "priority": priority,
        "source": "scenario_template_rules",
        "slots": slots or {},
    }


GLOBAL_RULES: List[Dict[str, Any]] = [
    _r(
        "TPL_GLOBAL_task_urgent_prefer", "all", "task",
        [{"feature": "task_urgency_boost", "op": ">", "value": 0}],
        "prefer", {"task_policy": "urgent_task"}, 120.0, "deadline",
        "如果任务被扰动模块标记为 urgent，则优先选择该任务。",
        {"adjust": [80, 160]},
    ),
    _r(
        "TPL_GLOBAL_task_train_not_arrived_avoid", "all", "task",
        [{"feature": "task_train_not_arrived", "op": "eq", "value": True}],
        "avoid", {"train_policy": "not_arrived_train"}, -150.0, "feasibility",
        "如果某列车尚未到达，则暂时避免选择该列车任务。",
        {"adjust": [-300, -80]},
    ),
    _r(
        "TPL_GLOBAL_task_train_planning_blocked_mask", "all", "task",
        [{"feature": "task_train_planning_blocked", "op": "eq", "value": True}],
        "mask", {"train_policy": "planning_blocked_train"}, -1_000_000.0, "feasibility",
        "如果某列车处于 planning_blocked，则屏蔽该列车任务。",
    ),
    _r(
        "TPL_GLOBAL_aqc_broken_mask", "all", "aqc",
        [{"feature": "aqc_is_broken", "op": "eq", "value": True}],
        "mask", {"aqc_policy": "broken_aqc"}, -1_000_000.0, "safety",
        "如果某台 AQC 处于故障或不可用状态，则屏蔽该 AQC。",
    ),
    _r(
        "TPL_GLOBAL_aqc_conflict_avoid", "all", "aqc",
        [{"feature": "aqc_conflict_risk", "op": ">", "value": 0.60}],
        "avoid", {"aqc_policy": "high_conflict_risk"}, -90.0, "safety",
        "如果 AQC 间距过近或容易冲突，则降低该 AQC 分数。",
        {"aqc_conflict_risk": [0.4, 0.8], "adjust": [-150, -50]},
    ),
]


SCENARIO_RULES: List[Dict[str, Any]] = [
    # =========================================================
    # 一装载一卸载：task
    # =========================================================
    _r(
        "TPL_1LU_task_load_pressure", "1load_1unload", "task",
        [
            {"feature": "load_minus_unload_pressure", "op": ">", "value": 0.25},
            {"feature": "task_kind", "op": "in", "value": ["load_yard", "load_truck"]},
        ],
        "prefer", {"task_kind_group": "load"}, 80.0, "waiting",
        "如果装载方向压力明显高于卸载方向，则优先选择装载类任务。",
        {"load_minus_unload_pressure": [0.1, 0.5], "adjust": [40, 120]},
    ),
    _r(
        "TPL_1LU_task_unload_pressure", "1load_1unload", "task",
        [
            {"feature": "unload_minus_load_pressure", "op": ">", "value": 0.22},
            {"feature": "task_kind", "op": "in", "value": ["unload_yard", "unload_truck"]},
        ],
        "prefer", {"task_kind_group": "unload"}, 75.0, "waiting",
        "如果卸载方向压力明显高于装载方向，则优先选择卸载类任务。",
        {"unload_minus_load_pressure": [0.1, 0.5], "adjust": [40, 120]},
    ),
    _r(
        "TPL_1LU_task_load_truck_wait", "1load_1unload", "task",
        [
            {"feature": "load_truck_wait_norm", "op": ">", "value": 0.001},
            {"feature": "task_kind", "op": "eq", "value": "load_truck"},
        ],
        "prefer", {"task_kind": "load_truck"}, 90.0, "waiting",
        "如果 load_truck 卡车等待时间过长，则优先选择 load_truck。",
        {"load_truck_wait_norm": [0.0001, 0.02], "adjust": [5, 15]},
    ),
    _r(
        "TPL_1LU_task_deadline_pressure", "1load_1unload", "task",
        [{"feature": "task_train_deadline_pressure", "op": ">", "value": 0.68}],
        "prefer", {"train_policy": "candidate_task_train"}, 100.0, "deadline",
        "如果某列车接近截止时间，则优先选择该列车相关任务。",
        {"task_train_deadline_pressure": [0.45, 0.85], "adjust": [70, 160]},
    ),

    # 一装载一卸载：destination
    _r(
        "TPL_1LU_dest_unload_yard_near_low_stack", "1load_1unload", "destination",
        [
            {"feature": "task_kind", "op": "eq", "value": "unload_yard"},
            {"feature": "dest_kind", "op": "eq", "value": "YardSlot"},
            {"feature": "dest_distance_to_task", "op": "<=", "value": 6},
            {"feature": "yard_stack_height", "op": "<=", "value": 2},
            {"feature": "dest_occupied", "op": "eq", "value": False},
        ],
        "prefer", {"dest_kind": "YardSlot"}, 65.0, "distance_balance",
        "unload_yard 优先选择距离近、堆高较低、未占用的堆场位。",
        {"dest_distance_to_task": [4, 12], "yard_stack_height": [1, 4], "adjust": [40, 100]},
    ),
    _r(
        "TPL_1LU_dest_load_yard_near_car", "1load_1unload", "destination",
        [
            {"feature": "task_kind", "op": "eq", "value": "load_yard"},
            {"feature": "dest_kind", "op": "eq", "value": "TrainCar"},
            {"feature": "dest_occupied", "op": "eq", "value": False},
            {"feature": "dest_distance_to_task", "op": "<=", "value": 8},
        ],
        "prefer", {"dest_kind": "TrainCar"}, 60.0, "distance_balance",
        "load_yard 优先选择距离任务起点近、且对应列车未满的车厢。",
        {"dest_distance_to_task": [4, 14], "adjust": [40, 100]},
    ),
    _r(
        "TPL_1LU_dest_load_truck_fast_cross", "1load_1unload", "destination",
        [
            {"feature": "task_kind", "op": "eq", "value": "load_truck"},
            {"feature": "tf_wait_cross", "op": "<=", "value": 25},
            {"feature": "tf_total_wait", "op": "<=", "value": 60},
        ],
        "prefer", {"dest_policy": "low_cross_wait"}, 70.0, "waiting",
        "load_truck 优先选择 cross 等待时间较短、AQC 可较快接入的位置。",
        {"tf_wait_cross": [10, 60], "tf_total_wait": [30, 120], "adjust": [40, 120]},
    ),
    _r(
        "TPL_1LU_dest_unload_truck_available_slot", "1load_1unload", "destination",
        [
            {"feature": "task_kind", "op": "eq", "value": "unload_truck"},
            {"feature": "dest_kind", "op": "eq", "value": "TruckSlot"},
            {"feature": "dest_occupied", "op": "eq", "value": False},
            {"feature": "dest_available_time", "op": "<=", "value": "current_time"},
            {"feature": "dest_distance_to_task", "op": "<=", "value": 8},
        ],
        "prefer", {"dest_kind": "TruckSlot"}, 65.0, "feasibility",
        "unload_truck 不计算等待时间，优先选择可用、未占用、距离近的 truck slot。",
        {"dest_distance_to_task": [4, 14], "adjust": [40, 100]},
    ),

    # 一装载一卸载：aqc
    _r(
        "TPL_1LU_aqc_early_finish", "1load_1unload", "aqc",
        [{"feature": "tf_finish", "op": "<=", "value": 520}],
        "prefer", {"aqc_policy": "early_finish"}, 85.0, "efficiency",
        "优先选择预计完成时间最早的 AQC。",
        {"tf_finish": [300, 900], "adjust": [50, 130]},
    ),
    _r(
        "TPL_1LU_aqc_underloaded", "1load_1unload", "aqc",
        [{"feature": "aqc_available_delta", "op": "<", "value": -80}],
        "prefer", {"aqc_policy": "underloaded"}, 55.0, "distance_balance",
        "如果某台 AQC 当前负载明显低于平均负载，则适当提高该 AQC 分数。",
        {"aqc_available_delta": [-200, -20], "adjust": [30, 90]},
    ),
    _r(
        "TPL_1LU_aqc_near_task", "1load_1unload", "aqc",
        [{"feature": "aqc_gap_to_task", "op": "<=", "value": 7}],
        "prefer", {"aqc_policy": "near_task"}, 60.0, "distance_balance",
        "如果某台 AQC 离任务起点较近，则优先选择该 AQC。",
        {"aqc_gap_to_task": [4, 12], "adjust": [30, 100]},
    ),

    # =========================================================
    # 两装载：task
    # =========================================================
    _r(
        "TPL_2LOAD_task_load_truck_wait", "2load", "task",
        [
            {"feature": "remaining_ratio_load_truck", "op": ">", "value": 0.10},
            {"feature": "load_truck_wait_norm", "op": ">", "value": 0.001},
            {"feature": "task_kind", "op": "eq", "value": "load_truck"},
        ],
        "prefer", {"task_kind": "load_truck"}, 95.0, "waiting",
        "如果 load_truck 剩余比例较高且卡车等待较大，则优先选择 load_truck。",
        {"remaining_ratio_load_truck": [0.1, 0.7], "load_truck_wait_norm": [0.0001, 0.02], "adjust": [5, 15]},
    ),
    _r(
        "TPL_2LOAD_task_load_yard_near", "2load", "task",
        [
            {"feature": "remaining_ratio_load_yard", "op": ">", "value": 0.38},
            {"feature": "task_best_destination_distance", "op": "<=", "value": 9},
            {"feature": "task_kind", "op": "eq", "value": "load_yard"},
        ],
        "prefer", {"task_kind": "load_yard"}, 75.0, "efficiency",
        "如果 load_yard 剩余比例较高且堆场到车厢距离较短，则优先选择 load_yard。",
        {"remaining_ratio_load_yard": [0.2, 0.7], "task_best_destination_distance": [5, 15], "adjust": [40, 110]},
    ),
    _r(
        "TPL_2LOAD_task_low_completion_train", "2load", "task",
        [
            {"feature": "task_train_completion_gap", "op": ">", "value": 0.25},
            {"feature": "task_kind", "op": "in", "value": ["load_yard", "load_truck"]},
        ],
        "prefer", {"train_policy": "low_completion_train"}, 85.0, "deadline",
        "如果某列车完成率明显低于另一列车，则优先选择低完成率列车的装载任务。",
        {"task_train_completion_gap": [0.1, 0.5], "adjust": [50, 130]},
    ),

    # 两装载：destination
    _r(
        "TPL_2LOAD_dest_load_yard_near_car", "2load", "destination",
        [
            {"feature": "task_kind", "op": "eq", "value": "load_yard"},
            {"feature": "dest_kind", "op": "eq", "value": "TrainCar"},
            {"feature": "dest_occupied", "op": "eq", "value": False},
            {"feature": "dest_distance_to_task", "op": "<=", "value": 7},
        ],
        "prefer", {"dest_kind": "TrainCar"}, 65.0, "distance_balance",
        "对于 load_yard，优先选择距离堆场起点近、且列车车厢空闲的目的车厢。",
        {"dest_distance_to_task": [4, 14], "adjust": [40, 100]},
    ),
    _r(
        "TPL_2LOAD_dest_load_truck_fast_cross", "2load", "destination",
        [
            {"feature": "task_kind", "op": "eq", "value": "load_truck"},
            {"feature": "tf_wait_cross", "op": "<=", "value": 20},
            {"feature": "tf_wait_aqc", "op": "<=", "value": 35},
        ],
        "prefer", {"dest_policy": "fast_cross_aqc_access"}, 80.0, "waiting",
        "对于 load_truck，优先选择 cross 等待时间短、AQC 可快速接入的车厢。",
        {"tf_wait_cross": [10, 60], "tf_wait_aqc": [15, 80], "adjust": [50, 130]},
    ),
    _r(
        "TPL_2LOAD_dest_long_aqc_move_avoid", "2load", "destination",
        [
            {"feature": "dest_kind", "op": "eq", "value": "TrainCar"},
            {"feature": "preferred_aqc_gap_to_dest", "op": ">", "value": 18},
        ],
        "avoid", {"dest_policy": "long_aqc_move"}, -70.0, "distance_balance",
        "如果某个目的车厢会导致 AQC 长距离横移，则降低该目的位置分数。",
        {"preferred_aqc_gap_to_dest": [12, 28], "adjust": [-120, -40]},
    ),

    # 两装载：aqc
    _r(
        "TPL_2LOAD_aqc_near_task", "2load", "aqc",
        [{"feature": "aqc_gap_to_task", "op": "<=", "value": 6}],
        "prefer", {"aqc_policy": "near_task"}, 70.0, "distance_balance",
        "如果某台 AQC 到任务起点距离最短，则优先选择。",
        {"aqc_gap_to_task": [4, 12], "adjust": [40, 110]},
    ),
    _r(
        "TPL_2LOAD_aqc_early_finish", "2load", "aqc",
        [{"feature": "tf_finish", "op": "<=", "value": 500}],
        "prefer", {"aqc_policy": "early_finish"}, 90.0, "efficiency",
        "如果某台 AQC 预计完成时间最早，则优先选择。",
        {"tf_finish": [300, 900], "adjust": [50, 140]},
    ),
    _r(
        "TPL_2LOAD_aqc_overloaded_avoid", "2load", "aqc",
        [{"feature": "aqc_overload", "op": ">", "value": 100}],
        "avoid", {"aqc_policy": "overloaded"}, -65.0, "distance_balance",
        "如果某台 AQC 已经执行任务过多或可用时间明显滞后，则降低其分数，避免负载失衡。",
        {"aqc_overload": [50, 250], "adjust": [-120, -30]},
    ),

    # =========================================================
    # 两卸载：task
    # =========================================================
    _r(
        "TPL_2UNLOAD_task_unload_yard_available", "2unload", "task",
        [
            {"feature": "remaining_ratio_unload_yard", "op": ">", "value": 0.36},
            {"feature": "available_yard_slot_ratio", "op": ">", "value": 0.25},
            {"feature": "task_kind", "op": "eq", "value": "unload_yard"},
        ],
        "prefer", {"task_kind": "unload_yard"}, 80.0, "feasibility",
        "如果 unload_yard 剩余比例较高，且可用堆场位充足，则优先选择 unload_yard。",
        {"remaining_ratio_unload_yard": [0.2, 0.7], "available_yard_slot_ratio": [0.1, 0.6], "adjust": [50, 130]},
    ),
    _r(
        "TPL_2UNLOAD_task_unload_truck_available", "2unload", "task",
        [
            {"feature": "remaining_ratio_unload_truck", "op": ">", "value": 0.35},
            {"feature": "available_truck_slot_ratio", "op": ">", "value": 0.30},
            {"feature": "task_kind", "op": "eq", "value": "unload_truck"},
        ],
        "prefer", {"task_kind": "unload_truck"}, 75.0, "feasibility",
        "unload_truck 不使用等待压力；若剩余比例高且可用 truck slot 充足，则优先选择 unload_truck。",
        {"remaining_ratio_unload_truck": [0.2, 0.7], "available_truck_slot_ratio": [0.1, 0.7], "adjust": [45, 120]},
    ),
    _r(
        "TPL_2UNLOAD_task_low_unload_completion_train", "2unload", "task",
        [
            {"feature": "task_train_unload_completion_gap", "op": ">", "value": 0.20},
            {"feature": "task_kind", "op": "in", "value": ["unload_yard", "unload_truck"]},
        ],
        "prefer", {"train_policy": "low_unload_completion_train"}, 80.0, "deadline",
        "如果某列车卸载完成率较低，则优先选择该列车卸载任务。",
        {"task_train_unload_completion_gap": [0.1, 0.5], "adjust": [50, 130]},
    ),
    _r(
        "TPL_2UNLOAD_task_low_completion_train", "2unload", "task",
        [
            {"feature": "task_train_completion_gap", "op": ">", "value": 0.24},
            {"feature": "task_kind", "op": "in", "value": ["unload_yard", "unload_truck"]},
        ],
        "prefer", {"train_policy": "low_completion_train"}, 75.0, "deadline",
        "如果某列车完成率明显低于另一列车，则优先选择低完成率列车的卸载任务。",
        {"task_train_completion_gap": [0.1, 0.5], "adjust": [45, 120]},
    ),

    # 两卸载：destination
    _r(
        "TPL_2UNLOAD_dest_unload_yard_near_low_stack", "2unload", "destination",
        [
            {"feature": "task_kind", "op": "eq", "value": "unload_yard"},
            {"feature": "dest_kind", "op": "eq", "value": "YardSlot"},
            {"feature": "dest_distance_to_task", "op": "<=", "value": 6},
            {"feature": "yard_stack_height", "op": "<=", "value": 2},
            {"feature": "dest_occupied", "op": "eq", "value": False},
        ],
        "prefer", {"dest_kind": "YardSlot"}, 70.0, "distance_balance",
        "对于 unload_yard，优先选择距离任务起点近、堆高较低、未占用的堆场位置。",
        {"dest_distance_to_task": [4, 12], "yard_stack_height": [1, 4], "adjust": [45, 110]},
    ),
    _r(
        "TPL_2UNLOAD_dest_unload_truck_available_slot", "2unload", "destination",
        [
            {"feature": "task_kind", "op": "eq", "value": "unload_truck"},
            {"feature": "dest_kind", "op": "eq", "value": "TruckSlot"},
            {"feature": "dest_occupied", "op": "eq", "value": False},
            {"feature": "dest_available_time", "op": "<=", "value": "current_time"},
            {"feature": "dest_distance_to_task", "op": "<=", "value": 8},
        ],
        "prefer", {"dest_kind": "TruckSlot"}, 65.0, "feasibility",
        "对于 unload_truck，优先选择 truck slot 可用、未占用、距离任务起点近的目的位置。",
        {"dest_distance_to_task": [4, 14], "adjust": [40, 100]},
    ),
    _r(
        "TPL_2UNLOAD_dest_long_aqc_move_avoid", "2unload", "destination",
        [{"feature": "min_aqc_gap_to_dest", "op": ">", "value": 16}],
        "avoid", {"dest_policy": "long_aqc_move"}, -60.0, "distance_balance",
        "如果某个目的位置会造成 AQC 横向移动过大，则降低该目的位置分数。",
        {"min_aqc_gap_to_dest": [12, 28], "adjust": [-120, -40]},
    ),

    # 两卸载：aqc
    _r(
        "TPL_2UNLOAD_aqc_early_available", "2unload", "aqc",
        [{"feature": "aqc_available_time", "op": "<=", "value": 300}],
        "prefer", {"aqc_policy": "early_available"}, 70.0, "efficiency",
        "优先选择可用时间早的 AQC。",
        {"aqc_available_time": [100, 700], "adjust": [40, 110]},
    ),
    _r(
        "TPL_2UNLOAD_aqc_near_unload_start", "2unload", "aqc",
        [
            {"feature": "task_kind", "op": "in", "value": ["unload_yard", "unload_truck"]},
            {"feature": "aqc_gap_to_task", "op": "<=", "value": 6},
        ],
        "prefer", {"aqc_policy": "near_unload_start"}, 65.0, "distance_balance",
        "优先选择离卸载起点近的 AQC。",
        {"aqc_gap_to_task": [4, 12], "adjust": [40, 105]},
    ),
]


def build_template_rules(scenarios: Optional[Iterable[str]] = None,
                         include_global: bool = True) -> List[Dict[str, Any]]:
    """返回可直接给 RuleGuidanceEngine 使用的模板规则。"""
    allowed = set(scenarios or ["1load_1unload", "2load", "2unload"])
    rules: List[Dict[str, Any]] = []
    if include_global:
        rules.extend(deepcopy(GLOBAL_RULES))
    for r in SCENARIO_RULES:
        if r.get("scenario") in allowed:
            rules.append(deepcopy(r))
    return rules


def write_template_rules(path: str, scenarios: Optional[Iterable[str]] = None,
                         include_global: bool = True) -> Dict[str, Any]:
    rules = build_template_rules(scenarios=scenarios, include_global=include_global)
    data = {
        "method": "scenario_aware_template_guided_rules",
        "description": "人工预设三场景×三阶段规则模板；阈值和adjust可由后续算法补全。unload_truck不建模卡车等待。",
        "priority_order": PRIORITY_ORDER,
        "n_rules": len(rules),
        "rules": rules,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

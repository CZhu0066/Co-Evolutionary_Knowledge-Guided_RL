# -*- coding: utf-8 -*-
"""
scoring.py
==========
评分函数：5 个 score_* + 3 个 候选选择器。

迁移自 KG-PPO_0_test.py，并加入 paradigm 1 的 3 处扰动语义扩展：
  1. score_task 跳过 canceled 任务
  2. score_task 跳过被阻塞列车的任务
  3. score_task 加入 urgency_boost 加分

5 个 score 函数对应 θ 的 9 维分解：
  - score_task         → θ_i (3维)
  - score_destination  → θ_g (2维) [空间]
  - score_cross        → θ_p (2维) [时序]  ← KG-PPO_0_test 已做的θ分解
  - score_aqc_normal   → θ_k (2维)
  - score_aqc_load_truck → θ_k (2维)
"""
from typing import Dict, List, Optional, Set

import numpy as np

from config.constants import (
    D1, V_IT,
    TRUCK_PRIORITY_BIAS, TRUCK_AGE_WEIGHT,
    TRUCK_WAIT_WEIGHT_TASK, TRUCK_AQC_WAIT_WEIGHT_TASK,
    TRUCK_CROSS_WAIT_WEIGHT_TASK,
    TRUCK_PRESSURE_THRESHOLD, TRUCK_PRESSURE_BONUS,
    TRUCK_SERVICE_AQC_K, CROSS_NEAR_AQC_K, CAR_NEAR_CROSS_K,
)
from src.core.data_classes import (
    Task, AQCState, TrainCar, YardSlot, TruckCross, TruckSlot,
)
from src.core.simulators import (
    simulate_task_times, simulate_load_truck_times,
    loaded_move_time, estimate_system_time, check_aqc_safety,
)


# ============================================================
# 候选选择器（candidate selectors）
# ============================================================
def get_truck_service_aqcs(task: Task, aqcs: List[AQCState],
                           top_k: int = TRUCK_SERVICE_AQC_K) -> List[AQCState]:
    """
    选最适合服务该 truck 的 top_k 个 AQC。

    排序键：
      1. wait_after_arrival 小（AQC 越早空闲越好）
      2. spatial_gap 小（AQC 离 truck 越近越好）
      3. available_time 早（早空闲优先）
    """
    arrival = float(task.arrival_time)

    def key(aqc: AQCState):
        wait_after = max(0.0, float(aqc.available_time) - arrival)
        spatial_gap = abs(float(aqc.cur_bay) - float(task.init_bay))
        return (wait_after, spatial_gap, float(aqc.available_time))

    return sorted(aqcs, key=key)[:top_k]


def get_crosses_near_aqc(aqc: AQCState, crosses: List[TruckCross],
                         top_k: int = CROSS_NEAR_AQC_K) -> List[TruckCross]:
    """选 AQC 当前 bay 附近的 top_k 个 cross"""
    return sorted(
        crosses,
        key=lambda c: abs(float(c.bay) - float(aqc.cur_bay))
    )[:top_k]


def get_cars_near_cross_or_task(task: Task, cross: TruckCross,
                                cars: List[TrainCar],
                                top_k: int = CAR_NEAR_CROSS_K) -> List[TrainCar]:
    """
    选合适车厢。优先：
      1. 离 cross 近（减少 AQC 带箱移动）
      2. 离 task 初始 bay 近
    """
    valid = [c for c in cars if not c.occupied]
    return sorted(
        valid,
        key=lambda c: (
            abs(float(c.bay) - float(cross.bay)),
            abs(float(c.bay) - float(task.init_bay)),
        )
    )[:top_k]


# ============================================================
# 1. score_task —— 任务选择（θ_i, 3维）
# ============================================================
def score_task(task: Task,
               aqcs: List[AQCState],
               cars_by_train: Dict[int, List[TrainCar]],
               slots: List[YardSlot],
               truck_slots: List[TruckSlot],
               crosses: List[TruckCross],
               current_height: Dict,
               done_tasks: List[Task],
               theta_i: np.ndarray,
               A1: Dict[int, float],
               phi: Dict[int, float],
               train_planning_blocked: Optional[Set[int]] = None) -> float:
    """
    任务选择评分函数。

    paradigm 1 扩展（vs 原版）：
      - 跳过 canceled 任务
      - 跳过 train_planning_blocked 中列车的任务（列车未到达就不能调度它的任务）
      - 加入 urgency_boost
    """
    # ---- paradigm 1: 跳过 canceled / done ----
    if task.done or getattr(task, "canceled", False):
        return -1e18

    # ---- paradigm 1: 跳过被阻塞列车的任务 ----
    if (train_planning_blocked is not None
            and task.train_id in train_planning_blocked):
        return -1e18

    # ---- 原有打分逻辑 ----
    theta_i = np.asarray(theta_i, dtype=np.float32)
    wi = np.exp(np.clip(theta_i, -2.0, 2.0))

    best_cost = 1e18

    if task.kind == "load_yard":
        for aqc in aqcs:
            for car in cars_by_train.get(task.train_id, []):
                if car.occupied:
                    continue
                s, f = simulate_task_times(aqc, task, car, A1)
                if check_aqc_safety(s, f, task.init_bay, car.bay, aqc.id, done_tasks):
                    best_cost = min(best_cost, f)

    elif task.kind == "load_truck":
        current_time = estimate_system_time(aqcs, done_tasks)
        truck_age = max(0.0, current_time - float(task.arrival_time))

        candidate_aqcs = get_truck_service_aqcs(task, aqcs, top_k=TRUCK_SERVICE_AQC_K)

        for aqc in candidate_aqcs:
            candidate_crosses = get_crosses_near_aqc(aqc, crosses, top_k=CROSS_NEAR_AQC_K)
            for cross in candidate_crosses:
                candidate_cars = get_cars_near_cross_or_task(
                    task, cross,
                    cars_by_train.get(task.train_id, []),
                    top_k=CAR_NEAR_CROSS_K,
                )
                for car in candidate_cars:
                    tf = simulate_load_truck_times(aqc, task, cross, car, A1)
                    if not check_aqc_safety(
                            tf["start"], tf["finish"],
                            task.init_bay, car.bay, aqc.id, done_tasks):
                        continue
                    pressure_bonus = 0.0
                    if truck_age > TRUCK_PRESSURE_THRESHOLD:
                        pressure_bonus = (TRUCK_PRESSURE_BONUS
                                          + 0.5 * (truck_age - TRUCK_PRESSURE_THRESHOLD))
                    cost = (
                        tf["finish"]
                        + TRUCK_WAIT_WEIGHT_TASK * tf["total_wait"]
                        + TRUCK_AQC_WAIT_WEIGHT_TASK * tf["wait_aqc"]
                        + TRUCK_CROSS_WAIT_WEIGHT_TASK * tf["wait_cross"]
                        - TRUCK_AGE_WEIGHT * truck_age
                        - pressure_bonus
                    )
                    best_cost = min(best_cost, cost)

    elif task.kind == "unload_yard":
        for aqc in aqcs:
            for slot in slots:
                if slot.occupied:
                    continue
                h0 = current_height.get((slot.row, slot.bay), -1)
                if int(slot.tier) != int(h0):
                    continue
                s, f = simulate_task_times(aqc, task, slot, A1)
                if check_aqc_safety(s, f, task.init_bay, slot.bay, aqc.id, done_tasks):
                    best_cost = min(best_cost, f)

    elif task.kind == "unload_truck":
        for aqc in aqcs:
            for slot in truck_slots:
                if slot.occupied:
                    continue
                s, f = simulate_task_times(aqc, task, slot, A1)
                if check_aqc_safety(s, f, task.init_bay, slot.bay, aqc.id, done_tasks):
                    best_cost = min(best_cost, f)

    if best_cost >= 1e17:
        return -1e9

    train_weight = float(phi.get(task.train_id, 0.0))
    truck_bias = TRUCK_PRIORITY_BIAS if task.kind == "load_truck" else 0.0

    base_score = (
        -float(wi[0]) * best_cost
        - 0.05 * float(wi[1]) * abs(float(task.init_bay))
        + 0.5 * float(wi[2]) * train_weight
        + truck_bias
    )

    # ---- paradigm 1: urgency_boost 直接加进总分 ----
    urgency = float(getattr(task, "urgency_boost", 0.0))

    return base_score + urgency


# ============================================================
# 2. score_destination —— 目的地选择（θ_g, 2维, 空间语义）
# ============================================================
def score_destination(task: Task, dest, theta_g: np.ndarray,
                      current_height: Dict) -> float:
    """
    目的地评分（堆场 slot / 车厢 / 卡车 slot）。

    θ_g 专管空间最优：
      - θ_g[0]: 载箱移动时间权重
      - θ_g[1]: bay 距离权重（短距离更优）
    """
    if task.kind == "unload_yard":
        h0 = current_height.get((dest.row, dest.bay), -1)
        if int(dest.tier) != int(h0):
            return -1e9

    move_t = loaded_move_time(task, dest)
    return (
        -float(theta_g[0]) * move_t
        - float(theta_g[1]) * abs(dest.bay - task.init_bay)
    )


# ============================================================
# 3. score_cross —— 交接点选择（θ_p, 2维, 时序语义）
# ============================================================
def score_cross(task: Task, cross: TruckCross, theta_p: np.ndarray) -> float:
    """
    交接点（cross）评分 —— 卡车时序协调语义。

    与 score_destination 解耦：θ_p 专管 cross 时序，θ_g 专管堆场空间。
    """
    theta_p = np.asarray(theta_p, dtype=np.float32)
    wp = np.exp(np.clip(theta_p, -2.0, 2.0))

    h2 = float(task.arrival_time) + abs(float(cross.bay) - float(task.init_bay)) * D1 / V_IT
    wait_cross = max(0.0, float(cross.available_time) - h2)
    dist = abs(float(cross.bay) - float(task.init_bay))

    return (
        -0.5 * float(wp[0]) * h2
        - 1.5 * wait_cross
        - 0.15 * dist
        - 0.05 * float(wp[1]) * dist
    )


# ============================================================
# 4. score_aqc_normal —— AQC 选择（θ_k, 2维, 一般任务）
# ============================================================
def score_aqc_normal(task: Task, dest, aqc: AQCState,
                     aqcs: List[AQCState], theta_k: np.ndarray,
                     A1: Dict[int, float]) -> float:
    """
    一般任务（load_yard, unload_yard, unload_truck）的 AQC 选择评分。

    θ_k[0]: finish_time 权重（早完成优先）
    θ_k[1]: imbalance 权重（负载均衡）
    """
    _, f = simulate_task_times(aqc, task, dest, A1)
    mean_t = float(np.mean([a.available_time for a in aqcs])) if aqcs else 0.0
    imbalance = abs(aqc.available_time - mean_t)
    return -float(theta_k[0]) * f - float(theta_k[1]) * imbalance


# ============================================================
# 5. score_aqc_load_truck —— AQC 选择（θ_k, 2维, load_truck 任务）
# ============================================================
def score_aqc_load_truck(task: Task, cross: TruckCross, car: TrainCar,
                         aqc: AQCState, aqcs: List[AQCState],
                         theta_k: np.ndarray,
                         A1=None) -> float:
    """
    load_truck 任务的 AQC 选择评分。

    v10 修正：
      - 增加 A1 参数
      - 让 load_truck 评分阶段也遵守火车到达时间约束
      - 避免火车未到时提前评分/调度

    设计：
      - 加大 wait_aqc 权重（卡车等待 AQC 是大问题）
      - 加入 AQC 负载不均衡和过载惩罚
      - exp(theta_k) 保证权重为正
    """
    theta_k = np.asarray(theta_k, dtype=np.float32)
    wk = np.exp(np.clip(theta_k, -2.0, 2.0))

    tf = simulate_load_truck_times(aqc, task, cross, car, A1)

    mean_t = float(np.mean([a.available_time for a in aqcs])) if aqcs else 0.0
    imbalance = abs(float(aqc.available_time) - mean_t)
    overload_penalty = max(0.0, float(aqc.available_time) - mean_t)

    return (
        -0.7 * float(wk[0]) * tf["finish"]
        - 0.6 * float(wk[1]) * imbalance
        - 2.5 * tf["wait_aqc"]
        - 1.5 * tf["total_wait"]
        - 0.8 * tf["wait_cross"]
        - 0.5 * overload_penalty
    )

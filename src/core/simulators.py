# -*- coding: utf-8 -*-
"""
simulators.py
=============
时间仿真 + 目标函数计算，纯函数模块。

迁移自 KG-PPO_0_test.py 的对应函数。所有函数无状态、不依赖 env 实例，
便于单元测试和在反事实评估（创新A4）中独立调用。
"""
from typing import Dict, List, Tuple

import numpy as np

from config.constants import (
    D1, D2, D3, V_AQC_B, V_AQC_S, V_IT, V_TROLLEY,
    H_AQC, GAMMA_AQC, ALPHA_OBJ,
)
from src.core.data_classes import Task, AQCState, TrainCar, TruckCross


# ============================================================
# 基础时间计算
# ============================================================
def max_move_time(row1: float, bay1: float, row2: float, bay2: float,
                  d1: float, d2: float, v_b: float, v_s: float) -> float:
    """
    AQC 从 (row1, bay1) 移动到 (row2, bay2) 的最大移动时间。

    AQC 大车（bay方向）和小车（row方向）可并行移动，所以总时间 = max(t_bay, t_row)。
    """
    t_row = abs(row1 - row2) * d2 / v_s if v_s > 0 else 0.0
    t_bay = abs(bay1 - bay2) * d1 / v_b if v_b > 0 else 0.0
    return max(t_row, t_bay)


def empty_move_time(aqc: AQCState, task: Task) -> float:
    """AQC 当前位置 → 任务起点的空载时间"""
    return max_move_time(
        aqc.cur_row, aqc.cur_bay, task.init_row, task.init_bay,
        D1, D2, V_AQC_B, V_AQC_S,
    )


def loaded_move_time(task: Task, dest) -> float:
    """任务起点 → 目的地的载箱时间。dest 可以是 TrainCar/YardSlot/TruckSlot 之一。"""
    return max_move_time(
        task.init_row, task.init_bay, dest.row, dest.bay,
        D1, D2, V_AQC_B, V_AQC_S,
    )


# ============================================================
# 完整任务时间仿真
# ============================================================
def simulate_task_times(aqc: AQCState, task: Task, dest,
                        A1: Dict[int, float]) -> Tuple[float, float]:
    """
    一般任务（load_yard / unload_yard / unload_truck）的开始/结束时间。

    考虑因素：
      - AQC 当前可用时间 + 空载移动到任务起点
      - 列车到达时间约束（A1[train_id]）
      - 载箱移动 + trolley 升降时间
    """
    e = empty_move_time(aqc, task)
    arrival = float(A1.get(task.train_id, 0.0))
    s = max(aqc.available_time + e, arrival + e)
    travel = loaded_move_time(task, dest)
    f = (s + travel
         + 1.5 * (H_AQC - task.init_tier * D3) / V_TROLLEY
         + 1.5 * (H_AQC - dest.tier * D3) / V_TROLLEY)
    return s, f


def simulate_load_truck_times(aqc: AQCState, task: Task,
                              cross: TruckCross, car: TrainCar,
                              A1: Dict[int, float] = None) -> Dict[str, float]:
    """
    load_truck 任务的复杂时间仿真。

    流程：
      1. 卡车从堆场入口（task.arrival_time）开到 cross 用 (cross.bay - task.init_bay) * D1 / V_IT
      2. h2 = 卡车到达 cross 的时刻
      3. o2 = max(h2, cross.available_time)  实际开始服务时刻
      4. AQC 从当前位置移动到 cross 用稍慢的速度（带箱）
      5. AQC 处理时间 = 移动到 cross 周围 + 4s 固定操作

    返回包含 start, finish, h2, o2, wait_cross, wait_aqc, total_wait 的字典。
    """
    move_to_cross = max_move_time(
        aqc.cur_row, aqc.cur_bay, cross.row, cross.bay,
        D1, D2, 0.25 * V_AQC_B, 0.3 * V_AQC_S,
    )
    h2 = float(task.arrival_time) + abs(cross.bay - task.init_bay) * D1 / V_IT
    o2 = max(h2, float(cross.available_time))

    # v10: load_truck 也必须满足列车到达约束。
    # 旧版本只考虑卡车到达 cross 和 AQC 可用时间，可能出现列车晚到但 AQC 提前装车。
    train_arrival = float((A1 or {}).get(task.train_id, 0.0))
    s = max(aqc.available_time + move_to_cross, o2, train_arrival)

    proc = (abs(car.row - cross.row) * D2 / (0.3 * V_AQC_S)
            + abs(car.bay - cross.bay) * D1 / (0.25 * V_AQC_B)
            + 4.0)
    f = s + proc
    wait_cross = o2 - h2
    wait_aqc = max(0.0, s - o2)
    return {
        "start": s,
        "finish": f,
        "h2": h2,
        "o2": o2,
        "wait_cross": wait_cross,
        "wait_aqc": wait_aqc,
        "total_wait": wait_cross + wait_aqc,
    }


# ============================================================
# 系统状态估计
# ============================================================
def estimate_system_time(aqcs: List[AQCState], tasks: List[Task]) -> float:
    """
    估计当前系统时间（用于计算卡车等待时间等）。

    若已有完成任务：返回最大完成时间
    若无完成任务：返回 AQC 平均可用时间
    """
    done_finish = [float(t.finish_time) for t in tasks if getattr(t, "done", False)]
    if done_finish:
        return max(done_finish)

    aqc_times = [float(a.available_time) for a in aqcs]
    if aqc_times:
        return float(np.mean(aqc_times))

    return 0.0


# ============================================================
# 安全约束检查
# ============================================================
def check_aqc_safety(new_s: float, new_f: float,
                    new_init_bay: float, new_final_bay: float,
                    new_aqc_idx: int, done_tasks: List[Task]) -> bool:
    """
    检查新分配是否与其他 AQC 的已完成任务发生 AQC 安全距离冲突。

    冲突条件（任一）：
      - 时间重叠 且 同方向 且 距离 < GAMMA_AQC
      - 时间重叠 且 反方向（任务路径交叉）
    """
    for t in done_tasks:
        if not t.done or t.assigned_aqc_idx == new_aqc_idx:
            continue
        sj, fj = t.start_time, t.finish_time
        # 时间不重叠则跳过
        if not (new_s < fj and sj < new_f):
            continue
        if t.final_bay is None:
            return False
        if abs(new_init_bay - t.init_bay) < GAMMA_AQC:
            return False
        if abs(new_final_bay - t.final_bay) < GAMMA_AQC:
            return False
        if (new_init_bay - t.init_bay) * (new_final_bay - t.final_bay) <= 0:
            return False
    return True


# ============================================================
# 目标函数
# ============================================================
def train_finish_times(tasks: List[Task], trains: List[int]) -> Dict[int, float]:
    """
    每列火车的实际完成时间 = 该列车所有已完成任务中最大的 finish_time。

    被 cancel 的任务不计入。
    """
    F = {u: 0.0 for u in trains}
    for t in tasks:
        if (t.done
                and not getattr(t, "canceled", False)
                and t.train_id in F):
            F[t.train_id] = max(F[t.train_id], t.finish_time)
    return F


def compute_objective(tasks: List[Task], trains: List[int],
                      phi: Dict[int, float]) -> Tuple[float, float, float]:
    """
    目标函数：ALPHA_OBJ * train_obj + (1 - ALPHA_OBJ) * truck_total_wait

    返回 (obj, train_obj, truck_total_wait)
    """
    F = train_finish_times(tasks, trains)
    train_obj = sum(float(phi[u]) * float(F[u]) for u in trains)
    truck_total_wait = sum(
        float(t.total_wait) for t in tasks
        if (t.done
            and not getattr(t, "canceled", False)
            and t.kind == "load_truck")
    )
    obj = ALPHA_OBJ * train_obj + (1.0 - ALPHA_OBJ) * truck_total_wait
    return obj, train_obj, truck_total_wait

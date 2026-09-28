# -*- coding: utf-8 -*-
"""
obs_packer.py
=============
观测向量构造器。把 env 状态打平成一个固定维度的 numpy 数组，喂给 PPO/SAC。

paradigm 1 扩展（方案B）：在原有 obs 基础上增加约 20 维扰动状态特征，
让 PPO/SAC 能"看到"扰动后果，从而学到防御性行为。

obs 维度组成：
    obs_dim = original_dim + disturbance_dim

  original_dim 子段（迁移自 KG-PPO_0_test.py）:
    - max_load    × task_dim(17)
    - max_unload  × task_dim(17)
    - max_cars    × car_dim(5)
    - max_slots   × slot_dim(4)
    - max_aqcs    × aqc_dim(4)
    - max_cols    × col_dim(4)
    - max_crosses × cross_dim(5)
    - max_truck_slots × truck_slot_dim(4)

  disturbance_dim 子段（★新增）:
    - max_aqcs    × 3   (is_broken + repair_remaining_norm + time_since_break_norm)
    - max_trains  × 3   (is_delayed + delay_amount_norm + time_until_arrival_norm)
    - 7 全局      (n_pending_inserts + n_canceled + n_urgent_pending + intensity_4维one_hot)

env 必须有的属性（duck typing）：
  - tasks, cars_by_train, slots, aqcs, columns, current_height, crosses, unload_truck_slots
  - A1（含可能被 train_delay 修改的当前值）
  - train_arrived, train_planning_blocked
  - disturbance_intensity（"clean"/"low"/"med"/"high"，可选，默认"clean"）
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.constants import A1_DEFAULT
from src.core.data_classes import (
    Task, TrainCar, YardSlot, TruckCross, TruckSlot, AQCState,
)
from src.core.simulators import estimate_system_time


# 归一化常数
_NORM_REPAIR = 600.0       # 修复时长归一化分母（约 BREAK_DURATION_RANGE 上界 × 2）
_NORM_BREAK_AGE = 1800.0   # 距上次故障归一化（30分钟为饱和）
_NORM_DELAY = 600.0        # 延误时长归一化
_NORM_ARRIVAL_WINDOW = 1500.0  # 距列车到达归一化（含正负，clip[-2,2]）
_NORM_COUNT = 10.0         # 计数类（pending_inserts等）归一化


class ObsPacker:
    """
    把 env 状态打平为固定维度的 numpy 数组。

    用法：
        packer = ObsPacker(max_load=20, max_unload=30, max_cars=50, max_slots=80,
                           max_aqcs=4, max_cols=200, max_crosses=50, max_truck_slots=50,
                           max_trains=4, enable_disturbance=True)
        obs = packer.build(env)
        assert obs.shape == (packer.obs_dim,)
    """

    def __init__(self,
                 max_load: int, max_unload: int, max_cars: int,
                 max_slots: int, max_aqcs: int, max_cols: int,
                 max_crosses: int, max_truck_slots: int,
                 max_trains: int = 4,
                 enable_disturbance: bool = True):
        # 容量
        self.max_load = int(max_load)
        self.max_unload = int(max_unload)
        self.max_cars = int(max_cars)
        self.max_slots = int(max_slots)
        self.max_aqcs = int(max_aqcs)
        self.max_cols = int(max_cols)
        self.max_crosses = int(max_crosses)
        self.max_truck_slots = int(max_truck_slots)
        self.max_trains = int(max_trains)
        self.enable_disturbance = bool(enable_disturbance)

        # 每元素维度
        self.task_dim = 17
        self.car_dim = 5
        self.slot_dim = 4
        self.aqc_dim = 4
        self.col_dim = 4
        self.cross_dim = 5
        self.truck_slot_dim = 4

        # 原始 obs 维度
        self.original_dim = (
            self.max_load * self.task_dim
            + self.max_unload * self.task_dim
            + self.max_cars * self.car_dim
            + self.max_slots * self.slot_dim
            + self.max_aqcs * self.aqc_dim
            + self.max_cols * self.col_dim
            + self.max_crosses * self.cross_dim
            + self.max_truck_slots * self.truck_slot_dim
        )

        # ★ 扰动 obs 维度（方案B）
        self.disturb_aqc_dim = 3 if self.enable_disturbance else 0
        self.disturb_train_dim = 3 if self.enable_disturbance else 0
        self.disturb_global_dim = 7 if self.enable_disturbance else 0
        self.disturbance_dim = (
            self.max_aqcs * self.disturb_aqc_dim
            + self.max_trains * self.disturb_train_dim
            + self.disturb_global_dim
        )

        self.obs_dim = self.original_dim + self.disturbance_dim

    # ============================================================
    # 主入口
    # ============================================================
    def build(self, env) -> np.ndarray:
        """构造完整 obs 向量。"""
        feat: List[float] = []
        self._append_original(feat, env)
        if self.enable_disturbance:
            self._append_disturbance(feat, env)
        x = np.array(feat, dtype=np.float32)
        return np.clip(x, -1e6, 1e6)

    # ============================================================
    # 原始 obs（迁移自 KG-PPO_0_test.py）
    # ============================================================
    def _task_kind_code(self, t: Task) -> Tuple[float, float]:
        return (1.0 if t.type == "load" else 0.0,
                1.0 if t.subtype == "truck" else 0.0)

    def _push_task(self, feat: List[float], t: Optional[Task]):
        if t is None:
            feat.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0,
                         -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            return
        type_code, subtype_code = self._task_kind_code(t)
        feat.extend([
            1.0 if t.done else 0.0, type_code, subtype_code,
            float(t.init_row), float(t.init_tier), float(t.init_bay),
            float(t.train_id), float(t.arrival_time),
            float(t.assigned_dest_idx), float(t.assigned_aqc_idx),
            float(t.assigned_cross_idx),
            float(t.start_time), float(t.finish_time),
            float(t.h2), float(t.o2), float(t.total_wait),
            float(t.final_bay if t.final_bay is not None else -1.0),
        ])

    def _append_original(self, feat: List[float], env):
        # tasks split by type
        load_tasks = [t for t in env.tasks if t.type == "load"]
        unload_tasks = [t for t in env.tasks if t.type == "unload"]

        for i in range(self.max_load):
            self._push_task(feat, load_tasks[i] if i < len(load_tasks) else None)
        for i in range(self.max_unload):
            self._push_task(feat, unload_tasks[i] if i < len(unload_tasks) else None)

        # cars
        cars_all: List[TrainCar] = []
        for arr in env.cars_by_train.values():
            cars_all.extend(arr)
        cars_all.sort(key=lambda c: (c.train_id, c.bay))
        for i in range(self.max_cars):
            if i < len(cars_all):
                c = cars_all[i]
                feat.extend([1.0 if c.occupied else 0.0,
                             float(c.row), float(c.tier), float(c.bay), float(c.train_id)])
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # slots
        for i in range(self.max_slots):
            if i < len(env.slots):
                s = env.slots[i]
                feat.extend([1.0 if s.occupied else 0.0,
                             float(s.row), float(s.tier), float(s.bay)])
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0])

        # aqcs (original 4-dim, disturbance dim is appended later)
        for i in range(self.max_aqcs):
            if i < len(env.aqcs):
                a = env.aqcs[i]
                feat.extend([1.0, float(a.cur_row), float(a.cur_bay),
                             float(a.available_time)])
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0])

        # columns
        for i in range(self.max_cols):
            if i < len(env.columns):
                r, b = env.columns[i]
                feat.extend([1.0, float(r), float(b),
                             float(env.current_height.get((r, b), -1))])
            else:
                feat.extend([0.0, 0.0, 0.0, -1.0])

        # crosses
        for i in range(self.max_crosses):
            if i < len(env.crosses):
                c = env.crosses[i]
                feat.extend([1.0, float(c.row), float(c.tier),
                             float(c.bay), float(c.available_time)])
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # truck slots
        for i in range(self.max_truck_slots):
            if i < len(env.unload_truck_slots):
                s = env.unload_truck_slots[i]
                feat.extend([1.0 if s.occupied else 0.0,
                             float(s.row), float(s.tier), float(s.bay)])
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0])

    # ============================================================
    # 扰动 obs（★ 方案B 新增 ~20 维）
    # ============================================================
    def _append_disturbance(self, feat: List[float], env):
        """构造扰动状态特征。"""
        current_time = estimate_system_time(env.aqcs, env.tasks)

        # ---- 1. 每台 AQC 的扰动状态（3维 × max_aqcs）----
        self._append_aqc_disturbance(feat, env, current_time)

        # ---- 2. 每列车的延误状态（3维 × max_trains）----
        self._append_train_disturbance(feat, env, current_time)

        # ---- 3. 全局扰动统计（7维）----
        self._append_global_disturbance(feat, env)

    def _append_aqc_disturbance(self, feat: List[float], env, current_time: float):
        for i in range(self.max_aqcs):
            if i < len(env.aqcs):
                a = env.aqcs[i]
                # is_broken: 当前是否处于故障期（available_time 在 last_repair_time 之前算 broken）
                broken_now = float(
                    a.is_broken
                    and a.last_repair_time > current_time
                )
                # repair_remaining_norm: 距修复还有多久（归一化到[0,1]）
                repair_remaining = max(0.0, a.last_repair_time - current_time)
                repair_remaining_norm = float(np.clip(
                    repair_remaining / _NORM_REPAIR, 0.0, 1.0,
                ))
                # time_since_break_norm: 距上次故障已过多久（归一化到[0,1]）
                if a.last_break_time < 0:
                    # 从未故障 → 视作距离很久（无影响）
                    time_since_norm = 1.0
                else:
                    age = max(0.0, current_time - a.last_break_time)
                    time_since_norm = float(np.clip(
                        age / _NORM_BREAK_AGE, 0.0, 1.0,
                    ))
                feat.extend([broken_now, repair_remaining_norm, time_since_norm])
            else:
                # padding（不存在的AQC）：全0表示"没扰动"
                feat.extend([0.0, 0.0, 1.0])  # time_since=1.0 表示"很久没故障"

    def _append_train_disturbance(self, feat: List[float], env, current_time: float):
        """
        每列车 3 维：is_delayed + delay_amount_norm + time_until_arrival_norm

        约定：列车 ID 范围 1..max_trains，索引为 (tid - 1)。
        未参与本 instance 的列车，所有特征为 0。
        """
        A1 = getattr(env, "A1", {}) or {}
        for tid in range(1, self.max_trains + 1):
            if tid in A1:
                a1_now = float(A1[tid])
                a1_default = float(A1_DEFAULT.get(tid, 0.0))
                # is_delayed: 当前A1 > 默认A1 + 1s 容差
                is_delayed = float(a1_now > a1_default + 1.0)
                # delay_amount_norm
                delay = max(0.0, a1_now - a1_default)
                delay_norm = float(np.clip(delay / _NORM_DELAY, 0.0, 1.0))
                # time_until_arrival_norm: 距实际到达还有多久
                # 正：还没到达；负：已经到达多久
                time_until = (a1_now - current_time) / _NORM_ARRIVAL_WINDOW
                time_until_clipped = float(np.clip(time_until, -2.0, 2.0))
                feat.extend([is_delayed, delay_norm, time_until_clipped])
            else:
                feat.extend([0.0, 0.0, 0.0])

    def _append_global_disturbance(self, feat: List[float], env):
        """
        7 维全局：
          - n_pending_inserts_norm: 已发生但还没做的插单数
          - n_canceled_norm: 累计取消数
          - n_urgent_pending_norm: 未完成且有 urgency_boost 的任务数
          - intensity_one_hot: clean/low/med/high (4维)
        """
        n_pending_inserts = sum(
            1 for t in env.tasks
            if getattr(t, "is_inserted", False) and not t.done
        )
        n_canceled = sum(1 for t in env.tasks if getattr(t, "canceled", False))
        n_urgent = sum(
            1 for t in env.tasks
            if (not t.done) and getattr(t, "urgency_boost", 0.0) > 0.0
        )

        feat.extend([
            float(np.clip(n_pending_inserts / _NORM_COUNT, 0.0, 1.0)),
            float(np.clip(n_canceled / _NORM_COUNT, 0.0, 1.0)),
            float(np.clip(n_urgent / _NORM_COUNT, 0.0, 1.0)),
        ])

        # intensity one-hot
        intensity = str(getattr(env, "disturbance_intensity", "clean"))
        intensity_one_hot = {
            "clean": [1.0, 0.0, 0.0, 0.0],
            "low":   [0.0, 1.0, 0.0, 0.0],
            "med":   [0.0, 0.0, 1.0, 0.0],
            "high":  [0.0, 0.0, 0.0, 1.0],
        }.get(intensity, [1.0, 0.0, 0.0, 0.0])
        feat.extend(intensity_one_hot)


# ============================================================
# 便利函数：从 instance 列表算出 ObsPacker 的容量
# ============================================================
def compute_maxima(json_files: List[str],
                   max_trains: int = 4,
                   enable_disturbance: bool = True) -> ObsPacker:
    """
    扫一遍所有 instance，找出各容量的最大值，返回配好的 ObsPacker。
    """
    from src.core.instance_parser import parse_instance

    mx = dict(load=0, unload=0, cars=0, slots=0,
              aqcs=0, cols=0, crosses=0, truck_slots=0)
    for p in json_files:
        inst = parse_instance(p)
        tasks = inst["tasks"]
        mx["load"]        = max(mx["load"],       sum(1 for t in tasks if t.type == "load"))
        mx["unload"]      = max(mx["unload"],     sum(1 for t in tasks if t.type == "unload"))
        mx["cars"]        = max(mx["cars"],       sum(len(arr) for arr in inst["cars_by_train"].values()))
        mx["slots"]       = max(mx["slots"],      len(inst["slots"]))
        mx["aqcs"]        = max(mx["aqcs"],       len(inst["aqc_init"]))
        mx["cols"]        = max(mx["cols"],       len(inst["columns"]))
        mx["crosses"]     = max(mx["crosses"],    len(inst["crosses"]))
        mx["truck_slots"] = max(mx["truck_slots"], len(inst["unload_truck_slots"]))

    return ObsPacker(
        max_load=mx["load"], max_unload=mx["unload"], max_cars=mx["cars"],
        max_slots=mx["slots"], max_aqcs=mx["aqcs"], max_cols=mx["cols"],
        max_crosses=mx["crosses"], max_truck_slots=mx["truck_slots"],
        max_trains=max_trains, enable_disturbance=enable_disturbance,
    )

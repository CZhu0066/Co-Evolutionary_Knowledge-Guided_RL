# -*- coding: utf-8 -*-
"""
disturbance.py
==============
扰动剧本采样 + 应用，paradigm 1（单阶段鲁棒调度）的核心新模块。

设计原则：
  - 事件状态无关（state-independent）：采样时不查 env 当前状态，
    只用 instance metadata（n_aqcs, trains_involved, A1, estimated_horizon等）。
  - 应用时由 env 处理状态相关的副作用（如把任务延后、选取消目标等）。
  - 完全可重现：固定 seed → 固定剧本。

迁移自 KG_PPO_Reschedule.py 的 8 个 _handle_* 函数，但做了 paradigm 1 简化：
  - 不区分 break-empty / break-loaded
  - 不需要 paused/resume 机制
  - 故障 = 把 AQC.available_time 推后 + 标记 is_broken

主要类：
  - DisturbanceSampler:  在 env.reset() 调用，根据 intensity 生成剧本
  - DisturbanceApplier:  在 env.step() 推进时间时，按 event.time 触发并应用
"""
from typing import Any, Dict, List, Optional

import numpy as np

from config.constants import (
    DISTURBANCE_INTENSITY,
    BREAK_DURATION_RANGE,
    TRAIN_DELAY_RANGE,
    INSERT_TASK_COUNT_RANGE,
    CANCEL_TASK_COUNT_RANGE,
    URGENT_TASK_COUNT_RANGE,
    LOAD_TRUCK_CROSS_BAYS,
)
from src.core.data_classes import (
    Task, AQCState, DisturbanceEvent, DisturbanceScript,
)


# ============================================================
# DisturbanceSampler
# ============================================================
class DisturbanceSampler:
    """
    根据强度配置，在 env.reset() 时为本 episode 采样一份扰动剧本。

    用法：
        sampler = DisturbanceSampler(intensity="med", seed=42)
        script = sampler.sample(env_meta={
            "n_aqcs": 2,
            "trains_involved": [1, 3],
            "A1": {1: 500.0, 3: 1500.0},
            "estimated_horizon": 8000.0,
        })
        env.disturbance_script = script
    """

    def __init__(self, intensity: str = "med", seed: int = 0):
        if intensity not in DISTURBANCE_INTENSITY:
            raise ValueError(
                f"未知 intensity '{intensity}'. "
                f"合法值: {list(DISTURBANCE_INTENSITY.keys())}"
            )
        self.intensity = intensity
        self.config = DISTURBANCE_INTENSITY[intensity]
        self.rng = np.random.default_rng(int(seed))
        self._seed = int(seed)

    def sample(self, env_meta: Dict[str, Any]) -> DisturbanceScript:
        """
        env_meta 必填字段：
          - n_aqcs: int
          - trains_involved: List[int]
          - A1: Dict[int, float]
          - estimated_horizon: float    estimated episode duration
        """
        # 校验 env_meta
        for k in ("n_aqcs", "trains_involved", "A1", "estimated_horizon"):
            if k not in env_meta:
                raise KeyError(f"env_meta 缺少必填字段: {k}")

        events: List[DisturbanceEvent] = []

        events.extend(self._sample_breaks(env_meta))
        events.extend(self._sample_train_delays(env_meta))
        events.extend(self._sample_inserts(env_meta))
        events.extend(self._sample_cancels(env_meta))
        events.extend(self._sample_urgents(env_meta))

        return DisturbanceScript(
            events=events,
            intensity=self.intensity,
            seed=self._seed,
        )

    # ----- 单类扰动采样 -----

    def _sample_breaks(self, env_meta) -> List[DisturbanceEvent]:
        """每台 AQC 独立采样：以概率 p_break_per_aqc 故障一次。"""
        events = []
        H = float(env_meta["estimated_horizon"])
        p = float(self.config["p_break_per_aqc"])
        if p <= 0.0:
            return events
        for aqc_id in range(int(env_meta["n_aqcs"])):
            if self.rng.random() < p:
                t = float(self.rng.uniform(100.0, max(101.0, 0.9 * H)))
                dur = float(self.rng.uniform(*BREAK_DURATION_RANGE))
                events.append(DisturbanceEvent(
                    time=t,
                    type="break",
                    aqc_id=int(aqc_id),
                    duration=dur,
                    intensity_tag=self.intensity,
                ))
        return events

    def _sample_train_delays(self, env_meta) -> List[DisturbanceEvent]:
        """每列车独立采样：以概率 p_train_delay 延误。生成 notice+arrive 一对事件。"""
        events = []
        p = float(self.config["p_train_delay"])
        if p <= 0.0:
            return events
        for train_id in env_meta["trains_involved"]:
            if self.rng.random() < p:
                plan_time = float(env_meta["A1"].get(int(train_id), 0.0))
                delay = float(self.rng.uniform(*TRAIN_DELAY_RANGE))
                actual = plan_time + delay
                events.append(DisturbanceEvent(
                    time=plan_time,
                    type="train_delay_notice",
                    train_id=int(train_id),
                    plan_time=plan_time,
                    actual_time=actual,
                    delay=delay,
                    intensity_tag=self.intensity,
                ))
                events.append(DisturbanceEvent(
                    time=actual,
                    type="train_arrive",
                    train_id=int(train_id),
                    actual_time=actual,
                    intensity_tag=self.intensity,
                ))
        return events

    def _sample_inserts(self, env_meta) -> List[DisturbanceEvent]:
        """以概率 p_insert 触发一批插单（数量在 INSERT_TASK_COUNT_RANGE 内）。"""
        events = []
        scenario = str(env_meta.get("scenario", "unknown"))

        # v10: 2unload 场景只有卸载任务，不生成 load_truck 插单。
        # 当前 insert 实现只支持临时新增 load_truck，因此 2unload 直接跳过插单。
        if scenario == "2unload":
            return events

        p = float(self.config["p_insert"])
        if p <= 0.0 or self.rng.random() >= p:
            return events

        n = int(self.rng.integers(
            INSERT_TASK_COUNT_RANGE[0],
            INSERT_TASK_COUNT_RANGE[1] + 1,
        ))
        H = float(env_meta["estimated_horizon"])
        for _ in range(n):
            t = float(self.rng.uniform(100.0, max(101.0, 0.6 * H)))
            train_id = int(self.rng.choice(env_meta["trains_involved"]))
            events.append(DisturbanceEvent(
                time=t,
                type="insert",
                insert_kind="load_truck",
                train_id=train_id,
                truck_arrival_time=t,
                truck_init_row=-4.0,
                truck_init_tier=0.0,
                truck_init_bay=float(self.rng.choice(LOAD_TRUCK_CROSS_BAYS)),
                intensity_tag=self.intensity,
            ))
        return events

    def _sample_cancels(self, env_meta) -> List[DisturbanceEvent]:
        """以概率 p_cancel 触发一次取消（批量取消若干个同 kind 的任务）。"""
        events = []
        p = float(self.config["p_cancel"])
        if p <= 0.0 or self.rng.random() >= p:
            return events

        n = int(self.rng.integers(
            CANCEL_TASK_COUNT_RANGE[0],
            CANCEL_TASK_COUNT_RANGE[1] + 1,
        ))
        H = float(env_meta["estimated_horizon"])
        t = float(self.rng.uniform(50.0, max(51.0, 0.5 * H)))
        scenario = str(env_meta.get("scenario", "unknown"))
        if scenario == "2load":
            cancel_candidates = ["load_yard", "load_truck"]
        elif scenario == "2unload":
            cancel_candidates = ["unload_yard", "unload_truck"]
        else:
            cancel_candidates = ["load_yard", "load_truck", "unload_yard", "unload_truck"]
        target_kind = str(self.rng.choice(cancel_candidates))
        events.append(DisturbanceEvent(
            time=t,
            type="cancel",
            target_kind=target_kind,
            n_cancel=n,
            intensity_tag=self.intensity,
        ))
        return events

    def _sample_urgents(self, env_meta) -> List[DisturbanceEvent]:
        """以概率 p_urgent 触发一次加急。"""
        events = []
        p = float(self.config["p_urgent"])
        if p <= 0.0 or self.rng.random() >= p:
            return events

        n = int(self.rng.integers(
            URGENT_TASK_COUNT_RANGE[0],
            URGENT_TASK_COUNT_RANGE[1] + 1,
        ))
        H = float(env_meta["estimated_horizon"])
        t = float(self.rng.uniform(50.0, max(51.0, 0.5 * H)))
        scenario = str(env_meta.get("scenario", "unknown"))
        if scenario == "2load":
            urgent_candidates = ["load_yard", "load_truck"]
        elif scenario == "2unload":
            urgent_candidates = ["unload_yard", "unload_truck"]
        else:
            urgent_candidates = ["load_yard", "load_truck", "unload_yard", "unload_truck"]
        target_kind = str(self.rng.choice(urgent_candidates))
        events.append(DisturbanceEvent(
            time=t,
            type="urgent",
            target_kind=target_kind,
            n_urgent=n,
            boost_value=200.0,
            intensity_tag=self.intensity,
        ))
        return events


# ============================================================
# DisturbanceApplier
# ============================================================
class DisturbanceApplier:
    """
    把单个 DisturbanceEvent 应用到 env 状态上。

    env 需要的接口（duck-typing）：
      - env.aqcs: List[AQCState]
      - env.tasks: List[Task]
      - env.A1: Dict[int, float]
      - env.train_planning_blocked: Set[int]    需 env 初始化为 set()
      - env.train_arrived: Dict[int, bool]       需 env 初始化
      - env.rng: numpy 随机数生成器（用于 cancel/urgent 的目标随机选择）

    返回 dict 记录实际效果，供 reward shaping 和 obs 使用。
    """

    @staticmethod
    def apply(env, event: DisturbanceEvent) -> Dict[str, Any]:
        """统一入口。根据 event.type 分发到具体的 apply_* 方法。"""
        handlers = {
            "break": DisturbanceApplier.apply_break,
            "train_delay_notice": DisturbanceApplier.apply_train_delay_notice,
            "train_arrive": DisturbanceApplier.apply_train_arrive,
            "insert": DisturbanceApplier.apply_insert,
            "cancel": DisturbanceApplier.apply_cancel,
            "urgent": DisturbanceApplier.apply_urgent,
        }
        handler = handlers.get(event.type)
        if handler is None:
            return {"success": False, "reason": f"unknown event type: {event.type}"}
        return handler(env, event)

    # ----- 各类型 -----

    @staticmethod
    def apply_break(env, event: DisturbanceEvent) -> Dict[str, Any]:
        """
        AQC 故障：把 available_time 推后，标记 is_broken。

        若 AQC 在故障时正在执行某任务（start_time <= t_break < finish_time），
        该任务的 finish_time 也要延后 duration（任务被卡了 duration 秒）。
        """
        if event.aqc_id is None or event.aqc_id >= len(env.aqcs):
            return {"success": False, "reason": "invalid aqc_id"}

        aqc = env.aqcs[int(event.aqc_id)]
        t_break = float(event.time)
        duration = float(event.duration or 0.0)

        old_available = aqc.available_time
        new_available = max(old_available, t_break) + duration
        aqc.available_time = new_available
        aqc.is_broken = True
        aqc.last_break_time = t_break
        aqc.last_repair_time = new_available

        # 若AQC正在执行任务，把该任务的 finish_time 延后
        affected_task_ids = []
        for t in env.tasks:
            if (t.done
                    and not getattr(t, "canceled", False)
                    and t.assigned_aqc_idx == int(event.aqc_id)
                    and t.start_time <= t_break < t.finish_time):
                t.finish_time += duration
                affected_task_ids.append(t.id)

        return {
            "success": True,
            "aqc_id": int(event.aqc_id),
            "old_available": float(old_available),
            "new_available": float(new_available),
            "affected_task_ids": affected_task_ids,
        }

    @staticmethod
    def apply_train_delay_notice(env, event: DisturbanceEvent) -> Dict[str, Any]:
        """
        列车延误通知：修改 A1[train_id] = actual_time，
        把该列车加入 planning_blocked（score_task 中需检查这个集合）。
        """
        train_id = int(event.train_id)
        env.A1[train_id] = float(event.actual_time)
        env.train_planning_blocked.add(train_id)
        env.train_arrived[train_id] = False
        return {
            "success": True,
            "train_id": train_id,
            "new_A1": float(event.actual_time),
            "delay": float(event.delay or 0.0),
        }

    @staticmethod
    def apply_train_arrive(env, event: DisturbanceEvent) -> Dict[str, Any]:
        """列车实际到达：解除 blocked。"""
        train_id = int(event.train_id)
        env.train_arrived[train_id] = True
        env.train_planning_blocked.discard(train_id)
        return {"success": True, "train_id": train_id}

    @staticmethod
    def apply_insert(env, event: DisturbanceEvent) -> Dict[str, Any]:
        """插单：往 env.tasks 添加一个新任务（默认 load_truck）。"""
        # 生成新ID（最大现有ID + 1，避免冲突）
        if env.tasks:
            try:
                max_id = max(int(t.id) for t in env.tasks if isinstance(t.id, (int, float)))
            except ValueError:
                max_id = 0
        else:
            max_id = 0
        new_id = max_id + 1

        kind = event.insert_kind or "load_truck"
        ttype = "load" if kind.startswith("load") else "unload"
        subtype = "truck" if kind.endswith("truck") else "yard"

        new_task = Task(
            id=new_id,
            type=ttype,
            subtype=subtype,
            kind=kind,
            train_id=int(event.train_id),
            init_row=float(event.truck_init_row or -4.0),
            init_tier=float(event.truck_init_tier or 0.0),
            init_bay=float(event.truck_init_bay or 0.0),
            arrival_time=float(event.truck_arrival_time or event.time),
            is_inserted=True,
        )
        env.tasks.append(new_task)
        return {
            "success": True,
            "new_task_id": new_id,
            "kind": kind,
            "train_id": int(event.train_id),
            "truck_arrival_time": float(event.truck_arrival_time or event.time),
            "truck_init_bay": float(event.truck_init_bay or 0.0),
        }

    @staticmethod
    def apply_cancel(env, event: DisturbanceEvent) -> Dict[str, Any]:
        """
        软取消：从未完成、kind 匹配的任务里随机挑 n_cancel 个标记为 canceled。

        关键：被取消的任务设置 done=True，让 planner 不再选它，
              但 canceled=True 让 compute_objective 不计入它的完成时间。
        """
        target_kind = event.target_kind
        if target_kind is None:
            return {"success": False, "reason": "missing target_kind"}

        candidates = [
            t for t in env.tasks
            if (not t.done)
            and (not getattr(t, "canceled", False))
            and (not getattr(t, "is_inserted", False))  # 不取消插单任务
            and t.kind == target_kind
            and (event.target_train_id is None or t.train_id == event.target_train_id)
        ]
        if not candidates:
            return {"success": True, "canceled_ids": [], "note": "no candidates"}

        n = min(int(event.n_cancel or 1), len(candidates))
        chosen_indices = env.rng.choice(len(candidates), size=n, replace=False)
        canceled_ids = []
        for idx in chosen_indices:
            t = candidates[int(idx)]
            t.canceled = True
            t.done = True       # 让 planner 跳过它
            canceled_ids.append(t.id)
        return {"success": True, "canceled_ids": canceled_ids}

    @staticmethod
    def apply_urgent(env, event: DisturbanceEvent) -> Dict[str, Any]:
        """
        加急：从未完成、kind 匹配的任务里挑 n_urgent 个设置 urgency_boost。

        score_task 函数将在评分时把这个 boost 加进任务的优先级。
        """
        target_kind = event.target_kind
        if target_kind is None:
            return {"success": False, "reason": "missing target_kind"}

        candidates = [
            t for t in env.tasks
            if (not t.done)
            and (not getattr(t, "canceled", False))
            and t.kind == target_kind
            and (event.target_train_id is None or t.train_id == event.target_train_id)
        ]
        if not candidates:
            return {"success": True, "urgent_ids": [], "note": "no candidates"}

        n = min(int(event.n_urgent or 1), len(candidates))
        chosen_indices = env.rng.choice(len(candidates), size=n, replace=False)
        boost = float(event.boost_value or 200.0)
        urgent_ids = []
        for idx in chosen_indices:
            t = candidates[int(idx)]
            t.urgency_boost = max(t.urgency_boost, boost)  # 取最大，避免覆盖更高的boost
            urgent_ids.append(t.id)
        return {"success": True, "urgent_ids": urgent_ids, "boost": boost}

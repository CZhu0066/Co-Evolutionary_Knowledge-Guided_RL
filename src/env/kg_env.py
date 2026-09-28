# -*- coding: utf-8 -*-
"""
kg_env.py
=========
KGUnifiedYardEnv 主类。整合 core / env 各组件成一个 gym.Env。

v7 范围：扰动 sampler/applier 已接入：
  - __init__, reset, step 完整可跑
  - 决策子流程：task → dest → aqc → execute → reward
  - 已初始化扰动相关属性，但 disturbance_script 默认为 None
    （v7 已通过 reset() 调用 sampler 生成）
  - rule_bonus_fn 默认 None（A3 闭环未启用）

模块化设计：
  - _pick_best_task    : 用 score_task 选任务
  - _pick_destination  : 按 task.kind 分发到 4 个子流程
  - _pick_aqc          : 用 score_aqc_* 选 AQC
  - _execute_normal / _execute_load_truck : 执行任务
  - reward 计算：用 src.env.reward 的 5 个独立子函数组合

设计变化（vs KG-PPO_0_test.py）：
  1. obs_packer 通过 build(env) 而非 build(*args) 调用
  2. 决策子流程拆成独立方法，便于阅读和测试
  3. reward 拆分为 4 个独立函数（step / shaping / terminal / rule_bonus），
     便于 A4 因果反事实评估独立调用
"""
import os
import random
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    import gym
    from gym import spaces
from config.constants import (
    D_I, D_G, D_P, D_K, THETA_DIM,
    TRUCK_SERVICE_AQC_K, CROSS_NEAR_AQC_K, CAR_NEAR_CROSS_K,SCALE_STEP_BASE,
)
from src.core.data_classes import (
    Task, AQCState, TrainCar, YardSlot, TruckCross, TruckSlot,
    DisturbanceScript,
)
from src.core.instance_parser import parse_instance
from src.core.simulators import (
    simulate_task_times, simulate_load_truck_times,
    check_aqc_safety, train_finish_times, compute_objective,
)
from src.env.obs_packer import ObsPacker
from src.env.scoring import (
    score_task, score_destination, score_cross,
    score_aqc_normal, score_aqc_load_truck,
    get_truck_service_aqcs, get_crosses_near_aqc, get_cars_near_cross_or_task,
)
from src.env.reward import (
    compute_step_reward, compute_truck_shaping,
    compute_terminal_reward, compute_rule_bonus, compose_total_reward,
)
from src.env.disturbance import DisturbanceSampler, DisturbanceApplier


class KGUnifiedYardEnv(gym.Env):
    """
    Knowledge-Guided 港口装卸调度 env。

    动作：9D 连续向量（[-1, 1]^9），分解为 θ_i(3) + θ_g(2) + θ_p(2) + θ_k(2)
    观测：通过 ObsPacker.build(self) 构造的扁平向量
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 json_files: List[str],
                 obs_packer: ObsPacker,
                 seed: int = 0,
                 shuffle_each_reset: bool = True,
                 enable_disturbance: bool = False,
                 disturbance_intensity: str = "clean",
                 disturbance_seed: int = 0,
                 rule_bonus_fn=None,
                 rule_guidance_fn=None):
        super().__init__()
        assert len(json_files) > 0, "json_files 不能为空"

        # 配置
        self.json_files = list(json_files)
        self.obs_packer = obs_packer
        self.shuffle_each_reset = bool(shuffle_each_reset)

        # 随机数（用于 instance 选择、扰动应用时取目标）
        self.rng = np.random.default_rng(int(seed))
        self._py_rng = random.Random(int(seed))   # 兼容部分API

        # 动作 + 观测空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(THETA_DIM,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6,
            shape=(obs_packer.obs_dim,), dtype=np.float32,
        )

        # 扰动配置（v7: 已接入 sampler/applier）
        self.enable_disturbance = bool(enable_disturbance)
        self.disturbance_intensity = str(disturbance_intensity)
        self._master_disturbance_seed = int(disturbance_seed)
        self._episode_count = 0   # 每次 reset 递增

        # 规则注入（A3 闭环，默认关闭）：执行后 reward bonus
        self.rule_bonus_fn = rule_bonus_fn
        # v14-RGCD：执行前规则引导连续动作解码（task/destination/AQC score guidance）
        self.rule_guidance_fn = rule_guidance_fn

        # 初始化所有 state 字段
        self._init_state_attrs()

    # ============================================================
    # State 初始化
    # ============================================================
    def _init_state_attrs(self):
        """声明所有 state 字段。在 reset()/_load_instance 中实际填充。"""
        # 静态/半静态
        self.instance: Optional[Dict] = None
        self.instance_path: str = ""
        self.tasks: List[Task] = []
        self.cars_by_train: Dict[int, List[TrainCar]] = {}
        self.slots: List[YardSlot] = []
        self.crosses: List[TruckCross] = []
        self.unload_truck_slots: List[TruckSlot] = []
        self.aqcs: List[AQCState] = []
        # Cumulative processing workload of each AQC in the current episode.
        # B_k = sum_i (finish_i - start_i) for tasks assigned to AQC k.
        self.aqc_workloads: Dict[int, float] = {}
        self.columns: List[Tuple[float, float]] = []
        self.initial_height: Dict[Tuple[float, float], int] = {}
        self.current_height: Dict[Tuple[float, float], int] = {}
        self.trains_involved: List[int] = []
        self.phi: Dict[int, float] = {}
        self.A1: Dict[int, float] = {}
        self.A2: Dict[int, float] = {}

        # 动态量
        self.F: Dict[int, float] = {}
        self.obj: float = 0.0
        self.train_obj: float = 0.0
        self.truck_total_wait: float = 0.0
        self.current_step: int = 0
        self.max_steps: int = 0
        self.t_now: float = 0.0   # v7: 仿真当前时间
        self._last_decision: dict = {}   # v10: 上一步决策记录
        # A3 规则注入诊断字段
        self._rule_bonus_features_before_action = None
        self._last_rule_bonus_detail: Dict[str, Any] = {}
        self._rule_bonus_stats: Dict[str, Any] = {
            "calls": 0,
            "no_decision": 0,
            "extract_error": 0,
            "no_match": 0,
            "low_confidence": 0,
            "hits": 0,
            "matched": 0,
            "mismatched": 0,
            "total_bonus": 0.0,
            "abs_total_bonus": 0.0,
            "feature_before_action": 0,
            "feature_after_action_fallback": 0,
        }
        # v14-RGCD：规则引导解码诊断字段（执行前 score guidance）
        self._last_rule_guidance_detail: Dict[str, Any] = {}
        self._rule_guidance_stats: Dict[str, Any] = {
            "calls": 0,
            "triggered": 0,
            "masked": 0,
            "task_triggered": 0,
            "destination_triggered": 0,
            "aqc_triggered": 0,
            "total_adjust": 0.0,
            "abs_total_adjust": 0.0,
            "match_strength_sum": 0.0,
            "matched_rule_count": 0,
            "trigger_strength_sum": 0.0,
            "trigger_strength_count": 0,
            "high_strength_triggered": 0,
        }
        # v14-AQC：候选级 AQC 样本缓存。每个 step 记录 _pick_aqc 中所有可行候选，
        # 用于后续 AQC candidate rule mining（不影响调度执行）。
        self._last_aqc_candidate_samples: List[Dict[str, Any]] = []
        # v14-DEST：候选级 destination 样本缓存。每个 step 记录目的地/slot/car/cross 组合候选，
        # 用于第三层 destination candidate rule mining（不影响调度执行）。
        self._last_destination_candidate_samples: List[Dict[str, Any]] = []

        # 扰动相关 state（供 scoring/reward/obs 读取）
        self.train_planning_blocked: set = set()
        self.train_arrived: Dict[int, bool] = {}
        self.disturbance_script: Optional[DisturbanceScript] = None
        self.next_event_idx: int = 0

        # v10_fix2: 缓存“执行任务前”触发的扰动事件。
        # 目的：step() 中遇到 pre_events 后会递归 return self.step(action)，
        # 若不缓存，run_demo 无法拿到加急/取消/插单的真实任务ID。
        self._carry_applied_events_for_info: List[Dict[str, Any]] = []

    # ============================================================
    # 加载 instance
    # ============================================================
    def _load_instance(self, json_path: str):
        """从 JSON 文件加载所有 env 状态。"""
        inst = parse_instance(json_path)
        self.instance = inst
        self.instance_path = json_path
        # A3 分场景规则注入：保存当前实例场景，供轨迹采集和 rule_bonus 选择规则使用
        self.scenario = str(inst.get("raw", {}).get("scenario", "unknown"))

        # 复制（避免共享引用）
        self.tasks = [Task(**{k: v for k, v in t.__dict__.items()})
                      for t in inst["tasks"]]
        self.cars_by_train = {
            tid: [TrainCar(**{k: v for k, v in c.__dict__.items()}) for c in arr]
            for tid, arr in inst["cars_by_train"].items()
        }
        self.slots = [YardSlot(**{k: v for k, v in s.__dict__.items()})
                      for s in inst["slots"]]
        self.crosses = [TruckCross(**{k: v for k, v in c.__dict__.items()})
                        for c in inst["crosses"]]
        self.unload_truck_slots = [
            TruckSlot(**{k: v for k, v in s.__dict__.items()})
            for s in inst["unload_truck_slots"]
        ]

        self.initial_height = dict(inst["initial_height"])
        self.current_height = dict(self.initial_height)
        self.columns = list(inst["columns"])

        self.aqcs = [
            AQCState(id=i, cur_row=float(r), cur_bay=float(b), available_time=0.0)
            for i, (r, b) in enumerate(inst["aqc_init"])
        ]
        self._reset_aqc_workloads()

        self.trains_involved = list(inst["trains_involved"])
        self.phi = dict(inst["phi"])
        self.A1 = dict(inst["A1"])
        self.A2 = dict(inst["A2"])
        self.F = {u: 0.0 for u in self.trains_involved}

        # 重置动态量
        self.obj = 0.0
        self.train_obj = 0.0
        self.truck_total_wait = 0.0
        self.current_step = 0
        self.max_steps = max(1, len(self.tasks) * 5)
        # ★ reward 归一化基准：reset 时锁定，episode 内不随 insert/cancel 变化

        self.obj_scale = float(SCALE_STEP_BASE) * max(1, len(self.tasks))
        self.t_now = 0.0   # v7: 每个 episode 仿真时间从 0 开始
        # 每个 episode 重置规则注入诊断信息
        self._rule_bonus_features_before_action = None
        self._last_rule_bonus_detail = {}
        self._rule_bonus_stats = {
            "calls": 0,
            "no_decision": 0,
            "extract_error": 0,
            "no_match": 0,
            "low_confidence": 0,
            "hits": 0,
            "matched": 0,
            "mismatched": 0,
            "total_bonus": 0.0,
            "abs_total_bonus": 0.0,
            "feature_before_action": 0,
            "feature_after_action_fallback": 0,
        }
        self._last_rule_guidance_detail = {}
        self._rule_guidance_stats = {
            "calls": 0,
            "triggered": 0,
            "masked": 0,
            "task_triggered": 0,
            "destination_triggered": 0,
            "aqc_triggered": 0,
            "total_adjust": 0.0,
            "abs_total_adjust": 0.0,
            "match_strength_sum": 0.0,
            "matched_rule_count": 0,
            "trigger_strength_sum": 0.0,
            "trigger_strength_count": 0,
            "high_strength_triggered": 0,
        }
        self._last_aqc_candidate_samples = []

        # 扰动相关：所有列车默认"未阻塞、已到达"
        self.train_planning_blocked = set()
        self.train_arrived = {tid: True for tid in self.trains_involved}

        # 扰动剧本：在 reset() 中通过 _sample_disturbance_script() 生成
        self.disturbance_script = None
        self.next_event_idx = 0
        self.disturbance_history = []

        # v10_fix2: 每个 episode 重置 pre_events 缓存
        self._carry_applied_events_for_info = []

    # ============================================================
    # gym.Env 接口
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 选择 instance
        if self.shuffle_each_reset:
            json_path = self._py_rng.choice(self.json_files)
        else:
            if not hasattr(self, "_reset_idx"):
                self._reset_idx = 0
            json_path = self.json_files[self._reset_idx % len(self.json_files)]
            self._reset_idx += 1

        self._load_instance(json_path)

        # ★ v7: 采样本 episode 的扰动剧本，并应用 t=0 时刻的事件
        if self.enable_disturbance:
            self._sample_disturbance_script()
            self._apply_pending_disturbances(0.0)

        return self._build_obs(), {
            "instance": os.path.basename(json_path),
            "n_disturbance_events": (
                self.disturbance_script.n_events
                if self.disturbance_script is not None else 0
            ),
        }

    def step(self, action):
        obj_before = float(self.obj)

        # A3 规则注入诊断：
        # 在任务选择和环境状态改变之前，提前保存规则匹配用的状态特征。
        # 这样 rule_bonus 使用的是“动作前状态”，与策略蒸馏时的特征定义一致。
        self._rule_bonus_features_before_action = None
        if self.rule_bonus_fn is not None and hasattr(self.rule_bonus_fn, "capture_features"):
            self._rule_bonus_features_before_action = self.rule_bonus_fn.capture_features(self)

        self.current_step += 1

        # 解析 θ
        theta = np.array(action, dtype=np.float32)
        theta_i = theta[:D_I]
        theta_g = theta[D_I:D_I + D_G]
        theta_p = theta[D_I + D_G:D_I + D_G + D_P]
        theta_k = theta[D_I + D_G + D_P:]

        done_tasks = [t for t in self.tasks if t.done]

        # 1. 选任务
        # ★ v7: 若所有任务被阻塞（如列车未到达），尝试推进到下一个扰动事件再重试
        best_task = self._pick_best_task(theta_i, done_tasks)
        retries = 0
        while best_task is None and retries < 3:
            if not self._advance_to_next_disturbance_event():
                break
            done_tasks = [t for t in self.tasks if t.done]
            best_task = self._pick_best_task(theta_i, done_tasks)
            retries += 1
        if best_task is None:
            return self._end_episode(-10.0, "no_task")

        # 2. 选目的地（+ cross + preferred_aqc）
        chosen_dest, chosen_dest_idx, chosen_cross, preferred_aqc = \
            self._pick_destination(best_task, theta_g, theta_p, done_tasks)
        if chosen_dest is None:
            return self._end_episode(-10.0, "no_destination")

        # 3. 选 AQC
        best_aqc = self._pick_aqc(
            best_task, chosen_dest, chosen_cross,
            preferred_aqc, theta_k, done_tasks,
        )
        if best_aqc is None:
            return self._end_episode(-10.0, "no_aqc")

        # v10: 执行前先处理“预计开始时间之前”的扰动。
        # 旧逻辑是在任务执行完成后才应用扰动，可能出现：
        #   火车500s通知晚到，但AQC已经安排510s开始处理该火车任务。
        # 这里若触发了任何扰动，则重新进入一次 step，让选任务/选AQC基于新状态重算。
        planned_start = self._estimate_planned_start(
            best_task, chosen_dest, chosen_cross, best_aqc
        )
        pre_events = self._apply_pending_disturbances(planned_start)
        if pre_events:
            # v11_fix1:
            # 执行任务前如果先触发了扰动，不再递归 self.step(action)。
            # 原递归写法会让同一个旧 action 在状态变化后继续执行，
            # 并且可能额外消耗内部 current_step。
            # 现在的处理：先触发扰动，返回新 obs，让智能体下一步基于新状态重新决策。
            self.t_now = max(
                self.t_now,
                max(float(x["event_time"]) for x in pre_events),
            )

            # 扰动可能取消/新增/加急任务，因此这里同步刷新目标函数。
            self.F = train_finish_times(self.tasks, self.trains_involved)
            self.obj, self.train_obj, self.truck_total_wait = compute_objective(
                self.tasks, self.trains_involved, self.phi,
            )

            all_done = all(t.done for t in self.tasks)
            step_over = self.current_step >= self.max_steps
            obs = self._build_obs()
            info = dict(
                reason="pre_disturbance_only",
                instance=os.path.basename(self.instance_path),
                F=dict(self.F),
                obj=float(self.obj),
                train_obj=float(self.train_obj),
                truck_total_wait=float(self.truck_total_wait),
                makespan=float(self._current_makespan()),
                aqc_balance=float(self._current_aqc_balance()),
                aqc_workloads=dict(self._get_aqc_workloads()),
                step_reward=0.0,
                truck_shaping=0.0,
                terminal_reward=0.0,
                rule_bonus=0.0,
                total_reward=0.0,
                rule_bonus_detail={"reason": "pre_disturbance_only", "bonus": 0.0},
                rule_bonus_stats=dict(getattr(self, "_rule_bonus_stats", {})),
                rule_guidance_detail=dict(getattr(self, "_last_rule_guidance_detail", {})),
                rule_guidance_stats=dict(getattr(self, "_rule_guidance_stats", {})),
                rule_hit_count=int(getattr(self, "_rule_bonus_stats", {}).get("hits", 0)),
                rule_match_count=int(getattr(self, "_rule_bonus_stats", {}).get("matched", 0)),
                rule_mismatch_count=int(getattr(self, "_rule_bonus_stats", {}).get("mismatched", 0)),
                rule_no_match_count=int(getattr(self, "_rule_bonus_stats", {}).get("no_match", 0)),
                rule_bonus_total=float(getattr(self, "_rule_bonus_stats", {}).get("total_bonus", 0.0)),
                rule_bonus_abs_total=float(getattr(self, "_rule_bonus_stats", {}).get("abs_total_bonus", 0.0)),
                all_done=all_done,
                step_over=step_over,
                t_now=float(self.t_now),
                n_applied_disturbances=len(getattr(self, "disturbance_history", [])),
                applied_events=list(pre_events),
                disturbance_history=list(getattr(self, "disturbance_history", [])),
                last_decision={
                    "task_id": int(best_task.id),
                    "task_kind": str(best_task.kind),
                    "train_id": int(getattr(best_task, "train_id", -1) or -1),
                    "aqc_id": int(best_aqc.id),
                    "dest_kind": str(getattr(chosen_dest, "__class__", type(chosen_dest)).__name__),
                    "note": "not_executed_due_to_pre_disturbance",
                },
                destination_candidates=list(getattr(self, "_last_destination_candidate_samples", [])),
                aqc_candidates=list(getattr(self, "_last_aqc_candidate_samples", [])),
            )
            # 不执行旧动作，让智能体下一步基于扰动后的新状态重新选择。
            return obs, 0.0, bool(all_done), bool((not all_done) and step_over), info

        # 4. 执行
        if best_task.kind == "load_truck":
            self._execute_load_truck(best_task, chosen_dest, chosen_dest_idx,
                                      chosen_cross, best_aqc)
        else:
            self._execute_normal(best_task, chosen_dest, chosen_dest_idx, best_aqc)

        # ★ v10: 记录刚执行的决策（供 Innovation A 蒸馏 + rule_bonus_fn 使用）
        self._last_decision = {
            "task_id": int(best_task.id),
            "task_kind": str(best_task.kind),
            "train_id": int(getattr(best_task, "train_id", -1) or -1),
            "aqc_id": int(best_aqc.id),
            "dest_kind": str(getattr(chosen_dest, "__class__", type(chosen_dest)).__name__),
        }

        # ★ v7: 推进 t_now，应用此期间触发的扰动事件
        if self.aqcs:
            new_t_now = max(self.t_now, max(a.available_time for a in self.aqcs))
        else:
            new_t_now = self.t_now
        applied_events = self._apply_pending_disturbances(new_t_now)
        self.t_now = new_t_now

        # v10_fix2:
        # 合并“执行前触发的扰动”和“执行后触发的扰动”，统一传给 demo。
        carry_events = getattr(self, "_carry_applied_events_for_info", [])
        all_applied_events = list(carry_events) + list(applied_events)
        self._carry_applied_events_for_info = []

        # 5. 更新目标函数
        self.F = train_finish_times(self.tasks, self.trains_involved)
        self.obj, self.train_obj, self.truck_total_wait = compute_objective(
            self.tasks, self.trains_involved, self.phi,
        )
        obj_after = float(self.obj)

        # 6. 计算 reward（用 reward 模块的独立子函数）
        step_r = compute_step_reward(obj_before, obj_after, self.obj_scale)
        truck_shaping = compute_truck_shaping(self, best_task)

        # 7. 终止判断
        all_done = all(t.done for t in self.tasks)
        step_over = self.current_step >= self.max_steps
        terminated = bool(all_done)
        truncated = bool((not all_done) and step_over)

        terminal_r = compute_terminal_reward(
            obj_after, self.truck_total_wait, terminated, truncated,
        )
        #rule_bonus = compute_rule_bonus(self, action, self.rule_bonus_fn)
        # A3 规则注入：
        # Round 0 没有规则，self.rule_bonus_fn 为 None，此时 rule_bonus 必须为 0。
        # Round >= 1 有规则时，才调用 rule_bonus_fn。
        if self.rule_bonus_fn is not None:
            try:
                rule_bonus = float(self.rule_bonus_fn(self, action))
            except Exception as e:
                rule_bonus = 0.0

                if not hasattr(self, "_last_rule_bonus_detail") or self._last_rule_bonus_detail is None:
                    self._last_rule_bonus_detail = {}

                self._last_rule_bonus_detail.update({
                    "reason": "rule_bonus_exception",
                    "exception_type": type(e).__name__,
                    "exception_msg": str(e),
                    "bonus": 0.0,
                })
        else:
            rule_bonus = 0.0

            if not hasattr(self, "_last_rule_bonus_detail") or self._last_rule_bonus_detail is None:
                self._last_rule_bonus_detail = {}

            self._last_rule_bonus_detail.update({
                "reason": "rule_bonus_fn_is_none",
                "bonus": 0.0,
            })


        total_reward = compose_total_reward(
            step_r, truck_shaping, terminal_r, rule_bonus,
        )

        # 8. obs + info
        obs = self._build_obs()
        info = dict(
            instance=os.path.basename(self.instance_path),
            F=dict(self.F),
            obj=float(self.obj),
            train_obj=float(self.train_obj),
            truck_total_wait=float(self.truck_total_wait),
            makespan=float(self._current_makespan()),
            aqc_balance=float(self._current_aqc_balance()),
            aqc_workloads=dict(self._get_aqc_workloads()),
            completion=float(1.0 if all_done else 0.0),
            step_reward=float(step_r),
            truck_shaping=float(truck_shaping),
            terminal_reward=float(terminal_r),
            rule_bonus=float(rule_bonus),
            total_reward=float(total_reward),
            rule_bonus_detail=dict(getattr(self, "_last_rule_bonus_detail", {})),
            rule_bonus_stats=dict(getattr(self, "_rule_bonus_stats", {})),
            rule_guidance_detail=dict(getattr(self, "_last_rule_guidance_detail", {})),
            rule_guidance_stats=dict(getattr(self, "_rule_guidance_stats", {})),
            rule_hit_count=int(getattr(self, "_rule_bonus_stats", {}).get("hits", 0)),
            rule_match_count=int(getattr(self, "_rule_bonus_stats", {}).get("matched", 0)),
            rule_mismatch_count=int(getattr(self, "_rule_bonus_stats", {}).get("mismatched", 0)),
            rule_no_match_count=int(getattr(self, "_rule_bonus_stats", {}).get("no_match", 0)),
            rule_bonus_total=float(getattr(self, "_rule_bonus_stats", {}).get("total_bonus", 0.0)),
            rule_bonus_abs_total=float(getattr(self, "_rule_bonus_stats", {}).get("abs_total_bonus", 0.0)),
            all_done=all_done,
            step_over=step_over,
            t_now=float(self.t_now),                            # v7
            n_applied_disturbances=len(getattr(self, "disturbance_history", [])),  # v10 cumulative
            applied_events=list(all_applied_events),            # v10_fix2: 包含执行前+执行后触发事件
            disturbance_history=list(getattr(self, "disturbance_history", [])),
            last_decision=dict(self._last_decision),            # v10
            destination_candidates=list(getattr(self, "_last_destination_candidate_samples", [])),  # v14-DEST candidate samples
            aqc_candidates=list(getattr(self, "_last_aqc_candidate_samples", [])),  # v14-AQC candidate samples
        )
        return obs, total_reward, terminated, truncated, info

    def _end_episode(self, reward: float, reason: str):
        """中断 episode：用于无任务/无目的地/无AQC的死局。

        v11_fix5：补充终止信息，方便 demo 分析“为什么任务没有完全完成”。
        """
        self.F = train_finish_times(self.tasks, self.trains_involved)
        self.obj, self.train_obj, self.truck_total_wait = compute_objective(
            self.tasks, self.trains_involved, self.phi,
        )

        unfinished_task_ids = [
            int(t.id) for t in self.tasks
            if (not t.done) and (not getattr(t, "canceled", False))
        ]
        canceled_task_ids = [
            int(t.id) for t in self.tasks
            if getattr(t, "canceled", False)
        ]
        real_done_task_ids = [
            int(t.id) for t in self.tasks
            if t.done and (not getattr(t, "canceled", False))
        ]

        return (self._build_obs(),
                float(reward),
                True, False,
                {
                    "reason": reason,
                    "instance": os.path.basename(self.instance_path),
                    "F": dict(self.F),
                    "obj": float(self.obj),
                    "train_obj": float(self.train_obj),
                    "truck_total_wait": float(self.truck_total_wait),
                    "makespan": float(self._current_makespan()),
                    "aqc_balance": float(self._current_aqc_balance()),
                    "aqc_workloads": dict(self._get_aqc_workloads()),
                    "completion": 0.0,
                    "all_done": False,
                    "step_over": self.current_step >= self.max_steps,
                    "t_now": float(getattr(self, "t_now", 0.0)),
                    "applied_events": [],
                    "disturbance_history": list(getattr(self, "disturbance_history", [])),
                    "n_applied_disturbances": len(getattr(self, "disturbance_history", [])),
                    "last_decision": dict(getattr(self, "_last_decision", {})),
                    "destination_candidates": list(getattr(self, "_last_destination_candidate_samples", [])),
                    "aqc_candidates": list(getattr(self, "_last_aqc_candidate_samples", [])),
                    "rule_guidance_detail": dict(getattr(self, "_last_rule_guidance_detail", {})),
                    "rule_guidance_stats": dict(getattr(self, "_rule_guidance_stats", {})),
                    "unfinished_task_ids": unfinished_task_ids,
                    "canceled_task_ids": canceled_task_ids,
                    "real_done_task_ids": real_done_task_ids,
                    "n_unfinished_tasks": len(unfinished_task_ids),
                    "n_canceled_tasks": len(canceled_task_ids),
                    "n_real_done_tasks": len(real_done_task_ids),
                })


    def _current_makespan(self) -> float:
        """Current episode makespan based on train finish times."""
        if getattr(self, "F", None):
            vals = [float(v) for v in self.F.values()]
            if vals:
                return float(max(vals))
        done_finish = [float(getattr(t, "finish_time", 0.0)) for t in self.tasks if getattr(t, "done", False)]
        return float(max(done_finish)) if done_finish else float(getattr(self, "t_now", 0.0))

    def _reset_aqc_workloads(self) -> None:
        """Initialize cumulative AQC processing workloads for one episode."""
        self.aqc_workloads = {}
        for idx, aqc in enumerate(list(getattr(self, "aqcs", []) or [])):
            try:
                aqc_id = int(getattr(aqc, "id", idx))
            except Exception:
                aqc_id = int(idx)
            self.aqc_workloads[aqc_id] = 0.0

    def _record_aqc_workload(self, aqc: AQCState, start_time: float, finish_time: float) -> None:
        """Accumulate the actual processing duration assigned to one AQC.

        This should be called only when a task is truly executed in the current
        environment state.  Candidate enumeration must not update this value.
        """
        try:
            if not hasattr(self, "aqc_workloads") or self.aqc_workloads is None:
                self._reset_aqc_workloads()
            aqc_id = int(getattr(aqc, "id", 0))
            duration = max(0.0, float(finish_time) - float(start_time))
            self.aqc_workloads[aqc_id] = float(self.aqc_workloads.get(aqc_id, 0.0)) + duration
        except Exception:
            pass

    def _get_aqc_workloads(self) -> Dict[int, float]:
        """Return workloads ordered by current AQC ids, filling missing ids with 0."""
        if not hasattr(self, "aqc_workloads") or self.aqc_workloads is None:
            self._reset_aqc_workloads()
        out: Dict[int, float] = {}
        for idx, aqc in enumerate(list(getattr(self, "aqcs", []) or [])):
            try:
                aqc_id = int(getattr(aqc, "id", idx))
            except Exception:
                aqc_id = int(idx)
            out[aqc_id] = float(self.aqc_workloads.get(aqc_id, 0.0))
        return out

    def _current_aqc_balance(self) -> float:
        """AQC workload balance: std(B_1,...,B_K) / mean(B_1,...,B_K).

        B_k is the cumulative processing workload assigned to AQC k:
        B_k = sum_{i in J_k} (finish_i - start_i).  Lower is better.
        """
        try:
            import numpy as _np
            workloads = [float(v) for v in self._get_aqc_workloads().values()]
            if len(workloads) <= 1:
                return 0.0
            mean_workload = float(_np.mean(workloads))
            if mean_workload <= 1e-9:
                return 0.0
            return float(_np.std(workloads) / mean_workload)
        except Exception:
            return 0.0

    def _build_obs(self) -> np.ndarray:
        return self.obs_packer.build(self)

    def close(self):
        pass

    # ============================================================
    # 决策子流程
    # ============================================================
    def _apply_rule_guidance(self, stage: str, candidate, base_score: float,
                             task: Optional[Task] = None, extra: Optional[Dict[str, Any]] = None) -> float:
        """v14-RGCD: 在连续动作解码阶段做规则引导。

        LEGIBLE 原文是对离散 action 做 enforce / block；本工程的 action 是 9维连续向量，
        因此这里把规则作用点放到候选对象评分层：
          - stage="task"        : 修正候选任务 score
          - stage="destination" : 修正目的地/车厢/slot/cross 组合 score
          - stage="aqc"         : 修正 AQC score

        rule_guidance_fn 可以返回：
          - float：直接作为 score 调整量
          - dict ：{"adjust": float, "mask": bool, "matched_rules": [...], ...}
        """
        fn = getattr(self, "rule_guidance_fn", None)
        if fn is None:
            return float(base_score)

        stats = getattr(self, "_rule_guidance_stats", None)
        if stats is None:
            stats = self._rule_guidance_stats = {
                "calls": 0, "triggered": 0, "masked": 0,
                "task_triggered": 0, "destination_triggered": 0, "aqc_triggered": 0,
                "total_adjust": 0.0, "abs_total_adjust": 0.0,
                "match_strength_sum": 0.0, "matched_rule_count": 0,
                "trigger_strength_sum": 0.0, "trigger_strength_count": 0,
                "high_strength_triggered": 0,
            }
        stats["calls"] = int(stats.get("calls", 0)) + 1

        try:
            out = fn(
                env=self,
                stage=str(stage),
                candidate=candidate,
                task=task,
                base_score=float(base_score),
                extra=extra or {},
            )
        except TypeError:
            # 兼容简单函数签名：fn(env, stage, candidate, task, base_score, extra)
            try:
                out = fn(self, stage, candidate, task, float(base_score), extra or {})
            except Exception as e:
                self._last_rule_guidance_detail = {
                    "stage": str(stage), "reason": "rule_guidance_exception",
                    "exception_type": type(e).__name__, "exception_msg": str(e),
                    "base_score": float(base_score), "final_score": float(base_score),
                }
                return float(base_score)
        except Exception as e:
            self._last_rule_guidance_detail = {
                "stage": str(stage), "reason": "rule_guidance_exception",
                "exception_type": type(e).__name__, "exception_msg": str(e),
                "base_score": float(base_score), "final_score": float(base_score),
            }
            return float(base_score)

        adjust = 0.0
        mask = False
        matched_rules = []
        matched_rule_details = []
        if isinstance(out, dict):
            adjust = float(out.get("adjust", out.get("bonus", 0.0)) or 0.0)
            mask = bool(out.get("mask", False))
            matched_rules = list(out.get("matched_rules", []))
            matched_rule_details = list(out.get("matched_rule_details", []))
        elif out is not None:
            adjust = float(out)

        if mask:
            final_score = -1e18
            stats["masked"] = int(stats.get("masked", 0)) + 1
        else:
            final_score = float(base_score) + adjust

        if mask or abs(adjust) > 1e-12 or matched_rules:
            stats["triggered"] = int(stats.get("triggered", 0)) + 1
            key = f"{stage}_triggered"
            stats[key] = int(stats.get(key, 0)) + 1
            stats["total_adjust"] = float(stats.get("total_adjust", 0.0)) + adjust
            stats["abs_total_adjust"] = float(stats.get("abs_total_adjust", 0.0)) + abs(adjust)
            strengths = []
            for d in matched_rule_details:
                try:
                    strengths.append(float(d.get("match_strength", 1.0)))
                except Exception:
                    pass
            if strengths:
                stats["match_strength_sum"] = float(stats.get("match_strength_sum", 0.0)) + float(sum(strengths))
                stats["matched_rule_count"] = int(stats.get("matched_rule_count", 0)) + int(len(strengths))
                trigger_strength = float(sum(strengths) / max(1, len(strengths)))
                stats["trigger_strength_sum"] = float(stats.get("trigger_strength_sum", 0.0)) + trigger_strength
                stats["trigger_strength_count"] = int(stats.get("trigger_strength_count", 0)) + 1
                if trigger_strength >= 0.70:
                    stats["high_strength_triggered"] = int(stats.get("high_strength_triggered", 0)) + 1
            self._last_rule_guidance_detail = {
                "stage": str(stage),
                "candidate_type": type(candidate).__name__,
                "task_id": int(getattr(task or candidate, "id", -1) or -1),
                "task_kind": str(getattr(task or candidate, "kind", "")),
                "base_score": float(base_score),
                "adjust": float(adjust),
                "mask": bool(mask),
                "final_score": float(final_score),
                "matched_rules": matched_rules,
                "matched_rule_details": matched_rule_details,
            }

        return float(final_score)

    def _pick_best_task(self, theta_i, done_tasks: List[Task]) -> Optional[Task]:
        """用 score_task 选当前最优任务。"""
        best_task = None
        best_score = -1e18
        for t in self.tasks:
            if t.done:
                continue
            base_s = score_task(
                t, self.aqcs, self.cars_by_train, self.slots,
                self.unload_truck_slots, self.crosses,
                self.current_height, done_tasks,
                theta_i, self.A1, self.phi,
                train_planning_blocked=self.train_planning_blocked,
            )
            s = self._apply_rule_guidance("task", t, base_s, task=t, extra={"done_tasks": done_tasks})
            if s > best_score:
                best_score = s
                best_task = t
        # 若所有任务都被屏蔽（-1e18），best_task 仍然是第一个 not done 的，
        # 但 best_score 会是 -1e18。这种情况我们也返回 None，触发 _end_episode
        if best_score <= -1e17:
            return None
        return best_task

    def _make_destination_candidate_sample(self, dest, task: Task, base_score: float, final_score: float,
                                           candidate_index: int, dest_kind: str = "",
                                           selected: bool = False, cross=None,
                                           preferred_aqc=None, tf: Optional[Dict[str, Any]] = None,
                                           extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """构造候选级 destination 样本，用于第三层 destination 规则挖掘。

        与 rule_guidance.extract_guidance_features(stage="destination") 保持同名特征，
        这样决策树挖出的规则可以直接转成 stage="destination" 的 RGCD 规则。
        """
        extra = extra or {}
        aqcs = list(getattr(self, "aqcs", []) or [])
        tasks = list(getattr(self, "tasks", []) or [])
        remaining = [t for t in tasks if (not getattr(t, "done", False)) and (not getattr(t, "canceled", False))]
        n_remaining = max(1, len(remaining))
        kind_counts: Dict[str, int] = {}
        for t in remaining:
            k = str(getattr(t, "kind", "unknown"))
            kind_counts[k] = kind_counts.get(k, 0) + 1
        aqc_times = [float(getattr(a, "available_time", 0.0)) for a in aqcs] or [0.0]
        mean_aqc_t = float(np.mean(aqc_times))
        max_aqc_t = float(np.max(aqc_times))
        min_aqc_t = float(np.min(aqc_times))
        dest_bay = float(getattr(dest, "bay", getattr(task, "init_bay", 0.0)))
        task_init_bay = float(getattr(task, "init_bay", 0.0))
        arrival = float(getattr(task, "arrival_time", 0.0))
        row: Dict[str, Any] = {
            "stage": "destination",
            "candidate_index": int(candidate_index),
            "selected": bool(selected),
            "selected_dest_index": -1,
            "scenario": str(getattr(self, "scenario", "unknown")),
            "current_step": int(getattr(self, "current_step", 0)),
            "t_now": float(getattr(self, "t_now", 0.0)),
            "obj": float(getattr(self, "obj", 0.0)),
            "train_obj": float(getattr(self, "train_obj", 0.0)),
            "truck_total_wait": float(getattr(self, "truck_total_wait", 0.0)),
            "truck_total_wait_norm": float(getattr(self, "truck_total_wait", 0.0)) / 10000.0,
            "load_truck_wait_norm": float(getattr(self, "truck_total_wait", 0.0)) / 10000.0,
            "unload_truck_wait_norm": 0.0,
            "n_remaining_tasks": int(len(remaining)),
            "mean_aqc_available_time": mean_aqc_t,
            "max_aqc_available_time": max_aqc_t,
            "min_aqc_available_time": min_aqc_t,
            "aqc_time_spread": max_aqc_t - min_aqc_t,
            "task_id": int(getattr(task, "id", -1) or -1),
            "task_kind": str(getattr(task, "kind", "")),
            "task_train_id": int(getattr(task, "train_id", -1) or -1),
            "task_arrival_time": arrival,
            "task_age": max(0.0, float(getattr(self, "t_now", 0.0)) - arrival),
            "task_init_bay": task_init_bay,
            "task_urgency_boost": float(getattr(task, "urgency_boost", 0.0)),
            "dest_kind": str(dest_kind or type(dest).__name__),
            "dest_row": float(getattr(dest, "row", 0.0)),
            "dest_bay": dest_bay,
            "dest_tier": float(getattr(dest, "tier", 0.0)),
            "dest_occupied": 1.0 if bool(getattr(dest, "occupied", False)) else 0.0,
            "dest_available_time": float(getattr(dest, "available_time", getattr(self, "t_now", 0.0) if not bool(getattr(dest, "occupied", False)) else 1e18)),
            "yard_stack_height": float(self.current_height.get((getattr(dest, "row", 0.0), getattr(dest, "bay", 0.0)), getattr(dest, "tier", 0.0))),
            "dest_distance_to_task": abs(dest_bay - task_init_bay),
            "available_yard_slot_ratio": float(sum(1 for s in getattr(self, "slots", []) if (not getattr(s, "occupied", False))) / max(1, len(getattr(self, "slots", []) or []))),
            "available_truck_slot_ratio": float(sum(1 for s in getattr(self, "unload_truck_slots", []) if (not getattr(s, "occupied", False))) / max(1, len(getattr(self, "unload_truck_slots", []) or []))),
            "base_score": float(base_score),
            "guided_score": float(final_score),
        }
        for k in ["load_yard", "load_truck", "unload_yard", "unload_truck"]:
            row[f"remaining_{k}"] = int(kind_counts.get(k, 0))
            row[f"remaining_ratio_{k}"] = float(kind_counts.get(k, 0)) / float(n_remaining)
        if cross is not None:
            row.update({
                "cross_id": int(getattr(cross, "id", -1) or -1),
                "cross_bay": float(getattr(cross, "bay", 0.0)),
                "cross_available_time": float(getattr(cross, "available_time", 0.0)),
                "cross_distance_to_task": abs(float(getattr(cross, "bay", 0.0)) - task_init_bay),
                "cross_distance_to_dest": abs(float(getattr(cross, "bay", 0.0)) - dest_bay),
            })
        if preferred_aqc is not None:
            row.update({
                "preferred_aqc_id": int(getattr(preferred_aqc, "id", -1) or -1),
                "preferred_aqc_available_time": float(getattr(preferred_aqc, "available_time", 0.0)),
                "preferred_aqc_cur_bay": float(getattr(preferred_aqc, "cur_bay", 0.0)),
                "preferred_aqc_gap_to_task": abs(float(getattr(preferred_aqc, "cur_bay", 0.0)) - task_init_bay),
                "preferred_aqc_gap_to_dest": abs(float(getattr(preferred_aqc, "cur_bay", 0.0)) - dest_bay),
            })
        if aqcs:
            row["min_aqc_gap_to_dest"] = float(min(abs(float(getattr(a, "cur_bay", 0.0)) - dest_bay) for a in aqcs))
        if isinstance(tf, dict):
            for k in ["finish", "start", "wait_aqc", "wait_cross", "total_wait", "h2", "o2"]:
                if k in tf:
                    row[f"tf_{k}"] = float(tf[k])
        for k, v in extra.items():
            if isinstance(v, (int, float, bool, str)):
                row[str(k)] = v
        return row

    def _pick_destination(self, best_task: Task, theta_g, theta_p,
                          done_tasks: List[Task]):
        """根据 task.kind 分发到 4 个子流程。"""
        if best_task.kind == "load_yard":
            chosen, idx = self._pick_dest_load_yard(best_task, theta_g, done_tasks)
            return chosen, idx, None, None
        elif best_task.kind == "load_truck":
            return self._pick_dest_load_truck(best_task, theta_g, theta_p, done_tasks)
        elif best_task.kind == "unload_yard":
            chosen, idx = self._pick_dest_unload_yard(best_task, theta_g, done_tasks)
            return chosen, idx, None, None
        elif best_task.kind == "unload_truck":
            chosen, idx = self._pick_dest_unload_truck(best_task, theta_g, done_tasks)
            return chosen, idx, None, None
        return None, -1, None, None

    def probe_destination_candidates_by_task_kind(self, max_tasks_per_kind: int = 1) -> List[Dict[str, Any]]:
        """Collect destination candidates for unfinished tasks of each task_kind without executing them.

        This is only for mining/template completion.  It prevents the destination
        sample table from containing only the task kind preferred by the current
        policy.  All rows produced by this probe should be treated as
        selected=False by the external collector.
        """
        old_last = list(getattr(self, "_last_destination_candidate_samples", []) or [])
        rows: List[Dict[str, Any]] = []
        try:
            done_tasks = [t for t in self.tasks if getattr(t, "done", False)]
            theta_g = np.zeros((D_G,), dtype=np.float32)
            theta_p = np.zeros((D_P,), dtype=np.float32)
            used: Dict[str, int] = {}
            for task in list(getattr(self, "tasks", []) or []):
                if getattr(task, "done", False) or getattr(task, "canceled", False):
                    continue
                kind = str(getattr(task, "kind", "unknown"))
                if used.get(kind, 0) >= int(max_tasks_per_kind):
                    continue
                used[kind] = used.get(kind, 0) + 1
                try:
                    self._pick_destination(task, theta_g, theta_p, done_tasks)
                    for r in list(getattr(self, "_last_destination_candidate_samples", []) or []):
                        rr = dict(r)
                        rr["probe_kind"] = f"probe_{kind}"
                        rr["selected"] = False
                        rows.append(rr)
                except Exception:
                    continue
        finally:
            self._last_destination_candidate_samples = old_last
        return rows

    def _pick_dest_load_yard(self, best_task, theta_g, done_tasks):
        """load_yard: 在该列车的车厢中选最优。"""
        best_score = -1e18
        chosen_dest = None
        chosen_idx = -1
        candidate_records: List[Dict[str, Any]] = []
        for idx, car in enumerate(self.cars_by_train.get(best_task.train_id, [])):
            if car.occupied:
                continue
            # 可行性检查
            feasible = False
            for aqc in self.aqcs:
                s_i, f_i = simulate_task_times(aqc, best_task, car, self.A1)
                if check_aqc_safety(s_i, f_i, best_task.init_bay, car.bay,
                                    aqc.id, done_tasks):
                    feasible = True
                    break
            if not feasible:
                continue
            base_s = score_destination(best_task, car, theta_g, self.current_height)
            s = self._apply_rule_guidance("destination", car, base_s, task=best_task,
                                          extra={"dest_kind": "TrainCar", "done_tasks": done_tasks})
            candidate_records.append(self._make_destination_candidate_sample(
                dest=car, task=best_task, base_score=float(base_s), final_score=float(s),
                candidate_index=int(idx), dest_kind="TrainCar", selected=False,
            ))
            if s > best_score:
                best_score = s
                chosen_dest = car
                chosen_idx = idx
        for r in candidate_records:
            r["selected"] = bool(int(r.get("candidate_index", -1)) == int(chosen_idx))
            r["selected_dest_index"] = int(chosen_idx)
        self._last_destination_candidate_samples = candidate_records
        return chosen_dest, chosen_idx

    def _pick_dest_load_truck(self, best_task, theta_g, theta_p, done_tasks):
        """
        load_truck: 三层候选筛选（AQC → cross → car），用 score_cross 评分。

        返回 (chosen_car, car_idx, chosen_cross, preferred_aqc)
        """
        best_score = -1e18
        chosen_dest = None
        chosen_dest_idx = -1
        chosen_cross = None
        preferred_aqc = None
        candidate_records: List[Dict[str, Any]] = []

        cars_all = self.cars_by_train.get(best_task.train_id, [])
        candidate_aqcs = get_truck_service_aqcs(
            best_task, self.aqcs, top_k=TRUCK_SERVICE_AQC_K,
        )

        for aqc in candidate_aqcs:
            candidate_crosses = get_crosses_near_aqc(
                aqc, self.crosses, top_k=CROSS_NEAR_AQC_K,
            )
            for cross in candidate_crosses:
                candidate_cars = get_cars_near_cross_or_task(
                    best_task, cross, cars_all, top_k=CAR_NEAR_CROSS_K,
                )
                for car in candidate_cars:
                    try:
                        car_idx = cars_all.index(car)
                    except ValueError:
                        continue
                    tf = simulate_load_truck_times(aqc, best_task, cross, car, self.A1)
                    if not check_aqc_safety(
                        tf["start"], tf["finish"],
                        best_task.init_bay, car.bay, aqc.id, done_tasks,
                    ):
                        continue
                    mean_t = float(np.mean([a.available_time for a in self.aqcs])) \
                             if self.aqcs else 0.0
                    imbalance = abs(float(aqc.available_time) - mean_t)
                    overload_penalty = max(0.0, float(aqc.available_time) - mean_t)
                    cross_score = score_cross(best_task, cross, theta_p)
                    s_local = (
                        cross_score
                        - 1.0 * tf["finish"]
                        - 3.0 * tf["wait_aqc"]
                        - 1.5 * tf["total_wait"]
                        - 0.6 * tf["wait_cross"]
                        - 0.5 * imbalance
                        - 0.5 * overload_penalty
                        - 0.05 * abs(float(car.bay) - float(cross.bay))
                    )
                    s_local = self._apply_rule_guidance(
                        "destination", car, s_local, task=best_task,
                        extra={
                            "dest_kind": "TrainCar", "cross": cross, "preferred_aqc": aqc,
                            "tf": tf, "imbalance": imbalance, "overload_penalty": overload_penalty,
                            "done_tasks": done_tasks,
                        },
                    )
                    candidate_records.append(self._make_destination_candidate_sample(
                        dest=car, task=best_task, base_score=float(cross_score), final_score=float(s_local),
                        candidate_index=int(car_idx), dest_kind="TrainCar", selected=False,
                        cross=cross, preferred_aqc=aqc, tf=tf,
                        extra={"imbalance": imbalance, "overload_penalty": overload_penalty},
                    ))
                    if s_local > best_score:
                        best_score = s_local
                        chosen_dest = car
                        chosen_dest_idx = int(car_idx)
                        chosen_cross = cross
                        preferred_aqc = aqc

        for r in candidate_records:
            r["selected"] = bool(
                int(r.get("candidate_index", -1)) == int(chosen_dest_idx)
                and int(r.get("cross_id", -999)) == int(getattr(chosen_cross, "id", -999) if chosen_cross is not None else -999)
                and int(r.get("preferred_aqc_id", -999)) == int(getattr(preferred_aqc, "id", -999) if preferred_aqc is not None else -999)
            )
            r["selected_dest_index"] = int(chosen_dest_idx)
        self._last_destination_candidate_samples = candidate_records
        return chosen_dest, chosen_dest_idx, chosen_cross, preferred_aqc

    def _pick_dest_unload_yard(self, best_task, theta_g, done_tasks):
        """unload_yard: 在堆场 slot 中选最优（tier 必须匹配当前堆叠高度）。"""
        best_score = -1e18
        chosen_dest = None
        chosen_idx = -1
        candidate_records: List[Dict[str, Any]] = []
        for idx, slot in enumerate(self.slots):
            if slot.occupied:
                continue
            h0 = self.current_height.get((slot.row, slot.bay), -1)
            if int(slot.tier) != int(h0):
                continue
            feasible = False
            for aqc in self.aqcs:
                s_i, f_i = simulate_task_times(aqc, best_task, slot, self.A1)
                if check_aqc_safety(s_i, f_i, best_task.init_bay, slot.bay,
                                    aqc.id, done_tasks):
                    feasible = True
                    break
            if not feasible:
                continue
            base_s = score_destination(best_task, slot, theta_g, self.current_height)
            s = self._apply_rule_guidance("destination", slot, base_s, task=best_task,
                                          extra={"dest_kind": "YardSlot", "done_tasks": done_tasks})
            candidate_records.append(self._make_destination_candidate_sample(
                dest=slot, task=best_task, base_score=float(base_s), final_score=float(s),
                candidate_index=int(idx), dest_kind="YardSlot", selected=False,
            ))
            if s > best_score:
                best_score = s
                chosen_dest = slot
                chosen_idx = idx
        for r in candidate_records:
            r["selected"] = bool(int(r.get("candidate_index", -1)) == int(chosen_idx))
            r["selected_dest_index"] = int(chosen_idx)
        self._last_destination_candidate_samples = candidate_records
        return chosen_dest, chosen_idx

    def _pick_dest_unload_truck(self, best_task, theta_g, done_tasks):
        """unload_truck: 在卡车 slot 中选最优。"""
        best_score = -1e18
        chosen_dest = None
        chosen_idx = -1
        candidate_records: List[Dict[str, Any]] = []
        for idx, slot in enumerate(self.unload_truck_slots):
            if slot.occupied:
                continue
            feasible = False
            for aqc in self.aqcs:
                s_i, f_i = simulate_task_times(aqc, best_task, slot, self.A1)
                if check_aqc_safety(s_i, f_i, best_task.init_bay, slot.bay,
                                    aqc.id, done_tasks):
                    feasible = True
                    break
            if not feasible:
                continue
            base_s = score_destination(best_task, slot, theta_g, self.current_height)
            s = self._apply_rule_guidance("destination", slot, base_s, task=best_task,
                                          extra={"dest_kind": "TruckSlot", "done_tasks": done_tasks})
            candidate_records.append(self._make_destination_candidate_sample(
                dest=slot, task=best_task, base_score=float(base_s), final_score=float(s),
                candidate_index=int(idx), dest_kind="TruckSlot", selected=False,
            ))
            if s > best_score:
                best_score = s
                chosen_dest = slot
                chosen_idx = idx
        for r in candidate_records:
            r["selected"] = bool(int(r.get("candidate_index", -1)) == int(chosen_idx))
            r["selected_dest_index"] = int(chosen_idx)
        self._last_destination_candidate_samples = candidate_records
        return chosen_dest, chosen_idx

    def _make_aqc_candidate_sample(self, aqc: AQCState, task: Task, chosen_dest,
                                   chosen_cross, base_score: float, final_score: float,
                                   tf: Optional[Dict[str, Any]], candidate_index: int,
                                   selected: bool = False) -> Dict[str, Any]:
        """构造候选级 AQC 样本，用于 AQC 规则挖掘。

        字段名尽量与 rule_guidance.extract_guidance_features 保持一致，
        这样挖出来的阈值规则可以直接在 stage="aqc" 中触发。
        """
        aqcs = list(getattr(self, "aqcs", []) or [])
        tasks = list(getattr(self, "tasks", []) or [])
        remaining = [t for t in tasks if (not getattr(t, "done", False)) and (not getattr(t, "canceled", False))]
        n_remaining = max(1, len(remaining))
        kind_counts: Dict[str, int] = {}
        for t in remaining:
            k = str(getattr(t, "kind", "unknown"))
            kind_counts[k] = kind_counts.get(k, 0) + 1
        aqc_times = [float(getattr(a, "available_time", 0.0)) for a in aqcs] or [0.0]
        mean_aqc_t = float(np.mean(aqc_times))
        max_aqc_t = float(np.max(aqc_times))
        min_aqc_t = float(np.min(aqc_times))
        dest_bay = float(getattr(chosen_dest, "bay", getattr(task, "init_bay", 0.0)))
        task_init_bay = float(getattr(task, "init_bay", 0.0))
        arrival = float(getattr(task, "arrival_time", 0.0))
        row: Dict[str, Any] = {
            "stage": "aqc",
            "candidate_index": int(candidate_index),
            "selected": bool(selected),
            "selected_aqc_id": -1,
            "scenario": str(getattr(self, "scenario", "unknown")),
            "current_step": int(getattr(self, "current_step", 0)),
            "t_now": float(getattr(self, "t_now", 0.0)),
            "obj": float(getattr(self, "obj", 0.0)),
            "train_obj": float(getattr(self, "train_obj", 0.0)),
            "truck_total_wait": float(getattr(self, "truck_total_wait", 0.0)),
            "n_remaining_tasks": int(len(remaining)),
            "mean_aqc_available_time": mean_aqc_t,
            "max_aqc_available_time": max_aqc_t,
            "min_aqc_available_time": min_aqc_t,
            "aqc_time_spread": max_aqc_t - min_aqc_t,
            "task_id": int(getattr(task, "id", -1) or -1),
            "task_kind": str(getattr(task, "kind", "")),
            "task_train_id": int(getattr(task, "train_id", -1) or -1),
            "task_arrival_time": arrival,
            "task_age": max(0.0, float(getattr(self, "t_now", 0.0)) - arrival),
            "task_init_bay": task_init_bay,
            "task_urgency_boost": float(getattr(task, "urgency_boost", 0.0)),
            "dest_bay": dest_bay,
            "dest_row": float(getattr(chosen_dest, "row", 0.0)),
            "dest_tier": float(getattr(chosen_dest, "tier", 0.0)),
            "aqc_id": int(getattr(aqc, "id", -1) or -1),
            "aqc_cur_bay": float(getattr(aqc, "cur_bay", 0.0)),
            "aqc_available_time": float(getattr(aqc, "available_time", 0.0)),
            "aqc_available_delta": float(getattr(aqc, "available_time", 0.0)) - mean_aqc_t,
            "aqc_underload": max(0.0, mean_aqc_t - float(getattr(aqc, "available_time", 0.0))),
            "aqc_imbalance": abs(float(getattr(aqc, "available_time", 0.0)) - mean_aqc_t),
            "aqc_overload": max(0.0, float(getattr(aqc, "available_time", 0.0)) - mean_aqc_t),
            "aqc_task_count": int(len(getattr(aqc, "tasks", []) or [])),
            "mean_aqc_task_count": float(np.mean([len(getattr(a, "tasks", []) or []) for a in aqcs])) if aqcs else 0.0,
            "aqc_is_broken": bool(getattr(aqc, "is_broken", False)),
            "aqc_gap_to_task": abs(float(getattr(aqc, "cur_bay", 0.0)) - task_init_bay),
            "aqc_gap_to_dest": abs(float(getattr(aqc, "cur_bay", 0.0)) - dest_bay),
            "base_score": float(base_score),
            "guided_score": float(final_score),
        }
        for k in ["load_yard", "load_truck", "unload_yard", "unload_truck"]:
            row[f"remaining_{k}"] = int(kind_counts.get(k, 0))
            row[f"remaining_ratio_{k}"] = float(kind_counts.get(k, 0)) / float(n_remaining)
        row["aqc_task_overload"] = float(row.get("aqc_task_count", 0) - row.get("mean_aqc_task_count", 0.0))
        other_gaps = [abs(float(getattr(a, "cur_bay", 0.0)) - float(getattr(aqc, "cur_bay", 0.0))) for a in aqcs if int(getattr(a, "id", -1)) != int(getattr(aqc, "id", -2))]
        min_gap = float(min(other_gaps)) if other_gaps else 9999.0
        row["aqc_min_gap_to_other"] = min_gap
        row["aqc_conflict_risk"] = float(1.0 / (min_gap + 1.0))
        if chosen_cross is not None:
            row.update({
                "cross_id": int(getattr(chosen_cross, "id", -1) or -1),
                "cross_bay": float(getattr(chosen_cross, "bay", 0.0)),
                "cross_available_time": float(getattr(chosen_cross, "available_time", 0.0)),
                "cross_distance_to_task": abs(float(getattr(chosen_cross, "bay", 0.0)) - task_init_bay),
            })
        if isinstance(tf, dict):
            for k in ["finish", "start", "wait_aqc", "wait_cross", "total_wait", "h2", "o2"]:
                if k in tf:
                    row[f"tf_{k}"] = float(tf[k])
        return row

    def _pick_aqc(self, best_task, chosen_dest, chosen_cross,
                  preferred_aqc, theta_k, done_tasks):
        """
        选最优 AQC。

        v14-AQC：在不改变原调度逻辑的前提下，记录所有可行 AQC 候选的特征
        和最终 selected 标记，用于候选级 AQC 规则挖掘。
        """
        best_aqc = None
        best_score = -1e18
        candidate_records: List[Dict[str, Any]] = []

        candidates = (
            [preferred_aqc]
            if (best_task.kind == "load_truck" and preferred_aqc is not None)
            else self.aqcs
        )

        for cand_idx, aqc in enumerate(candidates):
            if aqc is None:
                continue
            if best_task.kind == "load_truck":
                tf = simulate_load_truck_times(aqc, best_task, chosen_cross, chosen_dest, self.A1)
                if not check_aqc_safety(
                    tf["start"], tf["finish"],
                    best_task.init_bay, chosen_dest.bay,
                    aqc.id, done_tasks,
                ):
                    continue
                base_s = score_aqc_load_truck(
                    best_task, chosen_cross, chosen_dest,
                    aqc, self.aqcs, theta_k, self.A1,
                )
                s = self._apply_rule_guidance("aqc", aqc, base_s, task=best_task,
                                              extra={"chosen_dest": chosen_dest, "chosen_cross": chosen_cross, "tf": tf,
                                                     "done_tasks": done_tasks})
            else:
                s_i, f_i = simulate_task_times(aqc, best_task, chosen_dest, self.A1)
                if not check_aqc_safety(
                    s_i, f_i, best_task.init_bay, chosen_dest.bay,
                    aqc.id, done_tasks,
                ):
                    continue
                tf = {"start": float(s_i), "finish": float(f_i), "wait_aqc": max(0.0, float(s_i) - float(getattr(aqc, "available_time", 0.0)))}
                base_s = score_aqc_normal(
                    best_task, chosen_dest, aqc, self.aqcs, theta_k, self.A1,
                )
                s = self._apply_rule_guidance("aqc", aqc, base_s, task=best_task,
                                              extra={"chosen_dest": chosen_dest, "chosen_cross": chosen_cross,
                                                     "tf": tf, "done_tasks": done_tasks})
            candidate_records.append(
                self._make_aqc_candidate_sample(
                    aqc=aqc, task=best_task, chosen_dest=chosen_dest, chosen_cross=chosen_cross,
                    base_score=float(base_s), final_score=float(s), tf=tf, candidate_index=cand_idx,
                    selected=False,
                )
            )
            if s > best_score:
                best_score = s
                best_aqc = aqc

        if best_aqc is not None:
            for r in candidate_records:
                is_sel = int(r.get("aqc_id", -1)) == int(getattr(best_aqc, "id", -999))
                r["selected"] = bool(is_sel)
                r["selected_aqc_id"] = int(getattr(best_aqc, "id", -1) or -1)
        self._last_aqc_candidate_samples = candidate_records
        return best_aqc


    # ============================================================
    # 执行任务（修改 env 状态）
    # ============================================================
    def _estimate_planned_start(self, task: Task, dest, cross, aqc: AQCState) -> float:
        """
        v10: 估计当前候选决策的任务开始时间，用于在执行前触发所有
        time <= planned_start 的扰动事件。

        关键修复：
        如果列车在计划到达时刻后被通知晚到，且本次任务的预计开始时间
        已经跨过通知时刻，则必须先应用晚到通知，再重新选任务。
        """
        if task.kind == "load_truck":
            tf = simulate_load_truck_times(aqc, task, cross, dest, self.A1)
            return float(tf["start"])
        s_i, _ = simulate_task_times(aqc, task, dest, self.A1)
        return float(s_i)

    def _execute_normal(self, task: Task, dest, dest_idx: int, aqc: AQCState):
        """load_yard / unload_yard / unload_truck 的执行"""
        s_i, f_i = simulate_task_times(aqc, task, dest, self.A1)
        task.done = True
        task.assigned_aqc_idx = aqc.id
        task.assigned_dest_idx = int(dest_idx)
        task.start_time = float(s_i)
        task.finish_time = float(f_i)
        task.final_row = float(dest.row)
        task.final_tier = float(dest.tier)
        task.final_bay = float(dest.bay)
        if task.kind == "load_yard":
            dest.occupied = True
        elif task.kind == "unload_yard":
            dest.occupied = True
            self.current_height[(dest.row, dest.bay)] = max(
                self.current_height.get((dest.row, dest.bay), -1),
                int(dest.tier),
            )
        elif task.kind == "unload_truck":
            dest.occupied = True
        aqc.cur_row = float(dest.row)
        aqc.cur_bay = float(dest.bay)
        aqc.available_time = float(f_i)
        aqc.tasks.append(task.id)
        self._record_aqc_workload(aqc, task.start_time, task.finish_time)

    def _execute_load_truck(self, task: Task, car, car_idx: int,
                            cross, aqc: AQCState):
        """load_truck 的执行：涉及 cross 占用 + truck等待统计"""
        tf = simulate_load_truck_times(aqc, task, cross, car, self.A1)
        task.done = True
        task.assigned_aqc_idx = aqc.id
        task.assigned_dest_idx = int(car_idx)
        task.assigned_cross_idx = int(cross.id)
        task.start_time = float(tf["start"])
        task.finish_time = float(tf["finish"])
        task.final_row = float(car.row)
        task.final_tier = float(car.tier)
        task.final_bay = float(car.bay)
        task.cross_row = float(cross.row)
        task.cross_tier = float(cross.tier)
        task.cross_bay = float(cross.bay)
        task.h2 = float(tf["h2"])
        task.o2 = float(tf["o2"])
        task.wait_cross = float(tf["wait_cross"])
        task.wait_aqc = float(tf["wait_aqc"])
        task.total_wait = float(tf["total_wait"])
        car.occupied = True
        cross.available_time = float(task.o2)
        aqc.cur_row = float(car.row)
        aqc.cur_bay = float(car.bay)
        aqc.available_time = float(task.finish_time)
        aqc.tasks.append(task.id)
        self._record_aqc_workload(aqc, task.start_time, task.finish_time)

    # ============================================================
    # ★ v7 新增：扰动剧本采样 + 应用
    # ============================================================
    def _sample_disturbance_script(self):
        """
        采样本 episode 的扰动剧本。

        原逻辑：
            seed = master_seed + episode_count

        问题：
            测试时每个文件都单独创建 env，episode_count 都从 0 开始；
            如果所有测试都用 --seed 42，那么每个文件的扰动剧本会高度相似。

        新逻辑：
            seed = master_seed + episode_count + instance_hash

        好处：
            1. 同一个实例、同一个 seed 下，Round0/Round1/Round2/Round4 扰动一致，保证公平；
            2. 不同实例即使使用同一个 seed，也会得到不同扰动剧本；
            3. 训练时不同 episode 和不同实例的扰动更丰富。
        """
        # 用文件名生成稳定 hash，不能用 Python 内置 hash()，
        # 因为 hash() 每次进程可能不同。
        instance_name = os.path.basename(str(self.instance_path))
        h = hashlib.md5(instance_name.encode("utf-8")).hexdigest()
        instance_offset = int(h[:8], 16) % 1_000_000

        episode_seed = (
                int(self._master_disturbance_seed)
                + int(self._episode_count)
                + int(instance_offset)
        )

        sampler = DisturbanceSampler(
            intensity=self.disturbance_intensity,
            seed=episode_seed,
        )

        # ------------------------------------------------------------
        # Disturbance timing reference horizon
        # ------------------------------------------------------------
        # 旧逻辑：
        #     estimated_horizon = n_tasks * 300 s
        #
        # 在当前 AQC=4 测试集中，这个值通常远大于实际作业时长，
        # 会把 break / insert / cancel / urgent 大量安排到 episode
        # 已经完成之后。
        #
        # 新逻辑使用“实例固定、算法无关”的参考时间窗：
        #
        #   H_ref = max(
        #       1800 s,
        #       latest_train_arrival + 300 s
        #       + 160 s * n_tasks / n_aqcs
        #   )
        #
        # 说明：
        # - latest_train_arrival：保留列车到达时间尺度；
        # - 300 s：固定缓冲；
        # - 160*n_tasks/n_aqcs：按任务规模和并行 AYC 数量缩放；
        # - 它不是某个算法的 makespan，因此同一 instance/seed/intensity
        #   在不同算法之间仍使用相同扰动剧本，保持公平性。
        n_tasks = int(len(self.tasks))
        n_aqcs = max(1, int(len(self.aqcs)))

        latest_train_arrival = max(
            (
                float(self.A1.get(int(tid), 0.0))
                for tid in self.trains_involved
            ),
            default=0.0,
        )

        disturbance_reference_horizon = max(
            1800.0,
            latest_train_arrival
            + 300.0
            + 160.0 * float(n_tasks) / float(n_aqcs),
        )

        # 保存到 env，便于图7/图2审计输出。
        self.disturbance_reference_horizon = float(
            disturbance_reference_horizon
        )

        self.disturbance_script = sampler.sample({
            "n_aqcs": len(self.aqcs),
            "trains_involved": list(self.trains_involved),
            "A1": dict(self.A1),
            "scenario": str(getattr(self, "scenario", "unknown")),
            # 为保持 DisturbanceSampler 接口不变，仍沿用字段名
            # estimated_horizon；其语义现在是固定 disturbance reference horizon。
            "estimated_horizon": float(disturbance_reference_horizon),
        })
        self.next_event_idx = 0

        # 记录扰动 seed，方便 demo / debug 时确认每个文件 seed 不同
        self.disturbance_episode_seed = int(episode_seed)

        self._episode_count += 1

    def _apply_pending_disturbances(self, target_t: float):
        """
        应用剧本中所有 time <= target_t 的事件。
        返回已应用事件列表，供 info / reward 使用。
        """
        if self.disturbance_script is None:
            return []
        applied = []
        events = self.disturbance_script.events
        while self.next_event_idx < len(events):
            ev = events[self.next_event_idx]
            if ev.time > target_t:
                break
            result = DisturbanceApplier.apply(self, ev)
            rec = {
                "event_index": int(self.next_event_idx),
                "event_type": ev.type,
                "event_time": float(ev.time),
                "event": ev,
                "result": result,
            }
            applied.append(rec)
            self.disturbance_history.append(rec)
            self.next_event_idx += 1
        return applied

    def _advance_to_next_disturbance_event(self) -> bool:
        """
        强制应用剧本中的下一个事件（用于所有任务被阻塞时推进）。
        返回 True 表示应用了一个事件，False 表示剧本已空。
        """
        if self.disturbance_script is None:
            return False
        if self.next_event_idx >= len(self.disturbance_script.events):
            return False
        ev = self.disturbance_script.events[self.next_event_idx]
        result = DisturbanceApplier.apply(self, ev)
        rec = {
            "event_index": int(self.next_event_idx),
            "event_type": ev.type,
            "event_time": float(ev.time),
            "event": ev,
            "result": result,
        }
        self.disturbance_history.append(rec)
        self.t_now = max(self.t_now, ev.time)
        self.next_event_idx += 1
        return True

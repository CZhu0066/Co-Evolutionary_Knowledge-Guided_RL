# -*- coding: utf-8 -*-
"""
feature_extractor.py
====================
从 env 状态 + 扰动状态提取扁平特征向量，供决策树蒸馏使用。

设计目标：
  - 特征大小固定（不依赖实例规模），可跨实例迁移
  - 全部归一化到 [0, 1] 或附近
  - 既反映「当前状态」又反映「已发生的扰动情况」
  - 可读：用 feature_names 属性暴露特征名，便于规则解释

总特征维度 = 25
"""
from __future__ import annotations
from typing import List

import numpy as np


# 特征名（顺序必须与 extract() 输出顺序一致）
FEATURE_NAMES: List[str] = [
    # ===== 进度类 (3) =====
    "progress",                # current_step / max_steps
    "tasks_done_ratio",        # done / total
    "tasks_remaining_norm",    # 剩余任务数 / 总任务数

    # ===== 任务种类分布 (4) =====
    "n_remaining_load_yard_ratio",
    "n_remaining_load_truck_ratio",
    "n_remaining_unload_yard_ratio",
    "n_remaining_unload_truck_ratio",

    # ===== 列车状态 (2) =====
    "n_trains_blocked_ratio",     # 阻塞列车 / 总列车
    "has_blocked_train",          # 0/1

    # ===== AQC 状态 (3) =====
    "n_aqcs_broken_ratio",
    "max_aqc_available_norm",     # 最大空闲时间 / horizon
    "aqc_available_spread_norm",  # (max - min) / horizon, AQC 之间不平衡度

    # ===== 扰动统计 (5) =====
    "n_breaks_applied_norm",      # 已应用的 break 数 / 10
    "n_pending_inserts_norm",     # 当前 pending insert 数 / 10
    "n_canceled_ratio",           # 已 cancel 任务比例
    "n_urgent_ratio",             # 紧急任务比例
    "n_events_progress",          # applied / total events

    # ===== 强度 one-hot (4) =====
    "intensity_clean",
    "intensity_low",
    "intensity_med",
    "intensity_high",

    # ===== 时间 (2) =====
    "t_now_norm",                 # t_now / horizon
    "time_to_next_event_norm",    # (next_event.time - t_now) / horizon

    # ===== 目标值 (2) =====
    "obj_norm",                   # obj / 1e4
    "truck_total_wait_norm",      # truck_wait / 1e4
]
N_FEATURES = len(FEATURE_NAMES)


class FeatureExtractor:
    """
    从 KGUnifiedYardEnv 提取 25 维特征向量。

    用法:
        extractor = FeatureExtractor(horizon=10000.0)
        feat = extractor.extract(env)   # ndarray (25,) float32
    """

    def __init__(self, horizon: float = 10000.0):
        self.horizon = float(horizon)

    @property
    def n_features(self) -> int:
        return N_FEATURES

    @property
    def feature_names(self) -> List[str]:
        return list(FEATURE_NAMES)

    def extract(self, env) -> np.ndarray:
        """提取一个 25 维 float32 向量"""
        feat: List[float] = []
        h = max(self.horizon, 1.0)

        # ===== 进度 =====
        total = max(1, len(env.tasks))
        n_done = sum(1 for t in env.tasks if t.done)
        n_remaining = total - n_done
        progress = env.current_step / max(1, env.max_steps)
        feat.append(float(np.clip(progress, 0.0, 2.0)))
        feat.append(n_done / total)
        feat.append(n_remaining / total)

        # ===== 任务种类 =====
        for kind in ["load_yard", "load_truck", "unload_yard", "unload_truck"]:
            n_kind = sum(1 for t in env.tasks
                         if not t.done and getattr(t, "kind", "") == kind)
            feat.append(n_kind / total)

        # ===== 列车 =====
        n_trains = max(1, len(env.trains_involved))
        n_blocked = len(env.train_planning_blocked)
        feat.append(n_blocked / n_trains)
        feat.append(1.0 if n_blocked > 0 else 0.0)

        # ===== AQC =====
        n_aqcs = max(1, len(env.aqcs))
        avails = [a.available_time for a in env.aqcs] if env.aqcs else [0.0]
        n_broken = sum(
            1 for a in env.aqcs
            if getattr(a, "last_break_time", -1.0) >= 0.0
            and getattr(a, "available_time", 0.0) > env.t_now
            and (getattr(a, "available_time", 0.0) - env.t_now) < 1e6
        )
        feat.append(n_broken / n_aqcs)
        feat.append(min(max(avails) / h, 5.0))
        feat.append(min((max(avails) - min(avails)) / h, 5.0))

        # ===== 扰动统计 =====
        # 已应用扰动事件按类型统计
        if env.disturbance_script is not None and env.disturbance_script.events:
            applied_events = env.disturbance_script.events[:env.next_event_idx]
            n_breaks = sum(1 for e in applied_events if e.type == "break")
            n_inserts_applied = sum(1 for e in applied_events if e.type == "insert")
            # canceled / urgent 直接看 task 标志
            n_canceled = sum(1 for t in env.tasks if getattr(t, "canceled", False))
            n_urgent = sum(1 for t in env.tasks
                           if getattr(t, "urgency_boost", 0.0) > 0)
            # 当前 pending insert（已 applied 但尚未完成的，简化为 applied 数）
            feat.append(min(n_breaks / 10.0, 1.0))
            feat.append(min(n_inserts_applied / 10.0, 1.0))
            feat.append(n_canceled / total)
            feat.append(n_urgent / total)
            feat.append(env.next_event_idx / max(1, env.disturbance_script.n_events))
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # ===== 强度 one-hot =====
        intensity = getattr(env, "disturbance_intensity", "clean")
        for lv in ["clean", "low", "med", "high"]:
            feat.append(1.0 if intensity == lv else 0.0)

        # ===== 时间 =====
        feat.append(min(env.t_now / h, 5.0))
        # 下个事件距离
        if (env.disturbance_script is not None
                and env.next_event_idx < env.disturbance_script.n_events):
            next_t = env.disturbance_script.events[env.next_event_idx].time
            feat.append(min(max(0.0, next_t - env.t_now) / h, 5.0))
        else:
            feat.append(1.0)  # 没有未来事件 → 默认 1.0

        # ===== 目标值 =====
        feat.append(min(env.obj / 1e4, 1e3))
        feat.append(min(env.truck_total_wait / 1e4, 1e3))

        arr = np.asarray(feat, dtype=np.float32)
        # 防御：clip + nan_to_num
        arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)
        assert arr.shape[0] == N_FEATURES, f"feature dim mismatch: {arr.shape[0]} != {N_FEATURES}"
        return arr

    def explain_feature(self, idx: int) -> str:
        """获取索引为 idx 的特征的名称"""
        return FEATURE_NAMES[idx]


# 标签编码：决策树要离散标签
TASK_KIND_TO_LABEL = {
    "load_yard": 0,
    "load_truck": 1,
    "unload_yard": 2,
    "unload_truck": 3,
}
LABEL_TO_TASK_KIND = {v: k for k, v in TASK_KIND_TO_LABEL.items()}


def encode_decision(last_decision: dict) -> int:
    """
    把 step() info 里的 last_decision 编码为整数标签。

    当前用 task_kind 作为分类目标。
    返回 -1 表示无效（无决策）。
    """
    if not last_decision or "task_kind" not in last_decision:
        return -1
    kind = last_decision["task_kind"]
    return TASK_KIND_TO_LABEL.get(kind, -1)


def decode_label(label: int) -> str:
    """整数标签 → 可读字符串"""
    return LABEL_TO_TASK_KIND.get(int(label), "unknown")

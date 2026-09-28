# -*- coding: utf-8 -*-
"""
reward.py
=========
奖励计算模块（重构版）。

重构要点（解决"模型回避卡车任务"+"critic 学不稳"）：
  1. truck 等待只在 obj 的 delta 里算一次，删除 shaping 情形B、删除 terminal 重复项
  2. step reward 按实例规模归一化（obj_scale），解决不同规模实例 critic 难收敛
  3. 完成 load_truck 给净正奖励（R_TRUCK_DONE - 等待惩罚），让模型有动机做卡车任务

reward 组成：
    total = clip(step + shaping + terminal + rule_bonus)
"""
from typing import Optional

import numpy as np

from config.constants import (
    R_COMPLETE, R_TRUCK_DONE, SCALE_STEP_BASE, SCALE_TERM, TRUNCATED_PENALTY,
    REWARD_CLIP_LOW, REWARD_CLIP_HIGH,
    TOTAL_REWARD_CLIP_LOW, TOTAL_REWARD_CLIP_HIGH,
)
from src.core.data_classes import Task


# ============================================================
# 1. step reward —— 任务完成 + 目标函数变化（按实例规模归一化）
# ============================================================
def compute_step_reward(obj_before: float, obj_after: float,
                        obj_scale: float) -> float:
    """
    每步基础奖励 = R_COMPLETE + 归一化的 delta目标函数。

    delta = -(obj_after - obj_before)，obj 下降 → delta 为正 → 奖励为正
    obj_scale = SCALE_STEP_BASE × n_tasks_init（reset 时锁定，episode 内不变）

    归一化的意义：不同规模实例的单步 reward 量级一致，critic 才学得稳。
    """
    delta = -(float(obj_after) - float(obj_before))
    scale = max(1.0, float(obj_scale))   # 防御除零
    return float(R_COMPLETE + delta / scale)


# ============================================================
# 2. 卡车等待 shaping —— 只保留"做了 load_truck"的情形，且净值可正
# ============================================================
def compute_truck_shaping(env, best_task: Optional[Task]) -> float:
    """
    Truck shaping（重构版）。

    旧版问题：
      - 情形B（卡车在等但没做它）返回 0 → "拖着不做"成了偷懒最优解
      - 情形A 纯惩罚 → 做卡车任务永远净亏
    重构：
      - 删除情形B（卡车等待已在 obj 的 delta 里体现，不重复算）
      - 情形A 改成 R_TRUCK_DONE - 等待惩罚，让完成卡车任务净值可正

    paradigm 1：跳过 canceled 任务。
    """
    if best_task is None:
        return 0.0
    if getattr(best_task, "canceled", False):
        return 0.0

    # 只在"这一步执行了 load_truck"时给信号
    if best_task.kind == "load_truck":
        wait_penalty = (
            0.004 * float(best_task.total_wait)
            + 0.003 * float(best_task.wait_aqc)
            + 0.001 * float(best_task.wait_cross)
        )
        # 净值 = 固定完成奖励 - 等待惩罚。
        # R_TRUCK_DONE=2.5，典型 wait_penalty（total_wait~300）约 1.2~1.5，
        # 所以正常情况下做卡车任务净值为正 (~+1.0)；
        # 只有等待极端大（total_wait>600）时才会净负，这是合理的——
        # 那种情况本就该早点做。
        return float(R_TRUCK_DONE - wait_penalty)

    # 其他任务类型：不在 shaping 里给信号（obj 的 delta 已覆盖）
    return 0.0


# ============================================================
# 3. terminal reward —— 只保留完成奖励 / 超时惩罚
# ============================================================
def compute_terminal_reward(obj_final: float, truck_total_wait: float,
                             terminated: bool, truncated: bool) -> float:
    """
    Episode 终局奖励（重构版）。

    旧版问题：terminated 分支里 -0.0005×truck_total_wait 是 truck 等待的
              第三次重复计分，导致 critic 收到自相矛盾信号。
    重构：删除 truck_total_wait 项。terminal 只负责两件事——
      - terminated（全部完成）：基于最终 obj 给奖励
      - truncated（超时未完成）：固定大惩罚

    注意：truck_total_wait 参数保留在签名里（兼容调用方），但不再使用。
    """
    if terminated:
        return float(-float(obj_final) / SCALE_TERM)
    if truncated:
        return float(TRUNCATED_PENALTY)
    return 0.0


# ============================================================
# 4. rule bonus —— A3 知识协同进化注入点（不变）
# ============================================================
def compute_rule_bonus(env, action: np.ndarray,
                       rule_bonus_fn,
                       pre_features: Optional[dict] = None) -> float:
    """A3 闭环注入。rule_bonus_fn=None 时返回 0。"""
    if rule_bonus_fn is None:
        return 0.0
    try:
        return float(rule_bonus_fn(env, action, pre_features=pre_features))
    except Exception:
        return 0.0


# ============================================================
# 5. 总奖励组合（不变）
# ============================================================
def compose_total_reward(step_r: float,
                         truck_shaping: float,
                         terminal_r: float,
                         rule_bonus: float) -> float:
    """组合并裁剪总奖励。"""
    step_part = float(np.clip(
        step_r + truck_shaping,
        REWARD_CLIP_LOW, REWARD_CLIP_HIGH,
    ))
    total = step_part + float(terminal_r) + float(rule_bonus)
    return float(np.clip(total, TOTAL_REWARD_CLIP_LOW, TOTAL_REWARD_CLIP_HIGH))
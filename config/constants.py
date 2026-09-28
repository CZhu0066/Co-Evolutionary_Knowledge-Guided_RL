# -*- coding: utf-8 -*-
"""
constants.py
============
全部物理 + 算法 + 扰动常数集中管理。

迁移自 KG-PPO_0_test.py 顶部 + 扩展扰动相关常数。
"""

# ============================================================
# AQC 物理参数
# ============================================================
GAMMA_AQC = 3           # 安全距离
V_AQC_B = 1.0           # 大车速度
V_AQC_S = 1.0           # 小车速度
D1 = 6.0                # 长
D2 = 2.5                # 宽
D3 = 2.53               # 高
H_AQC = 20.0            # AQC 高度
V_TROLLEY = 0.375       # trolley 速度
V_IT = 3.0              # 卡车速度

# ============================================================
# 任务时间窗口（火车上/下窗口）
# ============================================================
A1_DEFAULT = {1: 0.0,   2: 0.0,   3: 500.0,  4: 500.0}
A2_DEFAULT = {1: 10000.0, 2: 10000.0, 3: 10500.0, 4: 10500.0}

# # ============================================================
# # Reward shaping
# # ============================================================
# SCALE_STEP = 200.0
# SCALE_TERM = 100000.0
# R_COMPLETE = 0.05
# ALPHA_OBJ = 0.4
#
# # Reward 裁剪
# REWARD_CLIP_LOW = -100.0
# REWARD_CLIP_HIGH = 100.0
# TOTAL_REWARD_CLIP_LOW = -500.0
# TOTAL_REWARD_CLIP_HIGH = 200.0
# TRUNCATED_PENALTY = -500.0

# ============================================================
# Reward shaping
# ============================================================
SCALE_STEP_BASE = 15.0      # 归一化基准：obj_scale = SCALE_STEP_BASE × n_tasks_init
SCALE_TERM = 100000.0
R_COMPLETE = 0.3            # 0.05 → 0.3：完成任务基础奖励，不再被 delta 淹没
R_TRUCK_DONE = 2.5         # 新增：完成 load_truck 的额外正奖励，压过等待惩罚
ALPHA_OBJ = 0.4

# Reward 裁剪
REWARD_CLIP_LOW = -100.0
REWARD_CLIP_HIGH = 100.0
TOTAL_REWARD_CLIP_LOW = -500.0
TOTAL_REWARD_CLIP_HIGH = 200.0
TRUNCATED_PENALTY = -500.0


# ============================================================
# Truck 处理参数（迁移自原代码）
# ============================================================
# TRUCK_PRIORITY_BIAS = 300.0
TRUCK_AGE_WEIGHT = 0.8
TRUCK_WAIT_WEIGHT_TASK = 2.5
TRUCK_AQC_WAIT_WEIGHT_TASK = 2.5
TRUCK_CROSS_WAIT_WEIGHT_TASK = 0.8
TRUCK_PRESSURE_THRESHOLD = 600.0
# TRUCK_PRESSURE_BONUS = 500.0
TRUCK_PRIORITY_BIAS = 40.0     # 300 → 40：保留弱先验，不再碾压 θ
TRUCK_PRESSURE_BONUS = 80.0    # 500 → 80：超时压力不再阶跃式爆发
# Load_truck 候选规模
TRUCK_SERVICE_AQC_K = 4
CROSS_NEAR_AQC_K = 20
CAR_NEAR_CROSS_K = 10

# ============================================================
# Cross / Truck slot 位置参数
# ============================================================
LOAD_TRUCK_CROSS_ROW = -3.0
LOAD_TRUCK_CROSS_TIER = 0.0
LOAD_TRUCK_CROSS_BAYS = list(range(50))

UNLOAD_TRUCK_ROW = -3.0
UNLOAD_TRUCK_TIER = 0.0
UNLOAD_TRUCK_BAYS = list(range(50))

# ============================================================
# θ 维度（9D 解耦后）
# ============================================================
D_I = 3   # task selection: best_cost / init_bay / train_weight
D_G = 2   # destination spatial: move_time / bay_distance
D_P = 2   # cross temporal: h2 / cross_bay_distance
D_K = 2   # AQC: finish_time / imbalance
THETA_DIM = D_I + D_G + D_P + D_K   # = 9

# ============================================================
# ★ 新增：扰动参数（范式①核心）
# ============================================================
DISTURBANCE_TYPES = [
    "break",            # AQC 故障
    "train_delay",      # 列车延误
    "insert",           # 临时插单
    "cancel",           # 任务取消
    "urgent",           # 紧急任务（提高优先级）
]

# 强度分级：每种扰动在 episode 中触发的概率/数量
# 设计原则：
# 1. 任务级扰动优先：insert / cancel / urgent 概率最高
# 2. 火车晚到次之：train_delay 中等
# 3. AQC 故障最低：break 作为设备扰动，不宜过高，否则会掩盖任务扰动效果
DISTURBANCE_INTENSITY = {
    "clean": {
        "p_break_per_aqc": 0.0,
        "p_train_delay": 0.0,
        "p_insert": 0.0,
        "p_cancel": 0.0,
        "p_urgent": 0.0,
    },

    "low": {
        # 低扰动：少量任务变化，较少火车晚到，少量AQC故障
        "p_break_per_aqc": 0.05,
        "p_train_delay": 0.10,
        "p_insert": 0.20,
        "p_cancel": 0.15,
        "p_urgent": 0.15,
    },

    "med": {
        # 中等扰动：任务扰动明显增加，火车晚到次之，AQC故障保持较低
        "p_break_per_aqc": 0.15,
        "p_train_delay": 0.25,
        "p_insert": 0.50,
        "p_cancel": 0.35,
        "p_urgent": 0.35,
    },

    "high": {
        # 高扰动：任务扰动成为主要扰动来源，火车晚到较高，AQC故障仍低于任务扰动
        "p_break_per_aqc": 0.25,
        "p_train_delay": 0.40,
        "p_insert": 0.70,
        "p_cancel": 0.50,
        "p_urgent": 0.50,
    },

    "extreme": {
        # 极端扰动：任务扰动几乎必然发生，火车晚到很高，AQC故障也明显但仍低于任务扰动
        "p_break_per_aqc": 0.40,
        "p_train_delay": 0.70,
        "p_insert": 0.90,
        "p_cancel": 0.75,
        "p_urgent": 0.75,
    },
}


# 各扰动参数范围
BREAK_DURATION_RANGE = (180.0, 300.0)     # AQC 修复时长
TRAIN_DELAY_RANGE = (200.0, 500.0)        # 列车延误时长
INSERT_TASK_COUNT_RANGE = (1, 4)          # 插单批次大小
CANCEL_TASK_COUNT_RANGE = (1, 3)          # 取消批次大小
URGENT_TASK_COUNT_RANGE = (1, 3)          # 紧急批次大小

# ============================================================
# Obs 扰动状态维度
# ============================================================
# 每台 AQC 的扰动状态维度
OBS_DISTURB_DIM_PER_AQC = 2  # broken_flag + repair_remaining_norm
# 每列火车的扰动状态维度
OBS_DISTURB_DIM_PER_TRAIN = 1  # accumulated_delay_norm
# 全局扰动统计维度
OBS_DISTURB_DIM_GLOBAL = 6
# = n_breaks_w30 + n_breaks_w60 + n_pending_inserts + n_pending_urgents
#   + total_canceled + intensity_one_hot (实际4维但合成1个index)

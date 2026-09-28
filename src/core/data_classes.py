# -*- coding: utf-8 -*-
"""
data_classes.py
===============
全部领域对象的 dataclass 定义。

迁移自 KG-PPO_0_test.py 的对应类，并新增 DisturbanceEvent 用于范式①。

设计原则：
  - 所有字段都 type-hint，便于 IDE/类型检查
  - mutable default 用 None + __post_init__ 初始化
  - 不在这里写业务逻辑，业务逻辑在 simulators.py / env/
"""
from dataclasses import dataclass, field
from typing import Any, Optional, List


# ============================================================
# 任务（统一 Task，含 load / unload 两大类、4 个 kind）
# ============================================================
@dataclass
class Task:
    """
    统一的任务对象，覆盖 load_yard, load_truck, unload_yard, unload_truck 四种 kind。

    生命周期字段：
      - done=False, finish_time=0.0: 未完成
      - done=True, assigned_aqc_idx>=0, finish_time>0: 已完成
      - canceled=True: 被扰动取消（永远跳过）
      - urgency_boost>0: 被标记为紧急（评分时额外加分）
    """
    id: Any
    type: str              # "load" / "unload"
    subtype: str           # "yard" / "truck"
    kind: str              # "load_yard" / "load_truck" / "unload_yard" / "unload_truck"
    train_id: int
    init_row: float
    init_tier: float
    init_bay: float
    arrival_time: float = 0.0    # 仅 load_truck 的卡车到达堆场时间

    # ---- 调度结果 ----
    done: bool = False
    assigned_dest_idx: int = -1
    assigned_aqc_idx: int = -1
    assigned_cross_idx: int = -1
    start_time: float = 0.0
    finish_time: float = 0.0
    final_row: Optional[float] = None
    final_tier: Optional[float] = None
    final_bay: Optional[float] = None

    # ---- load_truck 专用 ----
    cross_row: Optional[float] = None
    cross_tier: Optional[float] = None
    cross_bay: Optional[float] = None
    h2: float = 0.0
    o2: float = 0.0
    wait_cross: float = 0.0
    wait_aqc: float = 0.0
    total_wait: float = 0.0

    # ---- ★ 新增：扰动相关状态 ----
    canceled: bool = False             # 被 cancel 扰动作废
    urgency_boost: float = 0.0         # urgent 扰动加分（>0 提高任务选择优先级）
    is_inserted: bool = False          # 是否为 insert 扰动加入的任务


# ============================================================
# 列车 / 堆场 / Cross / 卡车槽
# ============================================================
@dataclass
class TrainCar:
    """火车上的一个车厢位置（load_yard / load_truck 的目的地）"""
    id: Any
    row: float
    tier: float
    bay: float
    train_id: int
    occupied: bool = False


@dataclass
class YardSlot:
    """堆场槽位（unload_yard 的目的地）"""
    id: int
    row: float
    tier: float
    bay: float
    occupied: bool = False


@dataclass
class TruckCross:
    """卡车交接点（load_truck 的中转）"""
    id: int
    row: float
    tier: float
    bay: float
    available_time: float = 0.0


@dataclass
class TruckSlot:
    """卡车装卸槽位（unload_truck 的目的地）"""
    id: int
    row: float
    tier: float
    bay: float
    occupied: bool = False


# ============================================================
# AQC 状态
# ============================================================
@dataclass
class AQCState:
    """
    AQC 调度状态。

    扰动相关字段：
      - is_broken: 当前是否处于故障状态（available_time 已被推后但还没"修好"）
      - last_break_time: 最近一次故障开始时间（用于 obs 中统计窗口）
      - last_repair_time: 最近一次预计修复完成时间
    """
    id: int
    cur_row: float
    cur_bay: float
    available_time: float = 0.0
    tasks: Optional[list] = None

    # ---- ★ 新增：扰动状态 ----
    is_broken: bool = False
    last_break_time: float = -1.0
    last_repair_time: float = -1.0

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []


# ============================================================
# ★★★ 新增：扰动事件
# ============================================================
@dataclass
class DisturbanceEvent:
    """
    单个扰动事件。状态无关：事件参数在采样时确定，与env当前状态无关。

    事件类型与必填字段：

      type="break":
        aqc_id: int           哪台 AQC 故障
        duration: float       修复时长（秒）

      type="train_delay_notice":
        train_id: int         哪列火车
        plan_time: float      原计划到达时间（A1 值）
        actual_time: float    实际到达时间
        delay: float          延误时长

      type="train_arrive":
        train_id: int         配对的 notice
        actual_time: float    实际到达时间（= time）

      type="insert":
        insert_kind: str      新任务的 kind（如 "load_truck"）
        train_id: int         新任务所属列车
        truck_arrival_time: float    若是 load_truck，卡车到达时间
        truck_init_row/tier/bay: float

      type="cancel":
        target_kind: str      要取消的任务kind（"load_yard" 等）
        target_train_id: Optional[int]   可选：限定列车
        n_cancel: int         批次取消数量

      type="urgent":
        target_kind: str      要标记紧急的任务kind
        target_train_id: Optional[int]
        n_urgent: int         批次紧急数量
        boost_value: float    加分大小（默认 200）

    注：cancel 和 urgent 在采样时不指定 task_id，因为不依赖env具体状态。
        env 在应用时按 (target_kind, target_train_id) 选还没done的任务来取消/标记。
    """
    time: float
    type: str

    # ---- 各类型可选字段 ----
    aqc_id: Optional[int] = None
    duration: Optional[float] = None

    train_id: Optional[int] = None
    plan_time: Optional[float] = None
    actual_time: Optional[float] = None
    delay: Optional[float] = None

    insert_kind: Optional[str] = None
    truck_arrival_time: Optional[float] = None
    truck_init_row: Optional[float] = None
    truck_init_tier: Optional[float] = None
    truck_init_bay: Optional[float] = None

    target_kind: Optional[str] = None
    target_train_id: Optional[int] = None
    n_cancel: Optional[int] = None
    n_urgent: Optional[int] = None
    boost_value: float = 200.0

    # ---- 元信息（debug用） ----
    event_id: int = 0
    intensity_tag: str = ""    # 标记本事件来自哪个 intensity（"low" / "med" / "high"）


# ============================================================
# 扰动剧本：一组按时间排序的事件
# ============================================================
@dataclass
class DisturbanceScript:
    """
    完整的扰动剧本。env.reset() 时由 DisturbanceSampler 生成，
    env.step() 时由 DisturbanceApplier 按时间触发并应用。
    """
    events: List[DisturbanceEvent] = field(default_factory=list)
    intensity: str = "clean"
    seed: int = 0

    def __post_init__(self):
        # 按时间排序
        self.events.sort(key=lambda e: e.time)
        # 重新分配 event_id
        for i, ev in enumerate(self.events):
            ev.event_id = i

    @property
    def n_events(self) -> int:
        return len(self.events)

    def is_empty(self) -> bool:
        return len(self.events) == 0

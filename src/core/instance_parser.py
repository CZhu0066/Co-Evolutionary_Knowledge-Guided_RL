# -*- coding: utf-8 -*-
"""
instance_parser.py
==================
JSON instance 解析器，迁移自 KG-PPO_0_test.py 的 parse_instance() 及辅助函数。

支持三种 instance 格式：
  - two_load:    J1_yard, J3_yard, J1_truck, J3_truck, ...
  - two_unload:  J2_to_truck, J2_to_yard, J4_to_truck, J4_to_yard, ...
  - load_unload: 两者混合

输入：json 文件路径
输出：dict，包含 env 初始化所需的全部数据
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from config.constants import (
    A1_DEFAULT, A2_DEFAULT,
    LOAD_TRUCK_CROSS_ROW, LOAD_TRUCK_CROSS_TIER, LOAD_TRUCK_CROSS_BAYS,
    UNLOAD_TRUCK_ROW, UNLOAD_TRUCK_TIER, UNLOAD_TRUCK_BAYS,
)
from src.core.data_classes import (
    Task, TrainCar, YardSlot, TruckCross, TruckSlot,
)


# ============================================================
# 辅助函数（私有）
# ============================================================
def _train_id_from_key(key: str) -> Optional[int]:
    """从 JSON key 提取列车 ID。
    支持 "J1", "J1_yard", "J3_truck", "C_1", "C_3" 等。
    """
    if key.startswith("J"):
        m = re.match(r"J(\d+)", key)
    elif key.startswith("C_"):
        m = re.match(r"C_(\d+)", key)
    else:
        m = None
    return int(m.group(1)) if m else None


def _safe_get_columns(data) -> List[Tuple[float, float]]:
    """读取 COL（柱子列表），兼容两种 schema。"""
    if "COL" in data:
        col_raw = data["COL"]
        if col_raw and isinstance(col_raw[0], list):
            return [tuple(item) for item in col_raw]
        return [(float(item["row"]), float(item["bay"])) for item in col_raw]
    if "C" in data:
        col_raw = data["C"]
        if col_raw and isinstance(col_raw[0], list):
            return [tuple(item) for item in col_raw]
        return [(float(item["row"]), float(item["bay"])) for item in col_raw]
    return []


def _read_stack_data(data):
    """读取堆叠数据：columns + g_col + d_col"""
    columns = _safe_get_columns(data)
    g_col_raw = data.get("G_COL", data.get("G_c", {}))
    d_col_raw = data.get("D_COL", data.get("D_c", {}))

    g_col = {}
    for key, lst in g_col_raw.items():
        r, b = map(float, key.split(","))
        g_col[(r, b)] = [int(x) for x in lst]

    d_col = {}
    for key, val in d_col_raw.items():
        r, b = map(float, key.split(","))
        d_col[(r, b)] = int(val)

    return columns, g_col, d_col


# ============================================================
# 固定结构构造（cross / unload truck slots）
# ============================================================
def build_cross_points() -> List[TruckCross]:
    """构造 50 个固定的卡车交接点。"""
    return [
        TruckCross(id=i, row=LOAD_TRUCK_CROSS_ROW,
                   tier=LOAD_TRUCK_CROSS_TIER, bay=float(b))
        for i, b in enumerate(LOAD_TRUCK_CROSS_BAYS)
    ]


def build_unload_truck_slots() -> List[TruckSlot]:
    """构造 50 个固定的卸车 slot。"""
    return [
        TruckSlot(id=i, row=UNLOAD_TRUCK_ROW,
                  tier=UNLOAD_TRUCK_TIER, bay=float(b))
        for i, b in enumerate(UNLOAD_TRUCK_BAYS)
    ]


# ============================================================
# 主入口
# ============================================================
def parse_instance(json_path: str) -> Dict[str, Any]:
    """
    从 JSON 文件解析出 env 初始化所需的全部数据。

    返回字典字段：
      path                : json 文件路径
      raw                 : 原始 json dict
      tasks               : List[Task]
      cars_by_train       : Dict[int, List[TrainCar]]
      slots               : List[YardSlot]
      initial_height      : Dict[(row, bay), int]
      columns             : List[(row, bay)]
      aqc_init            : List[(row, bay)]
      trains_involved     : List[int]
      phi                 : Dict[int, float]  列车权重
      A1, A2              : Dict[int, float]  列车时间窗口
      g_col, d_col        : 堆叠位置/高度
      crosses             : List[TruckCross]
      unload_truck_slots  : List[TruckSlot]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---- 解析任务 ----
    tasks: List[Task] = []
    trains_involved = set()
    explicit_job_keys = set()

    def add_task(arr, train_id, kind):
        if not isinstance(arr, list):
            return
        for x in arr:
            subtype = "truck" if "truck" in kind else "yard"
            ttype = "load" if kind.startswith("load") else "unload"
            tasks.append(Task(
                id=x.get("id"),
                type=ttype,
                subtype=subtype,
                kind=kind,
                train_id=int(train_id),
                init_row=float(x.get("row", 0.0)),
                init_tier=float(x.get("tier", 0.0)),
                init_bay=float(x.get("bay", 0.0)),
                arrival_time=float(x.get("arrival_time", 0.0)),
            ))
            trains_involved.add(int(train_id))

    # 显式 key 解析
    for key in list(data.keys()):
        if key == "J1_yard":
            explicit_job_keys.add(key); add_task(data[key], 1, "load_yard")
        elif key == "J1_truck":
            explicit_job_keys.add(key); add_task(data[key], 1, "load_truck")
        elif key == "J4_to_yard":
            explicit_job_keys.add(key); add_task(data[key], 4, "unload_yard")
        elif key == "J4_to_truck":
            explicit_job_keys.add(key); add_task(data[key], 4, "unload_truck")

    # 其他 J*_ 类 key
    for key in list(data.keys()):
        if not isinstance(key, str) or not key.startswith("J") or key in explicit_job_keys:
            continue
        tid = _train_id_from_key(key)
        if tid is None:
            continue
        if key.endswith("_truck") and "_to_truck" not in key:
            explicit_job_keys.add(key); add_task(data[key], tid, "load_truck")
        elif key.endswith("_to_yard"):
            explicit_job_keys.add(key); add_task(data[key], tid, "unload_yard")
        elif key.endswith("_to_truck"):
            explicit_job_keys.add(key); add_task(data[key], tid, "unload_truck")

    # 兜底：J1, J3 → load_yard；J2, J4 → unload_yard
    for key, arr in data.items():
        if not isinstance(key, str) or not key.startswith("J"):
            continue
        if key in explicit_job_keys or key.endswith("_all"):
            continue
        tid = _train_id_from_key(key)
        if tid is None or not isinstance(arr, list):
            continue
        if tid in (1, 3):
            add_task(arr, tid, "load_yard")
        else:
            add_task(arr, tid, "unload_yard")

    trains_involved = sorted(list(trains_involved)) or [1]

    # ---- 解析火车车厢 ----
    cars_by_train: Dict[int, List[TrainCar]] = {}
    for key, arr in data.items():
        if not isinstance(key, str) or not key.startswith("C_"):
            continue
        tid = _train_id_from_key(key)
        if tid is None or not isinstance(arr, list):
            continue
        cars_by_train[tid] = []
        for c in arr:
            bay = c.get("bay")
            cid = f"{tid}_{bay}"
            cars_by_train[tid].append(TrainCar(
                id=cid,
                row=float(c.get("row", 0.0)),
                tier=float(c.get("tier", 0.0)),
                bay=float(c.get("bay", 0.0)),
                train_id=tid,
                occupied=False,
            ))

    # ---- 解析堆场 slot ----
    slots: List[YardSlot] = []
    if isinstance(data.get("G", None), list):
        for i, g in enumerate(data["G"]):
            slots.append(YardSlot(
                id=i,
                row=float(g.get("row", 0.0)),
                tier=float(g.get("tier", 0.0)),
                bay=float(g.get("bay", 0.0)),
                occupied=False,
            ))

    # ---- 解析堆叠高度 ----
    initial_height: Dict[Tuple[float, float], int] = {}
    stack_info = data.get("stack_info", [])
    if isinstance(stack_info, list):
        for item in stack_info:
            initial_height[(
                float(item.get("row", 0.0)),
                float(item.get("bay", 0.0)),
            )] = int(item.get("height", -1))

    columns, g_col, d_col = _read_stack_data(data)
    for c, h in d_col.items():
        initial_height[c] = h

    col_set = set(columns)
    for slot in slots:
        col_set.add((slot.row, slot.bay))
    for k in initial_height.keys():
        col_set.add(k)
    columns = sorted(list(col_set))

    # ---- 解析 AQC 初始位置 ----
    aqc_raw = data.get("K_aqc", [])
    aqc_init: List[Tuple[float, float]] = []
    if isinstance(aqc_raw, list) and len(aqc_raw) > 0:
        for a in aqc_raw:
            if isinstance(a, dict):
                aqc_init.append((
                    float(a.get("initial_row", 0.0)),
                    float(a.get("initial_bay", 0.0)),
                ))
            else:
                aqc_init.append((0.0, 0.0))
    else:
        # 默认 2 台 AQC
        aqc_init = [(0.0, 9.0), (0.0, 19.0)]

    # ---- 列车权重 ----
    phi = {tid: 1.0 / len(trains_involved) for tid in trains_involved}
    A1 = dict(A1_DEFAULT)
    A2 = dict(A2_DEFAULT)

    return {
        "path": json_path,
        "raw": data,
        "tasks": tasks,
        "cars_by_train": cars_by_train,
        "slots": slots,
        "initial_height": initial_height,
        "columns": columns,
        "aqc_init": aqc_init,
        "trains_involved": trains_involved,
        "phi": phi,
        "A1": A1,
        "A2": A2,
        "g_col": g_col,
        "d_col": d_col,
        "crosses": build_cross_points(),
        "unload_truck_slots": build_unload_truck_slots(),
    }

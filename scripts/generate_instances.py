# -*- coding: utf-8 -*-
"""
generate_instances.py  （v15.4 三场景版）
================================================================
严格对齐用户三套数据生成器：
  - data_generator_load.py        → 2load (U_1=[1,3])
  - data_generator_unload.py      → 2unload (U_2=[2,4])
  - data_generator_load_unload.py → 1load_1unload (U_1=[1], U_2=[4])

三种场景每种 8 train + 8 test = 16 个文件
共 48 个文件（24 train + 24 test），跟你原代码一一对应。

用法：
  # 单场景单配置
  python scripts/generate_instances.py \\
      --scenario 1load_1unload --dataset_type test --config_idx 2 \\
      --n_instances 5 --out data/instances/exp2/

  # 一键生成所有 48 个文件（每种场景 8 组配置各 1 个实例）
  python scripts/generate_all_scenarios.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# 物理参数（与原代码完全一致）
# ============================================================
# 2load:  ROWS = list(range(6, 13)) + [0]
# 2unload/1load_1unload: ROWS = [0, 6, 7, 8, 9, 10, 11, 12]
YARD_ROWS = [0, 6, 7, 8, 9, 10, 11, 12]
YARD_BAYS = list(range(0, 50))
YARD_TIERS = list(range(0, 6))

TRUCK_INIT_ROW = -4
TRUCK_INIT_TIER = 0
TRUCK_INIT_BAY = 0

TOTAL_COLUMNS = len(YARD_ROWS) * len(YARD_BAYS)   # 400
NUM_TRAIN_DEFAULT = 50


# ============================================================
# 预设配置组（来自你三套原代码）
# ============================================================

# 2load 装载场景 (n1, n2, n3, n4, num_aqc, num_train1, num_train3)
#   n1=J1_yard, n2=J3_yard, n3=J1_truck, n4=J3_truck
CONFIGS_2LOAD = {
    "train": [
        (10, 10, 2, 2, 4, 14, 14),
        (15, 15, 3, 3, 4, 20, 20),
        (20, 20, 5, 5, 4, 28, 28),
        (25, 25, 8, 8, 4, 37, 37),
        (15, 25, 4, 6, 4, 21, 35),
        (25, 15, 6, 4, 4, 35, 21),
        (20, 30, 5, 8, 4, 28, 42),
        (30, 20, 8, 5, 4, 42, 28),
    ],
    "test": [
        (12, 12, 3, 3, 4, 17, 17),
        (18, 18, 5, 5, 4, 26, 26),
        (25, 25, 8, 8, 4, 37, 37),
        (30, 30, 10, 10, 4, 44, 44),
        (20, 30, 5, 8, 4, 28, 42),
        (30, 20, 8, 5, 4, 42, 28),
        (32, 28, 10, 12, 4, 47, 44),
        (28, 32, 12, 10, 4, 44, 47),
    ],
}

# 2unload 卸载场景 (n1, n2, n3, n4, num_aqc, available_positions_ratio)
#   n1=J2_to_truck, n2=J2_to_yard, n3=J4_to_truck, n4=J4_to_yard
CONFIGS_2UNLOAD = {
    "train": [
        (1, 10, 1, 10, 4, 0.2),
        (2, 20, 2, 20, 4, 0.3),
        (3, 30, 3, 30, 4, 0.4),
        (4, 40, 4, 40, 4, 0.5),
        (1, 15, 2, 25, 4, 0.3),
        (2, 25, 1, 15, 4, 0.3),
        (3, 30, 4, 40, 4, 0.4),
        (4, 40, 3, 30, 4, 0.4),
    ],
    "test": [
        (1, 15, 1, 15, 4, 0.2),
        (2, 25, 2, 25, 4, 0.3),
        (3, 35, 3, 35, 4, 0.4),
        (4, 45, 4, 45, 4, 0.5),
        (2, 20, 3, 30, 4, 0.3),
        (3, 30, 2, 20, 4, 0.3),
        (4, 40, 5, 45, 4, 0.4),
        (5, 45, 4, 40, 4, 0.4),
    ],
}

# 1load_1unload 一装一卸场景 (n1, n2, n3, n4, num_aqc, num_train1, available_positions_ratio)
#   n1=J1_yard, n2=J1_truck, n3=J4_to_truck, n4=J4_to_yard
CONFIGS_1LOAD_1UNLOAD = {
    "train": [
        (10, 2, 1, 10, 4, 14, 0.2),
        (15, 3, 1, 15, 4, 20, 0.3),
        (20, 5, 2, 20, 4, 28, 0.4),
        (25, 8, 2, 25, 4, 37, 0.5),
        (15, 4, 2, 20, 4, 21, 0.3),
        (25, 6, 1, 15, 4, 35, 0.3),
        (20, 5, 3, 30, 4, 28, 0.4),
        (30, 8, 2, 20, 4, 42, 0.4),
    ],
    "test": [
        (12, 3, 1, 12, 4, 17, 0.2),
        (18, 5, 2, 18, 4, 26, 0.3),
        (25, 8, 2, 25, 4, 37, 0.4),
        (30, 10, 3, 30, 4, 44, 0.5),
        (20, 5, 3, 30, 4, 28, 0.3),
        (30, 8, 2, 20, 4, 42, 0.3),
        (32, 10, 3, 28, 4, 47, 0.4),
        (28, 12, 2, 25, 4, 44, 0.4),
    ],
}


# ============================================================
# 公共辅助
# ============================================================
def _gen_aqc(num_aqc: int) -> List[Dict[str, Any]]:
    """生成 K_aqc 列表（与原代码完全一致）"""
    return [
        {"id": i, "initial_row": 0, "initial_bay": int(49 / (num_aqc + 1) * i)}
        for i in range(1, num_aqc + 1)
    ]


def _gen_train_cars(num_cars: int, row: int, rng: random.Random) -> List[Dict[str, Any]]:
    """生成车厢位置"""
    bays = rng.sample(list(range(50)), num_cars)
    return [{"row": row, "tier": 0, "bay": b} for b in bays]


def _gen_yard_positions(occupied_columns: List, ratio: float, rng: random.Random):
    """生成堆场可用位置 G/stack_info/G_c/D_c/COL（与 unload/load_unload 一致）"""
    all_columns = [(r, b) for r in YARD_ROWS for b in YARD_BAYS]
    remaining = [c for c in all_columns if c not in occupied_columns]
    n_g = max(1, min(int(len(remaining) * ratio), len(remaining)))
    selected = rng.sample(remaining, n_g)

    G, stack_info = [], []
    for (row, bay) in selected:
        start_tier = rng.randint(0, 5)
        avail_tier = rng.randint(start_tier, 5)
        G.append({"row": row, "tier": avail_tier, "bay": bay, "index": len(G)})
        stack_info.append({"row": row, "bay": bay, "height": avail_tier})

    G_c, D_c = {}, {}
    for idx, pos in enumerate(G):
        key = f"{pos['row']},{pos['bay']}"
        G_c.setdefault(key, []).append(idx)
        D_c[key] = pos["tier"]
    COL_set = set((p["row"], p["bay"]) for p in G)
    COL = sorted([[r, b] for r, b in COL_set])
    return G, stack_info, G_c, D_c, COL


# ============================================================
# 场景 1: 2load （对齐 data_generator_load.py）
# ============================================================
def generate_2load_instance(
    n1: int, n2: int, n3: int, n4: int,
    num_aqc: int, num_train1: int, num_train3: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    2load: U_1 = [1, 3] 两个装载火车
      n1: J1 堆场装载箱 (火车1)
      n2: J3 堆场装载箱 (火车3)
      n3: J1 卡车装载箱 (火车1)
      n4: J3 卡车装载箱 (火车3)
    """
    rng = random.Random(seed)
    total_yard = n1 + n2
    total_truck = n3 + n4
    total_containers = total_yard + total_truck

    # 合法性
    max_yard = TOTAL_COLUMNS
    assert total_yard <= max_yard
    assert num_train1 <= 50 and num_train3 <= 50
    assert num_train1 >= math.ceil((n1 + n3) * 1.1)
    assert num_train3 >= math.ceil((n2 + n4) * 1.1)

    latest_arrival = max(1000, 50 * total_containers)

    # 堆场任务（保证每列只一个箱子）
    all_columns = [(r, b) for r in YARD_ROWS for b in YARD_BAYS]
    selected = rng.sample(all_columns, total_yard)
    yard_containers = []
    for (r, b) in selected:
        t = rng.choice(YARD_TIERS)
        yard_containers.append({"row": r, "tier": t, "bay": b, "type": "load"})
    J1_yard = yard_containers[:n1]
    J3_yard = yard_containers[n1:n1 + n2]
    for i, c in enumerate(J1_yard): c["id"] = 1001 + i
    for i, c in enumerate(J3_yard): c["id"] = 3001 + i

    # 卡车任务
    truck_containers = []
    for _ in range(total_truck):
        truck_containers.append({
            "row": TRUCK_INIT_ROW, "tier": TRUCK_INIT_TIER, "bay": TRUCK_INIT_BAY,
            "type": "load",
            "arrival_time": rng.randint(1000, latest_arrival),
        })
    J1_truck = truck_containers[:n3]
    J3_truck = truck_containers[n3:n3 + n4]
    for i, c in enumerate(J1_truck): c["id"] = 5001 + i
    for i, c in enumerate(J3_truck): c["id"] = 7001 + i

    K_aqc = _gen_aqc(num_aqc)
    U_1 = [1, 3]
    C_1 = _gen_train_cars(num_train1, row=4, rng=rng)
    C_3 = _gen_train_cars(num_train3, row=2, rng=rng)

    return {
        "scenario": "2load",
        "J1_yard": J1_yard,
        "J3_yard": J3_yard,
        "J1_truck": J1_truck,
        "J3_truck": J3_truck,
        "K_aqc": K_aqc,
        "U_1": U_1,
        "C_1": C_1,
        "C_3": C_3,
        "_meta": {
            "scenario": "2load",
            "n1_J1_yard": n1, "n2_J3_yard": n2, "n3_J1_truck": n3, "n4_J3_truck": n4,
            "num_aqc": num_aqc, "num_train1": num_train1, "num_train3": num_train3,
            "n_tasks": n1 + n2 + n3 + n4, "seed": seed,
        },
    }


# ============================================================
# 场景 2: 2unload （对齐 data_generator_unload.py）
# ============================================================
def generate_2unload_instance(
    n1: int, n2: int, n3: int, n4: int,
    num_aqc: int, available_positions_ratio: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    2unload: U_2 = [2, 4] 两个卸载火车
      n1: J2 卸到卡车 (火车2)
      n2: J2 卸到堆场 (火车2)
      n3: J4 卸到卡车 (火车4)
      n4: J4 卸到堆场 (火车4)
    """
    rng = random.Random(seed)
    total_j2 = n1 + n2
    total_j4 = n3 + n4
    assert total_j2 <= 50 and total_j4 <= 50
    assert 0 < available_positions_ratio <= 1

    # J2
    J2_bays = rng.sample(YARD_BAYS, total_j2)
    J2_to_truck = [
        {"id": 2001 + i, "row": 4, "tier": 0, "bay": J2_bays[i], "type": "unload_to_truck"}
        for i in range(n1)
    ]
    J2_to_yard = [
        {"id": 2501 + i, "row": 4, "tier": 0, "bay": J2_bays[n1 + i], "type": "unload_to_yard"}
        for i in range(n2)
    ]
    J2_all = J2_to_truck + J2_to_yard

    # J4
    J4_bays = rng.sample(YARD_BAYS, total_j4)
    J4_to_truck = [
        {"id": 4001 + i, "row": 2, "tier": 0, "bay": J4_bays[i], "type": "unload_to_truck"}
        for i in range(n3)
    ]
    J4_to_yard = [
        {"id": 4501 + i, "row": 2, "tier": 0, "bay": J4_bays[n3 + i], "type": "unload_to_yard"}
        for i in range(n4)
    ]
    J4_all = J4_to_truck + J4_to_yard

    K_aqc = _gen_aqc(num_aqc)
    U_2 = [2, 4]
    C_2 = _gen_train_cars(50, row=4, rng=rng)
    C_4 = _gen_train_cars(50, row=2, rng=rng)

    # 堆场可用位置（2unload 没有装载火车占位，全部都可用）
    G, stack_info, G_c, D_c, COL = _gen_yard_positions(
        occupied_columns=[], ratio=available_positions_ratio, rng=rng,
    )
    assert len(G) >= n2 + n4, f"可用堆场位置不足: G={len(G)} < n2+n4={n2+n4}"

    return {
        "scenario": "2unload",
        "J2_to_truck": J2_to_truck,
        "J2_to_yard": J2_to_yard,
        "J2_all": J2_all,
        "J4_to_truck": J4_to_truck,
        "J4_to_yard": J4_to_yard,
        "J4_all": J4_all,
        "K_aqc": K_aqc,
        "U_2": U_2,
        "C_2": C_2,
        "C_4": C_4,
        "G": G,
        "stack_info": stack_info,
        "G_c": G_c,
        "D_c": D_c,
        "COL": COL,
        "available_positions_ratio": available_positions_ratio,
        "total_columns": TOTAL_COLUMNS,
        "_meta": {
            "scenario": "2unload",
            "n1_J2_to_truck": n1, "n2_J2_to_yard": n2,
            "n3_J4_to_truck": n3, "n4_J4_to_yard": n4,
            "num_aqc": num_aqc, "available_positions_ratio": available_positions_ratio,
            "n_tasks": n1 + n2 + n3 + n4, "seed": seed,
        },
    }


# ============================================================
# 场景 3: 1load_1unload （对齐 data_generator_load_unload.py）
# ============================================================
def generate_1load_1unload_instance(
    n1: int, n2: int, n3: int, n4: int,
    num_aqc: int, num_train1: int,
    available_positions_ratio: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    1load_1unload: U_1=[1], U_2=[4] 一装一卸
      n1: J1 堆场装载箱
      n2: J1 卡车装载箱
      n3: J4 卸到卡车
      n4: J4 卸到堆场
    """
    rng = random.Random(seed)
    total_containers = n1 + n2 + n3 + n4

    assert 0 < available_positions_ratio <= 1
    assert num_train1 <= 50
    assert num_train1 >= math.ceil((n1 + n2) * 1.1)
    assert n3 + n4 <= 50
    assert n1 <= TOTAL_COLUMNS

    latest_arrival = max(1000, 50 * total_containers)

    # J1_yard
    all_columns = [(r, b) for r in YARD_ROWS for b in YARD_BAYS]
    selected_J1 = rng.sample(all_columns, n1)
    J1_yard = []
    for i, (r, b) in enumerate(selected_J1):
        t = rng.choice(YARD_TIERS)
        J1_yard.append({
            "id": 1001 + i, "row": r, "tier": t, "bay": b, "type": "load",
        })

    # J1_truck
    J1_truck = []
    for i in range(n2):
        J1_truck.append({
            "id": 5001 + i,
            "row": TRUCK_INIT_ROW, "tier": TRUCK_INIT_TIER, "bay": TRUCK_INIT_BAY,
            "arrival_time": rng.randint(1000, latest_arrival), "type": "load",
        })

    # J4
    total_j4 = n3 + n4
    j4_bays = rng.sample(YARD_BAYS, total_j4)
    J4_to_truck = [
        {"id": 4001 + i, "row": 2, "tier": 0, "bay": j4_bays[i], "type": "unload_to_truck"}
        for i in range(n3)
    ]
    J4_to_yard = [
        {"id": 4501 + i, "row": 2, "tier": 0, "bay": j4_bays[n3 + i], "type": "unload_to_yard"}
        for i in range(n4)
    ]

    K_aqc = _gen_aqc(num_aqc)
    U_1 = [1]; U_2 = [4]
    C_1 = _gen_train_cars(num_train1, row=4, rng=rng)
    C_4 = _gen_train_cars(50, row=2, rng=rng)

    # G 不能用 J1 占用的柱子
    G, stack_info, G_c, D_c, COL = _gen_yard_positions(
        occupied_columns=selected_J1, ratio=available_positions_ratio, rng=rng,
    )
    assert len(G) >= n4, f"可用堆场位置不足: G={len(G)} < n4={n4}"

    return {
        "scenario": "1load_1unload",
        "J1_yard": J1_yard,
        "J1_truck": J1_truck,
        "J1_all": J1_yard + J1_truck,
        "J4_to_truck": J4_to_truck,
        "J4_to_yard": J4_to_yard,
        "J4_all": J4_to_truck + J4_to_yard,
        "K_aqc": K_aqc,
        "U_1": U_1,
        "U_2": U_2,
        "C_1": C_1,
        "C_4": C_4,
        "G": G,
        "stack_info": stack_info,
        "G_c": G_c,
        "D_c": D_c,
        "COL": COL,
        "available_positions_ratio": available_positions_ratio,
        "total_columns": TOTAL_COLUMNS,
        "_meta": {
            "scenario": "1load_1unload",
            "n1_J1_yard": n1, "n2_J1_truck": n2,
            "n3_J4_to_truck": n3, "n4_J4_to_yard": n4,
            "num_aqc": num_aqc, "num_train1": num_train1,
            "available_positions_ratio": available_positions_ratio,
            "n_tasks": n1 + n2 + n3 + n4, "seed": seed,
        },
    }


# ============================================================
# 详细打印（对齐原 generator 的打印格式）
# ============================================================
def print_instance(data: Dict[str, Any], filename: str = ""):
    """打印实例详细信息"""
    scenario = data.get("scenario", "?")
    meta = data.get("_meta", {})

    print(f"\n{'#'*80}")
    print(f"# 实例: {filename}")
    print(f"# 场景: {scenario}")
    print(f"# 任务总数: {meta.get('n_tasks', '?')}")
    print(f"{'#'*80}")

    # 按场景打印各类任务
    sections = []
    if scenario == "2load":
        sections = [("J1_yard", "load"), ("J3_yard", "load"),
                    ("J1_truck", "load"), ("J3_truck", "load")]
    elif scenario == "2unload":
        sections = [("J2_to_truck", None), ("J2_to_yard", None),
                    ("J4_to_truck", None), ("J4_to_yard", None)]
    elif scenario == "1load_1unload":
        sections = [("J1_yard", "load"), ("J1_truck", "load"),
                    ("J4_to_truck", None), ("J4_to_yard", None)]

    for key, _ in sections:
        items = data.get(key, [])
        if not items:
            continue
        print(f"\n################ {key} ({len(items)} containers) ##########################")
        first = items[0]
        has_arrival = "arrival_time" in first
        if has_arrival:
            print("id     row    tier    bay     arrival_time    type")
        else:
            print("id     row    tier    bay     type")
        for c in items[:5]:
            if has_arrival:
                print(f"{c['id']:4d}    {c['row']:4d}    {c['tier']:4d}    {c['bay']:4d}    "
                      f"{c['arrival_time']:8d}    {c.get('type','?')}")
            else:
                print(f"{c['id']:4d}    {c['row']:4d}    {c['tier']:4d}    {c['bay']:4d}    "
                      f"{c.get('type','?')}")
        if len(items) > 5:
            print(f"... 共 {len(items)} 个")

    # K_aqc
    K_aqc = data.get("K_aqc", [])
    if K_aqc:
        print(f"\n################ K_aqc ({len(K_aqc)} AQCs) ##########################")
        print("id    row    bay")
        for a in K_aqc:
            print(f"{a['id']:4d}    {a['initial_row']:4d}    {a['initial_bay']:4d}")

    # U_1, U_2
    if "U_1" in data:
        print(f"\n################ U_1 (装载火车) ##########################")
        print(f"  {data['U_1']}")
    if "U_2" in data:
        print(f"\n################ U_2 (卸载火车) ##########################")
        print(f"  {data['U_2']}")

    # 车厢
    for key in ("C_1", "C_2", "C_3", "C_4"):
        cars = data.get(key, [])
        if cars:
            print(f"\n################ {key} ({len(cars)} 节车厢) ##########################")
            print("row    tier    bay")
            for c in cars[:3]:
                print(f"{c['row']:4d}    {c['tier']:4d}    {c['bay']:4d}")
            if len(cars) > 3:
                print(f"... 共 {len(cars)} 个车厢位置")

    # G (堆场可用位置)
    G = data.get("G", [])
    if G:
        ratio = data.get("available_positions_ratio", 0)
        total = data.get("total_columns", 400)
        print(f"\n################ G (堆场可用位置 {len(G)} 个, 占总柱{ratio*100:.0f}%) ##########################")
        print("index   row   tier   bay")
        for i, g in enumerate(G[:3]):
            print(f"{i:4d}   {g['row']:4d}   {g['tier']:4d}   {g['bay']:4d}")
        if len(G) > 3:
            print(f"... 共 {len(G)} 个可用位置")


def save_instance(data: Dict[str, Any], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# 统一入口
# ============================================================
def generate_one(scenario: str, cfg: tuple, seed: int) -> Dict[str, Any]:
    """根据场景调用对应生成函数"""
    if scenario == "2load":
        n1, n2, n3, n4, num_aqc, num_train1, num_train3 = cfg
        return generate_2load_instance(n1, n2, n3, n4, num_aqc, num_train1, num_train3, seed)
    if scenario == "2unload":
        n1, n2, n3, n4, num_aqc, ratio = cfg
        return generate_2unload_instance(n1, n2, n3, n4, num_aqc, ratio, seed)
    if scenario == "1load_1unload":
        n1, n2, n3, n4, num_aqc, num_train1, ratio = cfg
        return generate_1load_1unload_instance(n1, n2, n3, n4, num_aqc, num_train1, ratio, seed)
    raise ValueError(f"未知场景: {scenario}. 合法: 2load | 2unload | 1load_1unload")


def get_filename(scenario: str, cfg: tuple, dataset_type: str, seed: int) -> str:
    """对齐原代码的文件命名"""
    if scenario == "2load":
        n1, n2, n3, n4, num_aqc, num_train1, num_train3 = cfg
        return (f"{dataset_type}_2load_J1yard{n1}_J3yard{n2}_J1truck{n3}_J3truck{n4}_"
                f"AQC{num_aqc}_Train1{num_train1}_Train3{num_train3}_seed{seed}.json")
    if scenario == "2unload":
        n1, n2, n3, n4, num_aqc, ratio = cfg
        return (f"{dataset_type}_2unload_J2truck{n1}_J2yard{n2}_J4truck{n3}_J4yard{n4}_"
                f"AQC{num_aqc}_Gratio{ratio}_seed{seed}.json")
    if scenario == "1load_1unload":
        n1, n2, n3, n4, num_aqc, num_train1, ratio = cfg
        return (f"{dataset_type}_1load_1unload_J1yard{n1}_J1truck{n2}_J4truck{n3}_J4yard{n4}_"
                f"AQC{num_aqc}_Train1{num_train1}_Gratio{ratio}_seed{seed}.json")
    raise ValueError(f"未知场景: {scenario}")


def get_configs(scenario: str, dataset_type: str) -> List[tuple]:
    if scenario == "2load":
        return CONFIGS_2LOAD[dataset_type]
    if scenario == "2unload":
        return CONFIGS_2UNLOAD[dataset_type]
    if scenario == "1load_1unload":
        return CONFIGS_1LOAD_1UNLOAD[dataset_type]
    raise ValueError(f"未知场景: {scenario}")


# ============================================================
# 命令行
# ============================================================
def _parse_args():
    p = argparse.ArgumentParser(
        description="生成港口调度实例（支持 3 种场景）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scenario", type=str, default="1load_1unload",
                   choices=["2load", "2unload", "1load_1unload"],
                   help="场景类型")
    p.add_argument("--dataset_type", type=str, default="test",
                   choices=["train", "test"])
    p.add_argument("--config_idx", type=int, default=2,
                   help="预设组里的索引 [0-7]")
    p.add_argument("--n_instances", type=int, default=5,
                   help="每个配置生成多少个实例（seed 递增）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/instances/")
    p.add_argument("--verbose", action="store_true", default=True,
                   help="打印每个实例的详细信息")
    p.add_argument("--quiet", action="store_true",
                   help="只打印汇总")
    return p.parse_args()


def main():
    args = _parse_args()
    verbose = args.verbose and not args.quiet

    configs = get_configs(args.scenario, args.dataset_type)
    cfg = configs[args.config_idx % len(configs)]

    print(f"\n{'='*80}")
    print(f"场景: {args.scenario}")
    print(f"数据集类型: {args.dataset_type}")
    print(f"配置 idx={args.config_idx}: {cfg}")
    print(f"生成 {args.n_instances} 个实例 → {args.out}")
    print(f"{'='*80}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(args.n_instances):
        seed = args.seed + i
        inst = generate_one(args.scenario, cfg, seed)
        fname = get_filename(args.scenario, cfg, args.dataset_type, seed)
        path = out_dir / fname
        save_instance(inst, path)
        paths.append(path)
        if verbose:
            print_instance(inst, fname)

    print(f"\n{'='*80}")
    print(f"✅ 共生成 {len(paths)} 个 {args.scenario} 实例到 {out_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

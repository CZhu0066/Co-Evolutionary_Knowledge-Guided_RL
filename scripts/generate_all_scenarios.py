# -*- coding: utf-8 -*-
"""
generate_all_scenarios.py
================================================================
一键生成你原代码的所有 48 个数据文件：
  - 3 个场景 (2load / 2unload / 1load_1unload)
  - 每种场景 8 train + 8 test = 16 个文件
  - 总计 48 个文件 (24 train + 24 test)

并对每个文件打印详细信息，让你完整看到数据全貌。

用法：
  # 每个配置 1 个实例 → 48 个文件（对齐你原代码数量）
  python scripts/generate_all_scenarios.py

  # 每个配置 3 个实例 → 144 个文件（更大数据集，扩展训练）
  python scripts/generate_all_scenarios.py --n_per_config 3

  # 只生成测试集（24 个）
  python scripts/generate_all_scenarios.py --only_test

  # 只生成训练集（24 个）
  python scripts/generate_all_scenarios.py --only_train
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_instances import (
    generate_one, get_filename, get_configs, save_instance, print_instance,
)

SCENARIOS = ["2load", "2unload", "1load_1unload"]


def gen_for_scenario(scenario: str, dataset_type: str, n_per_config: int,
                     out_root: Path, seed_base: int, verbose: bool):
    """为某场景某 dataset_type 生成所有配置 × n_per_config 个实例"""
    configs = get_configs(scenario, dataset_type)
    sub_dir = out_root / dataset_type / scenario
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f"# 场景: {scenario}    数据集类型: {dataset_type}")
    print(f"# 配置组数: {len(configs)}    每组实例数: {n_per_config}")
    print(f"# 输出目录: {sub_dir}")
    print(f"{'#'*80}\n")

    paths = []
    for cfg_idx, cfg in enumerate(configs):
        print(f"\n>>> 配置 [{cfg_idx + 1}/{len(configs)}]: {cfg}")
        for i in range(n_per_config):
            seed = seed_base + cfg_idx * 100 + i
            inst = generate_one(scenario, cfg, seed)
            fname = get_filename(scenario, cfg, dataset_type, seed)
            path = sub_dir / fname
            save_instance(inst, path)
            paths.append(path)
            if verbose:
                print_instance(inst, fname)
            else:
                print(f"  ✅ {fname}")
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_config", type=int, default=1,
                   help="每个预设配置生成多少个实例（默认 1 = 对齐你原代码 48 个）")
    p.add_argument("--out_root", type=str, default="data/instances_all/",
                   help="输出根目录，会自动按场景分子目录")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only_train", action="store_true")
    p.add_argument("--only_test", action="store_true")
    p.add_argument("--quiet", action="store_true",
                   help="不打印每个实例详情（只显示文件名）")
    args = p.parse_args()

    verbose = not args.quiet
    out_root = Path(args.out_root)

    all_paths = []

    for scenario in SCENARIOS:
        if not args.only_test:
            paths = gen_for_scenario(
                scenario, "train", args.n_per_config,
                out_root, args.seed, verbose,
            )
            all_paths.extend(paths)
        if not args.only_train:
            paths = gen_for_scenario(
                scenario, "test", args.n_per_config,
                out_root, args.seed + 10000, verbose,
            )
            all_paths.extend(paths)

    # 汇总
    print(f"\n{'='*80}")
    print(f"✅ 全部生成完成！共 {len(all_paths)} 个实例")
    print(f"{'='*80}")
    print(f"\n目录结构：")
    print(f"  {out_root}/")
    for kind in ("train", "test"):
        if args.only_train and kind == "test":
            continue
        if args.only_test and kind == "train":
            continue
        for sc in SCENARIOS:
            sub = out_root / kind / sc
            if sub.exists():
                n = len(list(sub.glob("*.json")))
                print(f"    {kind}/{sc:<18s} : {n} 个实例")


if __name__ == "__main__":
    main()

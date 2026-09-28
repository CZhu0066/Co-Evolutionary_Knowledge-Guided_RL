# -*- coding: utf-8 -*-
"""
generate_validation_instances.py
============================================================
重新生成一批独立 validation 实例，而不是从 train/test 中切分。

用途：
  train_instances : 训练模型、采集轨迹、挖规则
  val_instances   : 规则逐条评价、贪心筛选规则
  test_instances  : 最终测试，只在最后使用

默认做法：
  - 使用 train 配置模板（规模分布与训练集一致）
  - 使用全新的 seed_base，重新随机生成实例内容
  - 每个场景 8 个配置 × 每配置 1 个 = 共 24 个 validation JSON
  - 输出到 data/instances_combined_val_aqc4

示例：
  python scripts/generate_validation_instances.py ^
    --out data/instances_combined_val_aqc4 ^
    --base_config train ^
    --n_per_config 1 ^
    --seed_base 20042 ^
    --clean

如果只想先生成少量验证数据，例如每个场景只取 0,2,4 三组配置：
  python scripts/generate_validation_instances.py ^
    --out data/instances_combined_val_aqc4_small ^
    --config_indices 0,2,4 ^
    --n_per_config 1 ^
    --seed_base 20042 ^
    --clean
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

# 允许从工程根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_instances import (  # noqa: E402
    generate_one,
    get_configs,
    get_filename,
    save_instance,
)

DEFAULT_SCENARIOS = ["1load_1unload", "2load", "2unload"]


def parse_indices(text: str, n: int) -> List[int]:
    text = str(text).strip().lower()
    if text in ("", "all", "*"):
        return list(range(n))
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    # 去重并保持顺序
    seen = set()
    final = []
    for i in out:
        if i < 0 or i >= n:
            raise ValueError(f"config index 越界：{i}，合法范围 0~{n-1}")
        if i not in seen:
            seen.add(i)
            final.append(i)
    return final


def scenario_from_instance(inst: Dict) -> str:
    return str(inst.get("scenario", "unknown"))


def main():
    p = argparse.ArgumentParser(
        description="重新生成独立 validation 实例，不切分原训练集/测试集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=str, default="data/instances_combined_val_aqc4",
                   help="validation JSON 输出目录（平铺保存，方便现有评估脚本读取）")
    p.add_argument("--base_config", type=str, default="train", choices=["train", "test"],
                   help="使用哪套配置模板。正式建议 train：规模分布贴近训练，但 seed 独立")
    p.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
                   choices=["1load_1unload", "2load", "2unload"],
                   help="要生成哪些场景")
    p.add_argument("--config_indices", type=str, default="all",
                   help="选择每个场景的哪些配置。all 表示 0~7；也可写 0,2,4 或 0-3")
    p.add_argument("--n_per_config", type=int, default=1,
                   help="每个配置重新生成几个不同 seed 的实例")
    p.add_argument("--seed_base", type=int, default=20042,
                   help="validation 独立随机种子基数；不要和 train/test 的 seed 重复")
    p.add_argument("--clean", action="store_true",
                   help="生成前清空输出目录")
    p.add_argument("--quiet", action="store_true",
                   help="少打印一些信息")
    args = p.parse_args()

    out_dir = Path(args.out)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    summary: Dict[str, int] = {}

    print("\n" + "=" * 80)
    print("重新生成 validation 实例")
    print(f"输出目录      : {out_dir}")
    print(f"配置模板      : {args.base_config}")
    print(f"场景          : {', '.join(args.scenarios)}")
    print(f"config_indices: {args.config_indices}")
    print(f"n_per_config  : {args.n_per_config}")
    print(f"seed_base     : {args.seed_base}")
    print("=" * 80)

    for s_idx, scenario in enumerate(args.scenarios):
        configs = get_configs(scenario, args.base_config)
        indices = parse_indices(args.config_indices, len(configs))
        summary.setdefault(scenario, 0)

        if not args.quiet:
            print(f"\n[{scenario}] 配置数={len(indices)}，每配置={args.n_per_config}")

        for cfg_idx in indices:
            cfg = configs[cfg_idx]
            for rep in range(args.n_per_config):
                # 场景、配置、重复号都写进 seed，避免不同场景重复
                seed = int(args.seed_base) + s_idx * 100000 + cfg_idx * 1000 + rep
                inst = generate_one(scenario, cfg, seed)
                # 文件名前缀使用 val，和 train/test 明确区分
                fname = get_filename(scenario, cfg, "val", seed)
                path = out_dir / fname
                save_instance(inst, path)

                n_tasks = 0
                for key in ("J1_yard", "J3_yard", "J1_truck", "J3_truck",
                            "J2_truck", "J2_yard", "J4_truck", "J4_yard"):
                    v = inst.get(key, [])
                    if isinstance(v, list):
                        n_tasks += len(v)

                row = {
                    "file": fname,
                    "scenario": scenario_from_instance(inst),
                    "base_config": args.base_config,
                    "config_idx": cfg_idx,
                    "rep": rep,
                    "seed": seed,
                    "n_tasks_approx": n_tasks,
                }
                manifest_rows.append(row)
                summary[scenario] += 1
                if not args.quiet:
                    print(f"  ✅ {fname}")

    # 保存 manifest
    manifest_path = out_dir / "validation_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)

    summary_obj = {
        "out_dir": str(out_dir),
        "base_config": args.base_config,
        "scenarios": args.scenarios,
        "config_indices": args.config_indices,
        "n_per_config": args.n_per_config,
        "seed_base": args.seed_base,
        "total": len(manifest_rows),
        "by_scenario": summary,
        "note": "validation 实例为独立 seed 重新生成；不是从 train/test 切分。",
    }
    summary_path = out_dir / "validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"✅ validation 生成完成：{len(manifest_rows)} 个 JSON")
    for k, v in summary.items():
        print(f"  {k:<18s}: {v}")
    print(f"清单：{manifest_path}")
    print(f"汇总：{summary_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

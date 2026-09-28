# -*- coding: utf-8 -*-
"""生成场景感知模板规则 JSON。

用法：
    python scripts/generate_template_rules.py --out results/template_guidance_rules.json
    python scripts/generate_template_rules.py --scenarios 2load 2unload --out results/template_rules_load_unload.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.innovation_A.template_rules import write_template_rules


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/template_guidance_rules.json", type=str)
    p.add_argument("--scenarios", nargs="*", default=None,
                   choices=["1load_1unload", "2load", "2unload"],
                   help="不填则生成三个场景全部规则")
    p.add_argument("--no_global", action="store_true", help="不包含 urgent/故障AQC/planning_blocked 等全局规则")
    args = p.parse_args()
    data = write_template_rules(args.out, scenarios=args.scenarios, include_global=not args.no_global)
    print(f"已生成模板规则：{args.out}")
    print(f"规则数量：{data['n_rules']}")


if __name__ == "__main__":
    main()

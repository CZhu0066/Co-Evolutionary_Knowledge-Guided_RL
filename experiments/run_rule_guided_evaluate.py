# -*- coding: utf-8 -*-
"""
run_rule_guided_evaluate.py
===========================
v14-RGCD：规则引导连续动作解码评估脚本。

作用：
1. 加载 SAC/PPO/RPPO 连续动作模型；
2. 可选加载 task/destination/AQC 三层规则；
3. 在 KGUnifiedYardEnv 的候选评分阶段执行 rule_guided decoding；
4. 输出 baseline / guided 的 obj、接受情况和规则触发统计；
5. 可选执行贪心规则组合：只有加入规则后 mean_obj 降低才保留。
"""
from __future__ import annotations
from src.innovation_A.scenario_rule_utils import (
    make_scenario_aware_engine,
)
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.obs_packer import ObsPacker
from src.env.kg_env import KGUnifiedYardEnv
from src.innovation_A.rule_guidance import RuleGuidanceEngine, load_rule_guidance
from src.eval.metrics import (
    compute_aqc_balance, compute_completion, compute_makespan_from_info, summarize_rows,
)

OBS_PACKER_DEFAULTS = dict(
    max_load=40, max_unload=40, max_cars=60, max_slots=200,
    max_aqcs=4, max_cols=400, max_crosses=20, max_truck_slots=20,
    max_trains=4,
)


def collect_json_files(instances_path: str, max_instances: int = 0) -> List[str]:
    p = Path(instances_path)
    if p.is_file():
        files = [str(p)]
    elif p.is_dir():
        files = [str(x) for x in sorted(p.glob("*.json"))]
    else:
        raise FileNotFoundError(f"路径不存在：{instances_path}")
    if not files:
        raise FileNotFoundError(f"在 {instances_path} 找不到 .json 文件")
    if max_instances and max_instances > 0:
        files = files[: int(max_instances)]
    return files


def load_model(model_path: str, algo: str, device: str = "auto"):
    algo = str(algo).lower()
    if algo == "sac":
        from stable_baselines3 import SAC
        return SAC.load(model_path, device=device)
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path, device=device)
    if algo == "rppo":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(model_path, device=device)
    raise ValueError(f"unsupported algo: {algo}")

def load_complete_rule_engine(
    rule_path: Optional[str],
    enabled_rule_ids=None,
):
    """
    同时兼容：
    1. 普通扁平规则文件；
    2. CoEvo-RGCD生成的分场景规则文件。
    """

    if not rule_path:
        return None

    path = Path(rule_path)

    if not path.exists():
        raise FileNotFoundError(
            f"规则文件不存在：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # 分场景规则
    selected_by_scenario = data.get(
        "selected_rule_ids_by_scenario"
    )

    rules = data.get(
        "rules",
        [],
    )

    if (
        isinstance(selected_by_scenario, dict)
        and isinstance(rules, list)
    ):
        engine_kwargs = data.get(
            "soft_rule_guidance",
            {},
        )

        # 只保留规则引擎支持的参数
        allowed_keys = {
            "soft_rule_guidance",
            "soft_rule_temperature",
            "soft_rule_min_strength",
            "soft_rule_aggregation",
            "rule_memory_retrieval",
            "rule_retrieval_top_k",
            "rule_retrieval_dedup_templates",
        }

        engine_kwargs = {
            key: value
            for key, value in engine_kwargs.items()
            if key in allowed_keys
        }

        engine = make_scenario_aware_engine(
            rules=rules,
            selected_ids_by_scenario=(
                selected_by_scenario
            ),
            engine_kwargs=engine_kwargs,
        )

        print(
            "已加载分场景规则："
            f"{len(engine.active_rule_ids()) if engine else 0}条"
        )

        return engine

    # 兼容普通规则文件
    return load_rule_guidance(
        str(path),
        enabled_rule_ids=enabled_rule_ids,
    )


def run_one_episode(model, algo: str, json_file: str, packer: ObsPacker,
                    seed: int, enable_disturbance: bool, intensity: str,
                    guidance_engine: Optional[RuleGuidanceEngine] = None,
                    deterministic: bool = True) -> Dict[str, Any]:
    env = KGUnifiedYardEnv(
        json_files=[json_file],
        obs_packer=packer,
        seed=seed,
        shuffle_each_reset=False,
        enable_disturbance=enable_disturbance,
        disturbance_intensity=intensity,
        disturbance_seed=seed,
        rule_guidance_fn=guidance_engine,
    )
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    n_steps = 0
    state = None
    episode_start = True
    last_info: Dict[str, Any] = dict(info or {})
    t_episode0 = time.perf_counter()

    while not done:
        if str(algo).lower() == "rppo":
            action, state = model.predict(
                obs, state=state, episode_start=[episode_start], deterministic=deterministic
            )
            episode_start = False
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        n_steps += 1
        last_info = dict(info or {})
        done = bool(terminated or truncated)
        if n_steps > getattr(env, "max_steps", 100000) + 10:
            break

    runtime_s = float(time.perf_counter() - t_episode0)
    makespan = compute_makespan_from_info(last_info, env=env)
    aqc_balance = compute_aqc_balance(env=env, info=last_info)
    completion = compute_completion(last_info, env=env)

    # ========================================================
    # Task-count statistics
    # ========================================================

    tasks = list(
        getattr(env, "tasks", []) or []
    )

    # 原始任务：不是扰动插入的任务
    n_initial_tasks = sum(
        1
        for task in tasks
        if not bool(
            getattr(task, "is_inserted", False)
        )
    )

    # 扰动过程中新增的任务
    n_inserted_tasks = sum(
        1
        for task in tasks
        if bool(
            getattr(task, "is_inserted", False)
        )
    )

    # 被取消的任务
    n_canceled_tasks = sum(
        1
        for task in tasks
        if bool(
            getattr(task, "canceled", False)
        )
    )

    # 真正完成处理且没有被取消的任务
    n_processed_tasks = sum(
        1
        for task in tasks
        if (
                bool(getattr(task, "done", False))
                and not bool(
            getattr(task, "canceled", False)
        )
        )
    )

    # 最终需要完成的有效任务数
    n_effective_tasks = sum(
        1
        for task in tasks
        if not bool(
            getattr(task, "canceled", False)
        )
    )

    row = {
        "instance": Path(json_file).name,
        "scenario": str(getattr(env, "scenario", "unknown")),
        "reward": float(total_reward),
        "length": int(n_steps),
        "obj": float(last_info.get("obj", getattr(env, "obj", 0.0))),
        "train_obj": float(last_info.get("train_obj", getattr(env, "train_obj", 0.0))),
        "truck_total_wait": float(last_info.get("truck_total_wait", getattr(env, "truck_total_wait", 0.0))),
        "makespan": float(makespan),
        "aqc_balance": float(aqc_balance),
        "runtime_s": float(runtime_s),
        "completion": float(completion),
        "n_disturbances": int(last_info.get("n_applied_disturbances", 0)),
        "reason": str(last_info.get("reason", "")),
        "n_initial_tasks": int(n_initial_tasks),
        "n_inserted_tasks": int(n_inserted_tasks),
        "n_canceled_tasks": int(n_canceled_tasks),
        "n_processed_tasks": int(n_processed_tasks),
        "n_effective_tasks": int(n_effective_tasks),
    }

    gs = dict(last_info.get("rule_guidance_stats", {}) or {})
    matched_rule_count = int(gs.get("matched_rule_count", 0) or 0)
    trigger_strength_count = int(gs.get("trigger_strength_count", 0) or 0)
    row.update({
        "guidance_calls": int(gs.get("calls", 0)),
        "guidance_triggered": int(gs.get("triggered", 0)),
        "guidance_masked": int(gs.get("masked", 0)),
        "task_triggered": int(gs.get("task_triggered", 0)),
        "destination_triggered": int(gs.get("destination_triggered", 0)),
        "aqc_triggered": int(gs.get("aqc_triggered", 0)),
        "guidance_total_adjust": float(gs.get("total_adjust", 0.0)),
        "guidance_abs_total_adjust": float(gs.get("abs_total_adjust", 0.0)),
        "guidance_matched_rule_count": matched_rule_count,
        "guidance_match_strength_sum": float(gs.get("match_strength_sum", 0.0)),
        "guidance_trigger_strength_sum": float(gs.get("trigger_strength_sum", 0.0)),
        "guidance_high_strength_triggered": int(gs.get("high_strength_triggered", 0)),
        "guidance_mean_rule_strength": (float(gs.get("match_strength_sum", 0.0)) / matched_rule_count) if matched_rule_count > 0 else 0.0,
        "guidance_mean_trigger_strength": (float(gs.get("trigger_strength_sum", 0.0)) / trigger_strength_count) if trigger_strength_count > 0 else 0.0,
    })
    env.close()
    return row


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-instance rows with the main-comparison metrics."""
    return summarize_rows(rows)


def evaluate_set(model, algo: str, files: Sequence[str], packer: ObsPacker, seed: int,
                 enable_disturbance: bool, intensity: str,
                 guidance_engine: Optional[RuleGuidanceEngine],
                 deterministic: bool = True) -> List[Dict[str, Any]]:
    rows = []
    for i, f in enumerate(files, 1):
        row = run_one_episode(
            model=model, algo=algo, json_file=f, packer=packer, seed=seed,
            enable_disturbance=enable_disturbance, intensity=intensity,
            guidance_engine=guidance_engine, deterministic=deterministic,
        )
        rows.append(row)
        print(f"[{i}/{len(files)}] {row['instance']}  scenario={row['scenario']}  obj={row['obj']:.4f}  "
              f"triggered={row.get('guidance_triggered', 0)}")
    return rows


def save_csv(rows: Sequence[Dict[str, Any]], path: str):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV 已保存 → {p}")


def save_json(obj: Any, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"JSON 已保存 → {p}")


def run_greedy_selection(args, model, files, packer, base_engine: RuleGuidanceEngine):
    print("\n" + "=" * 72)
    print("v14-RGCD 贪心规则组合：加入规则后 mean_obj 降低才保留")
    print("=" * 72)

    # baseline
    print("\n[Baseline] 无规则引导评估...")
    baseline_rows = evaluate_set(
        model, args.algo, files, packer, args.seed, args.disturbance, args.intensity,
        guidance_engine=None, deterministic=not args.stochastic,
    )
    baseline_summary = summarize(baseline_rows)
    best_obj = float(baseline_summary["mean_obj"])
    selected: List[str] = []
    history = [{"step": 0, "try_rule": "BASELINE", "accepted": True, **baseline_summary}]
    print(f"Baseline mean_obj={best_obj:.4f}")

    candidates = base_engine.active_rule_ids()
    print(f"候选规则数：{len(candidates)}")
    for rid in candidates:
        trial_ids = selected + [rid]
        print(f"\n[Try] + {rid}  当前规则集大小={len(trial_ids)}")
        trial_engine = base_engine.subset(trial_ids)
        rows = evaluate_set(
            model, args.algo, files, packer, args.seed, args.disturbance, args.intensity,
            guidance_engine=trial_engine, deterministic=not args.stochastic,
        )
        sm = summarize(rows)
        mean_obj = float(sm["mean_obj"])
        accepted = mean_obj < best_obj - float(args.min_delta)
        history.append({"step": len(history), "try_rule": rid, "accepted": bool(accepted), **sm})
        print(f"Try mean_obj={mean_obj:.4f}, best={best_obj:.4f}, accepted={accepted}")
        if accepted:
            selected.append(rid)
            best_obj = mean_obj

    out = {
        "baseline_mean_obj": float(baseline_summary["mean_obj"]),
        "best_mean_obj": float(best_obj),
        "selected_rule_ids": selected,
        "history": history,
    }
    if args.greedy_json:
        save_json(out, args.greedy_json)
    print("\n" + "=" * 72)
    print("贪心选择完成")
    print(f"selected_rule_ids={selected}")
    print(f"baseline_mean_obj={baseline_summary['mean_obj']:.4f}")
    print(f"best_mean_obj={best_obj:.4f}")
    print("=" * 72)
    return out


def main():
    p = argparse.ArgumentParser(
        description="v14-RGCD 规则引导连续动作解码评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--algo", default="sac", choices=["sac", "ppo", "rppo"])
    p.add_argument("--instances", required=True, type=str)
    p.add_argument("--max_instances", type=int, default=0)
    p.add_argument("--disturbance", action="store_true")
    p.add_argument("--intensity", default="med", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--rules", default=None, type=str, help="规则 JSON；不填则只跑 baseline")
    p.add_argument("--rule_ids", nargs="*", default=None, help="只启用指定规则 id")
    p.add_argument("--csv", default=None, type=str)
    p.add_argument("--summary_json", default=None, type=str)
    p.add_argument("--stochastic", action="store_true", help="使用随机策略而非 deterministic predict")
    p.add_argument("--greedy_select", action="store_true", help="执行贪心规则组合")
    p.add_argument("--greedy_json", default="results/v14_rgcd_greedy_selected_rules.json", type=str)
    p.add_argument("--min_delta", default=1e-6, type=float, help="mean_obj 至少降低多少才接受规则")
    args = p.parse_args()

    files = collect_json_files(args.instances, args.max_instances)
    packer = ObsPacker(**OBS_PACKER_DEFAULTS, enable_disturbance=args.disturbance)
    model = load_model(args.model, args.algo, args.device)

    print("\n" + "=" * 72)
    print("v14-RGCD：规则引导连续动作解码评估")
    print(f"模型      : {args.model}")
    print(f"算法      : {args.algo}")
    print(f"实例数    : {len(files)}")
    print(f"扰动      : {'✅ ' + args.intensity if args.disturbance else '❌'}")
    print(f"规则文件  : {args.rules or 'None / baseline'}")
    print(f"输出 CSV  : {args.csv or 'None'}")
    print("=" * 72)

    engine = None
    if args.rules:
        engine = load_complete_rule_engine(
            rule_path=args.rules,
            enabled_rule_ids=args.rule_ids,
        )
        print(f"已加载规则数：{len(engine.active_rule_ids())}")

    if args.greedy_select:
        if engine is None:
            raise ValueError("--greedy_select 需要提供 --rules")
        run_greedy_selection(args, model, files, packer, engine)
        return

    t0 = time.time()
    rows = evaluate_set(
        model, args.algo, files, packer, args.seed, args.disturbance, args.intensity,
        guidance_engine=engine, deterministic=not args.stochastic,
    )
    sm = summarize(rows)
    sm.update({
        "model": args.model,
        "algo": args.algo,
        "n_instances": len(files),
        "disturbance": bool(args.disturbance),
        "intensity": args.intensity if args.disturbance else "none",
        "rules": args.rules,
        "eval_time_s": round(time.time() - t0, 2),
    })
    print("\n" + "=" * 72)
    print("v14-RGCD 评估汇总")
    print("-" * 72)
    print(f"case 数量         : {sm['n']}")
    print(f"mean_obj          : {sm['mean_obj']:.4f}")
    print(f"mean_reward       : {sm['mean_reward']:.4f}")
    print(f"mean_truck_wait   : {sm['mean_truck_wait']:.4f}")
    print(f"mean_triggered    : {sm['mean_guidance_triggered']:.4f}")
    print(f"mean_masked       : {sm['mean_guidance_masked']:.4f}")
    print(f"eval_time_s       : {sm['eval_time_s']:.2f}")
    print("=" * 72)

    if args.csv:
        save_csv(rows, args.csv)
    if args.summary_json:
        save_json(sm, args.summary_json)


if __name__ == "__main__":
    main()

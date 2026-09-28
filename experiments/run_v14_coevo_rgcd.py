# -*- coding: utf-8 -*-
"""
run_v14_coevo_rgcd.py
=====================
一键运行 v14-CoEvo-RGCD：规则引导连续动作解码的知识-强化学习协同进化闭环。

正式数据流：
  train_instances：训练 SAC/PPO/RPPO + 采集轨迹 + 蒸馏候选规则；
  val_instances  ：逐条评价候选规则 + 贪心筛选知识库；
  test_instances ：最终泛化测试，只用于报告 baseline / guided / 新模型效果。

推荐先使用 --no_train_feedback：
  Round 0 训练 → train 挖规则 → val 筛规则 → test 测旧模型+筛选规则。
确认规则在 test 上有泛化效果后，再去掉 --no_train_feedback 做下一轮训练反馈。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.obs_packer import ObsPacker
from src.env.kg_env import KGUnifiedYardEnv
from src.algos import TrainerConfig, get_trainer
from src.innovation_A.feature_extractor import FeatureExtractor
from src.innovation_A.trajectory_collector import collect_trajectories, save_trajectories, summarize_trajectories
from src.innovation_A.distillation import distill_rules_by_scenario, save_rules_by_scenario
from src.innovation_A.rgcd_rule_miner import convert_distilled_to_guidance, save_guidance_rules
from src.innovation_A.aqc_rule_miner import (
    collect_aqc_candidate_samples,
    save_aqc_candidate_samples,
    mine_aqc_rules_by_group,
    convert_aqc_rules_to_guidance,
    save_json as save_aqc_json,
)
from src.innovation_A.destination_rule_miner import (
    collect_destination_candidate_samples,
    save_destination_candidate_samples,
    mine_destination_rules_by_group,
    convert_destination_rules_to_guidance,
    save_json as save_destination_json,
)
from src.innovation_A.rule_guidance import RuleGuidanceEngine, load_rule_guidance
from src.innovation_A.template_rule_completion import (
    complete_template_guidance_rules,
    save_template_completion_json,
)
from src.innovation_A.scenario_rule_utils import (
    SCENARIO_ORDER, infer_rule_scenario, infer_rule_stage, annotate_rule, is_global_rule,
    infer_rule_family, apply_scenario_stage_quota, save_rules_by_scenario_json,
    make_scenario_aware_engine, group_files_by_scenario,
)
from src.eval.training_progress import make_sb3_callback

from experiments.run_rule_guided_evaluate import (
    load_model, evaluate_set, summarize, save_csv, save_json,
)


OBS_PACKER_DEFAULTS = dict(
    max_load=40, max_unload=40, max_cars=60, max_slots=200,
    max_aqcs=4, max_cols=400, max_crosses=20, max_truck_slots=20,
    max_trains=4,
)


def collect_json_files(path: str, max_instances: int = 0) -> List[str]:
    p = Path(path)
    if p.is_file():
        files = [str(p)]
    elif p.is_dir():
        files = [str(x) for x in sorted(p.glob("*.json"))]
    else:
        raise FileNotFoundError(f"路径不存在：{path}")
    if not files:
        raise FileNotFoundError(f"在 {path} 找不到 .json")
    if max_instances and max_instances > 0:
        files = files[:int(max_instances)]
    return files


def write_csv(rows: List[Dict[str, Any]], path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(p, "w", encoding="utf-8-sig") as f:
            f.write("")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)





def _rule_id(rule: Dict[str, Any]) -> str:
    return str(rule.get("id", rule.get("rule_id", rule.get("name", ""))) or "")


def _dedup_rules_by_id(rules: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for rule in rules:
        rid = _rule_id(rule)
        key = rid or json.dumps(rule, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(rule)
    return out


def _extract_rules_from_json_obj(obj: Any) -> List[Dict[str, Any]]:
    """Read rules from selected/candidate JSON structures used by this project."""
    if not isinstance(obj, dict):
        return []
    if isinstance(obj.get("rules"), list):
        return [r for r in obj.get("rules", []) if isinstance(r, dict)]
    if isinstance(obj.get("selected_rules"), list):
        return [r for r in obj.get("selected_rules", []) if isinstance(r, dict)]
    if isinstance(obj.get("rules_by_scenario"), dict):
        arr: List[Dict[str, Any]] = []
        for rs in obj.get("rules_by_scenario", {}).values():
            if isinstance(rs, list):
                arr.extend([r for r in rs if isinstance(r, dict)])
        return arr
    return []


def _load_rules_from_json(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            return _extract_rules_from_json_obj(json.load(f))
    except Exception:
        return []


def _limit_carryover_rules_by_family(rules: Sequence[Dict[str, Any]], max_per_family: int = 1) -> List[Dict[str, Any]]:
    """Keep only a small number of previous selected rules per family.

    Carry-over only gives previous good rules a chance to be re-evaluated.
    It does not force them to be selected by VAL.
    """
    if max_per_family is None or int(max_per_family) <= 0:
        return [annotate_rule(r) for r in rules]
    counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for rule in rules:
        r = annotate_rule(rule)
        fam = infer_rule_family(r)
        if counts.get(fam, 0) >= int(max_per_family):
            continue
        counts[fam] = counts.get(fam, 0) + 1
        out.append(r)
    return out


def summarize_by_scenario(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate evaluate rows by scenario for scenario-wise diagnosis."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows or []:
        groups.setdefault(str(r.get("scenario", "unknown")), []).append(dict(r))
    out: List[Dict[str, Any]] = []
    for scenario in sorted(groups.keys()):
        sm = summarize(groups[scenario])
        sm["scenario"] = scenario
        sm["n"] = len(groups[scenario])
        # explicit trigger diagnostics that summarize_rows may not expose in older versions
        for k in ["guidance_triggered", "task_triggered", "destination_triggered", "aqc_triggered"]:
            vals = [float(x.get(k, 0) or 0) for x in groups[scenario]]
            sm[f"mean_{k}"] = sum(vals) / max(1, len(vals))
            sm[f"sum_{k}"] = sum(vals)
        out.append(sm)
    return out


def save_scenario_summary(rows: Sequence[Dict[str, Any]], csv_path: str | Path, json_path: str | Path) -> List[Dict[str, Any]]:
    sm = summarize_by_scenario(rows)
    write_csv(sm, csv_path)
    save_json(sm, str(json_path))
    return sm


def rule_guidance_engine_kwargs(args) -> Dict[str, Any]:
    """Collect RuleGuidanceEngine options.

    Trend-6 uses soft/fuzzy numeric rule matching. The function keeps all
    engine construction sites consistent, including candidate evaluation,
    scenario-aware selection, test-time guidance and feedback training.
    """
    return {
        "soft_rule_guidance": bool(getattr(args, "soft_rule_guidance", False)),
        "soft_rule_temperature": float(getattr(args, "soft_rule_temperature", 0.05)),
        "soft_rule_min_strength": float(getattr(args, "soft_rule_min_strength", 0.05)),
        "soft_rule_aggregation": str(getattr(args, "soft_rule_aggregation", "min")),
        # Trend-4: context-aware rule-memory retrieval.
        "rule_memory_retrieval": bool(getattr(args, "rule_memory_retrieval", False)),
        "rule_retrieval_top_k": int(getattr(args, "rule_retrieval_top_k", 0) or 0),
        "rule_retrieval_dedup_templates": bool(getattr(args, "rule_retrieval_dedup_templates", False)),
    }


def _load_json_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _reuse_mined_rules_if_available(round_dir: Path) -> Optional[Dict[str, Any]]:
    """Reuse existing rule-mining outputs in round_dir so a crashed run can resume from VAL selection.

    This skips the expensive trajectory / destination / AQC collection and rule mining stages.
    """
    candidate_path = round_dir / "candidate_guidance_rules.json"
    if not candidate_path.exists():
        return None
    candidate_data = _load_json_file(candidate_path)
    raw_path = round_dir / "mined_rules_raw.json"
    raw_data = _load_json_file(raw_path) if raw_path.exists() else {}
    traj_path = round_dir / "trajectories.npz"
    return {
        "trajectories_path": str(traj_path) if traj_path.exists() else "",
        "distilled_rules_path": str(raw_path) if raw_path.exists() else "",
        "candidate_rules_path": str(candidate_path),
        "task_candidate_rules_path": str(round_dir / "candidate_task_guidance_rules.json"),
        "n_samples": int(candidate_data.get("n_samples", 0) or 0),
        "n_raw_rules": int(raw_data.get("n_rules", 0) or 0),
        "n_task_candidate_rules": int(candidate_data.get("task_rule_count", 0) or 0),
        "n_destination_candidate_samples": 0,
        "n_destination_raw_rules": int(candidate_data.get("destination_rules_meta", {}).get("n_rules", 0) or 0),
        "n_destination_guidance_rules": int(candidate_data.get("destination_rule_count", 0) or 0),
        "n_aqc_candidate_samples": 0,
        "n_aqc_raw_rules": int(candidate_data.get("aqc_rules_meta", {}).get("n_rules", 0) or 0),
        "n_aqc_guidance_rules": int(candidate_data.get("aqc_rule_count", 0) or 0),
        "n_candidate_rules": int(candidate_data.get("n_rules", 0) or len(candidate_data.get("rules", []) or [])),
        "reused_mined_rules": True,
    }

def make_env_factory(json_files: Sequence[str], packer: ObsPacker, args,
                     guidance_engine: Optional[RuleGuidanceEngine] = None,
                     shuffle_each_reset: bool = True):
    def _fn():
        return KGUnifiedYardEnv(
            json_files=list(json_files),
            obs_packer=packer,
            seed=args.seed,
            shuffle_each_reset=shuffle_each_reset,
            enable_disturbance=bool(args.disturbance),
            disturbance_intensity=args.intensity,
            disturbance_seed=args.seed,
            rule_bonus_fn=None,
            rule_guidance_fn=guidance_engine,
        )
    return _fn


def train_model(args, trainer_cls, packer: ObsPacker, train_files: Sequence[str],
                out_model: str | Path, init_model: Optional[str] = None,
                guidance_engine: Optional[RuleGuidanceEngine] = None,
                timesteps: int = 10000, timestep_offset: int = 0,
                progress_method: str = "") -> str:
    cfg = TrainerConfig(
        total_timesteps=int(timesteps),
        n_envs=1,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        n_eval_episodes=args.n_eval_episodes,
        algo_specific=json.loads(args.algo_specific) if args.algo_specific else {},
    )
    env_fn = make_env_factory(train_files, packer, args, guidance_engine=guidance_engine, shuffle_each_reset=True)
    trainer = trainer_cls(env_fn=env_fn, config=cfg)
    trainer.setup()
    if init_model:
        print(f"[Train] warm start: {init_model}")
        trainer.load(init_model)
    print(f"[Train] timesteps={timesteps}, guidance={'ON' if guidance_engine else 'OFF'}")
    out_model = Path(out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_model.parent / "training_progress.csv"
    callback = make_sb3_callback(
        progress_path,
        method=progress_method or ("SAC + CoEvo-RGCD" if guidance_engine else args.algo.upper()),
        timestep_offset=int(timestep_offset),
    ) if bool(getattr(args, "save_progress", True)) else None
    t0 = time.time()
    trainer.train(callback=callback, reset_num_timesteps=False if init_model else True)
    print(f"[Train] done in {time.time() - t0:.1f}s")
    if bool(getattr(args, "save_progress", True)):
        print(f"[Train] progress CSV → {progress_path}")
    trainer.save(str(out_model))
    trainer.close()
    return str(out_model)


def evaluate_model(args, model_path: str, packer: ObsPacker, files: Sequence[str],
                   guidance_engine: Optional[RuleGuidanceEngine], csv_path: str | Path,
                   summary_path: str | Path, title: str, split_name: str) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print(title)
    print(f"数据：{split_name}，实例数={len(files)}")
    print(f"模型：{model_path}")
    print(f"规则：{'ON' if guidance_engine else 'OFF'}")
    print("=" * 72)
    model = load_model(model_path, args.algo, args.device)
    rows = evaluate_set(
        model=model,
        algo=args.algo,
        files=files,
        packer=packer,
        seed=args.seed,
        enable_disturbance=bool(args.disturbance),
        intensity=args.intensity,
        guidance_engine=guidance_engine,
        deterministic=not args.stochastic,
    )
    sm = summarize(rows)
    sm.update({
        "model": str(model_path),
        "algo": args.algo,
        "guidance": bool(guidance_engine),
        "split": split_name,
        "n_instances": len(files),
    })
    save_csv(rows, str(csv_path))
    save_json(sm, str(summary_path))
    # Scenario-wise test/validation statistics: this is important for SA-CoEvo-RGCD.
    try:
        cp = Path(csv_path)
        sp = Path(summary_path)
        save_scenario_summary(
            rows,
            cp.with_name(cp.stem + "_by_scenario.csv"),
            sp.with_name(sp.stem + "_by_scenario.json"),
        )
    except Exception as e:
        print(f"[WARN] scenario summary save failed: {type(e).__name__}: {e}")
    print(f"[Eval:{split_name}] mean_obj={sm.get('mean_obj', float('nan')):.4f}, "
          f"mean_reward={sm.get('mean_reward', float('nan')):.4f}, "
          f"triggered={sm.get('mean_guidance_triggered', 0):.2f}")
    return sm




def _row_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, default)
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def compute_rule_safety_metrics(
    baseline_rows: Sequence[Dict[str, Any]],
    trial_rows: Sequence[Dict[str, Any]],
    eps: float = 1e-6,
    cvar_alpha: float = 0.25,
) -> Dict[str, Any]:
    """计算一组规则在验证集上的收益-风险诊断指标。

    obj 越低越好，因此 delta = trial_obj - baseline_obj：
      delta < 0 表示该 case 改善；delta > 0 表示该 case 恶化。

    Trend-3: 为软规则筛选额外计算尾部风险、恶化比例、触发强度等指标，
    使贪心选择可以从“硬阈值一票否决”升级为“风险预算式评分”。
    """
    base_by_inst = {str(r.get("instance", i)): r for i, r in enumerate(baseline_rows)}
    deltas: List[float] = []
    triggered: List[float] = []
    matched_rule_count: List[float] = []
    mean_rule_strength: List[float] = []
    mean_trigger_strength: List[float] = []
    trigger_strength_sum: List[float] = []
    high_strength_triggered: List[float] = []
    abs_total_adjust: List[float] = []
    completion_delta: List[float] = []
    detail_rows: List[Dict[str, Any]] = []
    for i, tr in enumerate(trial_rows):
        inst = str(tr.get("instance", i))
        br = base_by_inst.get(inst)
        if br is None and i < len(baseline_rows):
            br = baseline_rows[i]
        if br is None:
            continue
        base_obj = float(br.get("obj", 0.0))
        trial_obj = float(tr.get("obj", 0.0))
        delta = trial_obj - base_obj
        gain = base_obj - trial_obj
        trig = _row_float(tr, "guidance_triggered", 0.0)
        mrc = _row_float(tr, "guidance_matched_rule_count", 0.0)
        mrs = _row_float(tr, "guidance_mean_rule_strength", 0.0)
        mts = _row_float(tr, "guidance_mean_trigger_strength", 0.0)
        tss = _row_float(tr, "guidance_trigger_strength_sum", 0.0)
        hst = _row_float(tr, "guidance_high_strength_triggered", 0.0)
        ata = _row_float(tr, "guidance_abs_total_adjust", 0.0)
        comp_delta = _row_float(tr, "completion", 0.0) - _row_float(br, "completion", 0.0)
        deltas.append(delta)
        triggered.append(trig)
        matched_rule_count.append(mrc)
        mean_rule_strength.append(mrs)
        mean_trigger_strength.append(mts)
        trigger_strength_sum.append(tss)
        high_strength_triggered.append(hst)
        abs_total_adjust.append(ata)
        completion_delta.append(comp_delta)
        detail_rows.append({
            "instance": inst,
            "scenario": tr.get("scenario", ""),
            "baseline_obj": base_obj,
            "trial_obj": trial_obj,
            "delta_obj": delta,
            "gain": gain,
            "improved": bool(delta < -eps),
            "worsened": bool(delta > eps),
            "guidance_triggered": trig,
            "guidance_matched_rule_count": mrc,
            "guidance_mean_rule_strength": mrs,
            "guidance_mean_trigger_strength": mts,
            "guidance_trigger_strength_sum": tss,
            "guidance_high_strength_triggered": hst,
            "guidance_abs_total_adjust": ata,
            "completion_delta": comp_delta,
        })

    n = max(1, len(deltas))
    improved_cases = sum(1 for d in deltas if d < -eps)
    worsened_cases = sum(1 for d in deltas if d > eps)
    unchanged_cases = len(deltas) - improved_cases - worsened_cases
    worsen_vals = [d for d in deltas if d > eps]
    gain_vals = [-d for d in deltas if d < -eps]
    max_case_worsen = max(worsen_vals + [0.0])
    max_case_gain = max(gain_vals + [0.0])
    avg_triggered = sum(triggered) / max(1, len(triggered))
    max_case_triggered = max(triggered + [0.0])
    mean_delta_obj = sum(deltas) / n
    mean_gain = -mean_delta_obj

    # CVaR-style tail risk: average of the worst alpha fraction of positive deltas.
    alpha = float(cvar_alpha)
    if alpha <= 0 or alpha > 1:
        alpha = 0.25
    sorted_worsen = sorted(worsen_vals, reverse=True)
    if sorted_worsen:
        k = max(1, int(np.ceil(alpha * len(sorted_worsen))))
        cvar_worsen = float(np.mean(sorted_worsen[:k]))
    else:
        cvar_worsen = 0.0

    p_worsen = float(worsened_cases) / float(max(1, len(deltas)))
    p_improve = float(improved_cases) / float(max(1, len(deltas)))
    mean_completion_delta = float(np.mean(completion_delta)) if completion_delta else 0.0
    min_completion_delta = float(np.min(completion_delta)) if completion_delta else 0.0

    return {
        "n_cases_compared": len(deltas),
        "mean_delta_obj": mean_delta_obj,
        "mean_gain": mean_gain,
        "max_case_worsen": max_case_worsen,
        "max_case_gain": max_case_gain,
        "avg_triggered": avg_triggered,
        "max_case_triggered": max_case_triggered,
        "improved_cases": improved_cases,
        "worsened_cases": worsened_cases,
        "unchanged_cases": unchanged_cases,
        "p_worsen": p_worsen,
        "p_improve": p_improve,
        "cvar_worsen": cvar_worsen,
        "mean_completion_delta": mean_completion_delta,
        "min_completion_delta": min_completion_delta,
        "avg_matched_rule_count": float(np.mean(matched_rule_count)) if matched_rule_count else 0.0,
        "avg_rule_strength": float(np.mean(mean_rule_strength)) if mean_rule_strength else 0.0,
        "avg_trigger_strength": float(np.mean(mean_trigger_strength)) if mean_trigger_strength else 0.0,
        "avg_trigger_strength_sum": float(np.mean(trigger_strength_sum)) if trigger_strength_sum else 0.0,
        "max_trigger_strength_sum": float(np.max(trigger_strength_sum)) if trigger_strength_sum else 0.0,
        "avg_high_strength_triggered": float(np.mean(high_strength_triggered)) if high_strength_triggered else 0.0,
        "max_high_strength_triggered": float(np.max(high_strength_triggered)) if high_strength_triggered else 0.0,
        "avg_abs_total_adjust": float(np.mean(abs_total_adjust)) if abs_total_adjust else 0.0,
        "case_details": detail_rows,
    }

def risk_budget_enabled(args) -> bool:
    return bool(getattr(args, "risk_budget_greedy", False))


def compute_risk_aware_score(args, mean_obj: float, metrics: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
    """Trend-3 risk-budgeted greedy score. Lower is better.

    When disabled, this is exactly the original mean_obj objective.
    When enabled, normal risks are treated as penalties instead of hard rejections.
    """
    mean_obj = float(mean_obj)
    if not risk_budget_enabled(args):
        return mean_obj, {"risk_aware_score": mean_obj}

    cvar = float(metrics.get("cvar_worsen", 0.0))
    p_w = float(metrics.get("p_worsen", 0.0))
    max_w = float(metrics.get("max_case_worsen", 0.0))
    trig_strength = float(metrics.get("avg_trigger_strength_sum", 0.0))
    high_strength = float(metrics.get("avg_high_strength_triggered", 0.0))
    p_improve = float(metrics.get("p_improve", 0.0))

    cvar_pen = float(getattr(args, "risk_lambda_cvar", 0.05)) * cvar
    p_w_pen = float(getattr(args, "risk_lambda_p_worsen", 50.0)) * p_w
    max_w_pen = float(getattr(args, "risk_lambda_max_worsen", 0.01)) * max_w
    trig_pen = float(getattr(args, "risk_lambda_trigger_strength", 0.02)) * trig_strength
    high_trig_pen = float(getattr(args, "risk_lambda_high_strength", 0.05)) * high_strength
    improve_bonus = float(getattr(args, "risk_lambda_p_improve", 0.0)) * p_improve

    score = mean_obj + cvar_pen + p_w_pen + max_w_pen + trig_pen + high_trig_pen - improve_bonus
    return float(score), {
        "risk_aware_score": float(score),
        "risk_score_mean_obj": mean_obj,
        "risk_penalty_cvar": float(cvar_pen),
        "risk_penalty_p_worsen": float(p_w_pen),
        "risk_penalty_max_worsen": float(max_w_pen),
        "risk_penalty_trigger_strength": float(trig_pen),
        "risk_penalty_high_strength": float(high_trig_pen),
        "risk_bonus_p_improve": float(improve_bonus),
    }


def passes_safe_filter(args, metrics: Dict[str, Any]) -> tuple[bool, List[str]]:
    """根据命令行阈值判断规则组合是否安全。

    Original mode: all safety thresholds are hard filters.
    Trend-3 risk-budget mode: only catastrophic risks are hard rejected; normal
    risks are passed to compute_risk_aware_score() as penalties.
    """
    if not bool(getattr(args, "safe_rule_filter", False)):
        return True, []
    reasons: List[str] = []

    if risk_budget_enabled(args):
        catastrophic_worsen = float(getattr(args, "risk_catastrophic_worsen", 1500.0))
        max_w = float(metrics.get("max_case_worsen", 0.0))
        if max_w > catastrophic_worsen:
            reasons.append(f"catastrophic max_case_worsen {max_w:.4f} > {catastrophic_worsen}")

        min_comp_delta = float(metrics.get("min_completion_delta", 0.0))
        max_completion_drop = float(getattr(args, "risk_max_completion_drop", 0.01))
        if min_comp_delta < -abs(max_completion_drop):
            reasons.append(f"completion_drop {min_comp_delta:.4f} < -{abs(max_completion_drop):.4f}")

        # In risk-budget mode, avg_triggered/max_case_triggered and improved<worsened
        # are no longer one-vote vetoes. They are represented by strength-aware
        # risk penalties in compute_risk_aware_score().
        return (len(reasons) == 0), reasons

    if float(metrics.get("max_case_worsen", 0.0)) > float(args.max_case_worsen):
        reasons.append(f"max_case_worsen {metrics.get('max_case_worsen', 0):.4f} > {args.max_case_worsen}")
    if float(metrics.get("avg_triggered", 0.0)) > float(args.max_avg_triggered):
        reasons.append(f"avg_triggered {metrics.get('avg_triggered', 0):.2f} > {args.max_avg_triggered}")
    if float(metrics.get("max_case_triggered", 0.0)) > float(args.max_case_triggered):
        reasons.append(f"max_case_triggered {metrics.get('max_case_triggered', 0):.0f} > {args.max_case_triggered}")
    if bool(getattr(args, "require_improved_ge_worsened", False)):
        if int(metrics.get("improved_cases", 0)) < int(metrics.get("worsened_cases", 0)):
            reasons.append(f"improved_cases {metrics.get('improved_cases', 0)} < worsened_cases {metrics.get('worsened_cases', 0)}")
    return (len(reasons) == 0), reasons

def collect_and_mine_rules(args, trainer_cls, packer: ObsPacker, train_files: Sequence[str],
                           source_model: str, round_dir: Path,
                           collect_guidance: Optional[RuleGuidanceEngine],
                           prev_selected_rules_path: Optional[str | Path] = None) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print(f"[Mine] 用上一轮模型在 TRAIN 上采集轨迹并蒸馏规则：{source_model}")
    print(f"[Mine] train instances={len(train_files)}, collect episodes={args.n_collect_episodes}, max_steps/ep={args.collect_max_steps}")
    print("=" * 72)
    mining_mode = str(getattr(args, "rule_mining_mode", "dt")).lower()
    use_template_completion = mining_mode in ("template", "hybrid")
    use_dt_mining = mining_mode in ("dt", "hybrid")
    if use_template_completion:
        print(f"[Mine:TEMPLATE] 启用模板补全规则生成：mode={mining_mode}")
    else:
        print(f"[Mine:DT] 启用原决策树/RGCD规则挖掘：mode={mining_mode}")
    cfg = TrainerConfig(
        total_timesteps=0,
        n_envs=1,
        seed=args.seed,
        device=args.device,
        verbose=0,
        n_eval_episodes=args.n_eval_episodes,
        algo_specific=json.loads(args.algo_specific) if args.algo_specific else {},
    )
    env_fn = make_env_factory(train_files, packer, args, guidance_engine=collect_guidance, shuffle_each_reset=True)
    trainer = trainer_cls(env_fn=env_fn, config=cfg)
    trainer.setup()
    trainer.load(source_model)
    extractor = FeatureExtractor(horizon=args.horizon)
    data = collect_trajectories(
        predict_fn=lambda obs: trainer.predict(obs, deterministic=True),
        env_factory=env_fn,
        n_episodes=args.n_collect_episodes,
        max_steps_per_episode=args.collect_max_steps,
        extractor=extractor,
        horizon=args.horizon,
        verbose=True,
    )
    trainer.close()

    traj_path = round_dir / "trajectories.npz"
    save_trajectories(data, traj_path)
    (round_dir / "trajectories_summary.txt").write_text(summarize_trajectories(data), encoding="utf-8")

    n_samples = len(data.get("features", []))
    if (not use_dt_mining) or n_samples <= 0:
        rules_data = {
            "method": "template_mode_skip_decision_tree" if use_template_completion and not use_dt_mining else "decision_tree_by_scenario",
            "rules_by_scenario": {},
            "n_rules_by_scenario": {},
            "n_rules": 0,
            "feature_names": list(extractor.feature_names),
            "n_features": extractor.n_features,
            "n_samples": int(n_samples),
            "train_accuracy": 0.0,
        }
    else:
        rules_data = distill_rules_by_scenario(
            features=data["features"],
            labels=data["labels"],
            scenarios=data.get("scenarios", []),
            max_depth=args.distill_max_depth,
            min_samples_leaf=args.distill_min_samples_leaf,
            feature_names=list(data.get("feature_names", extractor.feature_names)),
            random_state=args.seed,
        )
    distilled_path = round_dir / "mined_rules_raw.json"
    save_rules_by_scenario(rules_data, distilled_path)

    if use_dt_mining:
        guidance_data = convert_distilled_to_guidance(
            rules_data,
            adjust=args.guidance_adjust,
            min_confidence=args.min_confidence,
            min_support=args.min_support,
            min_coverage=args.min_coverage,
            min_rule_conditions=args.min_rule_conditions,
            max_rule_conditions=args.max_rule_conditions,
            max_rules=args.max_rules,
        )
    else:
        guidance_data = {"rules": [], "n_rules": 0, "method": "template_mode_skip_task_dt_guidance"}
    # 先单独保存 task 层候选规则。
    task_candidate_path = round_dir / "candidate_task_guidance_rules.json"
    save_guidance_rules(guidance_data, task_candidate_path)

    # 可选：候选级 Destination 规则挖掘；模板补全模式也需要这些候选样本。
    dest_samples: List[Dict[str, Any]] = []
    aqc_samples: List[Dict[str, Any]] = []
    destination_info: Dict[str, Any] = {
        "enabled": bool(args.enable_destination_rules),
        "n_destination_candidate_samples": 0,
        "n_destination_raw_rules": 0,
        "n_destination_guidance_rules": 0,
        "destination_candidate_samples_path": "",
        "destination_rules_raw_path": "",
        "destination_guidance_rules_path": "",
    }
    destination_guidance_data: Dict[str, Any] = {"rules": [], "n_rules": 0}
    if bool(args.enable_destination_rules) or use_template_completion:
        print("\n" + "=" * 72)
        print("[Mine:DEST] 采集候选级 Destination 样本" + ("并挖掘 destination stage 决策树规则" if bool(args.enable_destination_rules) and use_dt_mining else "用于模板补全"))
        n_dest_eps = int(args.n_destination_collect_episodes or args.n_collect_episodes)
        print(f"[Mine:DEST] episodes={n_dest_eps}, max_steps/ep={args.collect_max_steps}")
        print("=" * 72)
        dest_cfg = TrainerConfig(
            total_timesteps=0,
            n_envs=1,
            seed=args.seed,
            device=args.device,
            verbose=0,
            n_eval_episodes=args.n_eval_episodes,
            algo_specific=json.loads(args.algo_specific) if args.algo_specific else {},
        )
        dest_env_fn = make_env_factory(
            train_files, packer, args, guidance_engine=collect_guidance,
            shuffle_each_reset=not bool(getattr(args, "destination_reuse_env", False)),
        )
        dest_trainer = trainer_cls(env_fn=dest_env_fn, config=dest_cfg)
        dest_trainer.setup()
        dest_trainer.load(source_model)
        dest_samples = collect_destination_candidate_samples(
            predict_fn=lambda obs: dest_trainer.predict(obs, deterministic=True),
            env_factory=dest_env_fn,
            n_episodes=n_dest_eps,
            max_steps_per_episode=args.collect_max_steps,
            verbose=True,
            reuse_env=bool(getattr(args, "destination_reuse_env", False)),
            probe_all_task_kinds=bool(getattr(args, "destination_probe_all_task_kinds", False)),
            probe_max_tasks_per_kind=int(getattr(args, "destination_probe_max_tasks_per_kind", 1)),
        )
        dest_trainer.close()
        dest_samples_path = round_dir / "destination_candidate_samples.csv"
        save_destination_candidate_samples(dest_samples, dest_samples_path)
        try:
            from collections import Counter
            print(f"[Mine:DEST] scenario distribution: {dict(Counter(str(x.get('scenario', 'unknown')) for x in dest_samples))}")
            print(f"[Mine:DEST] task_kind distribution: {dict(Counter(str(x.get('task_kind', 'unknown')) for x in dest_samples))}")
        except Exception:
            pass
        dest_rules_raw = {"method": "template_mode_skip_destination_dt", "n_rules": 0, "groups": {}}
        dest_raw_path = round_dir / "destination_rules_raw.json"
        dest_guidance_path = round_dir / "candidate_destination_guidance_rules.json"
        if bool(args.enable_destination_rules) and use_dt_mining:
            dest_rules_raw = mine_destination_rules_by_group(
                dest_samples,
                max_depth=args.destination_distill_max_depth,
                min_samples_leaf=args.destination_distill_min_samples_leaf,
                random_state=args.seed,
            )
            save_destination_json(dest_rules_raw, dest_raw_path, label="save_destination_rules_raw")
            destination_guidance_data = convert_destination_rules_to_guidance(
                dest_rules_raw,
                adjust=args.destination_guidance_adjust,
                min_confidence=args.destination_min_confidence,
                min_support=args.destination_min_support,
                min_coverage=args.destination_min_coverage,
                min_rule_conditions=args.destination_min_rule_conditions,
                max_rule_conditions=args.destination_max_rule_conditions,
                max_rules=args.destination_max_rules,
            )
            save_destination_json(destination_guidance_data, dest_guidance_path, label="save_destination_guidance_rules")
        else:
            save_destination_json(dest_rules_raw, dest_raw_path, label="save_destination_rules_raw")
            save_destination_json(destination_guidance_data, dest_guidance_path, label="save_destination_guidance_rules")
        destination_info.update({
            "n_destination_candidate_samples": int(len(dest_samples)),
            "n_destination_raw_rules": int(dest_rules_raw.get("n_rules", 0)),
            "n_destination_guidance_rules": int(destination_guidance_data.get("n_rules", 0)),
            "destination_candidate_samples_path": str(dest_samples_path),
            "destination_rules_raw_path": str(dest_raw_path),
            "destination_guidance_rules_path": str(dest_guidance_path),
        })

    # 可选：候选级 AQC 规则挖掘。第二层 AQC。
    aqc_info: Dict[str, Any] = {
        "enabled": bool(args.enable_aqc_rules),
        "n_aqc_candidate_samples": 0,
        "n_aqc_raw_rules": 0,
        "n_aqc_guidance_rules": 0,
        "aqc_candidate_samples_path": "",
        "aqc_rules_raw_path": "",
        "aqc_guidance_rules_path": "",
    }
    aqc_guidance_data: Dict[str, Any] = {"rules": [], "n_rules": 0}
    if bool(args.enable_aqc_rules) or use_template_completion:
        print("\n" + "=" * 72)
        print("[Mine:AQC] 采集候选级 AQC 样本" + ("并挖掘 AQC stage 决策树规则" if bool(args.enable_aqc_rules) and use_dt_mining else "用于模板补全"))
        n_aqc_eps = int(args.n_aqc_collect_episodes or args.n_collect_episodes)
        print(f"[Mine:AQC] episodes={n_aqc_eps}, max_steps/ep={args.collect_max_steps}")
        print("=" * 72)
        aqc_cfg = TrainerConfig(
            total_timesteps=0,
            n_envs=1,
            seed=args.seed,
            device=args.device,
            verbose=0,
            n_eval_episodes=args.n_eval_episodes,
            algo_specific=json.loads(args.algo_specific) if args.algo_specific else {},
        )
        aqc_env_fn = make_env_factory(train_files, packer, args, guidance_engine=collect_guidance, shuffle_each_reset=True)
        aqc_trainer = trainer_cls(env_fn=aqc_env_fn, config=aqc_cfg)
        aqc_trainer.setup()
        aqc_trainer.load(source_model)
        aqc_samples = collect_aqc_candidate_samples(
            predict_fn=lambda obs: aqc_trainer.predict(obs, deterministic=True),
            env_factory=aqc_env_fn,
            n_episodes=n_aqc_eps,
            max_steps_per_episode=args.collect_max_steps,
            verbose=True,
        )
        aqc_trainer.close()
        aqc_samples_path = round_dir / "aqc_candidate_samples.csv"
        save_aqc_candidate_samples(aqc_samples, aqc_samples_path)
        aqc_rules_raw = {"method": "template_mode_skip_aqc_dt", "n_rules": 0, "groups": {}}
        aqc_raw_path = round_dir / "aqc_rules_raw.json"
        aqc_guidance_path = round_dir / "candidate_aqc_guidance_rules.json"
        if bool(args.enable_aqc_rules) and use_dt_mining:
            aqc_rules_raw = mine_aqc_rules_by_group(
                aqc_samples,
                max_depth=args.aqc_distill_max_depth,
                min_samples_leaf=args.aqc_distill_min_samples_leaf,
                random_state=args.seed,
            )
            save_aqc_json(aqc_rules_raw, aqc_raw_path, label="save_aqc_rules_raw")
            aqc_guidance_data = convert_aqc_rules_to_guidance(
                aqc_rules_raw,
                adjust=args.aqc_guidance_adjust,
                min_confidence=args.aqc_min_confidence,
                min_support=args.aqc_min_support,
                min_coverage=args.aqc_min_coverage,
                min_rule_conditions=args.aqc_min_rule_conditions,
                max_rule_conditions=args.aqc_max_rule_conditions,
                max_rules=args.aqc_max_rules,
            )
            save_aqc_json(aqc_guidance_data, aqc_guidance_path, label="save_aqc_guidance_rules")
        else:
            save_aqc_json(aqc_rules_raw, aqc_raw_path, label="save_aqc_rules_raw")
            save_aqc_json(aqc_guidance_data, aqc_guidance_path, label="save_aqc_guidance_rules")
        aqc_info.update({
            "n_aqc_candidate_samples": int(len(aqc_samples)),
            "n_aqc_raw_rules": int(aqc_rules_raw.get("n_rules", 0)),
            "n_aqc_guidance_rules": int(aqc_guidance_data.get("n_rules", 0)),
            "aqc_candidate_samples_path": str(aqc_samples_path),
            "aqc_rules_raw_path": str(aqc_raw_path),
            "aqc_guidance_rules_path": str(aqc_guidance_path),
        })

    # 模板补全候选规则：模板结构固定，阈值由采样分位数补全，adjust 网格交给 VAL 贪心筛选。
    template_guidance_data: Dict[str, Any] = {"rules": [], "n_rules": 0, "meta": {}}
    template_completion_path = round_dir / "candidate_template_guidance_rules.json"
    if use_template_completion:
        print("\n" + "=" * 72)
        print("[Mine:TEMPLATE] 根据预设模板 + 采样数据补全候选规则")
        print("=" * 72)
        template_guidance_data = complete_template_guidance_rules(
            trajectory_data=data,
            destination_samples=dest_samples,
            aqc_samples=aqc_samples,
            quantiles=[float(x) for x in str(getattr(args, "template_quantiles", "0.3,0.4,0.5,0.6,0.7,0.8")).replace(";", ",").split(",") if str(x).strip()],
            prefer_adjust_candidates=getattr(args, "template_prefer_adjust_candidates", None),
            avoid_adjust_candidates=getattr(args, "template_avoid_adjust_candidates", None),
            min_support=int(getattr(args, "template_min_support", 20)),
            min_confidence=float(getattr(args, "template_min_confidence", 0.75)),
            min_coverage=float(getattr(args, "template_min_coverage", 0.002)),
            truck_wait_task_min_support=int(getattr(args, "truck_wait_task_min_support", getattr(args, "truck_wait_template_min_support", 2))),
            truck_wait_task_min_confidence=float(getattr(args, "truck_wait_task_min_confidence", 0.05)),
            truck_wait_task_min_coverage=float(getattr(args, "truck_wait_task_min_coverage", 0.00005)),
            truck_wait_dest_min_support=int(getattr(args, "truck_wait_dest_min_support", getattr(args, "truck_wait_template_min_support", 2))),
            truck_wait_dest_min_confidence=float(getattr(args, "truck_wait_dest_min_confidence", 0.001)),
            truck_wait_dest_min_coverage=float(getattr(args, "truck_wait_dest_min_coverage", 0.000001)),
            truck_wait_candidate_quota=int(getattr(args, "truck_wait_candidate_quota", 2)),
            max_rules=int(getattr(args, "template_max_rules", 120)),
            max_variants_per_template=int(getattr(args, "template_max_variants_per_template", 80)),
            include_fixed_rules=bool(getattr(args, "template_include_fixed_rules", True)),
        )
        save_template_completion_json(template_guidance_data, template_completion_path, label="save_template_guidance_rules")

    # 合并 task + destination + AQC 候选规则。
    # mode=dt: 只使用原决策树/RGCD规则；mode=template: 只使用模板补全规则；mode=hybrid: 两者合并。
    # SA-CoEvo-RGCD: annotate each rule with target_scenario/rule_stage, and optionally keep a balanced
    # quota for every scenario-stage bucket before VAL selection.
    if mining_mode == "template":
        raw_combined_rules = list(template_guidance_data.get("rules", []) or [])
    elif mining_mode == "hybrid":
        raw_combined_rules = (
            list(guidance_data.get("rules", []) or [])
            + list(destination_guidance_data.get("rules", []) or [])
            + list(aqc_guidance_data.get("rules", []) or [])
            + list(template_guidance_data.get("rules", []) or [])
        )
    else:
        raw_combined_rules = (
            list(guidance_data.get("rules", []) or [])
            + list(destination_guidance_data.get("rules", []) or [])
            + list(aqc_guidance_data.get("rules", []) or [])
        )
    n_carryover_rules = 0
    carryover_rules_path_used = ""
    if bool(getattr(args, "carryover_selected_rules", False)) and prev_selected_rules_path:
        prev_rules = _load_rules_from_json(prev_selected_rules_path)
        prev_rules = _limit_carryover_rules_by_family(
            prev_rules,
            max_per_family=int(getattr(args, "carryover_max_per_family", 1)),
        )
        carryover_rules: List[Dict[str, Any]] = []
        for rule in prev_rules:
            r = dict(annotate_rule(rule))
            r["source"] = str(r.get("source", "") or "carryover_previous_selected")
            r["carryover_previous_selected"] = True
            r["carryover_from"] = str(prev_selected_rules_path)
            carryover_rules.append(r)
        if carryover_rules:
            raw_combined_rules = list(raw_combined_rules) + carryover_rules
            n_carryover_rules = len(carryover_rules)
            carryover_rules_path_used = str(prev_selected_rules_path)
            print(f"[Carry-over] previous selected rules added to candidate pool: {n_carryover_rules} from {prev_selected_rules_path}")

    combined_rules = [annotate_rule(r) for r in raw_combined_rules]
    combined_rules = _dedup_rules_by_id(combined_rules)
    n_before_quota = len(combined_rules)
    if bool(getattr(args, "scenario_candidate_quota", False)) or bool(getattr(args, "scenario_wise_rules", False)):
        combined_rules = apply_scenario_stage_quota(
            combined_rules,
            max_rules_per_scenario=int(getattr(args, "max_rules_per_scenario", 30)),
            max_rules_per_stage_scenario=int(getattr(args, "max_rules_per_stage_scenario", 10)),
            keep_unknown=True,
            truck_wait_candidate_quota=int(getattr(args, "truck_wait_candidate_quota", 0)),
            family_candidate_quota=bool(getattr(args, "family_candidate_quota", False)),
            family_quota_load_pressure=int(getattr(args, "family_quota_load_pressure", 2)),
            family_quota_unload_pressure=int(getattr(args, "family_quota_unload_pressure", 2)),
            family_quota_truck_wait=int(getattr(args, "family_quota_truck_wait", getattr(args, "truck_wait_candidate_quota", 2))),
            family_quota_destination=int(getattr(args, "family_quota_destination", 2)),
            family_quota_aqc=int(getattr(args, "family_quota_aqc", 2)),
            family_quota_global=int(getattr(args, "family_quota_global", 2)),
            family_quota_other=int(getattr(args, "family_quota_other", 0)),
        )
        quota_tag = "family+scenario quota" if bool(getattr(args, "family_candidate_quota", False)) else "scenario quota"
        print(f"[Scenario quota] {quota_tag}: candidate rules {n_before_quota} → {len(combined_rules)}")

    combined_guidance_data = {
        "method": "combined_task_destination_aqc_template_or_rgcd_guidance_rules",
        "description": "Candidate rules from rule_mining_mode=dt/template/hybrid. Template mode uses template completion; DT mode uses original RGCD miners.",
        "rule_mining_mode": str(mining_mode),
        "task_rule_count": int(len(guidance_data.get("rules", []) or [])),
        "destination_rule_count": int(len(destination_guidance_data.get("rules", []) or [])),
        "aqc_rule_count": int(len(aqc_guidance_data.get("rules", []) or [])),
        "template_rule_count": int(len(template_guidance_data.get("rules", []) or [])),
        "n_rules_before_scenario_quota": int(n_before_quota),
        "n_rules": int(len(combined_rules)),
        "rules": combined_rules,
        "scenario_candidate_quota": bool(getattr(args, "scenario_candidate_quota", False) or getattr(args, "scenario_wise_rules", False)),
        "max_rules_per_scenario": int(getattr(args, "max_rules_per_scenario", 30)),
        "max_rules_per_stage_scenario": int(getattr(args, "max_rules_per_stage_scenario", 10)),
        "family_candidate_quota": bool(getattr(args, "family_candidate_quota", False)),
        "family_quotas": {
            "load_pressure": int(getattr(args, "family_quota_load_pressure", 2)),
            "unload_pressure": int(getattr(args, "family_quota_unload_pressure", 2)),
            "truck_wait": int(getattr(args, "family_quota_truck_wait", getattr(args, "truck_wait_candidate_quota", 2))),
            "destination": int(getattr(args, "family_quota_destination", 2)),
            "aqc": int(getattr(args, "family_quota_aqc", 2)),
            "global": int(getattr(args, "family_quota_global", 2)),
            "other": int(getattr(args, "family_quota_other", 0)),
        },
        "carryover_selected_rules": bool(getattr(args, "carryover_selected_rules", False)),
        "n_carryover_rules_added": int(n_carryover_rules),
        "carryover_rules_path": str(carryover_rules_path_used),
        "task_rules_meta": {k: v for k, v in guidance_data.items() if k != "rules"},
        "destination_rules_meta": {k: v for k, v in destination_guidance_data.items() if k != "rules"},
        "aqc_rules_meta": {k: v for k, v in aqc_guidance_data.items() if k != "rules"},
        "template_rules_meta": {k: v for k, v in template_guidance_data.items() if k != "rules"},
    }
    candidate_path = round_dir / "candidate_guidance_rules.json"
    save_guidance_rules(combined_guidance_data, candidate_path)
    save_rules_by_scenario_json(
        combined_rules,
        round_dir / "candidate_guidance_rules_by_scenario.json",
        method="scenario_wise_candidate_rgcd_rules",
        extra={"source_candidate_rules": str(candidate_path)},
    )
    return {
        "trajectories_path": str(traj_path),
        "distilled_rules_path": str(distilled_path),
        "candidate_rules_path": str(candidate_path),
        "task_candidate_rules_path": str(task_candidate_path),
        "template_candidate_rules_path": str(template_completion_path),
        "rule_mining_mode": str(mining_mode),
        "n_samples": int(n_samples),
        "n_raw_rules": int(rules_data.get("n_rules", 0)),
        "n_task_candidate_rules": int(guidance_data.get("n_rules", 0)),
        "n_destination_candidate_samples": int(destination_info.get("n_destination_candidate_samples", 0)),
        "n_destination_raw_rules": int(destination_info.get("n_destination_raw_rules", 0)),
        "n_destination_guidance_rules": int(destination_info.get("n_destination_guidance_rules", 0)),
        "n_aqc_candidate_samples": int(aqc_info.get("n_aqc_candidate_samples", 0)),
        "n_aqc_raw_rules": int(aqc_info.get("n_aqc_raw_rules", 0)),
        "n_aqc_guidance_rules": int(aqc_info.get("n_aqc_guidance_rules", 0)),
        "n_template_guidance_rules": int(template_guidance_data.get("n_rules", 0)),
        "n_candidate_rules": int(combined_guidance_data.get("n_rules", 0)),
        "n_carryover_rules_added": int(n_carryover_rules),
        "carryover_rules_path": str(carryover_rules_path_used),
        **destination_info,
        **aqc_info,
    }




def _scenario_mean_obj(rows: Sequence[Dict[str, Any]], scenario: str) -> float:
    vals = [float(r.get("obj", 0.0)) for r in rows if str(r.get("scenario", "unknown")) == str(scenario)]
    if not vals:
        return float("inf")
    return float(sum(vals) / len(vals))


def _global_scenario_safe_reasons(args, baseline_rows: Sequence[Dict[str, Any]], trial_rows: Sequence[Dict[str, Any]], target_scenario: str) -> List[str]:
    """Check that non-target scenarios are not strongly harmed."""
    reasons: List[str] = []
    max_other = float(getattr(args, "max_other_scenario_worsen", getattr(args, "max_scenario_mean_worsen", 30.0)))
    by_base = {s: _scenario_mean_obj(baseline_rows, s) for s in SCENARIO_ORDER}
    by_trial = {s: _scenario_mean_obj(trial_rows, s) for s in SCENARIO_ORDER}
    for s in SCENARIO_ORDER:
        if not np.isfinite(by_base.get(s, float("inf"))) or not np.isfinite(by_trial.get(s, float("inf"))):
            continue
        delta = by_trial[s] - by_base[s]
        if s != str(target_scenario) and delta > max_other:
            reasons.append(f"other_scenario_worsen {s}: {delta:.4f} > {max_other}")
    return reasons




def _rule_family_key(rule: Dict[str, Any]) -> str:
    tid = str(rule.get("template_id", "") or "").strip()
    if tid:
        return tid
    rid = str(rule.get("id", rule.get("rule_id", "")) or "")
    if "__q" in rid:
        return rid.split("__q", 1)[0]
    return rid


def _history_score_for_rule(history: Sequence[Dict[str, Any]], rid: str, scenario: Optional[str] = None) -> Dict[str, Any]:
    """Return the best accepted history row for rid. Lower score_delta is better."""
    rows = []
    for h in history or []:
        if not bool(h.get("accepted", False)):
            continue
        if str(h.get("try_rule", "")) != str(rid):
            continue
        if scenario is not None and str(h.get("scenario", "")) != str(scenario):
            continue
        rows.append(h)
    if not rows:
        return {}
    def key(h: Dict[str, Any]):
        # Prefer risk-aware score improvement; fallback to target_delta.
        return (
            float(h.get("target_score_delta", h.get("score_delta", 1e18)) or 1e18),
            float(h.get("target_delta", h.get("mean_delta_obj", 1e18)) or 1e18),
        )
    return dict(sorted(rows, key=key)[0])


def _attach_rule_memory(rule: Dict[str, Any], hist: Dict[str, Any]) -> Dict[str, Any]:
    r = annotate_rule(dict(rule))
    if not hist:
        return r
    mem = dict(r.get("memory", {}) or {})
    # Store compact validation evidence for Trend-4 retrieval. Missing fields are safe.
    for src, dst in [
        ("scenario", "validated_in_scenario"),
        ("target_delta", "target_delta"),
        ("target_score_delta", "target_score_delta"),
        ("target_risk_aware_score", "target_risk_aware_score"),
        ("target_best_score_before", "target_best_score_before"),
        ("local_p_worsen", "p_worsen"),
        ("local_cvar_worsen", "cvar_worsen"),
        ("local_max_case_worsen", "max_case_worsen"),
        ("local_avg_trigger_strength_sum", "avg_trigger_strength_sum"),
        ("local_avg_high_strength_triggered", "avg_high_strength_triggered"),
        ("local_avg_triggered", "avg_triggered"),
        ("local_max_case_triggered", "max_case_triggered"),
        ("local_improved_cases", "improved_cases"),
        ("local_worsened_cases", "worsened_cases"),
    ]:
        if src in hist:
            mem[dst] = hist.get(src)
    r["memory"] = mem
    return r


def _finalize_selected_rule_memory(
    args,
    selected_by_scenario: Dict[str, List[str]],
    rule_by_id: Dict[str, Dict[str, Any]],
    history: Sequence[Dict[str, Any]],
) -> tuple[Dict[str, List[str]], List[Dict[str, Any]], Dict[str, Any]]:
    """De-duplicate selected rules and build a Trend-4 rule memory bank.

    - exact duplicate rule_id is removed;
    - global/all rules are stored once under selected_by_scenario["all"];
    - optional template-family dedup keeps the best validation row per
      scenario/stage/template_id, preventing q/beta variants from stacking.
    """
    enable_dedup = bool(getattr(args, "dedup_selected_rules", False))
    dedup_template = bool(getattr(args, "dedup_selected_by_template", False))
    move_global = bool(getattr(args, "dedup_global_rules", False)) or enable_dedup

    out_map: Dict[str, List[str]] = {s: [] for s in SCENARIO_ORDER}
    out_map["all"] = []
    records: List[Dict[str, Any]] = []
    seen_exact = set()

    for scenario in list(SCENARIO_ORDER) + ["all"]:
        for rid in list(selected_by_scenario.get(scenario, []) or []):
            rid = str(rid)
            if rid not in rule_by_id:
                continue
            rule = rule_by_id[rid]
            bucket = "all" if (move_global and is_global_rule(rule)) else scenario
            exact_key = (bucket, rid) if not move_global else ("all" if bucket == "all" else bucket, rid)
            if enable_dedup and exact_key in seen_exact:
                continue
            seen_exact.add(exact_key)
            hist = _history_score_for_rule(history, rid, None if bucket == "all" else scenario)
            if not hist:
                hist = _history_score_for_rule(history, rid, None)
            rec = {
                "bucket": bucket,
                "rid": rid,
                "rule": _attach_rule_memory(rule, hist),
                "history": hist,
            }
            records.append(rec)

    if enable_dedup and dedup_template:
        best: Dict[tuple, Dict[str, Any]] = {}
        for rec in records:
            r = rec["rule"]
            key = (str(rec["bucket"]), str(r.get("rule_stage", infer_rule_stage(r))), _rule_family_key(r))
            old = best.get(key)
            def score(x: Dict[str, Any]):
                h = x.get("history", {}) or {}
                return (
                    float(h.get("target_score_delta", h.get("score_delta", 1e18)) or 1e18),
                    float(h.get("target_delta", h.get("mean_delta_obj", 1e18)) or 1e18),
                    -float(r.get("confidence", 0.0) or 0.0),
                    -float(r.get("support", 0.0) or 0.0),
                )
            if old is None or score(rec) < score(old):
                best[key] = rec
        # Preserve a stable order: concrete scenarios first, then global rules.
        order = {s: i for i, s in enumerate(SCENARIO_ORDER + ["all", "unknown"])}
        records = sorted(best.values(), key=lambda x: (order.get(str(x["bucket"]), 99), str(x["rid"])))

    final_map: Dict[str, List[str]] = {s: [] for s in SCENARIO_ORDER}
    final_map["all"] = []
    final_rules: List[Dict[str, Any]] = []
    final_seen = set()
    for rec in records:
        bucket = str(rec["bucket"])
        rid = str(rec["rid"])
        if (bucket, rid) in final_seen:
            continue
        final_seen.add((bucket, rid))
        final_map.setdefault(bucket, []).append(rid)
        final_rules.append(rec["rule"])

    report = {
        "enabled": bool(enable_dedup),
        "dedup_selected_by_template": bool(dedup_template),
        "dedup_global_rules": bool(move_global),
        "n_rules_before": int(sum(len(v) for v in selected_by_scenario.values())),
        "n_rules_after": int(sum(len(v) for v in final_map.values())),
        "n_rules_by_bucket_after": {k: len(v) for k, v in final_map.items()},
    }
    return final_map, final_rules, report


def _save_rule_memory_bank(
    rules: Sequence[Dict[str, Any]],
    selected_by_scenario: Dict[str, List[str]],
    path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    obj = {
        "method": "contextual_rule_memory_bank",
        "n_rules": int(len(rules)),
        "scenario_order": SCENARIO_ORDER,
        "selected_rule_ids_by_scenario": selected_by_scenario,
        "rules": [annotate_rule(r) for r in rules],
    }
    if extra:
        obj.update(extra)
    save_json(obj, str(path))
    return str(path)

def greedy_select_rules_by_scenario(args, model_path: str, packer: ObsPacker, val_files: Sequence[str],
                                    candidate_rules_path: str, round_dir: Path) -> Dict[str, Any]:
    """Scenario-wise VAL rule selection.

    Each scenario has its own selected rule set. A candidate rule is mainly accepted by
    target-scenario improvement; optionally, the selected-by-scenario rule library is
    rechecked on the full validation set to avoid cross-scenario damage.
    """
    print("\n" + "=" * 72)
    print("[Scenario Greedy:VAL] 分场景筛选规则：target scenario improves + global safe check")
    print(f"[Scenario Greedy:VAL] val instances={len(val_files)}")
    print("=" * 72)

    base_engine = load_rule_guidance(candidate_rules_path, min_confidence=0.0, **rule_guidance_engine_kwargs(args))
    all_rules = [annotate_rule(r) for r in base_engine.rules]
    rule_by_id = {str(r.get("id")): r for r in all_rules}
    val_by_scenario = group_files_by_scenario(val_files)
    model = load_model(model_path, args.algo, args.device)

    # Full validation baseline without rules.
    full_baseline_rows = evaluate_set(
        model, args.algo, val_files, packer, args.seed,
        bool(args.disturbance), args.intensity,
        guidance_engine=None,
        deterministic=not args.stochastic,
    )
    save_csv(full_baseline_rows, str(round_dir / "val_scenario_greedy_baseline_rows.csv"))
    base_sm = summarize(full_baseline_rows)
    save_json(base_sm, str(round_dir / "val_scenario_greedy_baseline_summary.json"))
    save_scenario_summary(
        full_baseline_rows,
        round_dir / "val_scenario_greedy_baseline_by_scenario.csv",
        round_dir / "val_scenario_greedy_baseline_by_scenario.json",
    )
    current_global_rows = list(full_baseline_rows)

    selected_by_scenario: Dict[str, List[str]] = {s: [] for s in SCENARIO_ORDER}
    selected_by_scenario["all"] = []
    history: List[Dict[str, Any]] = [{"step": 0, "scenario": "ALL", "try_rule": "BASELINE", "accepted": True, **base_sm}]

    # Group candidate ids by inferred target scenario.
    candidate_ids_by_scenario: Dict[str, List[str]] = {s: [] for s in SCENARIO_ORDER}
    candidate_ids_by_scenario["unknown"] = []
    for rid in base_engine.active_rule_ids():
        r = rule_by_id.get(str(rid), {})
        s = infer_rule_scenario(r)
        # scenario="all" 的模板规则（如 urgent、broken AQC、not_arrived 等）
        # 需要分别在每个场景的 VAL 子集上接受/拒绝；否则会被分到 all 桶但主循环不处理。
        if s in ("all", "*", ""):
            for ss in SCENARIO_ORDER:
                candidate_ids_by_scenario.setdefault(ss, []).append(str(rid))
        else:
            candidate_ids_by_scenario.setdefault(s, []).append(str(rid))

    print("Candidate rules by scenario:", {s: len(v) for s, v in candidate_ids_by_scenario.items()})
    step = 1

    for scenario in SCENARIO_ORDER:
        s_files = val_by_scenario.get(scenario, [])
        s_candidates = candidate_ids_by_scenario.get(scenario, [])
        if not s_files:
            print(f"[Scenario {scenario}] no val files, skip.")
            continue
        if not s_candidates:
            print(f"[Scenario {scenario}] no candidate rules, skip.")
            continue

        # local baseline for this scenario under current scenario-specific rules plus global memory rules.
        current_ids_for_scenario = list(selected_by_scenario.get("all", [])) + list(selected_by_scenario.get(scenario, []))
        local_current_engine = base_engine.subset(current_ids_for_scenario)
        local_baseline_rows = evaluate_set(
            model, args.algo, s_files, packer, args.seed,
            bool(args.disturbance), args.intensity,
            guidance_engine=local_current_engine if current_ids_for_scenario else None,
            deterministic=not args.stochastic,
        )
        local_best = float(summarize(local_baseline_rows).get("mean_obj", float("inf")))
        local_best_score = float(local_best)
        print(f"\n[Scenario {scenario}] candidates={len(s_candidates)}, local baseline mean_obj={local_best:.4f}" +
              (f", risk_score={local_best_score:.4f}" if risk_budget_enabled(args) else ""))

        for rid in s_candidates:
            rule_obj = rule_by_id.get(rid, {})
            target_bucket = "all" if (bool(getattr(args, "dedup_global_rules", False)) and is_global_rule(rule_obj)) else scenario
            trial_ids_s = list(selected_by_scenario.get("all", [])) + list(selected_by_scenario.get(scenario, []))
            if rid not in trial_ids_s:
                trial_ids_s.append(rid)
            trial_engine_s = base_engine.subset(trial_ids_s)
            rule_stage = infer_rule_stage(rule_obj)
            print(f"\n[Scenario Try:VAL] scenario={scenario} + {rid} stage={rule_stage} bucket={target_bucket} size={len(trial_ids_s)}")
            rows_s = evaluate_set(
                model, args.algo, s_files, packer, args.seed,
                bool(args.disturbance), args.intensity,
                guidance_engine=trial_engine_s,
                deterministic=not args.stochastic,
            )
            sm_s = summarize(rows_s)
            mean_s = float(sm_s.get("mean_obj", float("inf")))
            metrics_s = compute_rule_safety_metrics(
                local_baseline_rows, rows_s,
                eps=float(args.min_delta),
                cvar_alpha=float(getattr(args, "risk_cvar_alpha", 0.25)),
            )
            safe_ok_s, safe_reasons_s = passes_safe_filter(args, metrics_s)
            trial_score_s, score_parts_s = compute_risk_aware_score(args, mean_s, metrics_s)
            if risk_budget_enabled(args):
                min_score_improve = float(getattr(args, "min_risk_score_improve", getattr(args, "min_target_scenario_improve", 5.0)))
                target_improved = trial_score_s < local_best_score - min_score_improve
            else:
                target_improved = mean_s < local_best - float(getattr(args, "min_target_scenario_improve", 5.0))

            accepted = False
            global_safe_ok = True
            global_reasons: List[str] = []
            global_mean = ""
            if target_improved and safe_ok_s:
                # Optional full-val global safety recheck using scenario-aware rule library.
                if bool(getattr(args, "scenario_global_safety_check", True)):
                    trial_map = {k: list(v) for k, v in selected_by_scenario.items()}
                    if target_bucket == "all":
                        trial_map["all"] = list(selected_by_scenario.get("all", [])) + ([rid] if rid not in selected_by_scenario.get("all", []) else [])
                    else:
                        trial_map[scenario] = list(selected_by_scenario.get(scenario, [])) + ([rid] if rid not in selected_by_scenario.get(scenario, []) else [])
                    trial_scenario_engine = make_scenario_aware_engine(all_rules, trial_map, engine_kwargs=rule_guidance_engine_kwargs(args))
                    trial_global_rows = evaluate_set(
                        model, args.algo, val_files, packer, args.seed,
                        bool(args.disturbance), args.intensity,
                        guidance_engine=trial_scenario_engine,
                        deterministic=not args.stochastic,
                    )
                    global_metrics = compute_rule_safety_metrics(
                        current_global_rows, trial_global_rows,
                        eps=float(args.min_delta),
                        cvar_alpha=float(getattr(args, "risk_cvar_alpha", 0.25)),
                    )
                    global_safe_ok, global_reasons = passes_safe_filter(args, global_metrics)
                    global_reasons.extend(_global_scenario_safe_reasons(args, current_global_rows, trial_global_rows, scenario))
                    global_mean = summarize(trial_global_rows).get("mean_obj", "")
                    if global_safe_ok and not global_reasons:
                        accepted = True
                        current_global_rows = trial_global_rows
                else:
                    accepted = True

            hist_row = {
                "step": step,
                "scenario": scenario,
                "rule_stage": rule_stage,
                "try_rule": rid,
                "accepted": bool(accepted),
                "target_improved": bool(target_improved),
                "target_mean_obj": mean_s,
                "target_best_before": local_best,
                "target_delta": mean_s - local_best,
                "risk_budget_greedy": bool(risk_budget_enabled(args)),
                "target_risk_aware_score": float(trial_score_s),
                "target_best_score_before": float(local_best_score),
                "target_score_delta": float(trial_score_s - local_best_score),
                **score_parts_s,
                "local_safe_ok": bool(safe_ok_s),
                "local_safe_reasons": "; ".join(safe_reasons_s),
                "global_safe_ok": bool(global_safe_ok),
                "global_safe_reasons": "; ".join(global_reasons),
                "global_mean_obj_if_checked": global_mean,
                **{f"local_{k}": v for k, v in metrics_s.items() if k != "case_details"},
                **{f"target_{k}": v for k, v in sm_s.items()},
            }
            history.append(hist_row)
            # Case-level local diagnostics for this candidate.
            diag_rows = []
            for d in metrics_s.get("case_details", []):
                row = dict(d)
                # d 本身可能已经包含 scenario/accepted 等字段；先展开 d，再覆盖我们要写入的诊断标签，
                # 避免 dict(..., scenario=..., **d) 造成 duplicate keyword 报错。
                row.update({
                    "rule_id": rid,
                    "target_scenario": scenario,
                    "scenario_group": scenario,
                    "accepted": bool(accepted),
                })
                diag_rows.append(row)
            write_csv(diag_rows, round_dir / "safe_case_diagnostics" / f"{step:03d}_{scenario}_{rid}.csv")
            if risk_budget_enabled(args):
                print(
                    f"Scenario try mean_obj={mean_s:.4f}, best={local_best:.4f}, "
                    f"risk_score={trial_score_s:.4f}, best_score={local_best_score:.4f}, "
                    f"target_improved={target_improved}, catastrophic_safe={safe_ok_s}, accepted={accepted}"
                )
                print(
                    f"  Risk metrics: p_worsen={metrics_s.get('p_worsen', 0):.3f}, "
                    f"cvar_worsen={metrics_s.get('cvar_worsen', 0):.2f}, "
                    f"max_worsen={metrics_s.get('max_case_worsen', 0):.2f}, "
                    f"avg_strength_sum={metrics_s.get('avg_trigger_strength_sum', 0):.2f}, "
                    f"high_strength={metrics_s.get('avg_high_strength_triggered', 0):.2f}"
                )
            else:
                print(
                    f"Scenario try mean_obj={mean_s:.4f}, best={local_best:.4f}, "
                    f"target_improved={target_improved}, local_safe={safe_ok_s}, accepted={accepted}"
                )
            if safe_reasons_s:
                print("  Local reject reasons: " + "; ".join(safe_reasons_s))
            if global_reasons:
                print("  Global reject reasons: " + "; ".join(global_reasons))

            if accepted:
                selected_by_scenario.setdefault(target_bucket, [])
                if rid not in selected_by_scenario[target_bucket]:
                    selected_by_scenario[target_bucket].append(rid)
                local_best = mean_s
                local_best_score = trial_score_s if risk_budget_enabled(args) else mean_s
                local_baseline_rows = rows_s
            step += 1

    raw_selected_by_scenario = {k: list(v) for k, v in selected_by_scenario.items()}
    selected_by_scenario, selected_rules, dedup_report = _finalize_selected_rule_memory(
        args, selected_by_scenario, rule_by_id, history
    )
    selected_flat_ids: List[str] = []
    for s in SCENARIO_ORDER + ["all"]:
        selected_flat_ids.extend(selected_by_scenario.get(s, []))

    selected_engine = make_scenario_aware_engine(all_rules, selected_by_scenario, engine_kwargs=rule_guidance_engine_kwargs(args))

    # If de-duplication or Trend-4 retrieval changed the final executable rule set,
    # re-evaluate the final selected memory on the full validation split so saved
    # rows/summaries match exactly what will be used in test/feedback training.
    if selected_engine is not None:
        current_global_rows = evaluate_set(
            model, args.algo, val_files, packer, args.seed,
            bool(args.disturbance), args.intensity,
            guidance_engine=selected_engine,
            deterministic=not args.stochastic,
        )
    else:
        current_global_rows = list(full_baseline_rows)
    best_sm = summarize(current_global_rows)

    memory_bank_path = _save_rule_memory_bank(
        selected_rules,
        selected_by_scenario,
        round_dir / "rule_memory_bank.json",
        extra={
            "rule_memory_retrieval": rule_guidance_engine_kwargs(args),
            "dedup_report": dedup_report,
            "raw_selected_rule_ids_by_scenario": raw_selected_by_scenario,
        },
    )

    selected_data = {
        "method": "scenario_wise_greedy_selected_rgcd_rules_on_validation_set",
        "source_candidate_rules": str(candidate_rules_path),
        "selection_split": "val",
        "scenario_wise": True,
        "baseline_mean_obj": float(base_sm.get("mean_obj", float("nan"))),
        "best_mean_obj": float(best_sm.get("mean_obj", float("nan"))),
        "selected_rule_ids_by_scenario": selected_by_scenario,
        "raw_selected_rule_ids_by_scenario": raw_selected_by_scenario,
        "selected_rule_ids": selected_flat_ids,
        "n_selected_rules_by_scenario": {s: len(selected_by_scenario.get(s, [])) for s in SCENARIO_ORDER + ["all"]},
        "n_selected_rules": len(selected_flat_ids),
        "rules": selected_rules,
        "rule_memory_bank_path": str(memory_bank_path),
        "dedup_report": dedup_report,
        "history": history,
        "soft_rule_guidance": rule_guidance_engine_kwargs(args),
        "safe_filter": {
            "enabled": bool(args.safe_rule_filter),
            "scenario_global_safety_check": bool(getattr(args, "scenario_global_safety_check", True)),
            "min_target_scenario_improve": float(getattr(args, "min_target_scenario_improve", 5.0)),
            "max_other_scenario_worsen": float(getattr(args, "max_other_scenario_worsen", 30.0)),
            "risk_budget_greedy": bool(risk_budget_enabled(args)),
            "min_risk_score_improve": float(getattr(args, "min_risk_score_improve", 5.0)),
            "risk_catastrophic_worsen": float(getattr(args, "risk_catastrophic_worsen", 1500.0)),
            "risk_cvar_alpha": float(getattr(args, "risk_cvar_alpha", 0.25)),
        },
    }
    selected_path = round_dir / "selected_rules_from_val.json"
    save_json(selected_data, str(selected_path))
    save_json(selected_data, str(round_dir / "selected_rules.json"))
    save_rules_by_scenario_json(
        selected_rules,
        round_dir / "selected_rules_by_scenario.json",
        method="scenario_wise_selected_rgcd_rules",
        extra={
            "selected_rule_ids_by_scenario": selected_by_scenario,
            "raw_selected_rule_ids_by_scenario": raw_selected_by_scenario,
            "dedup_report": dedup_report,
            "rule_memory_bank_path": str(memory_bank_path),
        },
    )
    write_csv(history, round_dir / "val_greedy_history.csv")
    save_csv(current_global_rows, str(round_dir / "val_scenario_greedy_selected_rows.csv"))
    save_scenario_summary(
        current_global_rows,
        round_dir / "val_scenario_greedy_selected_by_scenario.csv",
        round_dir / "val_scenario_greedy_selected_by_scenario.json",
    )

    print(f"[Scenario Greedy:VAL] selected={len(selected_flat_ids)}, by_scenario={selected_data['n_selected_rules_by_scenario']}, best_mean_obj={selected_data['best_mean_obj']:.4f}")
    if dedup_report.get("enabled"):
        print(f"[Dedup] before={dedup_report.get('n_rules_before')} after={dedup_report.get('n_rules_after')} by_bucket={dedup_report.get('n_rules_by_bucket_after')}")
    return {
        "selected_rules_path": str(selected_path),
        "selected_engine": selected_engine,
        "n_selected_rules": len(selected_flat_ids),
        "baseline_mean_obj": float(base_sm.get("mean_obj", float("nan"))),
        "best_mean_obj": float(best_sm.get("mean_obj", float("nan"))),
        "history": history,
        "selected_rule_ids": selected_flat_ids,
        "selected_rule_ids_by_scenario": selected_by_scenario,
        "rule_memory_bank_path": str(memory_bank_path),
        "dedup_report": dedup_report,
    }

def greedy_select_rules(args, model_path: str, packer: ObsPacker, val_files: Sequence[str],
                        candidate_rules_path: str, round_dir: Path) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print("[Greedy:VAL] 在验证集逐条评价候选规则，mean_obj 降低才保留")
    print(f"[Greedy:VAL] val instances={len(val_files)}")
    print("=" * 72)
    engine = load_rule_guidance(candidate_rules_path, min_confidence=0.0, **rule_guidance_engine_kwargs(args))
    candidate_ids = engine.active_rule_ids()
    model = load_model(model_path, args.algo, args.device)

    baseline_rows = evaluate_set(
        model, args.algo, val_files, packer, args.seed,
        bool(args.disturbance), args.intensity,
        guidance_engine=None,
        deterministic=not args.stochastic,
    )
    save_csv(baseline_rows, str(round_dir / "val_greedy_baseline_rows.csv"))
    base_sm = summarize(baseline_rows)
    save_json(base_sm, str(round_dir / "val_greedy_baseline_summary.json"))
    best_obj = float(base_sm.get("mean_obj", float("inf")))
    best_score = float(best_obj)
    selected: List[str] = []
    history: List[Dict[str, Any]] = [{"step": 0, "try_rule": "BASELINE", "accepted": True, **base_sm}]

    print(f"候选规则数：{len(candidate_ids)}，VAL baseline mean_obj={best_obj:.4f}")
    for rid in candidate_ids:
        trial_ids = selected + [rid]
        trial_engine = engine.subset(trial_ids)
        print(f"\n[Greedy Try:VAL] + {rid}  size={len(trial_ids)}")
        rows = evaluate_set(
            model, args.algo, val_files, packer, args.seed,
            bool(args.disturbance), args.intensity,
            guidance_engine=trial_engine,
            deterministic=not args.stochastic,
        )
        sm = summarize(rows)
        mean_obj = float(sm.get("mean_obj", float("inf")))
        metrics = compute_rule_safety_metrics(
            baseline_rows, rows,
            eps=float(args.min_delta),
            cvar_alpha=float(getattr(args, "risk_cvar_alpha", 0.25)),
        )
        safe_ok, safe_reasons = passes_safe_filter(args, metrics)
        trial_score, score_parts = compute_risk_aware_score(args, mean_obj, metrics)
        if risk_budget_enabled(args):
            improved_mean = trial_score < best_score - float(getattr(args, "min_risk_score_improve", args.min_delta))
        else:
            improved_mean = mean_obj < best_obj - float(args.min_delta)
        accepted = bool(improved_mean and safe_ok)
        hist_row = {
            "step": len(history),
            "try_rule": rid,
            "accepted": bool(accepted),
            "mean_improved": bool(improved_mean),
            "safe_ok": bool(safe_ok),
            "safe_reasons": "; ".join(safe_reasons),
            "risk_budget_greedy": bool(risk_budget_enabled(args)),
            "risk_aware_score": float(trial_score),
            "best_score_before": float(best_score),
            "score_delta": float(trial_score - best_score),
            **score_parts,
            **{k: v for k, v in metrics.items() if k != "case_details"},
            **sm,
        }
        history.append(hist_row)
        # 保存每条候选规则/组合在每个验证 case 上的诊断，方便排查“哪一个 case 被伤害”。
        case_diag_path = round_dir / "safe_case_diagnostics" / f"{len(history)-1:03d}_{rid}.csv"
        diag_rows = []
        for d in metrics.get("case_details", []):
            row = dict(d)
            # d 里可能已有同名字段；用 update 明确覆盖诊断标签，避免重复关键字报错。
            row.update({
                "rule_id": rid,
                "accepted": bool(accepted),
                "safe_ok": bool(safe_ok),
            })
            diag_rows.append(row)
        write_csv(diag_rows, case_diag_path)
        print(
            f"VAL try mean_obj={mean_obj:.4f}, best={best_obj:.4f}, "
            + (f"risk_score={trial_score:.4f}, best_score={best_score:.4f}, " if risk_budget_enabled(args) else "")
            + f"mean_improved={improved_mean}, safe_ok={safe_ok}, accepted={accepted}"
        )
        print(
            f"  Safe metrics: max_worsen={metrics['max_case_worsen']:.4f}, "
            f"avg_triggered={metrics['avg_triggered']:.2f}, "
            f"max_triggered={metrics['max_case_triggered']:.0f}, "
            f"improved/worsened={metrics['improved_cases']}/{metrics['worsened_cases']}"
        )
        if safe_reasons:
            print("  Reject reasons: " + "; ".join(safe_reasons))
        if accepted:
            selected.append(rid)
            best_obj = mean_obj
            best_score = trial_score if risk_budget_enabled(args) else mean_obj

    selected_engine = engine.subset(selected)
    selected_rules = [r for r in engine.rules if str(r.get("id")) in set(selected)]
    selected_data = {
        "method": "greedy_selected_rgcd_rules_on_validation_set",
        "source_candidate_rules": str(candidate_rules_path),
        "selection_split": "val",
        "baseline_mean_obj": float(base_sm.get("mean_obj", float("nan"))),
        "best_mean_obj": float(best_obj),
        "selected_rule_ids": selected,
        "n_selected_rules": len(selected),
        "rules": selected_rules,
        "history": history,
        "soft_rule_guidance": rule_guidance_engine_kwargs(args),
        "safe_filter": {
            "enabled": bool(args.safe_rule_filter),
            "max_case_worsen": float(args.max_case_worsen),
            "max_avg_triggered": float(args.max_avg_triggered),
            "max_case_triggered": float(args.max_case_triggered),
            "require_improved_ge_worsened": bool(args.require_improved_ge_worsened),
            "risk_budget_greedy": bool(risk_budget_enabled(args)),
            "risk_catastrophic_worsen": float(getattr(args, "risk_catastrophic_worsen", 1500.0)),
            "risk_cvar_alpha": float(getattr(args, "risk_cvar_alpha", 0.25)),
        },
    }
    selected_path = round_dir / "selected_rules_from_val.json"
    save_json(selected_data, str(selected_path))
    # 兼容旧文件名：部分旧脚本可能还读取 selected_rules.json。
    save_json(selected_data, str(round_dir / "selected_rules.json"))
    write_csv(history, round_dir / "val_greedy_history.csv")
    print(f"[Greedy:VAL] selected={len(selected)}, best_mean_obj={best_obj:.4f}")
    return {
        "selected_rules_path": str(selected_path),
        "selected_rule_ids": selected,
        "n_selected_rules": len(selected),
        "best_mean_obj": float(best_obj),
        "baseline_mean_obj": float(base_sm.get("mean_obj", float("nan"))),
        "selected_engine": selected_engine if selected else None,
    }


def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_files = collect_json_files(args.train_instances, args.max_train_instances)
    val_path = args.val_instances or args.test_instances
    val_files = collect_json_files(val_path, args.max_val_instances)
    test_files = collect_json_files(args.test_instances, args.max_instances)
    packer = ObsPacker(**OBS_PACKER_DEFAULTS, enable_disturbance=bool(args.disturbance))
    trainer_cls = get_trainer(args.algo)

    if args.val_instances is None:
        print("\n⚠️  未提供 --val_instances：当前会临时用 test_instances 做规则筛选。正式实验请提供独立验证集。")

    print("\n" + "=" * 80)
    print("v14-CoEvo-RGCD：train挖规则 / val筛规则 / test最终评价")
    print(f"算法          : {args.algo}")
    print(f"训练实例      : {len(train_files)}  ({args.train_instances})")
    print(f"验证实例      : {len(val_files)}  ({val_path})")
    print(f"测试实例      : {len(test_files)}  ({args.test_instances})")
    print(f"闭环轮数      : {args.n_rounds} （不含 Round 0）")
    print(f"Round0 steps  : {args.round_0_timesteps}")
    print(f"每轮 steps    : {args.per_round_timesteps}")
    print(f"采集episodes  : {args.n_collect_episodes}")
    print(f"规则过滤      : conf>={args.min_confidence}, support>={args.min_support}, coverage>={args.min_coverage}, conditions={args.min_rule_conditions}~{args.max_rule_conditions}")
    print(f"规则强度      : guidance_adjust={args.guidance_adjust}")
    print(f"软规则触发    : {'ON' if args.soft_rule_guidance else 'OFF'}" + (
        f"  temperature={args.soft_rule_temperature}, min_strength={args.soft_rule_min_strength}, aggregation={args.soft_rule_aggregation}"
        if args.soft_rule_guidance else ""
    ))
    print(f"Destination规则: {'ON' if args.enable_destination_rules else 'OFF'}" + (
        f" (episodes={args.n_destination_collect_episodes or args.n_collect_episodes}, adjust={args.destination_guidance_adjust})"
        if args.enable_destination_rules else ""
    ))
    print(f"AQC规则挖掘   : {'ON' if args.enable_aqc_rules else 'OFF'}" + (
        f" (episodes={args.n_aqc_collect_episodes or args.n_collect_episodes}, adjust={args.aqc_guidance_adjust})"
        if args.enable_aqc_rules else ""
    ))
    print(f"安全筛选      : {'ON' if args.safe_rule_filter else 'OFF'}" + (
        f"  max_worsen<={args.max_case_worsen}, avg_trig<={args.max_avg_triggered}, max_trig<={args.max_case_triggered}, improved>=worsened={args.require_improved_ge_worsened}"
        if args.safe_rule_filter else ""
    ))
    print(f"风险预算贪心  : {'ON' if risk_budget_enabled(args) else 'OFF'}" + (
        f"  min_score_improve={args.min_risk_score_improve}, catastrophic_worsen={args.risk_catastrophic_worsen}, "
        f"cvar_alpha={args.risk_cvar_alpha}, lambdas=(cvar={args.risk_lambda_cvar}, p_w={args.risk_lambda_p_worsen}, "
        f"max_w={args.risk_lambda_max_worsen}, strength={args.risk_lambda_trigger_strength}, high={args.risk_lambda_high_strength})"
        if risk_budget_enabled(args) else ""
    ))
    print(f"规则去重      : {'ON' if args.dedup_selected_rules else 'OFF'}" + (
        f"  by_template={args.dedup_selected_by_template}, global_once={args.dedup_global_rules}"
        if args.dedup_selected_rules else ""
    ))
    print(f"规则记忆检索  : {'ON' if args.rule_memory_retrieval else 'OFF'}" + (
        f"  top_k={args.rule_retrieval_top_k}, dedup_templates={args.rule_retrieval_dedup_templates}"
        if args.rule_memory_retrieval else ""
    ))
    print(f"候选族保底    : {'ON' if getattr(args, 'family_candidate_quota', False) else 'OFF'}" + (
        f"  load={args.family_quota_load_pressure}, unload={args.family_quota_unload_pressure}, "
        f"truck_wait={args.family_quota_truck_wait}, aqc={args.family_quota_aqc}, global={args.family_quota_global}, dest={args.family_quota_destination}"
        if getattr(args, 'family_candidate_quota', False) else ""
    ))
    print(f"规则继承      : {'ON' if getattr(args, 'carryover_selected_rules', False) else 'OFF'}" + (
        f"  max_per_family={args.carryover_max_per_family}" if getattr(args, 'carryover_selected_rules', False) else ""
    ))
    print(f"训练反馈      : {'OFF (--no_train_feedback)' if args.no_train_feedback else 'ON'}")
    print(f"扰动          : {'✅ '+args.intensity if args.disturbance else '❌'}")
    print(f"输出          : {out_dir}")
    print("=" * 80)

    save_json({
        "algo": args.algo,
        "train_instances": args.train_instances,
        "val_instances": val_path,
        "test_instances": args.test_instances,
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "n_test_files": len(test_files),
        "no_train_feedback": bool(args.no_train_feedback),
        "disturbance": bool(args.disturbance),
        "intensity": args.intensity,
        "seed": args.seed,
        "n_collect_episodes": args.n_collect_episodes,
        "collect_max_steps": args.collect_max_steps,
        "distill_max_depth": args.distill_max_depth,
        "distill_min_samples_leaf": args.distill_min_samples_leaf,
        "min_confidence": args.min_confidence,
        "min_support": args.min_support,
        "min_coverage": args.min_coverage,
        "min_rule_conditions": args.min_rule_conditions,
        "max_rule_conditions": args.max_rule_conditions,
        "guidance_adjust": args.guidance_adjust,
        "soft_rule_guidance": bool(getattr(args, "soft_rule_guidance", False)),
        "soft_rule_temperature": float(getattr(args, "soft_rule_temperature", 0.05)),
        "soft_rule_min_strength": float(getattr(args, "soft_rule_min_strength", 0.05)),
        "soft_rule_aggregation": str(getattr(args, "soft_rule_aggregation", "min")),
        "dedup_selected_rules": bool(getattr(args, "dedup_selected_rules", False)),
        "dedup_selected_by_template": bool(getattr(args, "dedup_selected_by_template", False)),
        "dedup_global_rules": bool(getattr(args, "dedup_global_rules", False)),
        "rule_memory_retrieval": bool(getattr(args, "rule_memory_retrieval", False)),
        "rule_retrieval_top_k": int(getattr(args, "rule_retrieval_top_k", 0) or 0),
        "rule_retrieval_dedup_templates": bool(getattr(args, "rule_retrieval_dedup_templates", False)),
        "safe_rule_filter": bool(args.safe_rule_filter),
        "max_case_worsen": args.max_case_worsen,
        "max_avg_triggered": args.max_avg_triggered,
        "max_case_triggered": args.max_case_triggered,
        "require_improved_ge_worsened": bool(args.require_improved_ge_worsened),
        "risk_budget_greedy": bool(getattr(args, "risk_budget_greedy", False)),
        "min_risk_score_improve": float(getattr(args, "min_risk_score_improve", 5.0)),
        "risk_catastrophic_worsen": float(getattr(args, "risk_catastrophic_worsen", 1500.0)),
        "risk_max_completion_drop": float(getattr(args, "risk_max_completion_drop", 0.01)),
        "risk_cvar_alpha": float(getattr(args, "risk_cvar_alpha", 0.25)),
        "risk_lambda_cvar": float(getattr(args, "risk_lambda_cvar", 0.05)),
        "risk_lambda_p_worsen": float(getattr(args, "risk_lambda_p_worsen", 50.0)),
        "risk_lambda_max_worsen": float(getattr(args, "risk_lambda_max_worsen", 0.01)),
        "risk_lambda_trigger_strength": float(getattr(args, "risk_lambda_trigger_strength", 0.02)),
        "risk_lambda_high_strength": float(getattr(args, "risk_lambda_high_strength", 0.05)),
        "risk_lambda_p_improve": float(getattr(args, "risk_lambda_p_improve", 0.0)),
        "scenario_wise_rules": bool(getattr(args, "scenario_wise_rules", False)),
        "scenario_candidate_quota": bool(getattr(args, "scenario_candidate_quota", False)),
        "max_rules_per_scenario": int(getattr(args, "max_rules_per_scenario", 30)),
        "max_rules_per_stage_scenario": int(getattr(args, "max_rules_per_stage_scenario", 10)),
        "family_candidate_quota": bool(getattr(args, "family_candidate_quota", False)),
        "family_quota_load_pressure": int(getattr(args, "family_quota_load_pressure", 2)),
        "family_quota_unload_pressure": int(getattr(args, "family_quota_unload_pressure", 2)),
        "family_quota_truck_wait": int(getattr(args, "family_quota_truck_wait", getattr(args, "truck_wait_candidate_quota", 2))),
        "family_quota_destination": int(getattr(args, "family_quota_destination", 2)),
        "family_quota_aqc": int(getattr(args, "family_quota_aqc", 2)),
        "family_quota_global": int(getattr(args, "family_quota_global", 2)),
        "carryover_selected_rules": bool(getattr(args, "carryover_selected_rules", False)),
        "carryover_max_per_family": int(getattr(args, "carryover_max_per_family", 1)),
        "scenario_global_safety_check": bool(getattr(args, "scenario_global_safety_check", True)),
        "min_target_scenario_improve": float(getattr(args, "min_target_scenario_improve", 5.0)),
        "max_other_scenario_worsen": float(getattr(args, "max_other_scenario_worsen", 30.0)),
    }, str(out_dir / "run_config.json"))

    summary_rows: List[Dict[str, Any]] = []
    t_all = time.time()

    round0_dir = out_dir / "round_0"
    round0_dir.mkdir(exist_ok=True)
    if args.initial_model:
        model_path = args.initial_model
        print(f"[Round 0] 使用已有初始模型：{model_path}")
    else:
        model_path = train_model(
            args, trainer_cls, packer, train_files,
            out_model=round0_dir / "model.zip",
            init_model=None,
            guidance_engine=None,
            timesteps=args.round_0_timesteps,
            timestep_offset=0,
            progress_method="SAC + CoEvo-RGCD-R2",
        )

    round0_summary_path = round0_dir / "test_no_rules_summary.json"
    if bool(getattr(args, "reuse_existing_outputs", False)) and round0_summary_path.exists():
        sm0_test = _load_json_file(round0_summary_path)
        print(f"[Resume] 复用 Round 0 TEST baseline summary → {round0_summary_path}")
    else:
        sm0_test = evaluate_model(
            args, model_path, packer, test_files, guidance_engine=None,
            csv_path=round0_dir / "test_no_rules.csv",
            summary_path=round0_dir / "test_no_rules_summary.json",
            title="[Round 0] TEST 无规则 baseline 评估",
            split_name="test",
        )
    summary_rows.append({
        "round": 0,
        "phase": "round0_baseline",
        "model_path": model_path,
        "knowledge_path": "",
        "n_samples": 0,
        "n_raw_rules": 0,
        "n_candidate_rules": 0,
        "n_selected_rules": 0,
        "val_greedy_baseline_mean_obj": "",
        "val_greedy_best_mean_obj": "",
        "test_no_rules_mean_obj": sm0_test.get("mean_obj"),
        "test_guided_selected_mean_obj": "",
        "new_model_test_no_rules_mean_obj": "",
        "new_model_test_guided_mean_obj": "",
        "train_guidance": False,
    })

    current_model = model_path
    current_knowledge_engine: Optional[RuleGuidanceEngine] = None

    for r in range(1, int(args.n_rounds) + 1):
        round_dir = out_dir / f"round_{r}"
        round_dir.mkdir(exist_ok=True)
        print("\n" + "#" * 80)
        print(f"# Round {r}: TRAIN采集挖规则 → VAL筛规则 → TEST最终评价")
        print("#" * 80)

        mined = None
        if bool(getattr(args, "reuse_mined_rules", False)):
            mined = _reuse_mined_rules_if_available(round_dir)
            if mined is not None:
                print(f"[Resume] 复用已生成候选规则，跳过 TRAIN 采集/挖规则 → {mined['candidate_rules_path']}")
            else:
                print(f"[Resume] 未找到 {round_dir / 'candidate_guidance_rules.json'}，将重新采集/挖规则。")

        if mined is None:
            prev_selected_rules_path = (out_dir / f"round_{r-1}" / "selected_rules_by_scenario.json") if r > 1 else None
            mined = collect_and_mine_rules(
                args, trainer_cls, packer, train_files,
                source_model=current_model,
                round_dir=round_dir,
                collect_guidance=current_knowledge_engine,
                prev_selected_rules_path=prev_selected_rules_path,
            )

        # SA-CoEvo-RGCD: when enabled, do true scenario-wise independent greedy selection:
        #   1) candidates are grouped by target_scenario;
        #   2) every scenario is selected on its own VAL subset;
        #   3) a ScenarioAwareRuleGuidanceEngine is returned, so TEST and feedback training
        #      dynamically use only the selected rule pool of the current instance scenario.
        if bool(getattr(args, "scenario_wise_rules", False)):
            selected_info = greedy_select_rules_by_scenario(
                args, current_model, packer, val_files,
                candidate_rules_path=mined["candidate_rules_path"],
                round_dir=round_dir,
            )
        else:
            selected_info = greedy_select_rules(
                args, current_model, packer, val_files,
                candidate_rules_path=mined["candidate_rules_path"],
                round_dir=round_dir,
            )
        selected_engine = selected_info.get("selected_engine")
        selected_rules_path = selected_info.get("selected_rules_path", "")

        if selected_engine is not None:
            prev_guided_test = evaluate_model(
                args, current_model, packer, test_files, guidance_engine=selected_engine,
                csv_path=round_dir / "test_prev_model_with_val_selected_rules.csv",
                summary_path=round_dir / "test_prev_model_with_val_selected_rules_summary.json",
                title=f"[Round {r}] TEST 上一轮模型 + VAL筛选规则 泛化评估",
                split_name="test",
            )
        else:
            print(f"[Round {r}] VAL 没有筛出有效规则，TEST guided 结果记为 baseline。")
            prev_guided_test = {"mean_obj": sm0_test.get("mean_obj", float("nan"))}

        row = {
            "round": r,
            "phase": "val_select_then_test_guided",
            "model_path": current_model,
            "knowledge_path": selected_rules_path,
            "n_samples": mined.get("n_samples", 0),
            "n_raw_rules": mined.get("n_raw_rules", 0),
            "n_candidate_rules": mined.get("n_candidate_rules", 0),
            "n_selected_rules": selected_info.get("n_selected_rules", 0),
            "val_greedy_baseline_mean_obj": selected_info.get("baseline_mean_obj"),
            "val_greedy_best_mean_obj": selected_info.get("best_mean_obj"),
            "test_no_rules_mean_obj": sm0_test.get("mean_obj"),
            "test_guided_selected_mean_obj": prev_guided_test.get("mean_obj"),
            "new_model_test_no_rules_mean_obj": "",
            "new_model_test_guided_mean_obj": "",
            "train_guidance": False,
        }

        if args.no_train_feedback:
            print("\n[Skip Train Feedback] 已启用 --no_train_feedback：本轮只验证规则泛化，不把规则反馈进下一轮训练。")
            summary_rows.append(row)
            current_knowledge_engine = selected_engine
            continue

        new_model_path = train_model(
            args, trainer_cls, packer, train_files,
            out_model=round_dir / "model.zip",
            init_model=current_model,
            guidance_engine=selected_engine,
            timesteps=args.per_round_timesteps,
            timestep_offset=int(args.round_0_timesteps) + int(r - 1) * int(args.per_round_timesteps),
            progress_method="SAC + CoEvo-RGCD-R2",
        )

        new_no_rules = evaluate_model(
            args, new_model_path, packer, test_files, guidance_engine=None,
            csv_path=round_dir / "test_new_model_no_rules.csv",
            summary_path=round_dir / "test_new_model_no_rules_summary.json",
            title=f"[Round {r}] TEST 新模型 无规则评估",
            split_name="test",
        )
        new_guided = evaluate_model(
            args, new_model_path, packer, test_files, guidance_engine=selected_engine,
            csv_path=round_dir / "test_new_model_with_val_selected_rules.csv",
            summary_path=round_dir / "test_new_model_with_val_selected_rules_summary.json",
            title=f"[Round {r}] TEST 新模型 + VAL筛选规则 评估",
            split_name="test",
        )
        row.update({
            "phase": "train_feedback",
            "model_path": new_model_path,
            "new_model_test_no_rules_mean_obj": new_no_rules.get("mean_obj"),
            "new_model_test_guided_mean_obj": new_guided.get("mean_obj"),
            "train_guidance": bool(selected_engine),
        })
        summary_rows.append(row)
        current_model = new_model_path
        current_knowledge_engine = selected_engine

    write_csv(summary_rows, out_dir / "coevo_summary.csv")
    save_json({
        "method": "SA-CoEvo-RGCD-unified-policy-scenario-wise-rules" if bool(getattr(args, "scenario_wise_rules", False)) else "v14-CoEvo-RGCD-val-test-separated",
        "algo": args.algo,
        "n_rounds": args.n_rounds,
        "train_instances": args.train_instances,
        "val_instances": val_path,
        "test_instances": args.test_instances,
        "no_train_feedback": bool(args.no_train_feedback),
        "disturbance": bool(args.disturbance),
        "intensity": args.intensity,
        "total_time_s": round(time.time() - t_all, 1),
        "rounds": summary_rows,
    }, out_dir / "coevo_summary.json")

    print("\n" + "=" * 80)
    print("v14-CoEvo-RGCD 完成")
    print(f"总表：{out_dir / 'coevo_summary.csv'}")
    print(f"JSON：{out_dir / 'coevo_summary.json'}")
    print("=" * 80)


def parse_args():
    p = argparse.ArgumentParser(
        description="v14-CoEvo-RGCD：train挖规则、val筛规则、test最终评价",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo", default="sac", choices=["sac", "ppo", "rppo"])
    p.add_argument("--train_instances", required=True, type=str)
    p.add_argument("--val_instances", default=None, type=str, help="验证集：用于逐条评价/贪心筛选规则。正式实验必须提供。")
    p.add_argument("--test_instances", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--initial_model", default=None, type=str, help="可选：用已有模型作为 round_0")
    p.add_argument("--n_rounds", default=1, type=int, help="闭环轮数，不含 Round 0")
    p.add_argument("--round_0_timesteps", default=10000, type=int)
    p.add_argument("--per_round_timesteps", default=5000, type=int)
    p.add_argument("--n_collect_episodes", default=100, type=int, help="采集多少个完整调度episode用于规则挖掘。debug可用24，正式建议100/300/600。")
    p.add_argument("--collect_max_steps", default=1000, type=int, help="每个采集episode最多步数，防止异常卡死。")
    p.add_argument("--n_eval_episodes", default=5, type=int)
    p.add_argument("--max_instances", default=24, type=int, help="测试集最多评估多少个实例")
    p.add_argument("--max_val_instances", default=24, type=int, help="验证集最多评估多少个实例")
    p.add_argument("--max_train_instances", default=0, type=int, help="训练集最多使用多少个实例，0为全部")
    p.add_argument("--no_train_feedback", action="store_true", help="只做 train挖规则→val筛规则→test泛化评估，不把规则反馈进训练")
    p.add_argument("--reuse_mined_rules", action="store_true",
                   help="断点续跑：如果 round_x/candidate_guidance_rules.json 已存在，则跳过 TRAIN 采集、Destination/AQC 候选采集和规则挖掘，直接从 VAL 筛选继续。")
    p.add_argument("--reuse_existing_outputs", action="store_true",
                   help="断点续跑：如果已有 Round0 baseline summary，则复用该结果，避免重复 TEST baseline 评估。")
    p.add_argument("--disturbance", action="store_true")
    p.add_argument("--intensity", default="med", choices=["clean", "low", "med", "high"])
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--verbose", default=0, type=int)
    p.add_argument("--save_progress", action="store_true", default=True,
                   help="保存每段 training_progress.csv，用于拼接 CoEvo-RGCD 收敛曲线")
    p.add_argument("--no_save_progress", dest="save_progress", action="store_false",
                   help="不保存 training_progress.csv")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--algo_specific", default=None, type=str)
    p.add_argument("--rule_mining_mode", default="dt", choices=["dt", "template", "hybrid"],
                   help="规则生成方式：dt=原决策树/RGCD；template=模板补全；hybrid=两者合并。")
    p.add_argument("--template_quantiles", default="0.3,0.4,0.5,0.6,0.7,0.8",
                   help="模板补全连续阈值的分位数候选，逗号分隔。")
    p.add_argument("--template_min_confidence", default=0.75, type=float,
                   help="模板候选规则初筛最低置信度。后续还会经过VAL贪心筛选。")
    p.add_argument("--template_min_support", default=20, type=int,
                   help="模板候选规则初筛最低支持度。")
    p.add_argument("--template_min_coverage", default=0.002, type=float,
                   help="模板候选规则初筛最低覆盖率。")
    p.add_argument("--template_max_rules", default=120, type=int,
                   help="最多保留多少条模板候选规则进入VAL筛选。")
    p.add_argument("--template_max_variants_per_template", default=80, type=int,
                   help="每个模板最多测试多少个阈值组合，防止组合爆炸。")
    p.add_argument("--truck_wait_task_min_support", default=2, type=int,
                   help="task层truck_wait模板进入候选池的最低support。")
    p.add_argument("--truck_wait_task_min_confidence", default=0.05, type=float,
                   help="task层truck_wait模板进入候选池的最低confidence。")
    p.add_argument("--truck_wait_task_min_coverage", default=0.00005, type=float,
                   help="task层truck_wait模板进入候选池的最低coverage。")
    p.add_argument("--truck_wait_dest_min_support", default=2, type=int,
                   help="destination层truck_wait模板进入候选池的最低support。")
    p.add_argument("--truck_wait_dest_min_confidence", default=0.001, type=float,
                   help="destination层候选很多，selected比例天然很低，因此confidence需要远低于task层。")
    p.add_argument("--truck_wait_dest_min_coverage", default=0.000001, type=float,
                   help="destination层truck_wait模板进入候选池的最低coverage。")
    p.add_argument("--truck_wait_candidate_quota", default=2, type=int,
                   help="每个scenario-stage候选配额中为truck_wait规则预留的名额。")
    # Backward-compatible aliases used by earlier scripts.
    p.add_argument("--truck_wait_template_min_support", default=2, type=int,
                   help="兼容旧参数：未显式设置task/dest support时作为truck_wait support默认值。")
    p.add_argument("--truck_wait_template_min_confidence", default=None, type=float,
                   help="兼容旧参数；建议改用truck_wait_task_min_confidence/truck_wait_dest_min_confidence。")
    p.add_argument("--truck_wait_template_min_coverage", default=None, type=float,
                   help="兼容旧参数；建议改用truck_wait_task_min_coverage/truck_wait_dest_min_coverage。")
    p.add_argument("--template_prefer_adjust_candidates", default="30,50,70,90,120",
                   help="prefer模板规则候选加分β，逗号分隔；VAL阶段会筛选。")
    p.add_argument("--template_avoid_adjust_candidates", default="-30,-50,-70,-90,-120",
                   help="avoid模板规则候选扣分β，逗号分隔；VAL阶段会筛选。")
    p.add_argument("--template_include_fixed_rules", action="store_true", default=True,
                   help="保留故障AQC、planning_blocked等固定安全/可行性模板规则。")
    p.add_argument("--no_template_include_fixed_rules", dest="template_include_fixed_rules", action="store_false",
                   help="不强制保留固定安全/可行性模板规则。")

    p.add_argument("--distill_max_depth", default=6, type=int, help="决策树深度。越大规则越细；建议debug=5/6，正式=6/7。")
    p.add_argument("--distill_min_samples_leaf", default=10, type=int, help="树叶子节点最小样本数。越大越保守。")
    p.add_argument("--min_confidence", default=0.90, type=float, help="规则最低准确率/置信度，LEGIBLE常用0.90。")
    p.add_argument("--min_support", default=30, type=int, help="规则最低绝对覆盖样本数，防止只有几条样本的偶然规则。")
    p.add_argument("--min_coverage", default=0.003, type=float, help="规则最低覆盖率=support/该场景样本数。细粒度规则建议0.003~0.01。")
    p.add_argument("--min_rule_conditions", default=2, type=int, help="规则至少包含几个状态条件；越大规则越细，触发越少。")
    p.add_argument("--max_rule_conditions", default=8, type=int, help="规则最多包含几个状态条件；0表示不限制。")
    p.add_argument("--max_rules", default=80, type=int)
    p.add_argument("--guidance_adjust", default=100.0, type=float, help="task规则加分强度。之前500太强，建议先用50~100。")
    p.add_argument("--soft_rule_guidance", action="store_true",
                   help="启用趋势6：数值型规则条件使用连续软触发；类别/安全/mask条件仍保持硬判断。")
    p.add_argument("--soft_rule_temperature", default=0.05, type=float,
                   help="软规则 sigmoid 温度。越大越平滑、触发范围越宽；建议 0.03~0.10。")
    p.add_argument("--soft_rule_min_strength", default=0.05, type=float,
                   help="规则软匹配强度低于该值时视为未触发，避免极弱远距离触发。")
    p.add_argument("--soft_rule_aggregation", default="min", choices=["min", "product", "mean"],
                   help="多条件规则的软强度聚合方式。min最稳，product最严格，mean最宽松。")
    p.add_argument("--enable_destination_rules", action="store_true", help="启用候选级Destination规则挖掘：从info['destination_candidates']采集目的地候选样本并生成stage=destination规则。")
    p.add_argument("--n_destination_collect_episodes", default=0, type=int, help="Destination候选样本采集episode数；0表示沿用--n_collect_episodes。")
    p.add_argument("--destination_distill_max_depth", default=5, type=int, help="Destination二分类决策树深度。")
    p.add_argument("--destination_distill_min_samples_leaf", default=10, type=int, help="Destination树叶子节点最小样本数。")
    p.add_argument("--destination_min_confidence", default=0.85, type=float, help="Destination规则最低准确率/置信度。")
    p.add_argument("--destination_min_support", default=20, type=int, help="Destination规则最低绝对覆盖样本数。")
    p.add_argument("--destination_min_coverage", default=0.002, type=float, help="Destination规则最低覆盖率=support/当前scenario+task_kind+dest_kind候选样本数。")
    p.add_argument("--destination_min_rule_conditions", default=1, type=int, help="Destination规则至少包含几个候选特征条件。")
    p.add_argument("--destination_max_rule_conditions", default=8, type=int, help="Destination规则最多包含几个候选特征条件；0表示不限制。")
    p.add_argument("--destination_max_rules", default=80, type=int, help="最多保留多少条Destination候选规则进入VAL筛选。")
    p.add_argument("--destination_guidance_adjust", default=60.0, type=float, help="Destination规则加分强度，建议与AQC相近或略低于task规则。")
    p.add_argument("--destination_reuse_env", action="store_true",
                   help="采集Destination候选时复用同一个env，避免同seed反复抽到同一个场景。")
    p.add_argument("--destination_probe_all_task_kinds", action="store_true",
                   help="采集Destination候选时额外探测每类未完成任务的目的地候选，不执行任务，只补充样本。")
    p.add_argument("--destination_probe_max_tasks_per_kind", default=1, type=int,
                   help="每个step每类task_kind最多额外探测几个任务，建议1，防止CSV过大。")
    p.add_argument("--enable_aqc_rules", action="store_true", help="启用候选级AQC规则挖掘：从info['aqc_candidates']采集AQC候选样本并生成stage=aqc规则。")
    p.add_argument("--n_aqc_collect_episodes", default=0, type=int, help="AQC候选样本采集episode数；0表示沿用--n_collect_episodes。")
    p.add_argument("--aqc_distill_max_depth", default=5, type=int, help="AQC二分类决策树深度。建议先比task略浅。")
    p.add_argument("--aqc_distill_min_samples_leaf", default=10, type=int, help="AQC树叶子节点最小样本数。")
    p.add_argument("--aqc_min_confidence", default=0.90, type=float, help="AQC规则最低准确率/置信度。")
    p.add_argument("--aqc_min_support", default=20, type=int, help="AQC规则最低绝对覆盖样本数。")
    p.add_argument("--aqc_min_coverage", default=0.003, type=float, help="AQC规则最低覆盖率=support/当前scenario+task_kind候选样本数。")
    p.add_argument("--aqc_min_rule_conditions", default=2, type=int, help="AQC规则至少包含几个候选特征条件。")
    p.add_argument("--aqc_max_rule_conditions", default=8, type=int, help="AQC规则最多包含几个候选特征条件；0表示不限制。")
    p.add_argument("--aqc_max_rules", default=60, type=int, help="最多保留多少条AQC候选规则进入VAL筛选。")
    p.add_argument("--aqc_guidance_adjust", default=80.0, type=float, help="AQC规则加分强度，建议略低于task规则。")
    p.add_argument("--safe_rule_filter", action="store_true", help="启用 Safe Rule Filter：规则平均变好之外，还必须满足单case恶化、触发次数和改善/恶化case数量约束。")
    p.add_argument("--max_case_worsen", default=200.0, type=float, help="Safe过滤：任何单个验证case允许的最大obj恶化值。")
    p.add_argument("--max_avg_triggered", default=50.0, type=float, help="Safe过滤：验证集平均每个case的最大规则触发次数。")
    p.add_argument("--max_case_triggered", default=300.0, type=float, help="Safe过滤：任何单个验证case允许的最大规则触发次数。")
    p.add_argument("--require_improved_ge_worsened", action="store_true", help="Safe过滤：要求改善case数 >= 恶化case数。")
    p.add_argument("--risk_budget_greedy", action="store_true",
                   help="启用趋势3：风险预算式贪心筛选。普通触发/恶化风险进入评分，只有灾难风险硬拒绝。")
    p.add_argument("--min_risk_score_improve", default=5.0, type=float,
                   help="趋势3：risk-aware score 至少改善多少才接受；建议与 min_target_scenario_improve 保持一致。")
    p.add_argument("--risk_catastrophic_worsen", default=1500.0, type=float,
                   help="趋势3：灾难性单case恶化阈值；超过才硬拒绝。")
    p.add_argument("--risk_max_completion_drop", default=0.01, type=float,
                   help="趋势3：允许的单case completion 最大下降；默认基本不允许完工率下降。")
    p.add_argument("--risk_cvar_alpha", default=0.25, type=float,
                   help="趋势3：CVaR尾部风险比例，0.25表示取最坏25%%恶化case。")
    p.add_argument("--risk_lambda_cvar", default=0.05, type=float,
                   help="趋势3：尾部恶化风险惩罚系数。")
    p.add_argument("--risk_lambda_p_worsen", default=50.0, type=float,
                   help="趋势3：恶化case比例惩罚系数。")
    p.add_argument("--risk_lambda_max_worsen", default=0.01, type=float,
                   help="趋势3：最大单case恶化惩罚系数。")
    p.add_argument("--risk_lambda_trigger_strength", default=0.02, type=float,
                   help="趋势3：软触发强度总量惩罚系数，替代原来的纯触发次数硬过滤。")
    p.add_argument("--risk_lambda_high_strength", default=0.05, type=float,
                   help="趋势3：高强度触发次数惩罚系数。")
    p.add_argument("--risk_lambda_p_improve", default=0.0, type=float,
                   help="趋势3：改善case比例奖励系数，默认0表示不额外奖励。")

    p.add_argument("--dedup_selected_rules", action="store_true",
                   help="趋势3.1：VAL筛选结束后对 selected rules 做去重，防止同一global规则跨场景重复保存/重复启用。")
    p.add_argument("--dedup_selected_by_template", action="store_true",
                   help="趋势3.1：同一 scenario/all + stage + template_id 只保留验证风险评分改善最好的一条，减少q/adjust变体堆叠。")
    p.add_argument("--dedup_global_rules", action="store_true",
                   help="趋势3.1：把 scenario=all/global 的规则统一放入 all 规则池，并在各场景执行时共享一次。")

    p.add_argument("--rule_memory_retrieval", action="store_true",
                   help="趋势4：启用上下文感知规则记忆检索。规则库可保留更多规则，但每个候选评分时只注入Top-K相关规则。")
    p.add_argument("--rule_retrieval_top_k", default=3, type=int,
                   help="趋势4：每次候选评分最多注入多少条匹配规则；<=0表示不限制。建议2~4。")
    p.add_argument("--rule_retrieval_dedup_templates", action="store_true",
                   help="趋势4：执行时同一模板族多个变体同时匹配时，仅检索相关性最高的一条，减少规则堆叠。")
    # Scenario-aware CoEvo-RGCD: unified policy + scenario-wise rule libraries.
    p.add_argument("--scenario_wise_rules", action="store_true",
                   help="启用统一SAC策略+分场景规则库+分场景VAL筛选+分场景TEST统计+场景感知反馈训练。")
    p.add_argument("--scenario_candidate_quota", action="store_true",
                   help="候选阶段按 scenario-stage 保留规则配额，避免2unload规则独占候选池。")
    p.add_argument("--max_rules_per_scenario", default=30, type=int,
                   help="分场景候选配额：每个scenario最多保留多少条候选规则。")
    p.add_argument("--max_rules_per_stage_scenario", default=10, type=int,
                   help="分场景候选配额：每个scenario-stage最多保留多少条候选规则。")
    p.add_argument("--family_candidate_quota", action="store_true",
                   help="候选压缩阶段按规则族保底：每类先保留质量最好的若干条，再用剩余名额按质量补齐。")
    p.add_argument("--family_quota_load_pressure", default=2, type=int,
                   help="候选族保底：每个scenario-stage保留多少条load_pressure规则。")
    p.add_argument("--family_quota_unload_pressure", default=2, type=int,
                   help="候选族保底：每个scenario-stage保留多少条unload_pressure规则。")
    p.add_argument("--family_quota_truck_wait", default=2, type=int,
                   help="候选族保底：每个scenario-stage保留多少条truck_wait规则。")
    p.add_argument("--family_quota_destination", default=2, type=int,
                   help="候选族保底：每个scenario-stage保留多少条destination规则；若该类未生成则不占名额。")
    p.add_argument("--family_quota_aqc", default=2, type=int,
                   help="候选族保底：每个scenario-stage保留多少条AQC规则。")
    p.add_argument("--family_quota_global", default=2, type=int,
                   help="候选族保底：每个scenario-stage保留多少条global规则。")
    p.add_argument("--family_quota_other", default=0, type=int,
                   help="候选族保底：每个scenario-stage保留多少条other规则。")
    p.add_argument("--carryover_selected_rules", action="store_true",
                   help="将上一轮VAL筛选出的规则加入下一轮candidate_guidance候选池，再重新接受VAL筛选；不强制最终保留。")
    p.add_argument("--carryover_max_per_family", default=1, type=int,
                   help="规则继承：上一轮selected规则中每个规则族最多继承多少条；<=0表示不限制。")
    p.add_argument("--scenario_global_safety_check", action="store_true", default=True,
                   help="分场景局部接受后，再用全体验证集进行跨场景安全复查。")
    p.add_argument("--no_scenario_global_safety_check", dest="scenario_global_safety_check", action="store_false",
                   help="关闭跨场景全体验证安全复查，仅做目标场景局部筛选。")
    p.add_argument("--min_target_scenario_improve", default=5.0, type=float,
                   help="目标场景接受准则：该规则在目标scenario上至少改善多少obj才可接受。")
    p.add_argument("--max_other_scenario_worsen", default=30.0, type=float,
                   help="目标场景接受准则：其他scenario允许的最大平均obj恶化。")
    p.add_argument("--min_delta", default=1e-6, type=float)
    p.add_argument("--horizon", default=10000.0, type=float)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

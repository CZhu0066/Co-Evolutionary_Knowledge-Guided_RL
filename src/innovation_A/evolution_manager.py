# -*- coding: utf-8 -*-
"""
evolution_manager.py
=====================
Innovation A3：多轮协同进化闭环管理器。

工作流（每轮）：
  1. 用上一轮模型采集轨迹
  2. 蒸馏规则
  3. （可选）因果验证 + 过滤
  4. 构造 rule_bonus_fn 注入 env
  5. 训练新模型（warm start 自上一轮）
  6. 评估、保存所有产物

收敛检测：obj 在最近 patience 轮内的变化 < threshold → 提前停止。
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.innovation_A.feature_extractor import FeatureExtractor
from src.innovation_A.trajectory_collector import collect_trajectories, save_trajectories
from src.innovation_A.distillation import (
    distill_rules, save_rules,
    distill_rules_by_scenario, save_rules_by_scenario,
)
from src.innovation_A.rule_bonus import make_rule_bonus_fn
from src.innovation_A.causal_validator import (
    CausalValidator, save_report, filter_rules_by_validation,
)


# ============================================================
# 数据类
# ============================================================
@dataclass
class RoundResult:
    """单轮结果"""
    round_idx: int
    model_path: str
    trajectories_path: Optional[str]
    rules_path: Optional[str]
    validation_report_path: Optional[str]
    rules_filtered_path: Optional[str]

    n_rules_distilled: int
    n_rules_real: int                 # 因果验证通过的（若开启）
    n_rules_used: int                 # 实际注入的（real 或全部）
    n_samples_collected: int

    eval_metrics: Dict[str, float]    # mean_obj, mean_reward, ...

    train_timesteps: int
    train_time_s: float
    collect_time_s: float
    distill_time_s: float
    validation_time_s: float
    total_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvolutionHistory:
    """完整 N 轮历史"""
    rounds: List[RoundResult] = field(default_factory=list)
    converged: bool = False
    best_round: int = -1
    best_obj: float = float("inf")
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    total_time_s: float = 0.0
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rounds": [r.to_dict() for r in self.rounds],
            "converged": self.converged,
            "best_round": self.best_round,
            "best_obj": self.best_obj,
            "config_snapshot": self.config_snapshot,
            "total_time_s": self.total_time_s,
            "created_at": self.created_at,
        }

    def obj_curve(self) -> List[float]:
        return [r.eval_metrics.get("mean_obj", float("nan"))
                for r in self.rounds]


# ============================================================
# Manager 主类
# ============================================================
class EvolutionManager:
    """
    多轮协同进化管理器。

    用法：
        manager = EvolutionManager(
            env_factory_no_bonus=...,
            make_env_with_bonus_factory=...,
            trainer_class=SACTrainer,
            base_config=cfg,
            out_dir="results/evolution/",
        )
        history = manager.run_loop(
            n_rounds=5,
            initial_model_path=None,    # None 表示先训 round 0
            per_round_timesteps=10000,
        )
    """

    def __init__(
        self,
        env_factory_no_bonus: Callable[[], Any],
        make_env_with_bonus_factory: Callable[[Callable], Callable[[], Any]],
        trainer_class: type,
        base_config: Any,
        out_dir: str | Path,
        # 蒸馏 / 采集
        n_collect_episodes: int = 50,
        distill_max_depth: int = 5,
        distill_min_samples_leaf: int = 5,
        # 注入参数
        # bonus_match: float = 0.05,
        # bonus_mismatch: float = -0.02,
        # min_confidence: float = 0.5,
        bonus_match: float = 0.02,
        bonus_mismatch: float = -0.005,
        min_confidence: float = 0.6,
        # 因果验证（可选）
        use_causal_validation: bool = False,
        validation_train_timesteps: int = 2000,
        validation_threshold: float = 0.01,
        # 收敛检测
        convergence_threshold: float = 0.005,
        patience: int = 2,
        # 其他
        n_eval_episodes: int = 10,
        extractor: Optional[FeatureExtractor] = None,
        verbose: bool = True,
    ):
        self.env_factory_no_bonus = env_factory_no_bonus
        self.make_env_with_bonus_factory = make_env_with_bonus_factory
        self.trainer_class = trainer_class
        self.base_config = base_config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.n_collect_episodes = int(n_collect_episodes)
        self.distill_max_depth = int(distill_max_depth)
        self.distill_min_samples_leaf = int(distill_min_samples_leaf)

        self.bonus_match = float(bonus_match)
        self.bonus_mismatch = float(bonus_mismatch)
        self.min_confidence = float(min_confidence)

        self.use_causal_validation = bool(use_causal_validation)
        self.validation_train_timesteps = int(validation_train_timesteps)
        self.validation_threshold = float(validation_threshold)

        self.convergence_threshold = float(convergence_threshold)
        self.patience = int(patience)

        self.n_eval_episodes = int(n_eval_episodes)
        self.extractor = extractor or FeatureExtractor()
        self.verbose = bool(verbose)

    # ------------------------------------------------------------
    # Round 0：基线训练
    # ------------------------------------------------------------
    def train_round_0(self, train_timesteps: int) -> RoundResult:
        """跑 Round 0：基线训练（无规则）"""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  Round 0: 基线训练（无规则注入）  steps={train_timesteps:,}")
            print(f"{'='*70}\n")

        t_total = time.time()
        round_dir = self.out_dir / "round_0"
        round_dir.mkdir(exist_ok=True)

        # 训练
        cfg = deepcopy(self.base_config)
        cfg.total_timesteps = int(train_timesteps)
        cfg.verbose = 0

        trainer = self.trainer_class(env_fn=self.env_factory_no_bonus, config=cfg)
        trainer.setup()
        t0 = time.time()
        trainer.train()
        train_time = time.time() - t0

        model_path = round_dir / "model.zip"
        trainer.save(str(model_path))

        # 评估
        eval_metrics = trainer.evaluate(n_episodes=self.n_eval_episodes)
        trainer.close()

        if self.verbose:
            print(f"  [Round 0] obj={eval_metrics['mean_obj']:.4f}  "
                  f"reward={eval_metrics['mean_reward']:.2f}  ({train_time:.1f}s)")

        return RoundResult(
            round_idx=0,
            model_path=str(model_path),
            trajectories_path=None,
            rules_path=None,
            validation_report_path=None,
            rules_filtered_path=None,
            n_rules_distilled=0,
            n_rules_real=0,
            n_rules_used=0,
            n_samples_collected=0,
            eval_metrics=eval_metrics,
            train_timesteps=train_timesteps,
            train_time_s=round(train_time, 1),
            collect_time_s=0.0,
            distill_time_s=0.0,
            validation_time_s=0.0,
            total_time_s=round(time.time() - t_total, 1),
        )

    # ------------------------------------------------------------
    # Round N≥1：完整闭环
    # ------------------------------------------------------------
    def run_round(self,
                  round_idx: int,
                  init_model_path: str,
                  train_timesteps: int) -> RoundResult:
        """
        跑一轮闭环：采集 → 蒸馏 → (验证) → 训练 (warm start)。
        """
        assert round_idx >= 1, "run_round 用于 round >= 1；round 0 用 train_round_0"
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  Round {round_idx}: 闭环 (warm start = {Path(init_model_path).name})")
            print(f"{'='*70}\n")

        t_total = time.time()
        round_dir = self.out_dir / f"round_{round_idx}"
        round_dir.mkdir(exist_ok=True)

        # ===== Step 1: 用上一轮模型采集 =====
        if self.verbose:
            print(f"  [1/4] 采集轨迹 (n_episodes={self.n_collect_episodes})...")

        cfg_collect = deepcopy(self.base_config)
        cfg_collect.total_timesteps = 0
        cfg_collect.verbose = 0
        cfg_collect.n_eval_episodes = self.n_eval_episodes
        collect_trainer = self.trainer_class(
            env_fn=self.env_factory_no_bonus, config=cfg_collect,
        )
        collect_trainer.setup()
        collect_trainer.load(init_model_path)

        t0 = time.time()
        data = collect_trajectories(
            predict_fn=lambda obs: collect_trainer.predict(obs, deterministic=True),
            env_factory=self.env_factory_no_bonus,
            n_episodes=self.n_collect_episodes,
            extractor=self.extractor,
            verbose=False,
        )
        collect_time = time.time() - t0
        n_samples = len(data["features"])
        collect_trainer.close()

        traj_path = round_dir / "trajectories.npz"
        save_trajectories(data, traj_path)
        if self.verbose:
            print(f"        采集到 {n_samples} 样本 ({collect_time:.1f}s)")

        # ===== Step 2: 蒸馏 =====
        if self.verbose:
            print(f"  [2/4] 蒸馏规则 (depth<={self.distill_max_depth})...")
        t0 = time.time()
        if n_samples == 0:
            if self.verbose:
                print(f"        [WARN] 无样本可蒸馏，跳过")
            rules_data = {
                "method": "decision_tree_by_scenario",
                "rules_by_scenario": {},
                "n_rules_by_scenario": {},
                "n_rules": 0,
                "feature_names": list(self.extractor.feature_names),
                "n_features": self.extractor.n_features,
                "n_samples": 0,
                "train_accuracy": 0.0,
            }
        elif "scenarios" in data:
            rules_data = distill_rules_by_scenario(
                data["features"], data["labels"], data["scenarios"],
                max_depth=self.distill_max_depth,
                min_samples_leaf=self.distill_min_samples_leaf,
                feature_names=list(self.extractor.feature_names),
            )
        else:
            # 兼容旧 trajectories.npz：没有 scenarios 时退回单规则池
            rules_data = distill_rules(
                data["features"], data["labels"],
                max_depth=self.distill_max_depth,
                min_samples_leaf=self.distill_min_samples_leaf,
                feature_names=list(self.extractor.feature_names),
            )
        distill_time = time.time() - t0
        n_rules_distilled = int(rules_data.get("n_rules", 0))
        if "rules_by_scenario" in rules_data:
            rules_path = round_dir / "rules_by_scenario.json"
            save_rules_by_scenario(rules_data, rules_path)
            # 同时另存一份 rules.json，方便旧脚本/人工查看路径不变
            save_rules_by_scenario(rules_data, round_dir / "rules.json")
        else:
            rules_path = round_dir / "rules.json"
            save_rules(rules_data, rules_path)
        if self.verbose:
            if "n_rules_by_scenario" in rules_data:
                print(f"        分场景规则数：{rules_data.get('n_rules_by_scenario', {})}")
            print(f"        提取 {n_rules_distilled} 条规则 ({distill_time:.1f}s)")

        # ===== Step 3 (optional): 因果验证 =====
        rules_for_injection = rules_data
        validation_time = 0.0
        validation_report_path = None
        rules_filtered_path = None
        n_rules_real = n_rules_distilled

        if self.use_causal_validation and n_rules_distilled > 0 and "rules_by_scenario" not in rules_data:
            if self.verbose:
                print(f"  [3/4] 因果验证 ({n_rules_distilled} 条规则, "
                      f"validation_steps={self.validation_train_timesteps})...")
            t0 = time.time()
            validator = CausalValidator(
                env_factory_no_bonus=self.env_factory_no_bonus,
                make_env_with_bonus_factory=self.make_env_with_bonus_factory,
                trainer_class=self.trainer_class,
                base_config=self.base_config,
                n_eval_episodes=self.n_eval_episodes,
                extractor=self.extractor,
                verbose=False,
            )
            report = validator.validate(
                rules_data,
                train_timesteps=self.validation_train_timesteps,
                threshold=self.validation_threshold,
                bonus_match=self.bonus_match,
                bonus_mismatch=self.bonus_mismatch,
                min_confidence=self.min_confidence,
                include_no_rules=False,
            )
            validation_time = time.time() - t0

            validation_report_path = round_dir / "validation_report.json"
            save_report(report, validation_report_path)

            # 过滤：仅保留 real
            rules_for_injection = filter_rules_by_validation(
                rules_data, report, keep_classifications=("real",),
            )
            rules_filtered_path = round_dir / "rules_filtered.json"
            save_rules(rules_for_injection, rules_filtered_path)

            n_rules_real = report.n_real
            if self.verbose:
                print(f"        real={report.n_real}  "
                      f"spurious={report.n_spurious}  "
                      f"harmful={report.n_harmful} ({validation_time:.1f}s)")
        else:
            if self.verbose and n_rules_distilled > 0:
                if "rules_by_scenario" in rules_data and self.use_causal_validation:
                    print(f"  [3/4] 因果验证 ⏭️ 暂不支持分场景规则，跳过")
                else:
                    print(f"  [3/4] 因果验证 ⏭️ 跳过（use_causal_validation=False）")

        # ===== A3 规则过滤：只注入较可靠的规则 =====
        # 分场景规则会分别过滤，避免某个场景的低质量规则污染其他场景。
        min_rule_confidence_for_injection = 0.80
        min_rule_support_for_injection = 20

        if "rules_by_scenario" in rules_for_injection:
            rules_for_injection = deepcopy(rules_for_injection)
            total_before = 0
            total_after = 0
            n_rules_by_scenario = {}

            for sc, pack in rules_for_injection.get("rules_by_scenario", {}).items():
                old_rules = list(pack.get("rules", []))
                total_before += len(old_rules)
                filtered_rules = []
                for r in old_rules:
                    conf = float(r.get("confidence", 0.0))
                    support = int(r.get("support", 0))
                    if conf >= min_rule_confidence_for_injection and support >= min_rule_support_for_injection:
                        filtered_rules.append(r)

                pack["rules"] = filtered_rules
                pack["n_rules"] = len(filtered_rules)
                n_rules_by_scenario[str(sc)] = len(filtered_rules)
                total_after += len(filtered_rules)

                if self.verbose:
                    print(
                        f"        [RuleFilter][{sc}] 原始 {len(old_rules)} 条，"
                        f"过滤后 {len(filtered_rules)} 条 "
                        f"(confidence>={min_rule_confidence_for_injection}, "
                        f"support>={min_rule_support_for_injection})"
                    )

            rules_for_injection["n_rules_by_scenario"] = n_rules_by_scenario
            rules_for_injection["n_rules"] = int(total_after)
            if self.verbose:
                print(f"        [RuleFilter] 总计 原始 {total_before} 条，过滤后 {total_after} 条")
        elif rules_for_injection.get("rules"):
            filtered_rules = []
            for r in rules_for_injection["rules"]:
                conf = float(r.get("confidence", 0.0))
                support = int(r.get("support", 0))
                if conf >= min_rule_confidence_for_injection and support >= min_rule_support_for_injection:
                    filtered_rules.append(r)

            if self.verbose:
                print(
                    f"        [RuleFilter] 原始规则 {rules_for_injection['n_rules']} 条，"
                    f"过滤后 {len(filtered_rules)} 条 "
                    f"(confidence>={min_rule_confidence_for_injection}, "
                    f"support>={min_rule_support_for_injection})"
                )

            rules_for_injection = dict(rules_for_injection)
            rules_for_injection["rules"] = filtered_rules
            rules_for_injection["n_rules"] = len(filtered_rules)

        n_rules_used = int(rules_for_injection.get("n_rules", 0))
        # ===== Step 4: 训练（warm start + 注入） =====
        if self.verbose:
            print(f"  [4/4] 训练 Round {round_idx} (warm start, "
                  f"steps={train_timesteps:,}, 注入 {n_rules_used} 条规则)...")

        if n_rules_used > 0:
        # if False and n_rules_used > 0:
            bonus_fn = make_rule_bonus_fn(
                rules_for_injection, extractor=self.extractor,
                bonus_match=self.bonus_match,
                bonus_mismatch=self.bonus_mismatch,
                min_confidence=self.min_confidence,
            )
            env_factory = self.make_env_with_bonus_factory(bonus_fn)
        else:
            # 无可用规则，退化为无注入训练（仍 warm start）
            env_factory = self.env_factory_no_bonus
        # ===== A3 规则注入有效性诊断：训练前先跑少量 episode =====
        rule_diag = {}
        if n_rules_used > 0:
            if self.verbose:
                print("        [RuleDiag] 训练前检查规则是否命中...")

            rule_diag = self._diagnose_rule_bonus(
                model_path=init_model_path,
                env_factory=env_factory,
                n_episodes=min(3, self.n_eval_episodes),
                max_steps_per_episode=1000,
            )

            diag_path = round_dir / "rule_bonus_diagnostic_before_train.json"
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(rule_diag, f, indent=2, ensure_ascii=False)

            if self.verbose:
                print(
                    "        [RuleDiag] "
                    f"steps={rule_diag.get('steps', 0)}, "
                    f"calls={rule_diag.get('final_calls', 0)}, "
                    f"hits={rule_diag.get('final_hits', 0)}, "
                    f"matched={rule_diag.get('final_matched', 0)}, "
                    f"mismatched={rule_diag.get('final_mismatched', 0)}, "
                    f"hit_rate={rule_diag.get('hit_rate', 0):.3f}, "
                    f"match_rate={rule_diag.get('match_rate_when_hit', 0):.3f}, "
                    f"bonus_sum={rule_diag.get('rule_bonus_sum', 0):.4f}, "
                    f"mean_bonus={rule_diag.get('mean_rule_bonus_per_step', 0):.6f}"
                )

        cfg = deepcopy(self.base_config)
        cfg.total_timesteps = int(train_timesteps)
        cfg.verbose = 0

        trainer = self.trainer_class(env_fn=env_factory, config=cfg)
        trainer.setup()
        trainer.load(init_model_path)   # warm start

        t0 = time.time()
        trainer.train(reset_num_timesteps=False)
        train_time = time.time() - t0

        model_path = round_dir / "model.zip"
        trainer.save(str(model_path))
        eval_metrics = trainer.evaluate(n_episodes=self.n_eval_episodes)

        # 把规则诊断结果写进 history.json，方便后续分析
        if rule_diag:
            for k, v in rule_diag.items():
                if isinstance(v, (int, float)):
                    eval_metrics[f"rule_diag_{k}"] = float(v)

        trainer.close()

        if self.verbose:
            print(f"        ✅ obj={eval_metrics['mean_obj']:.4f}  "
                  f"reward={eval_metrics['mean_reward']:.2f}  ({train_time:.1f}s)")

        return RoundResult(
            round_idx=round_idx,
            model_path=str(model_path),
            trajectories_path=str(traj_path),
            rules_path=str(rules_path),
            validation_report_path=(str(validation_report_path)
                                     if validation_report_path else None),
            rules_filtered_path=(str(rules_filtered_path)
                                  if rules_filtered_path else None),
            n_rules_distilled=n_rules_distilled,
            n_rules_real=n_rules_real,
            n_rules_used=n_rules_used,
            n_samples_collected=n_samples,
            eval_metrics=eval_metrics,
            train_timesteps=train_timesteps,
            train_time_s=round(train_time, 1),
            collect_time_s=round(collect_time, 1),
            distill_time_s=round(distill_time, 1),
            validation_time_s=round(validation_time, 1),
            total_time_s=round(time.time() - t_total, 1),
        )

    def _diagnose_rule_bonus(
        self,
        model_path: str,
        env_factory: Callable[[], Any],
        n_episodes: int = 3,
        max_steps_per_episode: int = 1000,
    ) -> Dict[str, float]:
        """
        用当前模型 + 注入规则后的环境跑少量 episode，
        统计规则是否真的命中、匹配、产生 bonus。
        """
        cfg = deepcopy(self.base_config)
        cfg.total_timesteps = 0
        cfg.verbose = 0

        trainer = self.trainer_class(env_fn=env_factory, config=cfg)
        trainer.setup()
        trainer.load(model_path)

        env = env_factory()

        agg = {
            "episodes": 0,
            "steps": 0,
            "rule_bonus_sum": 0.0,
            "rule_bonus_abs_sum": 0.0,
            "final_calls": 0,
            "final_hits": 0,
            "final_matched": 0,
            "final_mismatched": 0,
            "final_no_match": 0,
            "final_low_confidence": 0,
            "final_feature_before_action": 0,
            "final_feature_after_action_fallback": 0,
        }

        try:
            for ep in range(int(n_episodes)):
                obs, info = env.reset()
                last_info = {}

                for step in range(int(max_steps_per_episode)):
                    action = trainer.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(np.asarray(action))

                    agg["steps"] += 1
                    agg["rule_bonus_sum"] += float(info.get("rule_bonus", 0.0))
                    agg["rule_bonus_abs_sum"] += abs(float(info.get("rule_bonus", 0.0)))
                    last_info = info

                    if terminated or truncated:
                        break

                stats = last_info.get("rule_bonus_stats", {})
                agg["episodes"] += 1
                agg["final_calls"] += int(stats.get("calls", 0))
                agg["final_hits"] += int(stats.get("hits", 0))
                agg["final_matched"] += int(stats.get("matched", 0))
                agg["final_mismatched"] += int(stats.get("mismatched", 0))
                agg["final_no_match"] += int(stats.get("no_match", 0))
                agg["final_low_confidence"] += int(stats.get("low_confidence", 0))
                agg["final_feature_before_action"] += int(stats.get("feature_before_action", 0))
                agg["final_feature_after_action_fallback"] += int(
                    stats.get("feature_after_action_fallback", 0)
                )

        finally:
            try:
                env.close()
            except Exception:
                pass
            trainer.close()

        calls = max(1, agg["final_calls"])
        hits = max(1, agg["final_hits"])
        steps = max(1, agg["steps"])

        agg["hit_rate"] = agg["final_hits"] / calls
        agg["match_rate_when_hit"] = agg["final_matched"] / hits
        agg["mismatch_rate_when_hit"] = agg["final_mismatched"] / hits
        agg["mean_rule_bonus_per_step"] = agg["rule_bonus_sum"] / steps
        agg["mean_abs_rule_bonus_per_step"] = agg["rule_bonus_abs_sum"] / steps

        return agg

    # ------------------------------------------------------------
    # 收敛检测
    # ------------------------------------------------------------
    def check_convergence(self, history: EvolutionHistory) -> bool:
        """
        最近 patience+1 轮的 obj 变化 < threshold → 已收敛。
        """
        n = len(history.rounds)
        if n < self.patience + 1:
            return False
        recent_objs = [r.eval_metrics.get("mean_obj", float("nan"))
                       for r in history.rounds[-(self.patience + 1):]]
        if any(np.isnan(o) for o in recent_objs):
            return False
        spread = max(recent_objs) - min(recent_objs)
        return spread < self.convergence_threshold

    # ------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------
    def run_loop(
        self,
        n_rounds: int,
        initial_model_path: Optional[str] = None,
        per_round_timesteps: int = 5000,
        round_0_timesteps: Optional[int] = None,
    ) -> EvolutionHistory:
        """
        跑完整 N 轮闭环。

        Args:
            n_rounds: 闭环轮数（不含 round 0）
            initial_model_path: 已训好的 round 0 模型；None 则先训
            per_round_timesteps: 每个闭环轮的训练步数
            round_0_timesteps: round 0 步数（仅 initial_model_path=None 时用）

        Returns:
            EvolutionHistory
        """
        t_start = time.time()
        history = EvolutionHistory(
            config_snapshot={
                "n_rounds": n_rounds,
                "per_round_timesteps": per_round_timesteps,
                "round_0_timesteps": round_0_timesteps,
                "n_collect_episodes": self.n_collect_episodes,
                "distill_max_depth": self.distill_max_depth,
                "use_causal_validation": self.use_causal_validation,
                "validation_train_timesteps": self.validation_train_timesteps,
                "bonus_match": self.bonus_match,
                "bonus_mismatch": self.bonus_mismatch,
                "min_confidence": self.min_confidence,
                "convergence_threshold": self.convergence_threshold,
                "patience": self.patience,
            },
            created_at=datetime.now().isoformat(),
        )

        # Round 0
        if initial_model_path is None:
            r0_steps = round_0_timesteps or per_round_timesteps
            r0 = self.train_round_0(r0_steps)
        else:
            # 用已有模型，但需要先评估一下作为 round 0
            if self.verbose:
                print(f"  [Round 0] 使用已有模型：{initial_model_path}")
            cfg = deepcopy(self.base_config)
            cfg.total_timesteps = 0
            cfg.verbose = 0
            tr = self.trainer_class(env_fn=self.env_factory_no_bonus, config=cfg)
            tr.setup()
            tr.load(initial_model_path)
            eval_m = tr.evaluate(n_episodes=self.n_eval_episodes)
            tr.close()
            r0 = RoundResult(
                round_idx=0,
                model_path=initial_model_path,
                trajectories_path=None, rules_path=None,
                validation_report_path=None, rules_filtered_path=None,
                n_rules_distilled=0, n_rules_real=0, n_rules_used=0,
                n_samples_collected=0,
                eval_metrics=eval_m,
                train_timesteps=0,
                train_time_s=0.0, collect_time_s=0.0,
                distill_time_s=0.0, validation_time_s=0.0,
                total_time_s=0.0,
            )
        history.rounds.append(r0)
        self._save_history(history)

        # Round 1..N
        for k in range(1, n_rounds + 1):
            prev_model = history.rounds[-1].model_path
            try:
                rk = self.run_round(k, prev_model, per_round_timesteps)
            except Exception as e:
                if self.verbose:
                    print(f"  [Round {k}] ✗ 异常：{type(e).__name__}: {e}")
                # 用上一轮结果做兜底
                rk = deepcopy(history.rounds[-1])
                rk.round_idx = k
                rk.train_timesteps = 0
            history.rounds.append(rk)
            self._save_history(history)

            # 收敛检测
            if self.check_convergence(history):
                history.converged = True
                if self.verbose:
                    print(f"\n  ✅ 提前收敛于 round {k}！")
                break

        # 找最佳
        objs = history.obj_curve()
        valid_objs = [(i, o) for i, o in enumerate(objs) if not np.isnan(o)]
        if valid_objs:
            best = min(valid_objs, key=lambda x: x[1])
            history.best_round = best[0]
            history.best_obj = float(best[1])

        history.total_time_s = round(time.time() - t_start, 1)
        self._save_history(history)

        if self.verbose:
            self._print_summary(history)
        return history

    def _save_history(self, history: EvolutionHistory):
        path = self.out_dir / "history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def _print_summary(history: EvolutionHistory):
        print(f"\n{'='*70}")
        print(f"  Evolution 完成！")
        print(f"  总轮数：{len(history.rounds) - 1}（含 round 0）")
        print(f"  收敛：{'✅' if history.converged else '❌（未提前停止）'}")
        print(f"  最佳轮：round {history.best_round}  obj={history.best_obj:.4f}")
        print(f"  总用时：{history.total_time_s:.1f}s")
        print(f"\n  Obj 演化曲线：")
        for i, r in enumerate(history.rounds):
            obj = r.eval_metrics.get("mean_obj", float("nan"))
            tag = "★" if i == history.best_round else " "
            n_r = r.n_rules_used if i > 0 else 0
            print(f"    {tag} round_{i}: obj={obj:.4f}  "
                  f"rules_used={n_r}  ({r.total_time_s:.1f}s)")
        print(f"{'='*70}\n")


# ============================================================
# 序列化
# ============================================================
def save_history(history: EvolutionHistory, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history.to_dict(), f, indent=2, ensure_ascii=False)


def load_history(path: str | Path) -> EvolutionHistory:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rounds = [RoundResult(**r) for r in data.get("rounds", [])]
    return EvolutionHistory(
        rounds=rounds,
        converged=data.get("converged", False),
        best_round=data.get("best_round", -1),
        best_obj=data.get("best_obj", float("inf")),
        config_snapshot=data.get("config_snapshot", {}),
        total_time_s=data.get("total_time_s", 0.0),
        created_at=data.get("created_at", ""),
    )

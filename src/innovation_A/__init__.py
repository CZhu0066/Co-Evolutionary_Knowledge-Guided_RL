# -*- coding: utf-8 -*-
"""
src/innovation_A/
==================
Innovation A 完整套件：4 个子贡献。

A1 扰动鲁棒训练       → 通过 enable_disturbance 训练（在 env 层）
A2 状态-扰动耦合蒸馏  → feature_extractor + trajectory_collector + distillation
A3 协同进化闭环       → evolution_manager (v13 新增)
A4 因果反事实验证     → causal_validator
"""
from src.innovation_A.feature_extractor import (
    FeatureExtractor, FEATURE_NAMES, N_FEATURES,
    TASK_KIND_TO_LABEL, LABEL_TO_TASK_KIND,
    encode_decision, decode_label,
)
from src.innovation_A.trajectory_collector import (
    collect_trajectories, save_trajectories, load_trajectories,
    summarize_trajectories,
)
from src.innovation_A.distillation import (
    Rule, RuleCondition,
    train_decision_tree, extract_rules, distill_rules,
    distill_rules_by_scenario, save_rules_by_scenario,
    save_rules, load_rules, print_rules,
)
from src.innovation_A.rule_bonus import (
    RuleEngine, make_rule_bonus_fn, make_rule_bonus_fn_from_json,
)
from src.innovation_A.causal_validator import (
    ValidationResult, ValidationReport,
    CausalValidator,
    save_report, load_report, filter_rules_by_validation,
)
from src.innovation_A.evolution_manager import (
    RoundResult, EvolutionHistory,
    EvolutionManager,
    save_history, load_history,
)

from src.innovation_A.template_rules import (
    build_template_rules, write_template_rules, PRIORITY_ORDER,
)
from src.innovation_A.template_rule_completion import (
    complete_template_guidance_rules, save_template_completion_json,
)

__all__ = [
    # feature_extractor
    "FeatureExtractor", "FEATURE_NAMES", "N_FEATURES",
    "TASK_KIND_TO_LABEL", "LABEL_TO_TASK_KIND",
    "encode_decision", "decode_label",
    # trajectory_collector
    "collect_trajectories", "save_trajectories", "load_trajectories",
    "summarize_trajectories",
    # distillation
    "Rule", "RuleCondition",
    "train_decision_tree", "extract_rules", "distill_rules",
    "distill_rules_by_scenario", "save_rules_by_scenario",
    "save_rules", "load_rules", "print_rules",
    # rule_bonus
    "RuleEngine", "make_rule_bonus_fn", "make_rule_bonus_fn_from_json",
    # causal_validator (A4)
    "ValidationResult", "ValidationReport", "CausalValidator",
    "save_report", "load_report", "filter_rules_by_validation",
    # evolution_manager (A3)
    "RoundResult", "EvolutionHistory", "EvolutionManager",
    "save_history", "load_history",
    # template rules
    "build_template_rules", "write_template_rules", "PRIORITY_ORDER",
    "complete_template_guidance_rules", "save_template_completion_json",
]

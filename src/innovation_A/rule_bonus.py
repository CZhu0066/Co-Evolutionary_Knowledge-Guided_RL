# -*- coding: utf-8 -*-
"""
rule_bonus.py
=============
把蒸馏出的规则转换为可注入 KGUnifiedYardEnv 的 rule_bonus_fn。

工作机制：
  1. 训练时，PPO/SAC 输出 θ，env 用 score 函数选了一个具体任务
  2. step() 末尾，rule_bonus_fn(env, action) 被调用
  3. 我们从 env 提取特征，让规则引擎预测「应该选什么 kind 的任务」
  4. 比较 env.last_decision.task_kind 与规则预测：
     - 一致 → +bonus
     - 不一致 → -bonus（或 0）

这样 RL 训练时会被「鼓励」走规则给出的方向（A3 协同进化闭环的关键）。

设计选择：
  - 默认 bonus 幅度小（0.05），避免压过原始 reward 信号
  - confidence 高的规则权重大（bonus *= confidence）
  - 只在「最匹配的规则 confidence >= min_conf」时才生效
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.innovation_A.feature_extractor import (
    FeatureExtractor, FEATURE_NAMES, TASK_KIND_TO_LABEL, decode_label,
)


class RuleEngine:
    """
    根据规则集对 env 当前状态预测 task_kind 标签。

    用法:
        engine = RuleEngine.from_json("rules.json")
        label, conf, rule_id = engine.predict(features)
    """

    def __init__(
        self,
        rules: List[Dict[str, Any]],
        feature_names: List[str],
        min_confidence: float = 0.6,
    ):
        self.rules = rules
        self.feature_names = list(feature_names)
        self.min_confidence = float(min_confidence)
        # 建立 feature_name → index 映射
        self._name_to_idx = {n: i for i, n in enumerate(self.feature_names)}

    @classmethod
    def from_dict(cls, rules_data: Dict[str, Any],
                  min_confidence: float = 0.6) -> "RuleEngine":
        return cls(
            rules=rules_data.get("rules", []),
            feature_names=rules_data.get("feature_names", list(FEATURE_NAMES)),
            min_confidence=min_confidence,
        )

    @classmethod
    def from_json(cls, path: str | Path,
                  min_confidence: float = 0.6) -> "RuleEngine":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, min_confidence=min_confidence)

    def _match_rule(self, rule: Dict[str, Any], features: np.ndarray) -> bool:
        """判断 features 是否满足规则的所有 conditions"""
        for c in rule.get("conditions", []):
            fname = c["feature"]
            op = c["op"]
            thresh = float(c["threshold"])
            idx = self._name_to_idx.get(fname)
            if idx is None or idx >= features.shape[0]:
                return False
            v = float(features[idx])
            if op == "<=" and not (v <= thresh):
                return False
            if op == ">" and not (v > thresh):
                return False
        return True

    def predict(self, features: np.ndarray) -> tuple:
        """
        预测当前特征下应该选什么 task_kind。

        Returns:
            (predicted_label, confidence, rule_id)
            predicted_label = -1 表示没有任何规则匹配
        """
        # 找到所有匹配的规则；DT 的叶子之间互斥，最多匹配一条
        # 但为防御，按 confidence 高的排序
        best = None
        for r in self.rules:
            if self._match_rule(r, features):
                if (best is None) or (r["confidence"] > best["confidence"]):
                    best = r
        if best is None:
            return -1, 0.0, -1
        return int(best["predicted_label"]), float(best["confidence"]), int(best["id"])


def make_rule_bonus_fn(
    rules_data: Dict[str, Any],
    extractor: Optional[FeatureExtractor] = None,
    # bonus_match: float = 0.02,
    # bonus_mismatch: float = -0.05,
    # min_confidence: float = 0.6,
    bonus_match: float = 0.02,
    bonus_mismatch: float = -0.005,
    min_confidence: float = 0.6,
    horizon: float = 10000.0,
) -> Callable[[Any, np.ndarray], float]:
    """
    构造可注入 env 的 rule_bonus_fn。

    诊断增强版：
    1. 支持动作前特征 capture_features(env)
    2. 记录规则调用/命中/匹配/不匹配/奖励总量
    3. 把最后一次规则判断细节写入 env._last_rule_bonus_detail
    """
    # 兼容两种输入：
    # 1) 单规则包：rules_data["rules"]
    # 2) 分场景规则包：rules_data["rules_by_scenario"][scenario]["rules"]
    is_by_scenario = isinstance(rules_data, dict) and "rules_by_scenario" in rules_data
    if is_by_scenario:
        engines_by_scenario = {
            str(sc): RuleEngine.from_dict(pack, min_confidence=min_confidence)
            for sc, pack in rules_data.get("rules_by_scenario", {}).items()
        }
        engine = None
        total_rules = sum(len(e.rules) for e in engines_by_scenario.values())
    else:
        engines_by_scenario = {}
        engine = RuleEngine.from_dict(rules_data, min_confidence=min_confidence)
        total_rules = len(engine.rules)

    if extractor is None:
        extractor = FeatureExtractor(horizon=horizon)

    def _ensure_stats(env):
        if not hasattr(env, "_rule_bonus_stats") or env._rule_bonus_stats is None:
            env._rule_bonus_stats = {
                "calls": 0,
                "no_decision": 0,
                "extract_error": 0,
                "no_match": 0,
                "low_confidence": 0,
                "hits": 0,
                "matched": 0,
                "mismatched": 0,
                "total_bonus": 0.0,
                "abs_total_bonus": 0.0,
                "feature_before_action": 0,
                "feature_after_action_fallback": 0,
            }
        return env._rule_bonus_stats

    def capture_features(env):
        """
        在 env.step() 刚开始、动作真正执行前调用。
        这样 rule_bonus 使用的是“动作前状态”，和蒸馏规则训练时一致。
        """
        try:
            return extractor.extract(env)
        except Exception:
            return None

    def rule_bonus_fn(env, action) -> float:
        stats = _ensure_stats(env)
        stats["calls"] += 1

        env._last_rule_bonus_detail = {
            "rule_id": -1,
            "predicted_label": -1,
            "predicted_kind": "none",
            "actual_label": -1,
            "actual_kind": "none",
            "confidence": 0.0,
            "matched": False,
            "bonus": 0.0,
            "reason": "",
            "feature_source": "",
            "scenario": str(getattr(env, "scenario", "unknown")),
        }

        # 没有上一步决策记录 → 无 bonus
        last_decision = getattr(env, "_last_decision", None)
        if not last_decision or "task_kind" not in last_decision:
            stats["no_decision"] += 1
            env._last_rule_bonus_detail["reason"] = "no_decision"
            return 0.0

        # 优先使用动作前保存的特征
        feat = getattr(env, "_rule_bonus_features_before_action", None)

        if feat is not None:
            stats["feature_before_action"] += 1
            feature_source = "before_action"
        else:
            # 兜底：如果没有动作前特征，才用动作后状态
            try:
                feat = extractor.extract(env)
                stats["feature_after_action_fallback"] += 1
                feature_source = "after_action_fallback"
            except Exception:
                stats["extract_error"] += 1
                env._last_rule_bonus_detail["reason"] = "extract_error"
                return 0.0

        # 用完后清空，避免下一步误用旧特征
        try:
            env._rule_bonus_features_before_action = None
        except Exception:
            pass

        # 分场景规则：根据当前 env.scenario 选择对应的规则引擎。
        # 若 env 没有 scenario，则从 env.instance["raw"]["scenario"] 兜底。
        scenario = str(getattr(env, "scenario", "unknown"))
        if scenario == "unknown":
            try:
                scenario = str(env.instance.get("raw", {}).get("scenario", "unknown"))
            except Exception:
                scenario = "unknown"

        if is_by_scenario:
            engine_this = engines_by_scenario.get(scenario)
            if engine_this is None or len(engine_this.rules) == 0:
                stats["no_match"] += 1
                env._last_rule_bonus_detail.update({
                    "reason": "no_engine_for_scenario",
                    "scenario": scenario,
                    "feature_source": feature_source,
                })
                return 0.0
        else:
            engine_this = engine

        predicted_label, conf, rule_id = engine_this.predict(feat)

        actual_kind = last_decision["task_kind"]
        actual_label = TASK_KIND_TO_LABEL.get(actual_kind, -2)

        if predicted_label < 0:
            stats["no_match"] += 1
            env._last_rule_bonus_detail.update({
                "actual_label": int(actual_label),
                "actual_kind": str(actual_kind),
                "reason": "no_rule_matched",
                "feature_source": feature_source,
                "scenario": scenario,
            })
            return 0.0

        if conf < min_confidence:
            stats["low_confidence"] += 1
            env._last_rule_bonus_detail.update({
                "rule_id": int(rule_id),
                "predicted_label": int(predicted_label),
                "predicted_kind": decode_label(int(predicted_label)),
                "actual_label": int(actual_label),
                "actual_kind": str(actual_kind),
                "confidence": float(conf),
                "reason": "low_confidence",
                "feature_source": feature_source,
                "scenario": scenario,
            })
            return 0.0

        stats["hits"] += 1

        matched = (predicted_label == actual_label)
        if matched:
            bonus = float(bonus_match * conf)
            stats["matched"] += 1
            reason = "matched"
        else:
            bonus = float(bonus_mismatch * conf)
            stats["mismatched"] += 1
            reason = "mismatched"

        stats["total_bonus"] += float(bonus)
        stats["abs_total_bonus"] += abs(float(bonus))

        env._last_rule_bonus_detail.update({
            "rule_id": int(rule_id),
            "predicted_label": int(predicted_label),
            "predicted_kind": decode_label(int(predicted_label)),
            "actual_label": int(actual_label),
            "actual_kind": str(actual_kind),
            "confidence": float(conf),
            "matched": bool(matched),
            "bonus": float(bonus),
            "reason": reason,
            "feature_source": feature_source,
            "scenario": scenario,
        })

        return float(bonus)

    # 给 env.step() 使用：动作执行前先抓取状态特征
    rule_bonus_fn.capture_features = capture_features
    rule_bonus_fn.engine = engine
    rule_bonus_fn.engines_by_scenario = engines_by_scenario
    rule_bonus_fn.n_rules = int(total_rules)

    return rule_bonus_fn


def make_rule_bonus_fn_from_json(
    rules_json_path: str | Path,
    **kwargs,
) -> Callable[[Any, np.ndarray], float]:
    """便利函数：直接从 JSON 文件构造"""
    with open(rules_json_path, "r", encoding="utf-8") as f:
        rules_data = json.load(f)
    return make_rule_bonus_fn(rules_data, **kwargs)

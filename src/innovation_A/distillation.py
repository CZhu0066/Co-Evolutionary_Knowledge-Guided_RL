# -*- coding: utf-8 -*-
"""
distillation.py
================
从收集到的轨迹蒸馏出人类可读的决策规则。

核心方法：
  1. 用 DecisionTreeClassifier 拟合 (features, task_kind_label)
  2. 限制树深度（推荐 4-6）保证规则可读
  3. 提取每条叶子路径作为一条规则
  4. 输出 JSON 格式的规则集

输出格式：
{
  "method": "decision_tree",
  "n_features": 25,
  "feature_names": [...],
  "n_rules": K,
  "tree_depth": 5,
  "train_accuracy": 0.82,
  "rules": [
    {
      "id": 0,
      "conditions": [
        {"feature": "n_trains_blocked_ratio", "op": "<=", "threshold": 0.25},
        {"feature": "intensity_high", "op": ">", "threshold": 0.5}
      ],
      "predicted_kind": "load_yard",
      "predicted_label": 0,
      "support": 120,
      "confidence": 0.91,
      "human_readable": "IF 阻塞列车 <= 25% AND 强度=高 THEN load_yard (support=120, conf=0.91)"
    },
    ...
  ]
}
"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.innovation_A.feature_extractor import FEATURE_NAMES, decode_label


@dataclass
class RuleCondition:
    feature: str       # 特征名
    op: str            # "<=" or ">"
    threshold: float   # 阈值


@dataclass
class Rule:
    id: int
    conditions: List[RuleCondition]
    predicted_kind: str
    predicted_label: int
    support: int                 # 训练集中走到这个叶子的样本数
    confidence: float            # 叶子节点的多数类占比
    human_readable: str = ""


def train_decision_tree(
    features: np.ndarray,
    labels: np.ndarray,
    max_depth: int = 5,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    class_weight: str = "balanced",
):
    """
    训练 DecisionTreeClassifier。

    Args:
        features: (N, n_features) 特征矩阵
        labels: (N,) 整数标签
        max_depth: 树深度上限（控制可读性）
        min_samples_leaf: 叶子节点最少样本数
        random_state: 随机种子
        class_weight: 类别不平衡处理（默认 "balanced"）

    Returns:
        tree: 训练好的 DecisionTreeClassifier
        train_acc: 训练集 accuracy
    """
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight=class_weight,
    )
    tree.fit(features, labels)
    train_acc = float(tree.score(features, labels))
    return tree, train_acc


def extract_rules(
    tree,
    feature_names: Optional[List[str]] = None,
) -> List[Rule]:
    """
    从 DecisionTree 提取规则（每个叶子一条）。

    Args:
        tree: 训练好的 sklearn DecisionTreeClassifier
        feature_names: 特征名列表（None 时用默认 FEATURE_NAMES）

    Returns:
        rules: list of Rule
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    t = tree.tree_
    rules: List[Rule] = []

    def _recurse(node_id: int, conditions: List[RuleCondition]):
        # 是叶子？
        if t.children_left[node_id] == t.children_right[node_id] == -1:
            # 叶子节点
            class_counts = t.value[node_id][0]   # shape (n_classes,)
            total = float(class_counts.sum())
            predicted_class_idx = int(class_counts.argmax())
            # tree.classes_ 是 [0,1,2,3]（已编码）
            predicted_label = int(tree.classes_[predicted_class_idx])
            support = int(total)
            confidence = float(class_counts[predicted_class_idx] / max(total, 1e-9))
            rule = Rule(
                id=len(rules),
                conditions=list(conditions),
                predicted_kind=decode_label(predicted_label),
                predicted_label=predicted_label,
                support=support,
                confidence=confidence,
            )
            rule.human_readable = _format_rule_text(rule)
            rules.append(rule)
            return
        # 内部节点
        feat_idx = int(t.feature[node_id])
        thresh = float(t.threshold[node_id])
        fname = feature_names[feat_idx]
        # 左分支：feature <= threshold
        left_cond = RuleCondition(feature=fname, op="<=", threshold=thresh)
        _recurse(int(t.children_left[node_id]), conditions + [left_cond])
        # 右分支：feature > threshold
        right_cond = RuleCondition(feature=fname, op=">", threshold=thresh)
        _recurse(int(t.children_right[node_id]), conditions + [right_cond])

    _recurse(0, [])
    return rules


def _format_rule_text(rule: Rule) -> str:
    """生成单条规则的中文可读字符串"""
    if not rule.conditions:
        return (f"IF (root) THEN {rule.predicted_kind} "
                f"(support={rule.support}, conf={rule.confidence:.2f})")
    parts = [f"{c.feature} {c.op} {c.threshold:.3f}" for c in rule.conditions]
    cond_str = " AND ".join(parts)
    return (f"IF {cond_str} THEN {rule.predicted_kind} "
            f"(support={rule.support}, conf={rule.confidence:.2f})")


def distill_rules(
    features: np.ndarray,
    labels: np.ndarray,
    max_depth: int = 5,
    min_samples_leaf: int = 5,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    一站式接口：训练树 → 提取规则 → 打包结果。

    Returns:
        dict（可直接 json.dump）
    """
    tree, train_acc = train_decision_tree(
        features, labels,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    rules = extract_rules(tree, feature_names=feature_names)

    if feature_names is None:
        feature_names = FEATURE_NAMES

    # 转 dict 表示
    rules_dict: List[Dict[str, Any]] = []
    for r in rules:
        rules_dict.append({
            "id": r.id,
            "conditions": [asdict(c) for c in r.conditions],
            "predicted_kind": r.predicted_kind,
            "predicted_label": r.predicted_label,
            "support": r.support,
            "confidence": r.confidence,
            "human_readable": r.human_readable,
        })

    result = {
        "method": "decision_tree",
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "n_features": features.shape[1],
        "n_samples": features.shape[0],
        "feature_names": list(feature_names),
        "n_rules": len(rules),
        "tree_depth_actual": int(tree.get_depth()),
        "train_accuracy": train_acc,
        "rules": rules_dict,
    }
    return result


def save_rules(rules_data: Dict[str, Any], path: str | Path):
    """保存规则到 JSON"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)
    print(f"[save_rules] {rules_data['n_rules']} 条规则 → {path}")


def load_rules(path: str | Path) -> Dict[str, Any]:
    """从 JSON 加载规则"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_rules(rules_data: Dict[str, Any], max_print: int = 20):
    """打印规则到控制台"""
    print(f"\n{'='*70}")
    print(f"  蒸馏规则集 (method={rules_data['method']}, "
          f"depth={rules_data.get('tree_depth_actual')}, "
          f"acc={rules_data.get('train_accuracy', 0):.3f})")
    print(f"  共 {rules_data['n_rules']} 条规则")
    print(f"{'='*70}")
    rules = rules_data.get("rules", [])
    rules_sorted = sorted(rules, key=lambda r: -r["support"])
    for i, r in enumerate(rules_sorted[:max_print]):
        print(f"\n  [{i+1}] {r['human_readable']}")
    if len(rules_sorted) > max_print:
        print(f"\n  ... (还有 {len(rules_sorted) - max_print} 条未显示)")
    print(f"{'='*70}\n")


# ============================================================
# 分场景规则蒸馏（方案 A）
# ============================================================
DEFAULT_SCENARIOS = ["1load_1unload", "2load", "2unload"]


def _empty_rule_pack(
    scenario: str,
    feature_names: List[str],
    n_features: int,
    n_samples: int = 0,
    reason: str = "",
    max_depth: int = 5,
    min_samples_leaf: int = 5,
) -> Dict[str, Any]:
    """构造空规则包，避免某个场景样本不足时流程中断。"""
    return {
        "method": "decision_tree",
        "scenario": scenario,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "n_features": int(n_features),
        "n_samples": int(n_samples),
        "feature_names": list(feature_names),
        "n_rules": 0,
        "tree_depth_actual": 0,
        "train_accuracy": 0.0,
        "rules": [],
        "skip_reason": reason,
    }


def distill_rules_by_scenario(
    features: np.ndarray,
    labels: np.ndarray,
    scenarios: np.ndarray,
    max_depth: int = 5,
    min_samples_leaf: int = 5,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
    scenario_order: Optional[List[str]] = None,
    min_samples_per_scenario: int = 20,
) -> Dict[str, Any]:
    """
    分场景蒸馏规则。

    输入的 trajectories.npz 需要包含 scenarios 字段。
    输出格式：
    {
      "method": "decision_tree_by_scenario",
      "rules_by_scenario": {
        "2load": {... 单场景 distill_rules 输出 ...},
        "2unload": {...},
        "1load_1unload": {...}
      },
      "n_rules": 总规则数,
      "n_rules_by_scenario": {...}
    }
    """
    if feature_names is None:
        feature_names = list(FEATURE_NAMES)
    else:
        feature_names = list(feature_names)

    scenarios_arr = np.asarray(scenarios).astype(str)
    if scenario_order is None:
        found = sorted(set(scenarios_arr.tolist()))
        # 固定三类场景在前，其他 unknown 放后面，方便日志稳定
        scenario_order = [s for s in DEFAULT_SCENARIOS if s in found]
        scenario_order += [s for s in found if s not in scenario_order]

    rules_by_scenario: Dict[str, Dict[str, Any]] = {}
    n_rules_by_scenario: Dict[str, int] = {}
    n_samples_by_scenario: Dict[str, int] = {}

    for sc in scenario_order:
        mask = scenarios_arr == sc
        X = features[mask]
        y = labels[mask]
        n_sc = int(len(y))
        n_samples_by_scenario[sc] = n_sc

        if n_sc < int(min_samples_per_scenario):
            pack = _empty_rule_pack(
                scenario=sc,
                feature_names=feature_names,
                n_features=features.shape[1] if features.ndim == 2 else len(feature_names),
                n_samples=n_sc,
                reason=f"too_few_samples<{min_samples_per_scenario}",
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            )
        elif len(set(y.tolist())) < 2:
            # 单类别也可以形成 root rule，但 sklearn tree 可读性有限；这里仍允许训练会更麻烦。
            # 为保守起见，直接构造一条 root 规则。
            label = int(y[0])
            pack = _empty_rule_pack(
                scenario=sc,
                feature_names=feature_names,
                n_features=features.shape[1],
                n_samples=n_sc,
                reason="single_class_root_rule",
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            )
            pack["rules"] = [{
                "id": 0,
                "conditions": [],
                "predicted_kind": decode_label(label),
                "predicted_label": label,
                "support": n_sc,
                "confidence": 1.0,
                "human_readable": f"IF (root; scenario={sc}) THEN {decode_label(label)} (support={n_sc}, conf=1.00)",
            }]
            pack["n_rules"] = 1
            pack["train_accuracy"] = 1.0
        else:
            pack = distill_rules(
                X, y,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                feature_names=feature_names,
                random_state=random_state,
            )
            pack["scenario"] = sc

        rules_by_scenario[sc] = pack
        n_rules_by_scenario[sc] = int(pack.get("n_rules", 0))

    result = {
        "method": "decision_tree_by_scenario",
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "n_features": features.shape[1] if features.ndim == 2 else len(feature_names),
        "n_samples": int(features.shape[0]),
        "feature_names": feature_names,
        "scenarios": list(scenario_order),
        "n_samples_by_scenario": n_samples_by_scenario,
        "n_rules_by_scenario": n_rules_by_scenario,
        "n_rules": int(sum(n_rules_by_scenario.values())),
        "rules_by_scenario": rules_by_scenario,
    }
    return result


def save_rules_by_scenario(rules_data: Dict[str, Any], path: str | Path):
    """保存分场景规则到 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)
    n_total = int(rules_data.get("n_rules", 0))
    by_sc = rules_data.get("n_rules_by_scenario", {})
    detail = ", ".join(f"{k}:{v}" for k, v in by_sc.items())
    print(f"[save_rules_by_scenario] {n_total} 条规则 ({detail}) → {path}")

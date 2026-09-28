# -*- coding: utf-8 -*-
"""
destination_rule_miner.py
=================
候选级 Destination 规则挖掘器。

为什么需要这个文件？
- 旧版规则主要是 state -> task_kind，只能得到 task 层 prefer 规则；
- Destination 规则需要观察“每一步有哪些 Destination 候选、每个候选有什么特征、最终哪台被选中”；
- 本文件从 env.step(info["destination_candidates"]) 中采集候选级样本，并用二分类决策树挖：
    candidate destination features -> selected(1/0)
- 只把 predicted selected=1 的高置信叶子转换为 stage="destination" 的 prefer 规则。

注意：这是第三层 Destination 规则：作用于 stage="destination" 的候选目的地/车厢/slot/cross组合评分。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# 不进入决策树的字段：ID/标签/字符串/诊断字段。
_NON_FEATURE_KEYS = {
    "episode", "step", "candidate_index", "selected", "selected_destination_id", "selected_dest_index",
    "scenario", "task_kind", "task_id", "dest_kind", "dest_id", "destination_id", "stage",
    "base_score", "guided_score", "final_score",
}

# 这些字段虽然是数字，但容易把规则变成“记住某个编号”，不利于泛化。
_ID_LIKE_FEATURES = {"task_train_id", "cross_id", "preferred_aqc_id", "preferred_destination_id"}


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return float(int(x))
        return float(x)
    except Exception:
        return default


def _is_number(x: Any) -> bool:
    return _safe_float(x, None) is not None


def _normalize_destination_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add stable truck-wait feature aliases used by template rules."""
    out = dict(row)
    tw = _safe_float(out.get("truck_total_wait", 0.0), 0.0) or 0.0
    out.setdefault("truck_total_wait_norm", float(tw) / 10000.0)
    out.setdefault("load_truck_wait_norm", float(out.get("truck_total_wait_norm", 0.0) or 0.0))
    out.setdefault("unload_truck_wait_norm", 0.0)
    return out


def collect_destination_candidate_samples(
    predict_fn,
    env_factory,
    n_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    verbose: bool = True,
    reuse_env: bool = False,
    probe_all_task_kinds: bool = False,
    probe_max_tasks_per_kind: int = 1,
) -> List[Dict[str, Any]]:
    """运行策略并从 info["destination_candidates"] 采集候选级 Destination 样本。

    v2 fixes:
    - reuse_env=True avoids recreating an env with the same seed every episode,
      which previously could keep sampling the same scenario such as 2unload.
    - probe_all_task_kinds=True additionally asks the env to *inspect* candidates
      for one unfinished task of each task_kind.  This does not execute those
      tasks; it only records candidate rows so load_truck destination templates
      have data even when the current policy rarely chooses load_truck.
    """
    samples: List[Dict[str, Any]] = []
    shared_env = env_factory() if bool(reuse_env) else None
    try:
        for ep in range(int(n_episodes)):
            env = shared_env if shared_env is not None else env_factory()
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            step = 0
            while (not done) and step < int(max_steps_per_episode):
                action = predict_fn(obs)
                if isinstance(action, tuple):
                    action = action[0]
                out = env.step(action)
                if len(out) == 5:
                    obs, reward, terminated, truncated, info = out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = out
                cands = list((info or {}).get("destination_candidates", []) or [])
                for c in cands:
                    row = _normalize_destination_row(dict(c))
                    row["episode"] = ep
                    row["step"] = step
                    row.setdefault("probe_kind", "executed_task")
                    samples.append(row)

                if bool(probe_all_task_kinds) and hasattr(env, "probe_destination_candidates_by_task_kind"):
                    try:
                        probe_rows = env.probe_destination_candidates_by_task_kind(
                            max_tasks_per_kind=int(probe_max_tasks_per_kind)
                        )
                    except Exception as e:
                        probe_rows = []
                        if verbose:
                            print(f"  [WARN] destination probe failed: {type(e).__name__}: {e}")
                    for c in probe_rows or []:
                        row = _normalize_destination_row(dict(c))
                        row["episode"] = ep
                        row["step"] = step
                        row.setdefault("probe_kind", "probe_task_kind")
                        # Probe rows are not actually chosen by the policy; keep selected=False.
                        row["selected"] = False
                        samples.append(row)
                step += 1
            if shared_env is None:
                try:
                    env.close()
                except Exception:
                    pass
            if verbose and ((ep + 1) % max(1, int(n_episodes) // 10) == 0 or ep + 1 == int(n_episodes)):
                print(f"  [collect_destination_candidates] {ep+1}/{n_episodes} episodes done, total candidate samples = {len(samples)}")
    finally:
        if shared_env is not None:
            try:
                shared_env.close()
            except Exception:
                pass
    return samples


def save_destination_candidate_samples(samples: Sequence[Dict[str, Any]], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in samples:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in samples:
            w.writerow(r)
    print(f"[save_destination_candidate_samples] {len(samples)} rows → {p}")
    return str(p)


def _numeric_feature_names(rows: Sequence[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for r in rows[: min(len(rows), 2000)]:
        for k, v in r.items():
            if k in _NON_FEATURE_KEYS or k in _ID_LIKE_FEATURES:
                continue
            if k.startswith("rule_") or k.startswith("matched_"):
                continue
            if _is_number(v) and k not in names:
                names.append(k)
    # 稳定排序，便于复现实验。
    return sorted(names)


def _build_matrix(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    X = np.zeros((len(rows), len(feature_names)), dtype=float)
    for i, r in enumerate(rows):
        for j, name in enumerate(feature_names):
            X[i, j] = float(_safe_float(r.get(name), 0.0) or 0.0)
    return X


def _extract_binary_tree_rules(tree, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    """从 binary DecisionTreeClassifier 中提取所有叶子规则。"""
    t = tree.tree_
    rules: List[Dict[str, Any]] = []

    def rec(node_id: int, conds: List[Dict[str, Any]]):
        if t.children_left[node_id] == t.children_right[node_id] == -1:
            counts = t.value[node_id][0]
            total = float(counts.sum())
            cls_idx = int(counts.argmax())
            pred = int(tree.classes_[cls_idx])
            support = int(total)
            conf = float(counts[cls_idx] / max(total, 1e-9))
            rules.append({
                "id": len(rules),
                "conditions": list(conds),
                "predicted_selected": int(pred),
                "support": support,
                "confidence": conf,
                "human_readable": "IF " + " AND ".join(
                    [f"{c['feature']} {c['op']} {c['threshold']:.4f}" for c in conds]
                ) + f" THEN selected={pred} (support={support}, conf={conf:.3f})",
            })
            return
        feat_idx = int(t.feature[node_id])
        thresh = float(t.threshold[node_id])
        fname = str(feature_names[feat_idx])
        rec(int(t.children_left[node_id]), conds + [{"feature": fname, "op": "<=", "threshold": thresh}])
        rec(int(t.children_right[node_id]), conds + [{"feature": fname, "op": ">", "threshold": thresh}])

    rec(0, [])
    return rules


def mine_destination_rules_by_group(
    samples: Sequence[Dict[str, Any]],
    max_depth: int = 6,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """按 scenario + task_kind 分组挖 Destination 二分类规则。"""
    from sklearn.tree import DecisionTreeClassifier

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in samples:
        scenario = str(r.get("scenario", "unknown"))
        task_kind = str(r.get("task_kind", "unknown"))
        dest_kind = str(r.get("dest_kind", "unknown"))
        key = f"{scenario}::{task_kind}::{dest_kind}"
        groups.setdefault(key, []).append(dict(r))

    out: Dict[str, Any] = {
        "method": "destination_candidate_binary_decision_tree_by_scenario_taskkind_destkind",
        "n_samples": int(len(samples)),
        "groups": {},
        "n_rules": 0,
    }
    total_rules = 0
    for key, rows in sorted(groups.items()):
        y = np.array([1 if bool(r.get("selected", False)) else 0 for r in rows], dtype=int)
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        parts = key.split("::")
        scenario = parts[0] if len(parts) > 0 else "unknown"
        task_kind = parts[1] if len(parts) > 1 else "unknown"
        dest_kind = parts[2] if len(parts) > 2 else "unknown"
        if len(rows) < max(20, int(min_samples_leaf) * 2) or positives <= 0 or negatives <= 0:
            out["groups"][key] = {
                "scenario": scenario,
                "task_kind": task_kind,
                "dest_kind": dest_kind,
                "n_samples": len(rows),
                "positives": positives,
                "negatives": negatives,
                "rules": [],
                "skip_reason": "too_few_samples_or_single_class",
            }
            continue
        feats = _numeric_feature_names(rows)
        if not feats:
            out["groups"][key] = {
                "scenario": scenario,
                "task_kind": task_kind,
                "dest_kind": dest_kind,
                "n_samples": len(rows),
                "positives": positives,
                "negatives": negatives,
                "rules": [],
                "skip_reason": "no_numeric_features",
            }
            continue
        X = _build_matrix(rows, feats)
        clf = DecisionTreeClassifier(
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            class_weight="balanced",
        )
        clf.fit(X, y)
        rules = _extract_binary_tree_rules(clf, feats)
        total_rules += len(rules)
        out["groups"][key] = {
            "scenario": scenario,
            "task_kind": task_kind,
            "dest_kind": dest_kind,
            "n_samples": len(rows),
            "positives": positives,
            "negatives": negatives,
            "positive_rate": float(positives / max(1, len(rows))),
            "feature_names": feats,
            "train_accuracy": float(clf.score(X, y)),
            "tree_depth_actual": int(clf.get_depth()),
            "rules": rules,
        }
    out["n_rules"] = int(total_rules)
    return out


def _cond_to_guidance(c: Dict[str, Any]) -> Dict[str, Any]:
    op = str(c.get("op", "eq"))
    return {
        "feature": str(c.get("feature", "")),
        "op": "le" if op == "<=" else ("gt" if op == ">" else op),
        "value": float(c.get("threshold", c.get("value", 0.0))),
    }


def convert_destination_rules_to_guidance(
    rules_data: Dict[str, Any],
    adjust: float = 80.0,
    min_confidence: float = 0.90,
    min_support: int = 30,
    min_coverage: float = 0.003,
    min_rule_conditions: int = 2,
    max_rule_conditions: int = 8,
    max_rules: int = 80,
) -> Dict[str, Any]:
    """把 Destination selected=1 规则转换为 stage='destination' 的 prefer 规则。"""
    candidates: List[Dict[str, Any]] = []
    rejected = {
        "predicted_not_selected": 0,
        "low_confidence": 0,
        "low_support": 0,
        "low_coverage": 0,
        "too_few_conditions": 0,
        "too_many_conditions": 0,
        "duplicate": 0,
    }
    seen = set()
    for key, pack in (rules_data.get("groups") or {}).items():
        scenario = str(pack.get("scenario", "unknown"))
        task_kind = str(pack.get("task_kind", "unknown"))
        dest_kind = str(pack.get("dest_kind", "unknown"))
        denom = max(1, int(pack.get("n_samples", 1) or 1))
        for r in pack.get("rules", []) or []:
            if int(r.get("predicted_selected", 0)) != 1:
                rejected["predicted_not_selected"] += 1
                continue
            conf = float(r.get("confidence", 0.0))
            support = int(r.get("support", 0) or 0)
            coverage = float(support) / float(denom)
            raw_conds = list(r.get("conditions", []) or [])
            n_conds = len(raw_conds)
            if conf < float(min_confidence):
                rejected["low_confidence"] += 1
                continue
            if support < int(min_support):
                rejected["low_support"] += 1
                continue
            if coverage < float(min_coverage):
                rejected["low_coverage"] += 1
                continue
            if n_conds < int(min_rule_conditions):
                rejected["too_few_conditions"] += 1
                continue
            if int(max_rule_conditions) > 0 and n_conds > int(max_rule_conditions):
                rejected["too_many_conditions"] += 1
                continue
            conditions: List[Dict[str, Any]] = [
                {"feature": "scenario", "op": "eq", "value": scenario},
                {"feature": "task_kind", "op": "eq", "value": task_kind},
                {"feature": "dest_kind", "op": "eq", "value": dest_kind},
            ]
            for c in raw_conds:
                gc = _cond_to_guidance(c)
                if gc["feature"]:
                    conditions.append(gc)
            sig = (scenario, task_kind, dest_kind, tuple((c["feature"], c["op"], str(c["value"])) for c in conditions))
            if sig in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(sig)
            candidates.append({
                "scenario": scenario,
                "task_kind": task_kind,
                "dest_kind": dest_kind,
                "conditions": conditions,
                "confidence": conf,
                "support": support,
                "coverage": coverage,
                "n_state_conditions": n_conds,
                "human_readable": r.get("human_readable", ""),
            })
    candidates.sort(key=lambda x: (x["confidence"], x["support"], x["n_state_conditions"]), reverse=True)
    out_rules: List[Dict[str, Any]] = []
    for item in candidates[: int(max_rules)]:
        rid = f"rgcd_destination_prefer_{item['scenario']}_{item['task_kind']}_{item.get('dest_kind', 'dest')}_{len(out_rules):03d}"
        out_rules.append({
            "id": rid,
            "stage": "destination",
            "effect": "prefer",
            "adjust": float(adjust),
            "confidence": float(item["confidence"]),
            "support": int(item["support"]),
            "coverage": float(item["coverage"]),
            "n_state_conditions": int(item["n_state_conditions"]),
            "source": "destination_candidate_decision_tree_filtered",
            "human_readable": item.get("human_readable", ""),
            "conditions": item["conditions"],
        })
    return {
        "method": "destination_candidate_rules_to_rgcd_guidance_filtered",
        "description": "Candidate-level Destination selected-rules converted to stage=destination prefer score guidance.",
        "filter": {
            "min_confidence": float(min_confidence),
            "min_support": int(min_support),
            "min_coverage": float(min_coverage),
            "min_rule_conditions": int(min_rule_conditions),
            "max_rule_conditions": int(max_rule_conditions),
            "adjust": float(adjust),
            "max_rules": int(max_rules),
        },
        "rejected": rejected,
        "n_rules_before_limit": len(candidates),
        "n_rules": len(out_rules),
        "rules": out_rules,
    }


def save_json(data: Dict[str, Any], path: str | Path, label: str = "save_json") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[{label}] {data.get('n_rules', data.get('n_samples', ''))} → {p}")
    return str(p)

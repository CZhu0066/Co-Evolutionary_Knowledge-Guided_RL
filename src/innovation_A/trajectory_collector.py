# -*- coding: utf-8 -*-
"""
trajectory_collector.py
========================
跑训练好的模型，收集 (features, decision, reward) 轨迹用于规则蒸馏。

输出格式（npz）:
    features:      (N, n_features) float32
    labels:        (N,) int          # encoded task_kind label
    rewards:       (N,) float32
    aqc_ids:       (N,) int
    train_ids:     (N,) int
    task_ids:      (N,) int
    episode_ids:   (N,) int          # 每个 sample 所属 episode
    step_indices:  (N,) int          # episode 内的步数索引
    scenarios:     (N,) str          # 每个 sample 所属场景：2load / 2unload / 1load_1unload
    episode_final_objs: (n_episodes,) float32  # 每个 episode 的终止 obj
    n_disturbances:    (n_episodes,) int       # 每个 episode 应用的扰动数
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.innovation_A.feature_extractor import FeatureExtractor, encode_decision


def collect_trajectories(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    env_factory: Callable[[], Any],
    n_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    extractor: Optional[FeatureExtractor] = None,
    horizon: float = 10000.0,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    收集轨迹。

    Args:
        predict_fn: 接受 obs 返回 action 的函数 (一般是 trainer.predict)
        env_factory: 工厂函数，每次调用返回一个新的 env 实例
        n_episodes: 收集多少条 episode
        max_steps_per_episode: 单 episode 最大步数（防卡死）
        extractor: 特征提取器实例（None 时新建）
        horizon: 归一化用 horizon
        verbose: 是否打印进度

    Returns:
        dict of arrays (见模块 docstring)
    """
    if extractor is None:
        extractor = FeatureExtractor(horizon=horizon)

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    all_rewards: List[float] = []
    all_aqc_ids: List[int] = []
    all_train_ids: List[int] = []
    all_task_ids: List[int] = []
    all_episode_ids: List[int] = []
    all_step_indices: List[int] = []
    all_scenarios: List[str] = []
    episode_final_objs: List[float] = []
    episode_n_disturbances: List[int] = []

    # for ep in range(n_episodes):
    #     env = env_factory()
    #     obs, info = env.reset()
    #     ep_dist_count = 0
    #     last_obj = 0.0
    #
    #     for step in range(max_steps_per_episode):
    #         # 提取 step 前的特征（用于决策预测）
    #         feat = extractor.extract(env)
    #
    #         # 预测动作
    #         action = predict_fn(obs)
    #
    #         # 执行
    #         obs, reward, terminated, truncated, info = env.step(np.asarray(action))
    #
    #         last_obj = float(info.get("obj", last_obj))
    #         ep_dist_count += int(info.get("n_applied_disturbances", 0))
    #
    #         # 编码决策标签
    #         last_decision = info.get("last_decision", {})
    #         label = encode_decision(last_decision)
    #
    #         # 只记录有效决策（label != -1）
    #         if label >= 0:
    #             all_features.append(feat)
    #             all_labels.append(label)
    #             all_rewards.append(float(reward))
    #             all_aqc_ids.append(int(last_decision.get("aqc_id", -1)))
    #             all_train_ids.append(int(last_decision.get("train_id", -1)))
    #             all_task_ids.append(int(last_decision.get("task_id", -1)))
    #             all_episode_ids.append(ep)
    #             all_step_indices.append(step)
    #
    #         if terminated or truncated:
    #             break
    #
    #     episode_final_objs.append(last_obj)
    #     episode_n_disturbances.append(ep_dist_count)
    #
    #     if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
    #         print(f"  [collect_trajectories] {ep + 1}/{n_episodes} "
    #               f"episodes done, total samples = {len(all_features)}")
    # 关键修改：
    # 原来是每个 episode 都 env_factory() 新建环境。
    # 如果 env_factory 使用固定 seed，就会导致每个 episode 都抽到同一个实例。
    # 现在改成只创建一次环境，让 reset() 连续推进随机状态，
    # 这样多个 episode 才有机会覆盖不同训练实例。
    env = env_factory()

    for ep in range(n_episodes):
        obs, info = env.reset()
        # 当前 episode 的场景名。优先读取 env.scenario；若没有，则从 env.instance["raw"] 兜底。
        ep_scenario = str(getattr(env, "scenario", "unknown"))
        if ep_scenario == "unknown":
            try:
                ep_scenario = str(env.instance.get("raw", {}).get("scenario", "unknown"))
            except Exception:
                ep_scenario = "unknown"
        ep_dist_count = 0
        last_obj = 0.0

        for step in range(max_steps_per_episode):
            # 提取 step 前的特征（用于决策预测）
            feat = extractor.extract(env)

            # 预测动作
            action = predict_fn(obs)

            # 执行
            obs, reward, terminated, truncated, info = env.step(np.asarray(action))

            last_obj = float(info.get("obj", last_obj))
            ep_dist_count += int(info.get("n_applied_disturbances", 0))

            # 编码决策标签
            last_decision = info.get("last_decision", {})
            label = encode_decision(last_decision)

            # 只记录有效决策（label != -1）
            if label >= 0:
                all_features.append(feat)
                all_labels.append(label)
                all_rewards.append(float(reward))
                all_aqc_ids.append(int(last_decision.get("aqc_id", -1)))
                all_train_ids.append(int(last_decision.get("train_id", -1)))
                all_task_ids.append(int(last_decision.get("task_id", -1)))
                all_episode_ids.append(ep)
                all_step_indices.append(step)
                all_scenarios.append(ep_scenario)

            if terminated or truncated:
                break

        episode_final_objs.append(last_obj)
        episode_n_disturbances.append(ep_dist_count)

        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            print(f"  [collect_trajectories] {ep + 1}/{n_episodes} "
                  f"episodes done, total samples = {len(all_features)}")

    if hasattr(env, "close"):
        env.close()
    return {
        "features": np.asarray(all_features, dtype=np.float32),
        "labels": np.asarray(all_labels, dtype=np.int32),
        "rewards": np.asarray(all_rewards, dtype=np.float32),
        "aqc_ids": np.asarray(all_aqc_ids, dtype=np.int32),
        "train_ids": np.asarray(all_train_ids, dtype=np.int32),
        "task_ids": np.asarray(all_task_ids, dtype=np.int32),
        "episode_ids": np.asarray(all_episode_ids, dtype=np.int32),
        "step_indices": np.asarray(all_step_indices, dtype=np.int32),
        "scenarios": np.asarray(all_scenarios),
        "episode_final_objs": np.asarray(episode_final_objs, dtype=np.float32),
        "episode_n_disturbances": np.asarray(episode_n_disturbances, dtype=np.int32),
        "feature_names": np.asarray(extractor.feature_names),
    }


def save_trajectories(data: Dict[str, np.ndarray], path: str | Path):
    """保存轨迹到 .npz 文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **data)
    print(f"[save_trajectories] {len(data['features'])} samples → {path}")


def load_trajectories(path: str | Path) -> Dict[str, np.ndarray]:
    """从 .npz 加载轨迹"""
    npz = np.load(str(path), allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def summarize_trajectories(data: Dict[str, np.ndarray]) -> str:
    """生成可读的轨迹统计文本"""
    n = len(data["features"])
    n_ep = len(data["episode_final_objs"])
    label_counts = {}
    for L in data["labels"]:
        label_counts[int(L)] = label_counts.get(int(L), 0) + 1

    from src.innovation_A.feature_extractor import decode_label
    lines = [
        f"轨迹统计：",
        f"  总样本数：{n}",
        f"  episode 数：{n_ep}",
        f"  每个 episode 平均决策数：{n / max(1, n_ep):.1f}",
        f"  平均终止 obj：{data['episode_final_objs'].mean():.2f}",
        f"  平均扰动数/episode：{data['episode_n_disturbances'].mean():.2f}",
        f"  决策分布：",
    ]
    for label, count in sorted(label_counts.items()):
        kind = decode_label(label)
        ratio = count / max(1, n) * 100
        lines.append(f"    {kind:>15}: {count} ({ratio:.1f}%)")

    if "scenarios" in data:
        scen_counts = {}
        for sc in data["scenarios"]:
            sc = str(sc)
            scen_counts[sc] = scen_counts.get(sc, 0) + 1
        lines.append("  场景分布：")
        for sc, count in sorted(scen_counts.items()):
            ratio = count / max(1, n) * 100
            lines.append(f"    {sc:>15}: {count} ({ratio:.1f}%)")

    return "\n".join(lines)

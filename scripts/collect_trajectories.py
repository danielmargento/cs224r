"""Collect offline teaching trajectories using the three rollout policies.

Outputs a single pickle file with a list of Trajectory objects. Each
trajectory carries the cloned final BKT simulator so reward functions can
probe post-episode mastery without rerunning the env.

Usage:
    python scripts/collect_trajectories.py
    python scripts/collect_trajectories.py --episodes_per_policy 200 --output data/trajectories/dataset.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from src.env import FinanceTutorEnv
from src.policies import (
    DifficultyIncreasingPolicy,
    PrereqRespectingPolicy,
    RandomPolicy,
)
from src.rewards.base import Trajectory
from src.simulators.bkt import BKTParams, BKTSimulator


def load_content(content_dir: Path) -> tuple[list[str], dict]:
    with open(content_dir / "constants.json") as f:
        constants = json.load(f)
    with open(content_dir / "knowledge_graph.json") as f:
        graph = json.load(f)["graph"]
    return constants["concepts"], graph


def run_episode(env: FinanceTutorEnv, policy, seed: int) -> Trajectory:
    obs, _ = env.reset(seed=seed)
    policy.reset(seed=seed)
    done = False
    while not done:
        action = policy.select_action(env, obs)
        result = env.step(action)
        obs = result.obs
        done = result.terminated or result.truncated
    return Trajectory(
        steps=env.trajectory_steps,
        final_simulator=env.simulator.clone(),
        initial_simulator=env.initial_simulator.clone() if env.initial_simulator else None,
    )


def collect(
    concepts: list[str],
    graph: dict,
    episodes_per_policy: int,
    max_steps: int,
    seed: int,
) -> list[dict]:
    """Returns a list of records: {policy_name, episode_idx, trajectory}."""
    sim = BKTSimulator(concept_ids=concepts, params=BKTParams())
    env = FinanceTutorEnv(simulator=sim, concept_ids=concepts, max_steps=max_steps)

    policies = [
        RandomPolicy(seed=seed),
        PrereqRespectingPolicy(knowledge_graph=graph, concept_ids=concepts, seed=seed + 1),
        DifficultyIncreasingPolicy(max_steps=max_steps, seed=seed + 2),
    ]

    records = []
    for p_idx, policy in enumerate(policies):
        print(f"  Rolling out {policy.name}: {episodes_per_policy} episodes ...")
        t0 = time.time()
        for ep in range(episodes_per_policy):
            ep_seed = seed + p_idx * 10_000 + ep
            traj = run_episode(env, policy, seed=ep_seed)
            records.append(
                {"policy_name": policy.name, "episode_idx": ep, "trajectory": traj}
            )
        print(f"    done in {time.time() - t0:.2f}s")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="content")
    parser.add_argument("--output", type=str, default="data/trajectories/dataset.pkl")
    parser.add_argument("--episodes_per_policy", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    content_dir = Path(args.content_dir)
    concepts, graph = load_content(content_dir)
    print(f"Loaded {len(concepts)} concepts; collecting {args.episodes_per_policy} eps/policy")

    records = collect(concepts, graph, args.episodes_per_policy, args.max_steps, args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "records": records,
                "concepts": concepts,
                "max_steps": args.max_steps,
                "episodes_per_policy": args.episodes_per_policy,
                "seed": args.seed,
            },
            f,
        )
    n = len(records)
    print(f"\nWrote {n} trajectories ({3 * args.episodes_per_policy} expected) to {out_path}")


if __name__ == "__main__":
    main()

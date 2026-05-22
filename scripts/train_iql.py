"""Train a from-scratch Discrete IQL policy on relabeled trajectories.

Loads the trajectory dataset and the per-episode relabeled reward, flattens
to transitions with reward placed at terminal step, fits a Discrete IQL
agent (Kostrikov et al. 2021), and evaluates by rolling out in the env.

Usage:
    PYTHONPATH=. python scripts/train_iql.py --reward score
    PYTHONPATH=. python scripts/train_iql.py --reward scenario --n_steps 20000
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from src.env import FinanceTutorEnv
from src.rewards.base import Trajectory
from src.simulators.bkt import BKTParams, BKTSimulator
from src.training import IQLConfig, action_index, build_mdp_dataset, train_offline


def evaluate_policy(algo, env: FinanceTutorEnv, n_episodes: int, seed: int = 0) -> list[Trajectory]:
    """Roll out the trained policy and return collected Trajectory objects."""
    trajectories: list[Trajectory] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = int(algo.predict(obs[None, :].astype(np.float32))[0])
            result = env.step(action)
            obs = result.obs
            done = result.terminated or result.truncated
        trajectories.append(
            Trajectory(
                steps=env.trajectory_steps,
                final_simulator=env.simulator.clone(),
                initial_simulator=env.initial_simulator.clone() if env.initial_simulator else None,
            )
        )
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="content")
    parser.add_argument("--dataset", type=str, default="data/trajectories/dataset.pkl")
    parser.add_argument("--rewards", type=str, default="data/trajectories/rewards.pkl")
    parser.add_argument("--reward", type=str, default="score", choices=["score", "scenario"])
    parser.add_argument("--n_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/checkpoints")
    args = parser.parse_args()

    content_dir = Path(args.content_dir)
    with open(content_dir / "constants.json") as f:
        concepts = json.load(f)["concepts"]

    with open(args.dataset, "rb") as f:
        ds = pickle.load(f)
    with open(args.rewards, "rb") as f:
        rw = pickle.load(f)
    records = ds["records"]
    episode_rewards = rw["rewards"][args.reward]
    print(f"Loaded {len(records)} trajectories. Training on '{args.reward}' reward.")

    t0 = time.time()
    print("Building transitions ...")
    dataset = build_mdp_dataset(records, episode_rewards, concepts)
    n_actions = len(concepts) * 3
    print(f"  transitions: {dataset['s'].shape[0]} | obs_dim: 16 | n_actions: {n_actions}")

    print(f"Training Discrete IQL for {args.n_steps} steps ...")
    cfg = IQLConfig(n_steps=args.n_steps, batch_size=args.batch_size, seed=args.seed)
    algo = train_offline(dataset, observation_shape=(16,), n_actions=n_actions, cfg=cfg)
    print(f"  trained in {time.time() - t0:.1f}s")

    # ----------------------------------------------------------- save model
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"iql_{args.reward}_seed{args.seed}.pt"
    algo.save(str(model_path))
    print(f"Saved checkpoint -> {model_path}")

    # -------------------------------------------------- in-env eval rollouts
    print(f"\nEvaluating policy with {args.eval_episodes} rollouts ...")
    sim = BKTSimulator(concept_ids=concepts, params=BKTParams())
    env = FinanceTutorEnv(simulator=sim, concept_ids=concepts, max_steps=50)
    eval_trajs = evaluate_policy(algo, env, n_episodes=args.eval_episodes, seed=args.seed)

    # Score the trained policy under both rewards using the same reward objects
    # used in relabel_rewards.py
    from src.rewards.scenario import Scenario, ScenarioReward
    from src.rewards.score import QuizItem, ScoreReward

    with open(content_dir / "item_bank.json") as f:
        ib = json.load(f)
    with open(content_dir / "scenarios.json") as f:
        sc = json.load(f)
    score_reward = ScoreReward(
        quiz_items=[QuizItem(id=q["id"], concept_id=q["concept"], difficulty=q["difficulty"]) for q in ib["quiz_items"]]
    )
    scenario_reward = ScenarioReward(
        scenarios=[Scenario(id=s["id"], concepts_required=s["concepts_tested"], difficulty=s["difficulty"]) for s in sc["scenarios"]],
        aggregation="product",
    )

    score_vals = np.array([score_reward(t) for t in eval_trajs])
    scenario_vals = np.array([scenario_reward(t) for t in eval_trajs])

    # Save eval trajectories + rewards for use by evaluate.py
    eval_out = {
        "policy_name": f"iql_{args.reward}",
        "trained_on": args.reward,
        "trajectories": eval_trajs,
        "score_rewards": score_vals.tolist(),
        "scenario_rewards": scenario_vals.tolist(),
        "n_steps": args.n_steps,
        "seed": args.seed,
    }
    eval_path = Path("data/trajectories") / f"iql_{args.reward}_eval.pkl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "wb") as f:
        pickle.dump(eval_out, f)
    print(f"Saved eval rollouts -> {eval_path}")

    print(
        f"\nLearned policy (trained on {args.reward}):"
        f"\n  score    : {score_vals.mean():.4f} ± {score_vals.std(ddof=1) / np.sqrt(len(score_vals)):.4f}"
        f"\n  scenario : {scenario_vals.mean():.4f} ± {scenario_vals.std(ddof=1) / np.sqrt(len(scenario_vals)):.4f}"
    )


if __name__ == "__main__":
    main()

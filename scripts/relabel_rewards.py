"""Apply each of the three reward functions to the trajectory dataset.

Loads the pickle produced by `collect_trajectories.py`, recomputes the
episode-end reward under each of:
  - ScoreReward      (held-out quiz items)
  - ScenarioReward   (multi-concept word problems, product aggregation)
  - PreferenceReward (untrained Bradley-Terry MLP -- placeholder until we
                     collect preference labels; produces a reproducible
                     scalar so the pipeline can be validated end-to-end)

Outputs:
  data/trajectories/rewards.pkl  -- dict[reward_name][record_idx] -> float

Usage:
    python scripts/relabel_rewards.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from src.rewards.preference import (
    BradleyTerryRewardModel,
    PreferenceReward,
    feature_dim,
)
from src.rewards.scenario import Scenario, ScenarioReward
from src.rewards.score import QuizItem, ScoreReward


def load_quiz_items(content_dir: Path) -> list[QuizItem]:
    with open(content_dir / "item_bank.json") as f:
        data = json.load(f)
    return [
        QuizItem(id=q["id"], concept_id=q["concept"], difficulty=q["difficulty"])
        for q in data["quiz_items"]
    ]


def load_scenarios(content_dir: Path) -> list[Scenario]:
    with open(content_dir / "scenarios.json") as f:
        data = json.load(f)
    return [
        Scenario(
            id=s["id"],
            concepts_required=s["concepts_tested"],
            difficulty=s["difficulty"],
        )
        for s in data["scenarios"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="content")
    parser.add_argument("--dataset", type=str, default="data/trajectories/dataset.pkl")
    parser.add_argument("--output", type=str, default="data/trajectories/rewards.pkl")
    parser.add_argument("--seed", type=int, default=0, help="Seed for preference-model init")
    args = parser.parse_args()

    content_dir = Path(args.content_dir)
    quiz_items = load_quiz_items(content_dir)
    scenarios = load_scenarios(content_dir)
    with open(content_dir / "constants.json") as f:
        concepts = json.load(f)["concepts"]

    with open(args.dataset, "rb") as f:
        ds = pickle.load(f)
    records = ds["records"]
    print(f"Loaded {len(records)} trajectories. Relabeling under 3 rewards ...")

    score = ScoreReward(quiz_items=quiz_items)
    scenario = ScenarioReward(scenarios=scenarios, aggregation="product")

    torch.manual_seed(args.seed)
    pref_model = BradleyTerryRewardModel(input_dim=feature_dim(len(concepts)))
    pref = PreferenceReward(model=pref_model, concept_ids=concepts)

    out: dict[str, list[float]] = {"score": [], "scenario": [], "preference": []}
    t0 = time.time()
    for i, rec in enumerate(records):
        traj = rec["trajectory"]
        out["score"].append(float(score(traj)))
        out["scenario"].append(float(scenario(traj)))
        out["preference"].append(float(pref(traj)))
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(records)} ({time.time() - t0:.1f}s)")
    print(f"Relabel done in {time.time() - t0:.1f}s")

    # Save rewards aligned by index with `records`.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "rewards": out,
                "policy_names": [r["policy_name"] for r in records],
                "concepts": concepts,
            },
            f,
        )
    print(f"Wrote rewards -> {out_path}")

    # Quick summary by policy.
    print("\n--- Mean reward by policy x reward ---")
    policies = sorted({r["policy_name"] for r in records})
    print(f"{'policy':<26} {'score':>8} {'scenario':>10} {'preference':>12}")
    for p in policies:
        mask = [r["policy_name"] == p for r in records]
        idxs = [i for i, m in enumerate(mask) if m]
        means = {k: float(np.mean([out[k][i] for i in idxs])) for k in out}
        print(f"{p:<26} {means['score']:>8.4f} {means['scenario']:>10.4f} {means['preference']:>12.4f}")


if __name__ == "__main__":
    main()

"""Evaluation: 3 rollout policies x 3 rewards + curriculum-structure metrics.

For the milestone we treat the three rollout policies as *baselines*. The
trained IQL policies (next milestone) will be evaluated through this same
harness. This script produces:
  - results/baseline_rewards.csv     reward x policy means & stderrs
  - results/baseline_structure.csv   curriculum-structure metrics per policy
  - results/figures/baselines.png    grouped bar chart

Usage:
    python scripts/evaluate.py
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from src.evaluation import (
    concept_entropy,
    first_hard_step,
    mean_difficulty,
    prerequisite_respect_rate,
    revisit_rate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="content")
    parser.add_argument("--dataset", type=str, default="data/trajectories/dataset.pkl")
    parser.add_argument("--rewards", type=str, default="data/trajectories/rewards.pkl")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    content_dir = Path(args.content_dir)
    with open(content_dir / "constants.json") as f:
        concepts = json.load(f)["concepts"]
    with open(content_dir / "knowledge_graph.json") as f:
        graph = json.load(f)["graph"]

    with open(args.dataset, "rb") as f:
        ds = pickle.load(f)
    with open(args.rewards, "rb") as f:
        rw = pickle.load(f)
    records = ds["records"]
    rewards = rw["rewards"]

    policies = sorted({r["policy_name"] for r in records})
    reward_names = ["score", "scenario"]  # preference omitted from milestone (untrained BT model)

    # -------------------------------------- Load trained policy eval rollouts
    trained_rows: list[dict] = []
    trained_traj_records: list[dict] = []
    for trained_reward in ["score", "scenario"]:
        path = Path(f"data/trajectories/iql_{trained_reward}_eval.pkl")
        if not path.exists():
            continue
        with open(path, "rb") as f:
            eval_data = pickle.load(f)
        score_arr = np.array(eval_data["score_rewards"])
        scen_arr = np.array(eval_data["scenario_rewards"])
        name = eval_data["policy_name"]  # e.g. "iql_score"
        trained_rows.append(
            {
                "policy": name,
                "n_episodes": len(score_arr),
                "score_mean": float(score_arr.mean()),
                "score_stderr": float(score_arr.std(ddof=1) / np.sqrt(len(score_arr))),
                "scenario_mean": float(scen_arr.mean()),
                "scenario_stderr": float(scen_arr.std(ddof=1) / np.sqrt(len(scen_arr))),
            }
        )
        for traj in eval_data["trajectories"]:
            trained_traj_records.append({"policy_name": name, "trajectory": traj})

    # -------------------------------------------------- reward summary table
    reward_rows = []
    for p in policies:
        idxs = [i for i, r in enumerate(records) if r["policy_name"] == p]
        row = {"policy": p, "n_episodes": len(idxs)}
        for rname in reward_names:
            arr = np.array([rewards[rname][i] for i in idxs])
            row[f"{rname}_mean"] = float(arr.mean())
            row[f"{rname}_stderr"] = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        reward_rows.append(row)
    reward_rows.extend(trained_rows)

    # -------------------------------------------- curriculum-structure metrics
    def structure_for(trajs, label: str) -> dict:
        entropies, revisits, prereqs, mean_diffs, first_hards = [], [], [], [], []
        for traj in trajs:
            entropies.append(concept_entropy(traj, concepts))
            revisits.append(revisit_rate(traj))
            prereqs.append(prerequisite_respect_rate(traj, graph, concepts))
            mean_diffs.append(mean_difficulty(traj))
            first_hards.append(first_hard_step(traj))
        return {
            "policy": label,
            "concept_entropy_mean": float(np.mean(entropies)),
            "revisit_rate_mean": float(np.mean(revisits)),
            "prereq_respect_mean": float(np.mean(prereqs)),
            "mean_difficulty_mean": float(np.mean(mean_diffs)),
            "first_hard_step_mean": float(np.mean(first_hards)),
        }

    structure_rows = []
    for p in policies:
        trajs = [records[i]["trajectory"] for i, r in enumerate(records) if r["policy_name"] == p]
        structure_rows.append(structure_for(trajs, p))
    for trained_p in sorted({t["policy_name"] for t in trained_traj_records}):
        trajs = [t["trajectory"] for t in trained_traj_records if t["policy_name"] == trained_p]
        structure_rows.append(structure_for(trajs, trained_p))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- CSV outputs
    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        headers = list(rows[0].keys())
        with open(path, "w") as f:
            f.write(",".join(headers) + "\n")
            for r in rows:
                f.write(",".join(str(r[h]) for h in headers) + "\n")

    rewards_csv = out_dir / "baseline_rewards.csv"
    structure_csv = out_dir / "baseline_structure.csv"
    write_csv(rewards_csv, reward_rows)
    write_csv(structure_csv, structure_rows)
    print(f"Wrote {rewards_csv}")
    print(f"Wrote {structure_csv}")
    print("(Run scripts/make_milestone_figure.py to produce the composite figure.)")

    # --------------------------------------------------- pretty print to stdout
    print("\n=== Reward by policy (mean ± stderr) ===")
    print(f"{'policy':<26} {'score':>15} {'scenario':>15}")
    for r in reward_rows:
        print(
            f"{r['policy']:<26}"
            f"  {r['score_mean']:.4f} ± {r['score_stderr']:.4f}"
            f"  {r['scenario_mean']:.4f} ± {r['scenario_stderr']:.4f}"
        )

    print("\n=== Curriculum-structure metrics (mean across episodes) ===")
    print(
        f"{'policy':<26} {'H(concept)':>11} {'revisit':>8} {'prereq_ok':>10}"
        f" {'mean_diff':>10} {'first_hard':>11}"
    )
    for r in structure_rows:
        print(
            f"{r['policy']:<26}"
            f"  {r['concept_entropy_mean']:>10.4f}"
            f"  {r['revisit_rate_mean']:>7.4f}"
            f"  {r['prereq_respect_mean']:>9.4f}"
            f"  {r['mean_difficulty_mean']:>9.4f}"
            f"  {r['first_hard_step_mean']:>10.2f}"
        )


if __name__ == "__main__":
    main()

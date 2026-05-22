"""Produce the milestone composite figure (2x2):
  top:     reward bars for score and scenario
  bottom:  difficulty-by-step line plot + concept-entropy / prereq-respect bars

Saves to results/figures/milestone.png. Run after evaluate.py.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import concept_entropy, prerequisite_respect_rate


CONTENT_DIR = Path("content")
RESULTS_DIR = Path("results")
DATA_DIR = Path("data/trajectories")

# Display labels (slightly nicer than the raw policy IDs)
LABELS = {
    "random": "random",
    "prereq_respecting": "prereq-respecting",
    "difficulty_increasing": "difficulty-increasing",
    "iql_score": "IQL (score)",
    "iql_scenario": "IQL (scenario)",
}

# A distinct categorical color + line-style per policy so the difficulty line
# plot is readable. Colors are colorblind-safe (Wong palette).
STYLES = {
    "random":               {"color": "#999999", "ls": ":",  "marker": "o", "lw": 1.6},
    "prereq_respecting":    {"color": "#0072B2", "ls": "-.", "marker": "s", "lw": 1.8},
    "difficulty_increasing":{"color": "#009E73", "ls": "-",  "marker": "^", "lw": 1.8},
    "iql_score":            {"color": "#D55E00", "ls": "--", "marker": "D", "lw": 2.2},
    "iql_scenario":         {"color": "#CC79A7", "ls": (0, (5, 1, 1, 1)), "marker": "v", "lw": 2.2},
}
DISPLAY_ORDER = list(LABELS.keys())
BASELINE_SET = {"random", "prereq_respecting", "difficulty_increasing"}


def load_all():
    with open(CONTENT_DIR / "constants.json") as f:
        concepts = json.load(f)["concepts"]
    with open(CONTENT_DIR / "knowledge_graph.json") as f:
        graph = json.load(f)["graph"]
    with open(DATA_DIR / "dataset.pkl", "rb") as f:
        ds = pickle.load(f)
    with open(DATA_DIR / "rewards.pkl", "rb") as f:
        rw = pickle.load(f)

    baseline_trajs: dict[str, list] = {}
    baseline_rewards: dict[str, dict[str, list[float]]] = {}
    for i, rec in enumerate(ds["records"]):
        p = rec["policy_name"]
        baseline_trajs.setdefault(p, []).append(rec["trajectory"])
        baseline_rewards.setdefault(p, {"score": [], "scenario": []})
        baseline_rewards[p]["score"].append(rw["rewards"]["score"][i])
        baseline_rewards[p]["scenario"].append(rw["rewards"]["scenario"][i])

    trained_trajs: dict[str, list] = {}
    trained_rewards: dict[str, dict[str, list[float]]] = {}
    for trained_reward in ("score", "scenario"):
        path = DATA_DIR / f"iql_{trained_reward}_eval.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as f:
            ed = pickle.load(f)
        name = ed["policy_name"]
        trained_trajs[name] = list(ed["trajectories"])
        trained_rewards[name] = {
            "score": list(ed["score_rewards"]),
            "scenario": list(ed["scenario_rewards"]),
        }

    all_trajs = {**baseline_trajs, **trained_trajs}
    all_rewards = {**baseline_rewards, **trained_rewards}
    return concepts, graph, all_trajs, all_rewards


def smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered rolling mean to clean up the per-step difficulty curve."""
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def main() -> None:
    # Typography & style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    concepts, graph, all_trajs, all_rewards = load_all()
    policies = [p for p in DISPLAY_ORDER if p in all_trajs]
    labels = [LABELS[p] for p in policies]

    # x-axis positions: insert a 0.6-unit gap between baselines and trained
    n_baseline = sum(1 for p in policies if p in BASELINE_SET)
    x = np.array([i if i < n_baseline else i + 0.6 for i in range(len(policies))])

    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32, top=0.92, bottom=0.08, left=0.08, right=0.95)
    ax_score = fig.add_subplot(gs[0, 0])
    ax_scen = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[1, 0])
    ax_struct = fig.add_subplot(gs[1, 1])

    # --------------------------------------------------- TOP: reward panels
    bar_colors = [STYLES[p]["color"] for p in policies]
    sep_x = (x[n_baseline - 1] + x[n_baseline]) / 2  # midpoint of the gap
    for ax, rname, title in [(ax_score, "score", "Score reward"), (ax_scen, "scenario", "Scenario reward")]:
        means, errs = [], []
        for p in policies:
            arr = np.asarray(all_rewards[p][rname])
            means.append(arr.mean())
            errs.append(arr.std(ddof=1) / np.sqrt(len(arr)))
        ax.bar(x, means, yerr=errs, color=bar_colors, capsize=4, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("mean episode reward")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax.axvline(sep_x, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
        # Group labels above bars
        y_top = ax.get_ylim()[1]
        ax.text((x[0] + x[n_baseline - 1]) / 2, y_top * 0.96, "baselines",
                ha="center", va="top", fontsize=9, style="italic", color="#444")
        ax.text((x[n_baseline] + x[-1]) / 2, y_top * 0.96, "trained (IQL)",
                ha="center", va="top", fontsize=9, style="italic", color="#444")

    # ----------------------------------- BOTTOM-LEFT: difficulty per timestep
    max_step = 50
    for p in policies:
        trajs = all_trajs[p]
        diff_grid = np.zeros((len(trajs), max_step), dtype=np.float32)
        for ti, traj in enumerate(trajs):
            for si, step in enumerate(traj.steps[:max_step]):
                diff_grid[ti, si] = step.difficulty
        mean_diff = diff_grid.mean(axis=0)
        # Light smoothing makes trends legible without distorting features
        smoothed = smooth(mean_diff, window=5)
        s = STYLES[p]
        ax_diff.plot(
            np.arange(max_step), smoothed,
            label=LABELS[p],
            color=s["color"], linestyle=s["ls"], linewidth=s["lw"],
            marker=s["marker"], markersize=5, markevery=7,
            markerfacecolor="white", markeredgewidth=1.4,
        )
    ax_diff.set_xlabel("step in episode")
    ax_diff.set_ylabel("mean selected difficulty")
    ax_diff.set_title("Difficulty selection over episode")
    ax_diff.set_ylim(0.9, 3.1)
    ax_diff.set_yticks([1, 2, 3])
    ax_diff.set_yticklabels(["1 (easy)", "2 (med.)", "3 (hard)"])
    ax_diff.legend(loc="upper left", fontsize=8.5, ncol=1, framealpha=0.92)
    ax_diff.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    # ----------------------- BOTTOM-RIGHT: concept entropy & prereq respect
    entropies, prereqs = [], []
    for p in policies:
        trajs = all_trajs[p]
        entropies.append(np.mean([concept_entropy(t, concepts) for t in trajs]))
        prereqs.append(np.mean([prerequisite_respect_rate(t, graph, concepts) for t in trajs]))

    w = 0.32
    ax2 = ax_struct.twinx()
    ax2.spines["top"].set_visible(False)
    bars_ent = ax_struct.bar(
        x - w / 2, entropies, width=w,
        color=bar_colors, edgecolor="black", linewidth=0.6,
    )
    bars_pr = ax2.bar(
        x + w / 2, prereqs, width=w,
        color=bar_colors, alpha=0.45, edgecolor="black", linewidth=0.6, hatch="///",
    )
    # Value labels on top of every bar so the side axes are secondary
    ax_struct.bar_label(bars_ent, fmt="%.2f", padding=2, fontsize=8)
    ax2.bar_label(bars_pr, fmt="%.2f", padding=2, fontsize=8)
    ax_struct.set_xticks(x)
    ax_struct.set_xticklabels(labels, rotation=22, ha="right")
    ax_struct.set_ylabel("concept entropy (nats)")
    ax2.set_ylabel("prerequisite-respect rate")
    ax_struct.set_title("Curriculum structure: diversity & prereq respect")
    ax_struct.set_ylim(0, max(entropies) * 1.18)
    ax2.set_ylim(0, 1.12)
    ax_struct.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax_struct.axvline(sep_x, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_struct.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    # Group labels omitted in this panel to avoid colliding with the bar-value
    # labels at the top; the dotted separator + grouped x-axis layout convey
    # baselines vs trained.

    # Axis labels themselves identify the two metrics; solid vs hatched bars
    # distinguish them. Caption explains in one sentence (no inline legend).
    ax_struct.set_ylabel("concept entropy (nats) — solid bars")
    ax2.set_ylabel("prerequisite-respect rate — hatched bars")

    fig.suptitle(
        "Reward design shapes both outcomes and curriculum structure\n"
        "Baselines: n = 200 episodes each.  Trained policies: n = 100 rollouts each.",
        fontsize=13, y=0.99,
    )
    out_path = RESULTS_DIR / "figures" / "milestone.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

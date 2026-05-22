"""Curriculum-structure metrics on completed trajectories.

These are pure functions over a Trajectory (no simulator queries). They
quantify HOW a policy teaches independent of WHAT mastery it produces:
    - concept_entropy        : diversity of taught concepts
    - revisit_rate           : fraction of steps spent on already-seen concepts
    - prerequisite_respect   : fraction of steps whose target concept has its
                               prereqs already mastered in the simulator's
                               belief at the time of the step
    - mean_difficulty        : mean of selected difficulties
    - first_hard_step        : step index when difficulty 3 is first picked
                               (max_steps if never)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.rewards.base import Trajectory


def concept_entropy(traj: Trajectory, concept_ids: Sequence[str]) -> float:
    """Shannon entropy (nats) of the empirical concept distribution."""
    n_c = len(concept_ids)
    counts = np.zeros(n_c, dtype=np.float64)
    idx = {c: i for i, c in enumerate(concept_ids)}
    for step in traj.steps:
        counts[idx[step.concept_id]] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def revisit_rate(traj: Trajectory) -> float:
    """Fraction of steps targeting a concept already seen earlier in the episode."""
    seen: set[str] = set()
    revisits = 0
    for step in traj.steps:
        if step.concept_id in seen:
            revisits += 1
        seen.add(step.concept_id)
    n = max(1, len(traj.steps))
    return revisits / n


def prerequisite_respect_rate(
    traj: Trajectory,
    knowledge_graph: dict,
    concept_ids: Sequence[str],
    threshold: float = 0.5,
) -> float:
    """Fraction of steps whose concept's prereqs are all >= `threshold` mastered
    in the agent's state at the time of the step."""
    n = max(1, len(traj.steps))
    if not traj.steps:
        return 1.0
    respected = 0
    idx = {c: i for i, c in enumerate(concept_ids)}
    for step in traj.steps:
        state = step.state
        prereqs = knowledge_graph[step.concept_id]["prerequisites"]
        if all(state[idx[p]] >= threshold for p in prereqs):
            respected += 1
    return respected / n


def mean_difficulty(traj: Trajectory) -> float:
    if not traj.steps:
        return 0.0
    return float(np.mean([s.difficulty for s in traj.steps]))


def first_hard_step(traj: Trajectory, hard_difficulty: int = 3) -> int:
    """Step index (0-based) of first difficulty==hard_difficulty selection,
    or len(traj.steps) if never reached."""
    for i, step in enumerate(traj.steps):
        if step.difficulty >= hard_difficulty:
            return i
    return len(traj.steps)

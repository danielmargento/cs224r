"""Score-based reward.

Standard signal from prior tutoring RL work. At the end of an episode, present
the held-out quiz set to the simulated learner and measure expected accuracy.

We use EXPECTED accuracy (E[correct] = P(correct)) rather than sampled
correctness because expectations have lower variance, which matters when the
reward feeds IQL on only ~10k trajectories. Sampling is available via the
`sample` flag if you want to study variance effects later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.rewards.base import BaseReward, Trajectory


@dataclass
class QuizItem:
    """Held-out quiz item. Not shown to the tutoring policy during episodes."""

    id: str
    concept_id: str
    difficulty: int = 1


@dataclass
class ScoreReward(BaseReward):
    """Reward = mean expected correctness on a fixed quiz set."""

    quiz_items: Sequence[QuizItem]
    sample: bool = False
    name: str = "score"

    def __call__(self, trajectory: Trajectory) -> float:
        sim = trajectory.final_simulator.clone()  # don't mutate live state
        if not self.quiz_items:
            return 0.0
        if self.sample:
            outcomes = [
                float(sim.sample_response(q.concept_id, q.difficulty)) for q in self.quiz_items
            ]
        else:
            outcomes = [sim.predict_correct(q.concept_id, q.difficulty) for q in self.quiz_items]
        return float(sum(outcomes) / len(outcomes))
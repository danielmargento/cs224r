"""Scenario-based reward.

Multi-concept word problems. A scenario requires the learner to apply several
concepts together; the proposal frames this as "operationalizing financial
decision-making rather than recall."

We model P(correct on scenario) as the product of P(correct) on each required
concept at the scenario's difficulty. This is the standard conjunctive model
used in cognitive diagnosis (Junker & Sijtsma, 2001): the learner must
succeed on every required component. It's strictly harder than single-concept
quiz items, which is the point.

We expose `aggregation` to swap in alternative models (mean instead of product,
weakest-link min, etc.) for ablations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from src.rewards.base import BaseReward, Trajectory


@dataclass
class Scenario:
    """Multi-concept word problem."""

    id: str
    concepts_required: Sequence[str]
    difficulty: int = 2


Aggregation = Literal["product", "mean", "min"]


@dataclass
class ScenarioReward(BaseReward):
    """Reward = mean expected scenario success across the scenario bank."""

    scenarios: Sequence[Scenario]
    aggregation: Aggregation = "product"
    name: str = "scenario"

    def __call__(self, trajectory: Trajectory) -> float:
        sim = trajectory.final_simulator.clone()
        if not self.scenarios:
            return 0.0
        scores = []
        for sc in self.scenarios:
            ps = [sim.predict_correct(c, sc.difficulty) for c in sc.concepts_required]
            if not ps:
                continue
            if self.aggregation == "product":
                scores.append(float(np.prod(ps)))
            elif self.aggregation == "mean":
                scores.append(float(np.mean(ps)))
            elif self.aggregation == "min":
                scores.append(float(np.min(ps)))
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
        return float(np.mean(scores)) if scores else 0.0
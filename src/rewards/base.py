"""Base interface for reward functions.

All three rewards (score, scenario, preference) operate on a completed
trajectory and return a scalar at episode end. This matches the proposal's
"reward not at every step, but rather calculated at the end of the episode"
spec.

A trajectory is a sequence of (state, action, response, next_state) tuples
plus the final post-episode simulator. Reward functions get both the action
sequence and a CLONE of the post-episode simulator they can probe freely
without mutating live state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.simulators.base import BaseSimulator


@dataclass
class Step:
    """One step of a teaching trajectory."""

    state: np.ndarray            # mastery vector before action
    concept_id: str              # action: which concept
    difficulty: int              # action: which difficulty
    correct: bool                # learner's response
    next_state: np.ndarray       # mastery vector after action


@dataclass
class Trajectory:
    """A complete teaching session (typically 50 steps).

    `final_simulator` should be a CLONE so reward functions can probe it
    without disturbing the original.
    """

    steps: Sequence[Step]
    final_simulator: BaseSimulator
    initial_simulator: BaseSimulator | None = None  # optional, for delta-mastery rewards


class BaseReward(ABC):
    """Abstract episode-end reward."""

    name: str

    @abstractmethod
    def __call__(self, trajectory: Trajectory) -> float:
        """Score a trajectory. Higher = better teaching."""
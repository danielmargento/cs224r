"""Base interface for student simulators.

Both BKT (training and eval) and any future DKT implementation conform to this
interface. The environment, reward functions, and rollout policies all program
against `BaseSimulator`, not against a concrete subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class BaseSimulator(ABC):
    """Abstract student simulator.

    A simulator tracks a learner's latent mastery over a fixed set of concepts.
    Practice items update mastery; predictions from mastery determine whether
    the learner answers a given item correctly.
    """

    #: Ordered concept IDs the simulator knows about. Index into mastery vectors.
    concept_ids: Sequence[str]

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Reset to a fresh learner. Optional seed for reproducibility."""

    @abstractmethod
    def observe(self, concept_id: str, correct: bool, difficulty: int = 1) -> None:
        """Update internal state given a learner's response to an item.

        Args:
            concept_id: Which concept the item targets.
            correct: Whether the learner answered correctly.
            difficulty: 1 (easy), 2 (medium), or 3 (hard). May modulate update.
        """

    @abstractmethod
    def predict_correct(self, concept_id: str, difficulty: int = 1) -> float:
        """Return P(correct) for an item on this concept at this difficulty.

        Does NOT mutate state. Use `sample_response` to draw an actual response.
        """

    def sample_response(
        self, concept_id: str, difficulty: int = 1, rng: np.random.Generator | None = None
    ) -> bool:
        """Bernoulli draw from `predict_correct`. Does NOT mutate state."""
        p = self.predict_correct(concept_id, difficulty)
        rng = rng if rng is not None else np.random.default_rng()
        return bool(rng.random() < p)

    @abstractmethod
    def get_mastery_vector(self) -> np.ndarray:
        """Return per-concept mastery probabilities aligned with `concept_ids`."""

    @abstractmethod
    def clone(self) -> "BaseSimulator":
        """Return a deep copy. Reward functions use this to evaluate post-
        episode state without disturbing the live simulator."""
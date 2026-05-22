"""Three behavioral rollout policies for offline trajectory collection.

Each policy implements `select_action(env, obs) -> int`. They are stateless
across episodes (or reset on episode start). Used by
`scripts/collect_trajectories.py`.

Policies
--------
- RandomPolicy: uniform over all (concept, difficulty) pairs.
- PrereqRespectingPolicy: at each step, pick a concept whose prerequisites
  are all "mastered" (mastery >= threshold) in the simulator's current
  belief; difficulty uniform.
- DifficultyIncreasingPolicy: split the episode into three phases and ramp
  difficulty 1 -> 2 -> 3; concept uniform.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


class RolloutPolicy(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None: ...
    def select_action(self, env, obs: np.ndarray) -> int: ...


# --------------------------------------------------------------------- helpers
def _action_index(env, concept: str, difficulty: int) -> int:
    return env.actions.index((concept, difficulty))


# ----------------------------------------------------------------- policy 1
@dataclass
class RandomPolicy:
    seed: int | None = None
    name: str = "random"
    _rng: np.random.Generator = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def select_action(self, env, obs: np.ndarray) -> int:
        return int(self._rng.integers(0, env.n_actions))


# ----------------------------------------------------------------- policy 2
@dataclass
class PrereqRespectingPolicy:
    """Pick concepts whose prereqs are mastered. Falls back to root concepts."""

    knowledge_graph: dict
    concept_ids: Sequence[str]
    mastery_threshold: float = 0.5
    seed: int | None = None
    name: str = "prereq_respecting"
    _rng: np.random.Generator = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        # Pre-compute root concepts (no prereqs) for fallback.
        self._roots = [
            c for c in self.concept_ids if not self.knowledge_graph[c]["prerequisites"]
        ]

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _unlocked(self, mastery: np.ndarray) -> list[str]:
        mastery_by_concept = dict(zip(self.concept_ids, mastery))
        unlocked = []
        for c in self.concept_ids:
            prereqs = self.knowledge_graph[c]["prerequisites"]
            if all(mastery_by_concept.get(p, 0.0) >= self.mastery_threshold for p in prereqs):
                unlocked.append(c)
        return unlocked

    def select_action(self, env, obs: np.ndarray) -> int:
        candidates = self._unlocked(obs) or self._roots
        concept = candidates[self._rng.integers(0, len(candidates))]
        difficulty = int(self._rng.integers(1, 4))
        return _action_index(env, concept, difficulty)


# ----------------------------------------------------------------- policy 3
@dataclass
class DifficultyIncreasingPolicy:
    """Ramp difficulty across the episode: easy -> medium -> hard."""

    max_steps: int = 50
    seed: int | None = None
    name: str = "difficulty_increasing"
    _rng: np.random.Generator = field(default=None, init=False, repr=False)
    _step_in_episode: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_in_episode = 0

    def _current_difficulty(self) -> int:
        third = self.max_steps // 3
        if self._step_in_episode < third:
            return 1
        elif self._step_in_episode < 2 * third:
            return 2
        return 3

    def select_action(self, env, obs: np.ndarray) -> int:
        difficulty = self._current_difficulty()
        concept = env.concept_ids[self._rng.integers(0, env.n_concepts)]
        self._step_in_episode += 1
        return _action_index(env, concept, difficulty)

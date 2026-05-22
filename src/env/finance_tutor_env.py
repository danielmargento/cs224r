"""Gym-style environment for the finance-tutoring offline RL problem.

The agent picks (concept, difficulty) practice items; the simulator decides
whether the learner gets each item right and updates its mastery. The episode
terminates after `max_steps` (default 50). Per the proposal, the reward is
computed only at episode end, but the env emits 0 at each step and an
"empty" terminal slot -- the reward functions are applied OFFLINE to
Trajectory objects during the relabel step, so the env doesn't need them.

Observation: per-concept mastery vector (length |concepts|), float32 in [0, 1].
Action: discrete index over (concept, difficulty) pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.rewards.base import Step
from src.simulators.base import BaseSimulator


@dataclass
class EnvStepResult:
    """A single env step's outcome. Mirrors gymnasium's (obs, reward, terminated, truncated, info)."""

    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class FinanceTutorEnv:
    """Minimal Gym-compatible env. Not a strict subclass of gymnasium.Env to avoid
    pinning version semantics during the milestone; we mirror the same surface."""

    def __init__(
        self,
        simulator: BaseSimulator,
        concept_ids: Sequence[str],
        difficulties: Sequence[int] = (1, 2, 3),
        max_steps: int = 50,
    ) -> None:
        self.simulator = simulator
        self.concept_ids = list(concept_ids)
        self.difficulties = list(difficulties)
        self.max_steps = max_steps

        self.actions = [(c, d) for c in self.concept_ids for d in self.difficulties]
        self.n_actions = len(self.actions)
        self.n_concepts = len(self.concept_ids)

        self._steps_taken = 0
        self._initial_simulator: BaseSimulator | None = None
        self._trajectory_steps: list[Step] = []

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self.simulator.reset(seed=seed)
        self._steps_taken = 0
        self._initial_simulator = self.simulator.clone()
        self._trajectory_steps = []
        return self._obs(), {}

    def step(self, action: int) -> EnvStepResult:
        if not 0 <= action < self.n_actions:
            raise IndexError(f"Action {action} out of range [0, {self.n_actions})")
        concept, difficulty = self.actions[action]
        state_before = self.simulator.get_mastery_vector()
        correct = self.simulator.sample_response(concept, difficulty)
        self.simulator.observe(concept, correct, difficulty=difficulty)
        state_after = self.simulator.get_mastery_vector()

        self._trajectory_steps.append(
            Step(
                state=state_before,
                concept_id=concept,
                difficulty=difficulty,
                correct=correct,
                next_state=state_after,
            )
        )

        self._steps_taken += 1
        terminated = self._steps_taken >= self.max_steps

        return EnvStepResult(
            obs=state_after,
            reward=0.0,  # rewards are applied offline by the relabeler
            terminated=terminated,
            truncated=False,
            info={"concept": concept, "difficulty": difficulty, "correct": correct},
        )

    # ---------------------------------------------------------------- helpers
    def _obs(self) -> np.ndarray:
        return self.simulator.get_mastery_vector()

    @property
    def trajectory_steps(self) -> list[Step]:
        """Steps emitted since the last reset (a complete episode)."""
        return list(self._trajectory_steps)

    @property
    def initial_simulator(self) -> BaseSimulator | None:
        return self._initial_simulator

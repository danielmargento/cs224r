"""Bayesian Knowledge Tracing (BKT) simulator.

Classical BKT (Corbett & Anderson 1995) models each concept as an independent
two-state HMM:
  - Latent state: mastered or not mastered
  - Per-concept parameters:
      p_init   : P(mastered at start)
      p_learn  : P(transition unmastered -> mastered after one practice)
      p_slip   : P(incorrect | mastered)
      p_guess  : P(correct | not mastered)

Update rule given observation `correct`:
  posterior_pre   = P(mastered | observation)
  posterior_post  = posterior_pre + (1 - posterior_pre) * p_learn

We extend the classical model with difficulty modulation: harder items reduce
the effective learn rate and increase the slip rate. Tunable but defaults to a
mild effect.

This implementation is intentionally simple and per-concept independent --
that is the property that makes it a structurally distinct evaluator vs a DKT
LSTM, per the proposal.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .base import BaseSimulator


@dataclass
class BKTParams:
    """Per-concept BKT parameters. Defaults chosen for "reasonable learner"."""

    p_init: float = 0.1
    p_learn: float = 0.15
    p_slip: float = 0.10
    p_guess: float = 0.20

    def __post_init__(self) -> None:
        for name, val in [
            ("p_init", self.p_init),
            ("p_learn", self.p_learn),
            ("p_slip", self.p_slip),
            ("p_guess", self.p_guess),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"BKT param {name}={val} must lie in [0, 1]")


@dataclass
class BKTSimulator(BaseSimulator):
    """Per-concept independent BKT.

    Args:
        concept_ids: Ordered concept IDs.
        params: Either a single BKTParams (applied to every concept) or a dict
            mapping concept_id -> BKTParams.
        difficulty_slip_boost: Added to p_slip per difficulty level above 1
            (capped at 1.0). Default 0.05 means hard items slip ~10 percentage
            points more than easy items.
        difficulty_learn_decay: Multiplied into p_learn per difficulty level
            above 1. Default 0.85 means hard items teach ~28% less per practice
            than easy items.
        seed: For reproducibility of `sample_response`.
    """

    concept_ids: Sequence[str]
    params: BKTParams | dict[str, BKTParams] = field(default_factory=BKTParams)
    difficulty_slip_boost: float = 0.05
    difficulty_learn_decay: float = 0.85
    seed: int | None = None

    # Initialized in __post_init__
    _params_by_concept: dict[str, BKTParams] = field(init=False, repr=False)
    _mastery: dict[str, float] = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.params, BKTParams):
            self._params_by_concept = {c: self.params for c in self.concept_ids}
        else:
            missing = set(self.concept_ids) - set(self.params.keys())
            if missing:
                raise ValueError(f"Missing BKT params for concepts: {missing}")
            self._params_by_concept = dict(self.params)
        self._rng = np.random.default_rng(self.seed)
        self._mastery = {c: self._params_by_concept[c].p_init for c in self.concept_ids}

    # ------------------------------------------------------------------ core API
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mastery = {c: self._params_by_concept[c].p_init for c in self.concept_ids}

    def observe(self, concept_id: str, correct: bool, difficulty: int = 1) -> None:
        if concept_id not in self._mastery:
            raise KeyError(f"Unknown concept {concept_id!r}")
        p = self._params_by_concept[concept_id]
        prior = self._mastery[concept_id]
        slip = min(1.0, p.p_slip + self.difficulty_slip_boost * max(0, difficulty - 1))
        learn = p.p_learn * (self.difficulty_learn_decay ** max(0, difficulty - 1))

        # Bayesian update: posterior on mastery given the observation.
        if correct:
            num = prior * (1 - slip)
            den = prior * (1 - slip) + (1 - prior) * p.p_guess
        else:
            num = prior * slip
            den = prior * slip + (1 - prior) * (1 - p.p_guess)
        posterior_pre = num / den if den > 0 else prior

        # Learning transition.
        self._mastery[concept_id] = posterior_pre + (1 - posterior_pre) * learn

    def predict_correct(self, concept_id: str, difficulty: int = 1) -> float:
        if concept_id not in self._mastery:
            raise KeyError(f"Unknown concept {concept_id!r}")
        p = self._params_by_concept[concept_id]
        m = self._mastery[concept_id]
        slip = min(1.0, p.p_slip + self.difficulty_slip_boost * max(0, difficulty - 1))
        # P(correct) = P(mastered) * (1 - slip) + P(not mastered) * guess
        return m * (1.0 - slip) + (1.0 - m) * p.p_guess

    def get_mastery_vector(self) -> np.ndarray:
        return np.array([self._mastery[c] for c in self.concept_ids], dtype=np.float32)

    def clone(self) -> "BKTSimulator":
        """Deep copy. Reward fns use this to score without mutating live state."""
        new = copy.copy(self)
        new._mastery = dict(self._mastery)
        new._params_by_concept = dict(self._params_by_concept)
        # Fresh RNG with current state so clones diverge in sampling.
        new._rng = np.random.default_rng(self._rng.integers(0, 2**32 - 1))
        return new
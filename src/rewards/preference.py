"""Preference-based reward via a learned Bradley-Terry reward model.

Follows the RLHF paradigm: collect pairwise comparisons between teaching
sessions, train a reward model under the Bradley-Terry objective, then use
its scalar output as the reward signal for RL.

Bradley-Terry objective on a comparison (A preferred over B):
    L = -log sigmoid(r(A) - r(B))

Trajectory featurization: we feed a fixed-size feature vector summarizing the
session (final mastery vector, action histogram, response correctness rate,
revisit count). This avoids the complications of running an LSTM/Transformer
over variable-length trajectories for a milestone deliverable. We can swap to
a sequence model later.

Data collection (TODO): The proposal calls for human-labeled pairwise
comparisons. For the milestone, this class supports loading preference data
from disk; the data-collection script is left for a later sprint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rewards.base import BaseReward, Trajectory


# ---------------------------------------------------------------- featurization
def featurize_trajectory(traj: Trajectory, concept_ids: Sequence[str]) -> np.ndarray:
    """Fixed-length summary vector for a trajectory.

    Features (length = 3 * |concepts| + 4):
      - final mastery per concept                                  (|concepts|)
      - action histogram: fraction of steps per concept            (|concepts|)
      - per-concept correctness rate (NaN -> 0 if concept unseen)  (|concepts|)
      - overall correctness rate                                   (1)
      - revisit rate: fraction of steps on already-seen concepts   (1)
      - mean difficulty                                            (1)
      - episode length                                             (1)
    """
    n_c = len(concept_ids)
    idx = {c: i for i, c in enumerate(concept_ids)}

    mastery = traj.final_simulator.get_mastery_vector()

    action_counts = np.zeros(n_c, dtype=np.float32)
    correct_counts = np.zeros(n_c, dtype=np.float32)
    total_correct = 0
    seen: set[str] = set()
    revisits = 0
    diffs: list[int] = []

    for step in traj.steps:
        i = idx[step.concept_id]
        action_counts[i] += 1
        if step.correct:
            correct_counts[i] += 1
            total_correct += 1
        if step.concept_id in seen:
            revisits += 1
        seen.add(step.concept_id)
        diffs.append(step.difficulty)

    n = max(1, len(traj.steps))
    action_hist = action_counts / n
    correctness_rate = np.divide(
        correct_counts,
        action_counts,
        out=np.zeros_like(correct_counts),
        where=action_counts > 0,
    )
    overall = total_correct / n
    revisit_rate = revisits / n
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    length = float(len(traj.steps))

    return np.concatenate(
        [
            mastery.astype(np.float32),
            action_hist,
            correctness_rate.astype(np.float32),
            np.array([overall, revisit_rate, mean_diff, length], dtype=np.float32),
        ]
    )


def feature_dim(n_concepts: int) -> int:
    return 3 * n_concepts + 4


# -------------------------------------------------------------- reward model NN
class BradleyTerryRewardModel(nn.Module):
    """MLP reward model: features -> scalar reward."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def bradley_terry_loss(r_preferred: torch.Tensor, r_other: torch.Tensor) -> torch.Tensor:
    """L = -log sigmoid(r_preferred - r_other), averaged over batch."""
    return -F.logsigmoid(r_preferred - r_other).mean()


# -------------------------------------------------------------- reward function
@dataclass
class PreferenceReward(BaseReward):
    """Reward = scalar output of a trained Bradley-Terry model.

    Usage:
        model = BradleyTerryRewardModel(feature_dim(20))
        model.load_state_dict(torch.load("results/checkpoints/pref_model.pt"))
        reward_fn = PreferenceReward(model=model, concept_ids=CONCEPTS)
    """

    model: BradleyTerryRewardModel
    concept_ids: Sequence[str]
    name: str = "preference"

    def __post_init__(self) -> None:
        self.model.eval()

    @torch.no_grad()
    def __call__(self, trajectory: Trajectory) -> float:
        feats = featurize_trajectory(trajectory, self.concept_ids)
        x = torch.from_numpy(feats).float().unsqueeze(0)
        return float(self.model(x).item())


# ------------------------------------------------------------ training utilities
def train_preference_model(
    model: BradleyTerryRewardModel,
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
) -> list[float]:
    """Train `model` on preference pairs. Each pair = (preferred_feats, other_feats).

    Returns the per-epoch loss curve. Data collection is TODO -- see
    `scripts/collect_preferences.py` (not yet implemented).
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    preferred = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32, device=device)
    other = torch.tensor(np.stack([p[1] for p in pairs]), dtype=torch.float32, device=device)
    n = len(pairs)

    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            r_pref = model(preferred[idx])
            r_other = model(other[idx])
            loss = bradley_terry_loss(r_pref, r_other)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(1, n_batches))
    model.eval()
    return losses
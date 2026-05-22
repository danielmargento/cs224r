"""From-scratch Discrete Implicit Q-Learning (IQL) for offline RL.

Implements Kostrikov et al. (2021), "Offline Reinforcement Learning with
Implicit Q-Learning", adapted to a discrete action space. Three networks:

  Q(s, ·) : discrete Q over n_actions (we use double Q for target stability)
  V(s)    : scalar value function
  π(a|s)  : categorical policy

Training (per batch of transitions (s, a, r, s', done)):

  1. V loss = expectile_loss(Q_target(s, a) - V(s); tau)
       expectile_loss_tau(u) = |tau - 1{u<0}| * u^2
       With tau > 0.5, V learns the upper expectile of Q -> approximates
       max_a Q(s,a) WITHOUT ever evaluating Q at off-distribution actions.

  2. Q loss = MSE(Q(s,a), r + gamma * V(s') * (1 - done))
       Bellman backup using V(s') on the next state. Stays in-sample.

  3. Policy loss = -E[ exp(beta * (Q_target(s,a) - V(s))) * log pi(a|s) ]
       Advantage-weighted regression: imitate dataset actions, weighted
       by how much above the value baseline they were.

Hyperparameters (defaults chosen for our 30k-transition, 50-step-episode,
sparse-terminal-reward setting):
    tau    = 0.7    (expectile)
    beta   = 3.0    (AWR temperature)
    gamma  = 0.99
    polyak = 0.005  (soft target update rate)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def action_index(concept_id: str, difficulty: int, concept_ids: Sequence[str]) -> int:
    """Map (concept, difficulty) to a flat action index."""
    return concept_ids.index(concept_id) * 3 + (difficulty - 1)


# ----------------------------------------------------------- networks
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------- dataset / batches
def build_transitions(records, episode_rewards, concept_ids, use_mc_returns: bool = True):
    """Flatten Trajectory records into transition tensors.

    With `use_mc_returns=True` (default), the trajectory's terminal reward is
    broadcast to every transition in that episode and `done` is set to 1
    everywhere. The Q update then degenerates to supervised regression of
    Q(s,a) toward the trajectory return (Monte Carlo target). This trades
    Bellman credit-assignment ability for tractable learning on small
    datasets with sparse terminal rewards. IQL's V (expectile of Q) and AWR
    policy extraction are unchanged.

    With `use_mc_returns=False`, reward is placed only at the terminal step
    (standard sparse-reward MDP); the trainer must propagate signal via
    Bellman backups, which requires many more SGD steps.

    Returns dict of tensors: s, a, r, s_next, done.
    """
    obs, acts, rews, obs_next, dones = [], [], [], [], []
    for rec, ep_r in zip(records, episode_rewards):
        traj = rec["trajectory"]
        n = len(traj.steps)
        for i, step in enumerate(traj.steps):
            obs.append(step.state.astype(np.float32))
            acts.append(action_index(step.concept_id, step.difficulty, concept_ids))
            is_last = i == n - 1
            if use_mc_returns:
                rews.append(float(ep_r))
                dones.append(1.0)
            else:
                rews.append(float(ep_r) if is_last else 0.0)
                dones.append(1.0 if is_last else 0.0)
            obs_next.append(step.next_state.astype(np.float32))
    return {
        "s": torch.tensor(np.stack(obs), dtype=torch.float32),
        "a": torch.tensor(acts, dtype=torch.int64),
        "r": torch.tensor(rews, dtype=torch.float32),
        "s_next": torch.tensor(np.stack(obs_next), dtype=torch.float32),
        "done": torch.tensor(dones, dtype=torch.float32),
    }


# Backwards-compat alias (script imports build_mdp_dataset).
def build_mdp_dataset(records, episode_rewards, concept_ids):
    return build_transitions(records, episode_rewards, concept_ids)


# ---------------------------------------------------------------- trainer
@dataclass
class IQLConfig:
    n_steps: int = 5000
    batch_size: int = 256
    lr: float = 3e-4
    tau: float = 0.7        # expectile
    beta: float = 3.0       # AWR temperature
    gamma: float = 0.99
    polyak: float = 0.005   # target net soft-update rate
    hidden: int = 128
    seed: int = 0


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """L_tau(u) = |tau - 1{u<0}| * u^2  (Kostrikov 2021, eq. 6)."""
    weight = torch.where(diff > 0, tau, 1 - tau)
    return weight * diff.pow(2)


class DiscreteIQL:
    """A fitted DiscreteIQL agent. Exposes `predict(obs) -> action_idx` for
    use by the env rollout loop in scripts/train_iql.py."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: IQLConfig) -> None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.cfg = cfg
        self.n_actions = n_actions

        self.q = MLP(obs_dim, n_actions, hidden=cfg.hidden)
        self.q_target = copy.deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.v = MLP(obs_dim, 1, hidden=cfg.hidden)
        self.pi = MLP(obs_dim, n_actions, hidden=cfg.hidden)

        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=cfg.lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr)

    # ------------------------------------------------------------ updates
    def _q_at(self, q_net: MLP, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    def update(self, batch: dict) -> dict:
        s, a, r, s_next, done = batch["s"], batch["a"], batch["r"], batch["s_next"], batch["done"]
        cfg = self.cfg

        # 1. V update (expectile regression on Q_target(s,a) - V(s))
        with torch.no_grad():
            q_t = self._q_at(self.q_target, s, a)
        v = self.v(s).squeeze(1)
        v_loss = expectile_loss(q_t - v, cfg.tau).mean()
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # 2. Q update (Bellman target uses V(s'))
        with torch.no_grad():
            v_next = self.v(s_next).squeeze(1)
            target = r + cfg.gamma * v_next * (1.0 - done)
        q_pred = self._q_at(self.q, s, a)
        q_loss = F.mse_loss(q_pred, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # 3. Policy update (advantage-weighted regression)
        with torch.no_grad():
            adv = q_t - v.detach()
            weights = torch.clamp((cfg.beta * adv).exp(), max=100.0)
        logits = self.pi(s)
        log_probs = F.log_softmax(logits, dim=-1).gather(1, a.unsqueeze(1)).squeeze(1)
        pi_loss = -(weights * log_probs).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        # 4. Polyak target update
        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.mul_(1 - cfg.polyak).add_(cfg.polyak * p.data)

        return {
            "v_loss": float(v_loss.item()),
            "q_loss": float(q_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "mean_adv": float(adv.mean().item()),
        }

    # ------------------------------------------------------------ inference
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Greedy action from the policy network. `obs` shape (B, obs_dim)."""
        self.pi.eval()
        with torch.no_grad():
            logits = self.pi(torch.as_tensor(obs, dtype=torch.float32))
            actions = logits.argmax(dim=-1).cpu().numpy()
        self.pi.train()
        return actions

    def save(self, path: str) -> None:
        torch.save(
            {"q": self.q.state_dict(), "v": self.v.state_dict(), "pi": self.pi.state_dict(),
             "cfg": self.cfg.__dict__, "n_actions": self.n_actions},
            path,
        )


def train_offline(dataset: dict, observation_shape, n_actions: int, cfg: IQLConfig) -> DiscreteIQL:
    """Train DiscreteIQL on a transitions dict (from build_transitions)."""
    obs_dim = int(observation_shape[0])
    agent = DiscreteIQL(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg)

    n = dataset["s"].shape[0]
    rng = np.random.default_rng(cfg.seed)
    log_every = max(1, cfg.n_steps // 10)
    for step in range(1, cfg.n_steps + 1):
        idx = rng.integers(0, n, size=cfg.batch_size)
        batch = {k: v[idx] for k, v in dataset.items()}
        info = agent.update(batch)
        if step % log_every == 0:
            print(
                f"  step {step:>6d} | "
                f"v_loss={info['v_loss']:.4f}  q_loss={info['q_loss']:.4f}"
                f"  pi_loss={info['pi_loss']:.4f}  mean_adv={info['mean_adv']:+.4f}"
            )
    return agent

# Finance Tutor RL

Offline reinforcement learning for personal finance tutoring. CS 224R final project.

We train tutoring policies that select practice items for a simulated learner, comparing three reward designs:

1. **Score-based** — held-out quiz accuracy
2. **Scenario-based** — performance on multi-concept word problems
3. **Preference-based** — Bradley-Terry model trained on pairwise session comparisons

Policies are trained with Implicit Q-Learning (IQL) on a fixed offline dataset of ~10,000 teaching trajectories. The simulated learner is a Deep Knowledge Tracing (DKT) LSTM trained on ASSISTments and adapted to a 20-concept personal-finance skill space. A structurally distinct Bayesian Knowledge Tracing (BKT) simulator is held out for transfer evaluation.

## Repo layout

```
content/                   Curriculum: knowledge graph, item bank, scenarios
data/
  raw/                     ASSISTments and other downloaded data (gitignored)
  processed/               Cleaned datasets ready for training
  trajectories/            Offline RL dataset (gitignored)
src/
  env/                     Gym-style environment wrapping a simulator
  simulators/              DKT (training-time) and BKT (eval-time) student models
  policies/                Rollout policies for trajectory collection
  rewards/                 Score, scenario, and preference reward functions
  training/                IQL trainer
  evaluation/              Eval harness, metrics, qualitative analysis
  utils/                   Shared helpers
scripts/                   Top-level entry points (see below)
configs/                   YAML configs per experiment
results/
  checkpoints/             Saved model weights (gitignored)
  figures/                 Plots and tables
  logs/                    Training logs (gitignored)
tests/                     Unit tests
notebooks/                 Exploratory analysis
```

## Pipeline

```
1. python scripts/download_assistments.py        # raw -> data/raw/
2. python scripts/build_content.py               # generate knowledge graph + item bank
3. python scripts/train_dkt.py                   # train simulator on ASSISTments
4. python scripts/adapt_dkt_to_finance.py        # remap skill space
5. python scripts/collect_trajectories.py        # produce ~10k offline trajectories
6. python scripts/relabel_rewards.py             # apply each reward function
7. python scripts/train_iql.py --reward score    # train one policy
8. python scripts/evaluate.py                    # full eval matrix
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Team

Stella Wu, Daniel Argento. Stanford CS 224R, 2026.

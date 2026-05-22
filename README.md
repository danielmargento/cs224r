# Finance Tutor RL

Offline reinforcement learning for personal finance tutoring. CS 224R final project (Stanford, 2026).

We train tutoring policies that select practice items for a simulated learner, comparing how reward design shapes the resulting curriculum. The proposal commits to three reward designs:

1. **Score-based** — expected accuracy on a held-out quiz set
2. **Scenario-based** — product-aggregated correctness on multi-concept word problems
3. **Preference-based** — Bradley-Terry model trained on pairwise session comparisons *(implemented; pairwise label collection scheduled for the next milestone)*

The simulated learner is a Bayesian Knowledge Tracing (BKT) HMM for the current milestone. A Deep Knowledge Tracing (DKT) LSTM trained on ASSISTments is planned for the final report as the in-distribution training simulator; BKT will remain as a structurally distinct transfer evaluator. Policies are trained with **Discrete Implicit Q-Learning (IQL)** implemented from scratch in `src/training/iql_trainer.py`.

## Repo layout

```
content/                    Curriculum: knowledge graph, item bank, scenarios (JSON)
data/
  raw/                      ASSISTments and other downloaded data (gitignored)
  processed/                Cleaned datasets ready for training
  trajectories/             Offline RL dataset + relabeled rewards + IQL eval rollouts (gitignored)
src/
  simulators/               BKT learner; DKT slot reserved for next milestone
  env/                      Gym-style episode environment
  policies/                 Three behavioral rollout policies (random, prereq, difficulty-increasing)
  rewards/                  Score, scenario, and preference reward functions
  training/                 Discrete IQL trainer (Q + V + π networks)
  evaluation/               Curriculum-structure metrics
  utils/                    Shared helpers
scripts/                    Top-level entry points (see Pipeline below)
configs/                    YAML configs per experiment (reserved)
results/
  checkpoints/              Trained IQL policies (gitignored)
  figures/                  milestone.png and any future plots
  logs/                     Training logs (gitignored)
  MILESTONE.md              Milestone-report writeup
  baseline_rewards.csv      Per-policy reward summary
  baseline_structure.csv    Per-policy curriculum-structure metrics
tests/                      Smoke tests
notebooks/                  Exploratory analysis
```

## Pipeline

End-to-end run, single seed (BKT only):

```bash
python scripts/content.py                                       # 1. curriculum JSON
PYTHONPATH=. python scripts/collect_trajectories.py             # 2. 600 trajectories
PYTHONPATH=. python scripts/relabel_rewards.py                  # 3. apply score & scenario rewards
PYTHONPATH=. python scripts/train_iql.py --reward score         # 4a. train Discrete IQL on score
PYTHONPATH=. python scripts/train_iql.py --reward scenario      # 4b. train Discrete IQL on scenario
PYTHONPATH=. python scripts/evaluate.py                         # 5. reward + structure CSVs
PYTHONPATH=. python scripts/make_milestone_figure.py            # 6. composite figure
```

Reproducible end-to-end runtime: ~1 minute on CPU.

Planned for next milestone (DKT + full evaluation matrix):

```
- scripts/download_assistments.py    # ASSISTments -> data/raw/
- scripts/train_dkt.py               # DKT LSTM simulator
- scripts/adapt_dkt_to_finance.py    # remap skill space to finance concepts
- scripts/collect_preferences.py     # pairwise labels for Bradley-Terry training
- Per-concept BKT heterogeneity, 10k trajectories, 3 seeds per reward (9 IQL policies)
```

## Smoke tests

```bash
PYTHONPATH=. pytest tests/ -v
```

Nine tests cover content shape, BKT clone-isolation, the three reward functions, and a sanity check on score-vs-mastery monotonicity.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Team

Daniel Argento, Stella Wu. Stanford CS 224R, 2026.

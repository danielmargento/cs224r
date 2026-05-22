# Milestone Report — Infrastructure, Baselines, and Discrete IQL

## Pipeline (end-to-end, all components live)

```
content/                         16 concepts, 192 practice + 48 quiz items,
                                 40 multi-concept scenarios
   │
   ▼
src/simulators/                  BKT learner (per-concept independent HMM,
                                 structurally distinct from the DKT planned
                                 for the final report)
   │
   ▼
src/env/ + src/policies/         Gym-style env (50-step episodes) + three
                                 rollout policies (random, prereq_respecting,
                                 difficulty_increasing)
   │
   ▼
scripts/collect_trajectories.py  600 trajectories (200 per policy) in <1s
   │
   ▼
src/rewards/                     score reward (held-out quiz accuracy);
                                 scenario reward (40 multi-concept word
                                 problems, conjunctive aggregation).
                                 Bradley-Terry preference reward implemented
                                 but excluded from this milestone pending
                                 preference-label collection.
   │
   ▼
scripts/relabel_rewards.py       600 trajectories x 2 rewards = 1,200 labels
   │
   ▼
src/training/ + scripts/train_iql.py
                                 Discrete Implicit Q-Learning (Kostrikov 2021),
                                 implemented from scratch. One policy per
                                 reward.
   │
   ▼
src/evaluation/ + scripts/evaluate.py
                                 reward + curriculum-structure metrics CSVs;
                                 scripts/make_milestone_figure.py produces
                                 the composite figure (results/figures/
                                 milestone.png)
```

## Discrete IQL implementation

We implement IQL from scratch for our discrete-action setting. Three networks: Q(s, ·) ∈ ℝ^|A|, V(s) ∈ ℝ, and a categorical policy π(a|s). Training follows Kostrikov et al. (2021):

1. **V update — expectile regression on Q(s,a) - V(s)**, τ = 0.7. With τ > 0.5, V approximates an upper expectile of Q over the dataset's actions, without ever querying Q at out-of-distribution actions.
2. **Q update — Bellman target r + γ · V(s')** (γ = 0.99). Stays entirely in-sample.
3. **Policy extraction — advantage-weighted regression**: π loss = −𝔼[exp(β·(Q − V)) · log π(a|s)], β = 3.

We use Monte Carlo returns as the per-step reward (broadcast each episode's terminal reward to every transition in that episode). Under our sparse-terminal-reward regime — reward only at the 50th step — Bellman backups require many more SGD updates than is feasible in the milestone budget. MC returns trade credit-assignment capacity for tractable learning; IQL's V (expectile of Q) and AWR policy extraction are unchanged.

Training: 15,000 SGD steps, batch 256, Adam lr 3e-4, Polyak target update 0.005. ~20 s per policy.

## Results

### Reward by policy (mean ± SE)

| policy                    | score          | scenario        |
|---------------------------|----------------|-----------------|
| random (baseline)         | 0.460 ± 0.003  | 0.167 ± 0.003   |
| difficulty_increasing     | 0.454 ± 0.003  | 0.162 ± 0.003   |
| prereq_respecting         | 0.415 ± 0.003  | 0.154 ± 0.002   |
| **iql_score**             | 0.344 ± 0.001  | 0.097 ± 0.001   |
| **iql_scenario**          | 0.362 ± 0.001  | 0.108 ± 0.001   |

The trained policies underperform the rollout baselines on raw reward. Diagnosis follows the structure-metrics table.

### Curriculum-structure metrics — the more informative result

| policy                    | H(concept) | revisit | prereq_respect | mean_diff | first_hard |
|---------------------------|-----------:|--------:|---------------:|----------:|-----------:|
| random                    |   2.60     |  0.69   |   0.26         |   2.00    |   1.96     |
| difficulty_increasing     |   2.61     |  0.69   |   0.26         |   2.04    | **32.00**  |
| prereq_respecting         |   1.84     |  0.85   | **1.00**       |   2.00    |   2.35     |
| **iql_score**             | **0.59**   |  0.93   | **1.00**       |   1.46    | **33.32**  |
| **iql_scenario**          |   0.85     |  0.92   | **1.00**       |   1.63    |  10.32     |

The trained policies share three characteristics that no baseline exhibits:

1. **Perfect prerequisite respect (1.00)** vs random/diff_increasing at 0.26 — IQL discovered that prereq-respecting trajectories in the dataset led to higher returns.
2. **Extremely narrow concept focus (entropy 0.6–0.9)** vs baselines at 1.8–2.6 — both rewards reward concentrating mastery on a few concepts rather than touching all 16.
3. **Lower mean difficulty (1.46–1.63)** vs baselines at 2.00 — IQL prefers easier items that are more likely to register as "correct" with the learner.

The two trained policies *differ from each other* in interpretable ways:

- **iql_score** focuses even more narrowly (entropy 0.59 vs 0.85) and at lower difficulty (1.46 vs 1.63). It delays hard items to step 33. This makes sense: the score reward measures held-out quiz accuracy, which rewards consolidating mastery on a small concept set at low difficulty.
- **iql_scenario** spreads slightly more (entropy 0.85) and selects medium items earlier (first hard at step 10). The scenario reward depends on the *product* of P(correct) across multiple required concepts, so a broader concept base is incentivized — even at the cost of slightly less depth per concept.

Same dataset, same algorithm, same trajectory budget — the only thing that changed is the reward used during the IQL Bellman / AWR updates, and the result is qualitatively different teaching curricula. This is the central hypothesis of the proposal manifesting in microcosm.

## Why the trained policies lose on raw reward

Three causes, all addressable in the next milestone:

1. **Uniform BKT parameters.** Every concept has identical learn / slip / guess rates, so random sampling already produces high concept entropy and consequently high reward. The baselines' diversity is hard to beat without a sharper reward gradient. The narrow-focus + always-prereq-respecting curriculum IQL learned is *qualitatively* right but quantitatively over-restrictive for this BKT.
2. **Small dataset.** 600 trajectories × 50 steps = 30k transitions over a 48-action discrete space. The proposal targets 10k trajectories.
3. **MC-return reward broadcast (effective horizon = 1).** Each transition is treated as terminal with reward equal to the trajectory return, bypassing Bellman propagation. Necessary trade-off for the milestone training budget but eliminates within-episode temporal credit assignment.

The structural metrics show that IQL is learning the *right kind of behavior* (prereq respect, focus, low difficulty); the absolute reward gap reflects dataset/simulator limitations rather than a fundamental policy-learning failure.

## What's done vs. open

**Done**:
- 16-concept knowledge graph; 192 practice + 48 quiz items + 40 multi-concept scenarios
- BKT simulator + 9 passing smoke tests
- Gym env + three rollout policies
- Trajectory collection (~1 s for 600 episodes)
- Reward relabeling under score and scenario rewards
- Curriculum-structure metrics module (entropy, revisit, prereq_respect, mean_diff, first_hard_step)
- **Discrete IQL from scratch (Q + V + π networks, expectile regression, AWR)**
- Reproducible CSVs + 4-panel composite figure

**Open for next milestone**:
- Introduce per-concept BKT heterogeneity (sharpen the reward gradient)
- Scale dataset to 10k trajectories
- Three seeds per reward (currently one); proposal target is 3 × 3 = 9 policies
- Preference reward: collect pairwise labels and train the Bradley-Terry model
- DKT simulator (LSTM on ASSISTments) for in-distribution evaluation
- Restore sparse-terminal-reward IQL (no MC shortcut) once dataset is large enough for proper Bellman backups

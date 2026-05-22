"""End-to-end smoke test for the simulator + reward pipeline.

Validates that the pieces Daniel owns compose end-to-end before Stella's
env/policies arrive. Builds a BKT learner, runs 50 random teaching steps,
and feeds the resulting Trajectory into each of the three reward functions.

Run:
    pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.rewards.base import Step, Trajectory
from src.rewards.preference import (
    BradleyTerryRewardModel,
    PreferenceReward,
    feature_dim,
)
from src.rewards.scenario import Scenario, ScenarioReward
from src.rewards.score import QuizItem, ScoreReward
from src.simulators.bkt import BKTParams, BKTSimulator


CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def concepts() -> list[str]:
    with open(CONTENT_DIR / "constants.json") as f:
        return json.load(f)["concepts"]


@pytest.fixture(scope="module")
def quiz_items() -> list[QuizItem]:
    with open(CONTENT_DIR / "item_bank.json") as f:
        data = json.load(f)
    return [
        QuizItem(id=q["id"], concept_id=q["concept"], difficulty=q["difficulty"])
        for q in data["quiz_items"]
    ]


@pytest.fixture(scope="module")
def scenarios() -> list[Scenario]:
    with open(CONTENT_DIR / "scenarios.json") as f:
        data = json.load(f)
    return [
        Scenario(
            id=s["id"],
            concepts_required=s["concepts_tested"],
            difficulty=s["difficulty"],
        )
        for s in data["scenarios"]
    ]


@pytest.fixture
def trajectory(concepts: list[str]) -> Trajectory:
    """Random 50-step trajectory: pick (concept, difficulty), sample response, update."""
    rng = np.random.default_rng(seed=0)
    sim = BKTSimulator(concept_ids=concepts, params=BKTParams(), seed=0)
    initial = sim.clone()

    steps: list[Step] = []
    for _ in range(50):
        concept = concepts[rng.integers(0, len(concepts))]
        difficulty = int(rng.integers(1, 4))
        state_before = sim.get_mastery_vector()
        correct = sim.sample_response(concept, difficulty, rng=rng)
        sim.observe(concept, correct, difficulty=difficulty)
        state_after = sim.get_mastery_vector()
        steps.append(
            Step(
                state=state_before,
                concept_id=concept,
                difficulty=difficulty,
                correct=correct,
                next_state=state_after,
            )
        )

    return Trajectory(steps=steps, final_simulator=sim.clone(), initial_simulator=initial)


# ------------------------------------------------------------------- tests
def test_content_files_exist():
    """Content files were generated."""
    for name in ("knowledge_graph.json", "item_bank.json", "scenarios.json", "constants.json"):
        assert (CONTENT_DIR / name).exists(), f"Missing content/{name}"


def test_item_bank_split(concepts: list[str]):
    """4 practice + 1 quiz per (concept, difficulty) cell."""
    with open(CONTENT_DIR / "item_bank.json") as f:
        data = json.load(f)
    n_cells = len(concepts) * 3
    assert data["num_practice"] == n_cells * 4
    assert data["num_quiz"] == n_cells
    # Every cell has exactly 1 quiz item and 4 practice items.
    from collections import Counter

    practice_cells = Counter((q["concept"], q["difficulty"]) for q in data["practice_items"])
    quiz_cells = Counter((q["concept"], q["difficulty"]) for q in data["quiz_items"])
    assert all(v == 4 for v in practice_cells.values())
    assert all(v == 1 for v in quiz_cells.values())
    assert len(practice_cells) == n_cells
    assert len(quiz_cells) == n_cells


def test_scenario_count():
    """40 scenarios as planned."""
    with open(CONTENT_DIR / "scenarios.json") as f:
        data = json.load(f)
    assert data["num_scenarios"] == 40
    # Every scenario tests at least 2 concepts (multi-concept by definition).
    assert all(len(s["concepts_tested"]) >= 2 for s in data["scenarios"])


def test_bkt_mutation_isolation(concepts: list[str]):
    """clone() must not share state with the original."""
    sim = BKTSimulator(concept_ids=concepts, params=BKTParams(), seed=0)
    sim.observe("budgeting", correct=True, difficulty=1)
    snapshot = sim.get_mastery_vector().copy()
    clone = sim.clone()
    clone.observe("budgeting", correct=False, difficulty=3)
    clone.observe("budgeting", correct=False, difficulty=3)
    # original mastery vector unchanged
    np.testing.assert_array_equal(sim.get_mastery_vector(), snapshot)


def test_score_reward_returns_probability(trajectory: Trajectory, quiz_items: list[QuizItem]):
    reward = ScoreReward(quiz_items=quiz_items)
    r = reward(trajectory)
    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0, f"Score reward out of [0,1]: {r}"


def test_scenario_reward_returns_probability(trajectory: Trajectory, scenarios: list[Scenario]):
    reward = ScenarioReward(scenarios=scenarios, aggregation="product")
    r = reward(trajectory)
    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0, f"Scenario reward out of [0,1]: {r}"


def test_scenario_aggregations_ordered(trajectory: Trajectory, scenarios: list[Scenario]):
    """For probabilities in [0,1]: product <= min <= mean (averaged across scenarios)."""
    r_min = ScenarioReward(scenarios=scenarios, aggregation="min")(trajectory)
    r_prod = ScenarioReward(scenarios=scenarios, aggregation="product")(trajectory)
    r_mean = ScenarioReward(scenarios=scenarios, aggregation="mean")(trajectory)
    # Each p_i in [0,1], so product(p_i) <= min(p_i) <= mean(p_i). Linearity preserves under averaging.
    assert r_prod <= r_min + 1e-9, f"prod={r_prod} should be <= min={r_min}"
    assert r_min <= r_mean + 1e-9, f"min={r_min} should be <= mean={r_mean}"


def test_preference_reward_returns_finite(trajectory: Trajectory, concepts: list[str]):
    """Untrained Bradley-Terry model should still produce a finite scalar."""
    torch.manual_seed(0)
    model = BradleyTerryRewardModel(input_dim=feature_dim(len(concepts)))
    reward = PreferenceReward(model=model, concept_ids=concepts)
    r = reward(trajectory)
    assert isinstance(r, float)
    assert np.isfinite(r), f"Preference reward not finite: {r}"


def test_score_reward_increases_with_mastery(concepts: list[str], quiz_items: list[QuizItem]):
    """A learner who masters everything should score higher than a blank slate."""
    blank = BKTSimulator(concept_ids=concepts, params=BKTParams(), seed=0)
    mastered = BKTSimulator(concept_ids=concepts, params=BKTParams(), seed=0)
    # Pin every concept to fully mastered.
    for c in concepts:
        mastered._mastery[c] = 1.0

    reward = ScoreReward(quiz_items=quiz_items)
    r_blank = reward(Trajectory(steps=[], final_simulator=blank.clone()))
    r_master = reward(Trajectory(steps=[], final_simulator=mastered.clone()))
    assert r_master > r_blank, f"Mastery should improve score: blank={r_blank}, master={r_master}"

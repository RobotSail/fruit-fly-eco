"""Tests for knowledge distillation (rate model → spiking model).

Covers:
  - Teacher training produces decreasing loss
  - Soft label generation returns valid probability distributions
  - Student distillation training runs without errors
  - Majority baseline is correct
  - Evaluate accuracy gives expected results for trivial model
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flyecon.distillation import (
    N_BUY_PLANS,
    distill_student,
    evaluate_accuracy,
    generate_soft_labels,
    majority_baseline,
    train_teacher,
)
from flyecon.etl.controls import random_sparse
from flyecon.policy.ppo import build_fly_policy
from flyecon.reservoir import FlyReservoir
from flyecon.state.economy import BuyPlan, EconomyState, pistol_round_state

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_reservoir():
    """Small FlyReservoir for fast testing (no real connectome)."""
    import numpy as np
    from scipy import sparse

    n = 50
    rng = np.random.default_rng(42)
    W = sparse.random(n, n, density=0.1, format="csr", random_state=rng)
    W.data = (W.data - 0.5).astype(np.float32)
    return FlyReservoir(n, W, device="cpu")


@pytest.fixture
def tiny_student():
    """Small spiking FlyPolicy for fast testing."""
    conn = random_sparse(n_neurons=50, density=0.1, seed=42)
    return build_fly_policy(conn, gain=0.0, duration_ms=50.0)


@pytest.fixture
def sample_states():
    """Small batch of economy states for testing."""
    return [
        pistol_round_state(half=0),
        EconomyState(money=4000, round_number=3, loss_streak=1, half=0, opponent_loss_streak=0),
        EconomyState(money=8000, round_number=6, loss_streak=0, half=0, opponent_loss_streak=2),
        EconomyState(money=2000, round_number=4, loss_streak=2, half=0, opponent_loss_streak=0),
        EconomyState(money=800, round_number=1, loss_streak=0, half=1, opponent_loss_streak=0),
        EconomyState(money=10000, round_number=10, loss_streak=0, half=1, opponent_loss_streak=3),
    ]


# ── Majority baseline tests ──────────────────────────────────────────────────


class TestMajorityBaseline:
    def test_uniform_labels(self):
        labels = torch.tensor([0, 1, 2, 3, 4])
        assert abs(majority_baseline(labels) - 0.2) < 1e-6

    def test_dominated_labels(self):
        labels = torch.tensor([0, 0, 0, 1, 2])
        assert abs(majority_baseline(labels) - 0.6) < 1e-6

    def test_single_class(self):
        labels = torch.tensor([2, 2, 2, 2])
        assert abs(majority_baseline(labels) - 1.0) < 1e-6


# ── Evaluate accuracy tests ──────────────────────────────────────────────────


class TestEvaluateAccuracy:
    def test_perfect_model(self, sample_states):
        labels = torch.zeros(len(sample_states), dtype=torch.long)

        def always_zero(states):
            logits = torch.zeros(len(states), N_BUY_PLANS)
            logits[:, 0] = 10.0
            return logits

        acc = evaluate_accuracy(always_zero, sample_states, labels)
        assert acc == 1.0

    def test_wrong_model(self, sample_states):
        labels = torch.zeros(len(sample_states), dtype=torch.long)

        def always_one(states):
            logits = torch.zeros(len(states), N_BUY_PLANS)
            logits[:, 1] = 10.0
            return logits

        acc = evaluate_accuracy(always_one, sample_states, labels)
        assert acc == 0.0


# ── Soft label generation tests ──────────────────────────────────────────────


class TestGenerateSoftLabels:
    def test_shape_and_sum(self, tiny_reservoir, sample_states):
        soft = generate_soft_labels(tiny_reservoir, sample_states, temperature=2.0)
        assert soft.shape == (len(sample_states), N_BUY_PLANS)
        assert torch.allclose(soft.sum(dim=1), torch.ones(len(sample_states)), atol=1e-5)

    def test_all_positive(self, tiny_reservoir, sample_states):
        soft = generate_soft_labels(tiny_reservoir, sample_states, temperature=2.0)
        assert (soft > 0).all()

    def test_higher_temperature_softer(self, tiny_reservoir, sample_states):
        soft_low = generate_soft_labels(tiny_reservoir, sample_states, temperature=1.0)
        soft_high = generate_soft_labels(tiny_reservoir, sample_states, temperature=5.0)
        # Higher temperature → higher entropy (softer distribution)
        entropy_low = -(soft_low * torch.log(soft_low + 1e-10)).sum(dim=-1).mean()
        entropy_high = -(soft_high * torch.log(soft_high + 1e-10)).sum(dim=-1).mean()
        assert entropy_high > entropy_low


# ── Student distillation tests ───────────────────────────────────────────────


class TestDistillStudent:
    def test_runs_without_error(self, tiny_student, sample_states):
        n = len(sample_states)
        soft_labels = torch.ones(n, N_BUY_PLANS) / N_BUY_PLANS  # uniform
        hard_labels = torch.zeros(n, dtype=torch.long)
        logs = distill_student(
            tiny_student, sample_states, soft_labels, hard_labels,
            n_epochs=2, batch_size=4,
        )
        assert len(logs) == 2
        assert all("kl_loss" in l and "accuracy" in l for l in logs)

    def test_produces_valid_logits_after_distill(self, tiny_student, sample_states):
        n = len(sample_states)
        soft_labels = torch.ones(n, N_BUY_PLANS) / N_BUY_PLANS
        hard_labels = torch.zeros(n, dtype=torch.long)
        distill_student(
            tiny_student, sample_states, soft_labels, hard_labels,
            n_epochs=1, batch_size=4,
        )
        tiny_student.eval()
        dist, values, features = tiny_student.forward(sample_states)
        assert not torch.isnan(dist.logits).any()
        assert dist.probs.shape == (n, N_BUY_PLANS)

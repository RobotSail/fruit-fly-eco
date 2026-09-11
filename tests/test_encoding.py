"""Tests for PopulationEncoder (Phase 5).

Covers:
  - Population code activations (left/mid/right of money range)
  - One-hot loss-streak encoding
  - Thermometer round encoding
  - Total input dimensionality
  - Deterministic output (no stochasticity)
  - map_to_connectome_inputs correctness
  - Gradient flow through trainable parameters
"""

from __future__ import annotations

import torch

from flyecon.encoding.population import (
    N_HALF_NEURONS,
    N_INPUT_NEURONS,
    N_LOSS_STREAK_NEURONS,
    N_MONEY_NEURONS,
    N_OPP_LOSS_STREAK_NEURONS,
    N_ROUND_NEURONS,
    PopulationEncoder,
    map_to_connectome_inputs,
)
from flyecon.state.economy import EconomyState

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_state(
    money: int = 4000,
    loss_streak: int = 0,
    round_number: int = 1,
    half: int = 0,
    opp_loss_streak: int = 0,
) -> EconomyState:
    return EconomyState(
        money=money,
        loss_streak=loss_streak,
        round_number=round_number,
        half=half,
        opponent_loss_streak=opp_loss_streak,
    )


# ── Total dimensionality ──────────────────────────────────────────────────


class TestDimensionality:
    """Total input dimensionality matches expected ~44 neurons."""

    def test_n_input_neurons_constant(self) -> None:
        expected = (
            N_MONEY_NEURONS
            + N_LOSS_STREAK_NEURONS
            + N_ROUND_NEURONS
            + N_HALF_NEURONS
            + N_OPP_LOSS_STREAK_NEURONS
        )
        assert N_INPUT_NEURONS == expected

    def test_encode_output_shape(self) -> None:
        enc = PopulationEncoder()
        state = _make_state()
        out = enc.encode(state)
        assert out.shape == (N_INPUT_NEURONS,)

    def test_n_input_neurons_property(self) -> None:
        enc = PopulationEncoder()
        assert enc.n_input_neurons == N_INPUT_NEURONS


# ── Money population code ─────────────────────────────────────────────────


class TestMoneyPopulationCode:
    """Population code activations: money→Gaussian activations."""

    def test_money_zero_activates_leftmost(self) -> None:
        """money=0 should maximally activate the leftmost neuron."""
        enc = PopulationEncoder()
        state = _make_state(money=0)
        out = enc.encode(state)
        money_block = out[:N_MONEY_NEURONS]
        # Leftmost neuron (center at 0) should have the highest activation
        assert money_block[0] == money_block.max()

    def test_money_max_activates_rightmost(self) -> None:
        """money=16000 should maximally activate the rightmost neuron."""
        enc = PopulationEncoder()
        state = _make_state(money=16000)
        out = enc.encode(state)
        money_block = out[:N_MONEY_NEURONS]
        assert money_block[-1] == money_block.max()

    def test_money_midrange_activates_middle(self) -> None:
        """money=8000 should activate middle neurons most."""
        enc = PopulationEncoder()
        state = _make_state(money=8000)
        out = enc.encode(state)
        money_block = out[:N_MONEY_NEURONS]
        peak_idx = int(money_block.argmax().item())
        # Peak should be in the middle third of neurons
        assert N_MONEY_NEURONS // 3 <= peak_idx <= 2 * N_MONEY_NEURONS // 3

    def test_all_activations_nonneg(self) -> None:
        """Gaussian activations are always ≥ 0."""
        enc = PopulationEncoder()
        for money in [0, 800, 4000, 8000, 12000, 16000]:
            out = enc.encode(_make_state(money=money))
            money_block = out[:N_MONEY_NEURONS]
            assert (money_block >= 0).all(), f"Negative activation at money={money}"

    def test_different_money_different_activations(self) -> None:
        """Different money values produce different activation patterns."""
        enc = PopulationEncoder()
        out_low = enc.encode(_make_state(money=800))
        out_high = enc.encode(_make_state(money=12000))
        money_low = out_low[:N_MONEY_NEURONS]
        money_high = out_high[:N_MONEY_NEURONS]
        assert not torch.allclose(money_low, money_high)


# ── Loss-streak one-hot ───────────────────────────────────────────────────


class TestLossStreakOneHot:
    """One-hot encoding: each rung activates exactly one neuron."""

    def test_each_rung_one_active(self) -> None:
        enc = PopulationEncoder()
        start = N_MONEY_NEURONS
        end = start + N_LOSS_STREAK_NEURONS
        for streak in range(5):
            out = enc.encode(_make_state(loss_streak=streak))
            streak_block = out[start:end]
            # Exactly one neuron should be non-zero
            n_active = (streak_block.abs() > 0).sum().item()
            assert n_active == 1, f"streak={streak}: {n_active} neurons active"

    def test_correct_neuron_active(self) -> None:
        enc = PopulationEncoder()
        start = N_MONEY_NEURONS
        for streak in range(5):
            out = enc.encode(_make_state(loss_streak=streak))
            streak_block = out[start : start + N_LOSS_STREAK_NEURONS]
            active_idx = int(streak_block.abs().argmax().item())
            assert active_idx == streak, f"streak={streak}: neuron {active_idx} active"


# ── Thermometer round encoding ────────────────────────────────────────────


class TestThermometerRound:
    """Thermometer code: round k activates neurons 0..k-1."""

    def test_round_6_activates_first_6(self) -> None:
        enc = PopulationEncoder()
        start = N_MONEY_NEURONS + N_LOSS_STREAK_NEURONS
        end = start + N_ROUND_NEURONS
        out = enc.encode(_make_state(round_number=6))
        round_block = out[start:end]
        n_active = (round_block.abs() > 0).sum().item()
        assert n_active == 6

    def test_round_1_activates_1(self) -> None:
        enc = PopulationEncoder()
        start = N_MONEY_NEURONS + N_LOSS_STREAK_NEURONS
        out = enc.encode(_make_state(round_number=1))
        round_block = out[start : start + N_ROUND_NEURONS]
        n_active = (round_block.abs() > 0).sum().item()
        assert n_active == 1

    def test_round_12_activates_all(self) -> None:
        enc = PopulationEncoder()
        start = N_MONEY_NEURONS + N_LOSS_STREAK_NEURONS
        out = enc.encode(_make_state(round_number=12))
        round_block = out[start : start + N_ROUND_NEURONS]
        n_active = (round_block.abs() > 0).sum().item()
        assert n_active == N_ROUND_NEURONS

    def test_thermometer_preserves_order(self) -> None:
        """More rounds → more active neurons (ordinality)."""
        enc = PopulationEncoder()
        start = N_MONEY_NEURONS + N_LOSS_STREAK_NEURONS
        end = start + N_ROUND_NEURONS
        prev_count = 0
        for r in range(1, 13):
            out = enc.encode(_make_state(round_number=r))
            n_active = int((out[start:end].abs() > 0).sum().item())
            assert n_active >= prev_count, f"round {r}: ordinality broken"
            prev_count = n_active


# ── Half one-hot ──────────────────────────────────────────────────────────


class TestHalfOneHot:
    """One-hot encoding for half (0 or 1)."""

    def test_half_0(self) -> None:
        enc = PopulationEncoder()
        start = N_MONEY_NEURONS + N_LOSS_STREAK_NEURONS + N_ROUND_NEURONS
        out = enc.encode(_make_state(half=0))
        half_block = out[start : start + N_HALF_NEURONS]
        assert half_block[0].abs() > 0
        assert half_block[1].abs() == 0

    def test_half_1(self) -> None:
        enc = PopulationEncoder()
        start = N_MONEY_NEURONS + N_LOSS_STREAK_NEURONS + N_ROUND_NEURONS
        out = enc.encode(_make_state(half=1))
        half_block = out[start : start + N_HALF_NEURONS]
        assert half_block[0].abs() == 0
        assert half_block[1].abs() > 0


# ── Opponent loss streak ──────────────────────────────────────────────────


class TestOpponentLossStreak:
    """One-hot encoding for opponent loss streak."""

    def test_each_rung_one_active(self) -> None:
        enc = PopulationEncoder()
        start = (
            N_MONEY_NEURONS
            + N_LOSS_STREAK_NEURONS
            + N_ROUND_NEURONS
            + N_HALF_NEURONS
        )
        for streak in range(5):
            out = enc.encode(_make_state(opp_loss_streak=streak))
            block = out[start : start + N_OPP_LOSS_STREAK_NEURONS]
            n_active = (block.abs() > 0).sum().item()
            assert n_active == 1, f"opp_streak={streak}: {n_active} neurons active"


# ── Determinism ───────────────────────────────────────────────────────────


class TestDeterminism:
    """Encode output is deterministic — no stochasticity."""

    def test_same_state_same_output(self) -> None:
        enc = PopulationEncoder()
        state = _make_state(money=5000, loss_streak=2, round_number=7, half=1)
        out1 = enc.encode(state)
        out2 = enc.encode(state)
        assert torch.allclose(out1, out2, atol=1e-7)

    def test_batch_matches_sequential(self) -> None:
        enc = PopulationEncoder()
        states = [
            _make_state(money=800),
            _make_state(money=5000, loss_streak=3),
            _make_state(money=12000, round_number=10, half=1),
        ]
        batch = enc.encode_batch(states)
        for i, s in enumerate(states):
            single = enc.encode(s)
            assert torch.allclose(batch[i], single, atol=1e-7)


# ── map_to_connectome_inputs ──────────────────────────────────────────────


class TestMapToConnectome:
    """map_to_connectome_inputs maps encoded vector to full neuron dim."""

    def test_single_vector(self) -> None:
        enc = PopulationEncoder()
        state = _make_state(money=4000)
        encoded = enc.encode(state)
        n_neurons = 200
        ids = list(range(N_INPUT_NEURONS))
        full = map_to_connectome_inputs(encoded, ids, n_neurons)
        assert full.shape == (n_neurons,)
        assert torch.allclose(full[:N_INPUT_NEURONS], encoded)
        assert (full[N_INPUT_NEURONS:] == 0).all()

    def test_batched(self) -> None:
        enc = PopulationEncoder()
        states = [_make_state(money=800), _make_state(money=12000)]
        batch = enc.encode_batch(states)
        n_neurons = 100
        ids = list(range(N_INPUT_NEURONS))
        full = map_to_connectome_inputs(batch, ids, n_neurons)
        assert full.shape == (2, n_neurons)

    def test_noncontiguous_ids(self) -> None:
        enc = PopulationEncoder()
        state = _make_state(money=4000)
        encoded = enc.encode(state)
        n_neurons = 500
        # Scatter encoded values at non-contiguous positions
        ids = [i * 10 for i in range(N_INPUT_NEURONS)]
        full = map_to_connectome_inputs(encoded, ids, n_neurons)
        for j, idx in enumerate(ids):
            assert torch.isclose(full[idx], encoded[j])

    def test_mismatched_ids_raises(self) -> None:
        enc = PopulationEncoder()
        encoded = enc.encode(_make_state())
        import pytest

        with pytest.raises(ValueError, match="input_neuron_ids length"):
            map_to_connectome_inputs(encoded, [0, 1, 2], 100)


# ── Gradient flow ─────────────────────────────────────────────────────────


class TestGradientFlow:
    """Trainable parameters receive gradients."""

    def test_grad_through_money_encoding(self) -> None:
        enc = PopulationEncoder()
        state = _make_state(money=5000)
        out = enc.encode(state)
        loss = out.sum()
        loss.backward()
        assert enc.gain_money.grad is not None
        assert enc.centers.grad is not None
        assert enc.log_widths.grad is not None

    def test_grad_through_streak_gain(self) -> None:
        enc = PopulationEncoder()
        state = _make_state(loss_streak=2)
        out = enc.encode(state)
        loss = out.sum()
        loss.backward()
        assert enc.gain_streak.grad is not None

    def test_grad_through_round_gain(self) -> None:
        enc = PopulationEncoder()
        state = _make_state(round_number=6)
        out = enc.encode(state)
        loss = out.sum()
        loss.backward()
        assert enc.gain_round.grad is not None

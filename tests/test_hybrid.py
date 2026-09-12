"""Tests for the hybrid architecture: rate model + spiking LIF mushroom body.

Uses a synthetic mini-connectome with KC, MBON, and PPL neurons to
verify the full pipeline without requiring the real MaleCNS v1.0 data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flyecon.etl.loader import Connectome, build_csr_tensor
from flyecon.etl.subcircuit import extract_mushroom_body
from flyecon.hybrid import (
    BUY_PLAN_TO_IDX,
    HybridModel,
    _RateReservoir,
    _find_indices_by_prefix,
    _map_kc_full_to_mb,
)
from flyecon.state.economy import EconomyState


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_synthetic_connectome(n: int = 50) -> Connectome:
    """Build a tiny synthetic connectome with KC, MBON, and PPL neurons.

    Layout (n=50):
      Neurons 0–9:   generic input neurons
      Neurons 10–29: KC neurons (20 Kenyon cells)
      Neurons 30–39: MBON neurons (10 mushroom body output neurons)
      Neurons 40–44: PPL neurons (5 dopaminergic)
      Neurons 45–49: generic output neurons
    """
    rng = np.random.default_rng(42)
    body_ids = np.arange(1000, 1000 + n, dtype=np.int64)

    neuron_types: dict[int, str] = {}
    for i in range(10, 30):
        neuron_types[1000 + i] = f"KC-s{i - 10}"
    for i in range(30, 40):
        neuron_types[1000 + i] = f"MBON-{i - 30}"
    for i in range(40, 45):
        neuron_types[1000 + i] = f"PPL-{i - 40}"

    # Sparse random connections (~200 edges)
    n_edges = 200
    rows = rng.integers(0, n, size=n_edges).astype(np.int64)
    cols = rng.integers(0, n, size=n_edges).astype(np.int64)
    vals = rng.normal(0, 1, size=n_edges).astype(np.float32)

    # Strong input→KC connections
    for i in range(10):
        for j in range(10, 30):
            if rng.random() > 0.5:
                rows = np.append(rows, i)
                cols = np.append(cols, j)
                vals = np.append(vals, rng.uniform(0.5, 2.0))

    # Strong KC→MBON connections
    for i in range(10, 30):
        for j in range(30, 40):
            if rng.random() > 0.3:
                rows = np.append(rows, i)
                cols = np.append(cols, j)
                vals = np.append(vals, rng.uniform(0.5, 2.0))

    weight_matrix = build_csr_tensor(
        rows.astype(np.int64), cols.astype(np.int64),
        vals.astype(np.float32), n,
    )
    dopamine_matrix = build_csr_tensor(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        np.array([], dtype=np.float32), n,
    )

    return Connectome(
        weight_matrix=weight_matrix,
        dopamine_matrix=dopamine_matrix,
        body_ids=body_ids,
        neuron_types=neuron_types,
        n_neurons=n,
        n_synapses=len(rows),
    )


@pytest.fixture
def synthetic_connectome() -> Connectome:
    return _make_synthetic_connectome()


@pytest.fixture
def sample_state() -> EconomyState:
    return EconomyState(
        money=4000, loss_streak=1, round_number=3,
        half=0, opponent_loss_streak=0,
    )


def _build_hybrid(conn: Connectome) -> HybridModel:
    """Build HybridModel from synthetic connectome using the real API."""
    mb_conn = extract_mushroom_body(conn, hops=0)
    return HybridModel(
        full_conn=conn,
        mb_conn=mb_conn,
        reservoir_gain=0.1,
        lif_gain=0.1,
    )


# ── _RateReservoir tests ───────────────────────────────────────────────────


class TestRateReservoir:
    def test_init(self, synthetic_connectome: Connectome) -> None:
        res = _RateReservoir(synthetic_connectome, gain=0.1)
        assert res.n == 50
        assert res.rates.shape == (50,)
        assert (res.rates == 0).all()

    def test_step_changes_state(self, synthetic_connectome: Connectome) -> None:
        res = _RateReservoir(synthetic_connectome, gain=0.1)
        inp = torch.zeros(50)
        inp[:10] = 1.0
        res.step(inp)
        assert res.rates.abs().sum() > 0

    def test_rates_bounded(self, synthetic_connectome: Connectome) -> None:
        """Rates should stay in [-1, 1] due to tanh."""
        res = _RateReservoir(synthetic_connectome, gain=0.1)
        for _ in range(10):
            res.step(torch.ones(50))
        assert res.rates.max() <= 1.0
        assert res.rates.min() >= -1.0

    def test_reset(self, synthetic_connectome: Connectome) -> None:
        res = _RateReservoir(synthetic_connectome, gain=0.1)
        res.step(torch.ones(50))
        assert res.rates.abs().sum() > 0
        res.reset()
        assert (res.rates == 0).all()


# ── _find_indices_by_prefix tests ───────────────────────────────────────────


class TestFindIndicesByPrefix:
    def test_kc_indices(self, synthetic_connectome: Connectome) -> None:
        indices = _find_indices_by_prefix(synthetic_connectome, "KC")
        assert len(indices) == 20
        assert all(10 <= i < 30 for i in indices)

    def test_mbon_indices(self, synthetic_connectome: Connectome) -> None:
        indices = _find_indices_by_prefix(synthetic_connectome, "MBON")
        assert len(indices) == 10
        assert all(30 <= i < 40 for i in indices)

    def test_ppl_indices(self, synthetic_connectome: Connectome) -> None:
        indices = _find_indices_by_prefix(synthetic_connectome, "PPL")
        assert len(indices) == 5

    def test_no_match(self, synthetic_connectome: Connectome) -> None:
        assert _find_indices_by_prefix(synthetic_connectome, "NONEXIST") == []


# ── _map_kc_full_to_mb tests ───────────────────────────────────────────────


class TestMapKcFullToMb:
    def test_mapping_nonempty(self, synthetic_connectome: Connectome) -> None:
        mb_conn = extract_mushroom_body(synthetic_connectome, hops=0)
        kc_full, kc_mb = _map_kc_full_to_mb(synthetic_connectome, mb_conn)
        assert len(kc_full) > 0
        assert len(kc_full) == len(kc_mb)

    def test_mapping_body_ids_match(self, synthetic_connectome: Connectome) -> None:
        """Every mapped KC pair should have the same body_id."""
        mb_conn = extract_mushroom_body(synthetic_connectome, hops=0)
        kc_full, kc_mb = _map_kc_full_to_mb(synthetic_connectome, mb_conn)
        for fi, mi in zip(kc_full, kc_mb):
            full_bid = synthetic_connectome.body_ids[fi]
            mb_bid = mb_conn.body_ids[mi]
            assert full_bid == mb_bid


# ── HybridModel tests ──────────────────────────────────────────────────────


class TestHybridModel:
    def test_build(self, synthetic_connectome: Connectome) -> None:
        model = _build_hybrid(synthetic_connectome)
        assert model._n_full == 50
        assert model._n_mb > 0
        assert model._n_mbon > 0
        assert model._n_kc > 0

    def test_forward_single(
        self, synthetic_connectome: Connectome, sample_state: EconomyState,
    ) -> None:
        model = _build_hybrid(synthetic_connectome)
        logits = model._forward_single(sample_state)
        assert logits.shape == (5,)
        assert torch.isfinite(logits).all()

    def test_forward_batch(self, synthetic_connectome: Connectome) -> None:
        model = _build_hybrid(synthetic_connectome)
        states = [
            EconomyState(money=800, loss_streak=0, round_number=1,
                         half=0, opponent_loss_streak=0),
            EconomyState(money=4000, loss_streak=2, round_number=5,
                         half=0, opponent_loss_streak=1),
        ]
        logits = model.forward(states)
        assert logits.shape == (2, 5)
        assert torch.isfinite(logits).all()

    def test_different_states_different_reservoir(
        self, synthetic_connectome: Connectome,
    ) -> None:
        """Different states produce different reservoir activations."""
        model = _build_hybrid(synthetic_connectome)
        s1 = EconomyState(money=800, loss_streak=0, round_number=1,
                          half=0, opponent_loss_streak=0)
        s2 = EconomyState(money=16000, loss_streak=4, round_number=12,
                          half=1, opponent_loss_streak=4)
        # Run rate reservoir for both states
        enc1 = model._encode_state(s1)
        enc2 = model._encode_state(s2)
        drive1 = model._project_input(enc1)
        drive2 = model._project_input(enc2)
        model._reservoir.reset()
        model._reservoir.step(drive1)
        r1 = model._reservoir.rates.clone()
        model._reservoir.reset()
        model._reservoir.step(drive2)
        r2 = model._reservoir.rates.clone()
        assert not torch.allclose(r1, r2)

    def test_readout_is_trainable(self, synthetic_connectome: Connectome) -> None:
        """Readout layer should be an nn.Module with trainable parameters."""
        model = _build_hybrid(synthetic_connectome)
        params = list(model.parameters())
        assert len(params) > 0
        # Should have weight and bias from readout
        assert any(p.shape[0] == 5 for p in params)

    def test_encode_state_deterministic(
        self, synthetic_connectome: Connectome, sample_state: EconomyState,
    ) -> None:
        model = _build_hybrid(synthetic_connectome)
        enc1 = model._encode_state(sample_state)
        enc2 = model._encode_state(sample_state)
        assert torch.allclose(enc1, enc2)


# ── Training loop tests ────────────────────────────────────────────────────


class TestTrainingLoop:
    def test_readout_trains(self, synthetic_connectome: Connectome) -> None:
        """Training the readout on synthetic data should decrease loss."""
        model = _build_hybrid(synthetic_connectome)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        rng = np.random.default_rng(42)
        n_mbon = model._n_mbon
        X = torch.randn(20, n_mbon)
        y = torch.tensor(rng.integers(0, 5, 20), dtype=torch.long)

        model.train()
        losses = []
        for _ in range(5):
            logits = model.readout(X)
            loss = torch.nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease or at least be finite
        assert all(l < 100 for l in losses)
        assert losses[-1] <= losses[0] + 0.1  # some tolerance


# ── Label map tests ────────────────────────────────────────────────────────


class TestLabelMap:
    def test_all_buy_plans_mapped(self) -> None:
        expected = {"FULL_BUY", "FORCE_BUY", "HALF_BUY", "ECO", "SAVE"}
        assert set(BUY_PLAN_TO_IDX.keys()) == expected

    def test_unique_indices(self) -> None:
        assert len(set(BUY_PLAN_TO_IDX.values())) == len(BUY_PLAN_TO_IDX)

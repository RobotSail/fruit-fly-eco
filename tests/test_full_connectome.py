"""Tests for full connectome scale-up support.

Since we can't download the real 166K-neuron connectome in CI, these
tests validate the --connectome full code path against larger synthetic
connectomes (1000+ neurons) to verify:

  1. Harness boots with connectome_mode='full' on synthetic data
  2. Gain calibration works on larger networks
  3. Preflight gate runs on larger networks
  4. DN-class neuron detection works
  5. build_fly_policy accepts explicit output_neuron_ids
  6. Full pipeline runs without OOM on 1000-neuron synthetic

The real 166K connectome would be tested manually via:
  python -m flyecon run --connectome full --avatar threejs
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from flyecon.etl.controls import random_sparse
from flyecon.etl.loader import Connectome
from flyecon.harness import (
    CONNECTOME_FULL,
    Harness,
    _find_dn_neuron_indices,
)
from flyecon.policy.ppo import build_fly_policy
from flyecon.sim.calibration import calibrate_gain, preflight_gate

# ── Shared fixtures ──────────────────────────────────────────────────────

_CACHED_ORACLE = None


def _get_oracle():
    global _CACHED_ORACLE
    if _CACHED_ORACLE is None:
        from flyecon.oracle.mdp import EconomyMDP
        from flyecon.oracle.solver import solve
        _CACHED_ORACLE = solve(EconomyMDP(), gamma=0.99, tol=1e-8)
    return _CACHED_ORACLE


def _make_large_connectome(
    n_neurons: int = 1000,
    density: float = 0.01,
    seed: int = 42,
    with_dn_types: bool = False,
) -> Connectome:
    """Build a larger synthetic connectome for full-scale testing.

    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    density : float
        Edge density.
    seed : int
        RNG seed.
    with_dn_types : bool
        If True, label some neurons as DN-class for readout expansion.
    """
    conn = random_sparse(n_neurons=n_neurons, density=density, seed=seed)

    if with_dn_types:
        # Label last 50 neurons as DN-class
        neuron_types = dict(conn.neuron_types)
        for i in range(max(0, n_neurons - 50), n_neurons):
            neuron_types[int(conn.body_ids[i])] = f"DNa{i:03d}"
        conn = Connectome(
            weight_matrix=conn.weight_matrix,
            dopamine_matrix=conn.dopamine_matrix,
            body_ids=conn.body_ids,
            neuron_types=neuron_types,
            n_neurons=conn.n_neurons,
            n_synapses=conn.n_synapses,
            metadata=conn.metadata,
        )

    return conn


# ── DN neuron detection ──────────────────────────────────────────────────


class TestDNNeuronDetection:
    """_find_dn_neuron_indices finds DN-class neurons correctly."""

    def test_finds_dn_neurons(self) -> None:
        """DN-type neurons are found by prefix match."""
        conn = _make_large_connectome(n_neurons=100, with_dn_types=True)
        dn_indices = _find_dn_neuron_indices(conn)
        assert len(dn_indices) > 0
        # Should find ~50 DN neurons (last 50 labelled)
        assert len(dn_indices) <= 50

    def test_no_dn_returns_empty(self) -> None:
        """Connectome without DN types returns empty list."""
        conn = _make_large_connectome(n_neurons=100, with_dn_types=False)
        dn_indices = _find_dn_neuron_indices(conn)
        assert dn_indices == []

    def test_dn_indices_sorted(self) -> None:
        """DN indices are returned sorted."""
        conn = _make_large_connectome(n_neurons=200, with_dn_types=True)
        dn_indices = _find_dn_neuron_indices(conn)
        assert dn_indices == sorted(dn_indices)


# ── Gain calibration on larger networks ──────────────────────────────────


class TestLargeScaleCalibration:
    """Gain calibration and preflight work on 1000+ neuron networks."""

    def test_calibration_1000_neurons(self) -> None:
        """Calibration finds a healthy gain on a 1000-neuron network."""
        conn = _make_large_connectome(n_neurons=1000, density=0.005)
        cal = calibrate_gain(
            conn, target_rate_hz=(1.0, 10.0), duration_ms=200.0
        )
        # Should find some gain (healthy or closest)
        assert cal.gain > 0
        assert len(cal.trials) > 0

    def test_preflight_1000_neurons(self) -> None:
        """Preflight gate runs without error on 1000-neuron network."""
        conn = _make_large_connectome(n_neurons=1000, density=0.005)
        # Use gain=0 to ensure zero-input stability at least
        pf = preflight_gate(conn, gain=0.0, duration_ms=100.0)
        # Gate 1 should pass (zero gain → zero synaptic current)
        assert pf.gate1_pass


# ── build_fly_policy with explicit output IDs ───────────────────────────


class TestBuildFlyPolicyExpanded:
    """build_fly_policy accepts explicit output_neuron_ids."""

    def test_explicit_output_ids(self) -> None:
        """Policy built with explicit DN indices uses them."""
        conn = _make_large_connectome(n_neurons=200, with_dn_types=True)
        dn_indices = _find_dn_neuron_indices(conn)
        assert len(dn_indices) > 0

        policy = build_fly_policy(
            conn, gain=0.0, output_neuron_ids=dn_indices,
        )
        # Readout should use the DN indices
        assert policy.readout.n_output_neurons == len(dn_indices)

    def test_default_output_ids(self) -> None:
        """Policy built without explicit IDs uses last-N default."""
        conn = _make_large_connectome(n_neurons=200)
        policy = build_fly_policy(conn, gain=0.0)
        # Default: last min(20, n) neurons
        assert policy.readout.n_output_neurons == 20

    def test_empty_output_ids_falls_back(self) -> None:
        """Empty output_neuron_ids falls back to default."""
        conn = _make_large_connectome(n_neurons=200)
        policy = build_fly_policy(conn, gain=0.0, output_neuron_ids=[])
        assert policy.readout.n_output_neurons == 20


# ── Full harness with larger synthetic connectome ────────────────────────


class TestHarnessFullMode:
    """Harness boots and trains with connectome_mode='full' on synthetic data."""

    def test_boot_full_mode_synthetic(self, tmp_path: Path) -> None:
        """Harness boots with connectome_mode set, using injected connectome."""
        conn = _make_large_connectome(n_neurons=100, with_dn_types=True)

        harness = Harness(
            n_neurons=100,
            density=0.01,
            seed=42,
            checkpoint_dir=tmp_path / "ckpt",
            telemetry_path=tmp_path / "tel.jsonl",
            heartbeat_dir=tmp_path / "hb",
            dashboard_path=tmp_path / "dash.html",
            eval_interval=0,
            checkpoint_interval=0,
            dashboard_interval=0,
            n_steps=16,
            connectome=conn,
            connectome_mode=CONNECTOME_FULL,
        )
        harness._oracle = _get_oracle()
        harness.boot()

        assert harness.policy is not None
        assert harness.mission_state.stage.value == "TRAINING"

    def test_train_full_mode(self, tmp_path: Path) -> None:
        """3 training iterations work with full-mode connectome."""
        conn = _make_large_connectome(n_neurons=100, with_dn_types=True)

        harness = Harness(
            n_neurons=100,
            density=0.01,
            seed=42,
            checkpoint_dir=tmp_path / "ckpt",
            telemetry_path=tmp_path / "tel.jsonl",
            heartbeat_dir=tmp_path / "hb",
            dashboard_path=tmp_path / "dash.html",
            eval_interval=0,
            checkpoint_interval=0,
            dashboard_interval=0,
            n_steps=16,
            connectome=conn,
            connectome_mode=CONNECTOME_FULL,
        )
        harness._oracle = _get_oracle()
        harness.boot()

        metrics = harness.train_iterations(3)
        for k, v in metrics.items():
            assert math.isfinite(v), f"Non-finite metric: {k}={v}"

        assert harness.mission_state.training_step == 3

    def test_1000_neuron_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline runs on 1000-neuron connectome without OOM.

        This is the key scale test: 1000 neurons, ~10K edges,
        proving the sparse CSR path handles larger matrices.
        """
        conn = _make_large_connectome(
            n_neurons=1000, density=0.005, with_dn_types=True,
        )

        harness = Harness(
            n_neurons=1000,
            density=0.005,
            seed=42,
            checkpoint_dir=tmp_path / "ckpt",
            telemetry_path=tmp_path / "tel.jsonl",
            heartbeat_dir=tmp_path / "hb",
            dashboard_path=tmp_path / "dash.html",
            eval_interval=0,
            checkpoint_interval=0,
            dashboard_interval=0,
            n_steps=8,
            connectome=conn,
            connectome_mode=CONNECTOME_FULL,
        )
        harness._oracle = _get_oracle()
        harness.boot()

        metrics = harness.train_iterations(2)
        for k, v in metrics.items():
            assert math.isfinite(v), f"Non-finite metric: {k}={v}"


# ── Int32 indices verification ───────────────────────────────────────────


class TestInt32Indices:
    """Verify sparse CSR uses int32 indices for memory efficiency."""

    def test_large_connectome_int32(self) -> None:
        """1000-neuron connectome has int32 crow/col indices."""
        conn = _make_large_connectome(n_neurons=1000, density=0.005)
        crow = conn.weight_matrix.crow_indices()
        col = conn.weight_matrix.col_indices()
        assert crow.dtype == torch.int32
        assert col.dtype == torch.int32

    def test_weight_matrix_shape(self) -> None:
        """Weight matrix has expected shape for large connectome."""
        conn = _make_large_connectome(n_neurons=1000, density=0.005)
        assert conn.weight_matrix.shape == (1000, 1000)
        assert conn.n_neurons == 1000


# ── Avatar mode in harness ───────────────────────────────────────────────


class TestHarnessAvatarMode:
    """Harness propagates avatar_mode to dashboard rendering."""

    def test_threejs_avatar_mode(self, tmp_path: Path) -> None:
        """Harness accepts avatar_mode='threejs'."""
        harness = Harness(
            n_neurons=30,
            density=0.1,
            seed=42,
            checkpoint_dir=tmp_path / "ckpt",
            telemetry_path=tmp_path / "tel.jsonl",
            heartbeat_dir=tmp_path / "hb",
            dashboard_path=tmp_path / "dash.html",
            eval_interval=0,
            checkpoint_interval=0,
            dashboard_interval=0,
            n_steps=16,
            avatar_mode="threejs",
        )
        assert harness.avatar_mode == "threejs"

    def test_sprite_avatar_mode(self, tmp_path: Path) -> None:
        """Harness accepts avatar_mode='sprite'."""
        harness = Harness(
            n_neurons=30,
            density=0.1,
            seed=42,
            checkpoint_dir=tmp_path / "ckpt",
            telemetry_path=tmp_path / "tel.jsonl",
            heartbeat_dir=tmp_path / "hb",
            dashboard_path=tmp_path / "dash.html",
            eval_interval=0,
            checkpoint_interval=0,
            dashboard_interval=0,
            n_steps=16,
            avatar_mode="sprite",
        )
        assert harness.avatar_mode == "sprite"

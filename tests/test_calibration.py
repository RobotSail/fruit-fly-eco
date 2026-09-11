"""Tests for gain calibration and pre-flight gate.

Validates that calibration finds healthy gain on small networks,
and that preflight correctly rejects silent/saturated configurations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flyecon.etl.loader import Connectome, build_csr_tensor
from flyecon.etl.controls import random_sparse
from flyecon.sim.calibration import (
    CalibrationResult,
    PreflightResult,
    calibrate_gain,
    preflight_gate,
)
from flyecon.sim.lif import LIFNetwork


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_mixed_ei_connectome(
    n_neurons: int = 100,
    density: float = 0.1,
    ei_ratio: float = 0.8,
    inhibitory_weight: float = 1.0,
    seed: int = 42,
) -> Connectome:
    """Create a sparse connectome with mixed E/I connections.

    With inhibitory connections, increasing gain suppresses activity,
    making calibration meaningful: there exists a gain where the mean
    firing rate falls into a target range.

    Parameters
    ----------
    ei_ratio : float
        Fraction of excitatory connections (rest are inhibitory).
    inhibitory_weight : float
        Magnitude of inhibitory weights.  Excitatory weights are +1.0.
        Use values > 1.0 to create stronger inhibition that dominates
        at moderate gain values (needed for small test networks where
        sparse unit-weight synapses produce negligible feedback).
    """
    rng = np.random.default_rng(seed)
    n_possible = n_neurons * (n_neurons - 1)
    n_edges = int(n_possible * density)

    if n_edges == 0:
        empty = build_csr_tensor(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            n_neurons,
        )
        return Connectome(
            weight_matrix=empty,
            dopamine_matrix=empty,
            body_ids=np.arange(n_neurons, dtype=np.int64),
            neuron_types={},
            n_neurons=n_neurons,
            n_synapses=0,
            metadata={"control": "mixed_ei"},
        )

    flat_indices = rng.choice(n_possible, size=n_edges, replace=False)
    rows = flat_indices // (n_neurons - 1)
    cols = flat_indices % (n_neurons - 1)
    cols = np.where(cols >= rows, cols + 1, cols)

    # Assign signs: ei_ratio fraction are +1, rest are -inhibitory_weight
    is_excitatory = rng.random(n_edges) < ei_ratio
    vals = np.where(is_excitatory, 1.0, -inhibitory_weight).astype(np.float32)

    weight_matrix = build_csr_tensor(
        rows.astype(np.int64), cols.astype(np.int64), vals, n_neurons,
    )
    empty_dopa = build_csr_tensor(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float32),
        n_neurons,
    )

    return Connectome(
        weight_matrix=weight_matrix,
        dopamine_matrix=empty_dopa,
        body_ids=np.arange(n_neurons, dtype=np.int64),
        neuron_types={},
        n_neurons=n_neurons,
        n_synapses=n_edges,
        metadata={"control": "mixed_ei", "ei_ratio": ei_ratio},
    )


@pytest.fixture
def mixed_ei_connectome():
    """100-neuron mixed E/I connectome where gain calibration is meaningful.

    With 30% excitatory / 70% inhibitory at density 0.15 and strong
    inhibitory weights (5.0):
    - Low gain: direct input dominates, ~32 Hz (with input=10)
    - Higher gain: strong inhibition suppresses firing
    - There exists a gain where rate falls in [1, 10] Hz

    With n_in ≈ 15, ~4.5 E and ~10.5 I connections per neuron, and
    inhibitory weight 5.0:
      net I_syn ≈ gain * (4.5*1.0 - 10.5*5.0) * P(spike)
                = gain * (-48) * P(spike)
    At ~32 Hz: P(spike/step) ≈ 0.16, so I_syn ≈ -7.7 * gain.
    For I_syn to suppress 10 mV input to near-threshold (7 mV):
      gain ≈ 3/7.7 ≈ 0.39  (within calibration search range)
    """
    return _make_mixed_ei_connectome(
        n_neurons=100, density=0.15, ei_ratio=0.3,
        inhibitory_weight=5.0, seed=42,
    )


@pytest.fixture
def small_random_connectome():
    """100-neuron all-excitatory connectome (for preflight tests)."""
    return random_sparse(n_neurons=100, density=0.05, seed=42)


@pytest.fixture
def dense_connectome():
    """50-neuron dense connectome for saturation tests."""
    return random_sparse(n_neurons=50, density=0.3, seed=77)


# ── Test: calibration finds healthy gain on mixed E/I network ──────────────


def test_calibration_finds_healthy_gain(mixed_ei_connectome):
    """calibrate_gain returns a healthy gain for a mixed E/I network."""
    result = calibrate_gain(
        mixed_ei_connectome,
        target_rate_hz=(1.0, 10.0),
        duration_ms=500.0,
        n_trials=5,
        input_amplitude=10.0,
    )
    assert isinstance(result, CalibrationResult)
    assert result.is_healthy, (
        f"Calibration failed: gain={result.gain:.6f}, "
        f"rate={result.mean_rate_hz:.2f} Hz"
    )
    assert 1.0 <= result.mean_rate_hz <= 10.0
    assert result.gain > 0
    assert len(result.trials) > 0


def test_calibration_returns_trials(mixed_ei_connectome):
    """calibrate_gain logs trial details."""
    result = calibrate_gain(
        mixed_ei_connectome,
        target_rate_hz=(1.0, 10.0),
        duration_ms=500.0,
    )
    assert len(result.trials) >= 1
    for trial in result.trials:
        assert "gain" in trial
        assert "rate_hz" in trial
        assert trial["gain"] > 0


# ── Test: preflight rejects gain=0 (silence) ──────────────────────────────


def test_preflight_rejects_silence(small_random_connectome):
    """Preflight gate rejects gain=0 with zero input (all silent)."""
    result_silent = preflight_gate(
        small_random_connectome,
        gain=0.0,
        duration_ms=500.0,
        input_amplitude=0.0,
    )
    assert isinstance(result_silent, PreflightResult)
    assert result_silent.gate1_pass, "Gate 1 should pass with gain=0 (no spikes)"
    assert not result_silent.gate2_pass, (
        "Gate 2 should fail with gain=0 and zero input (network is silent)"
    )
    assert not result_silent.all_pass, "All gates should not pass for silent network"


# ── Test: preflight rejects extreme gain (saturation) ──────────────────────


def test_preflight_rejects_saturation(dense_connectome):
    """Preflight gate rejects extreme gain (spontaneous spiking or >10 Hz)."""
    result = preflight_gate(
        dense_connectome,
        gain=1.0,
        duration_ms=500.0,
        input_amplitude=10.0,
    )
    assert isinstance(result, PreflightResult)
    # With gain=1.0 on a dense all-excitatory network, either:
    # - Gate 1 fails (spontaneous spikes from strong recurrent excitation)
    # - Gate 2 fails (rate > 10 Hz saturated)
    assert not result.all_pass, (
        f"All gates should not pass for extreme gain=1.0: "
        f"g1={result.gate1_pass}, g2={result.gate2_pass}, g3={result.gate3_pass}"
    )


# ── Test: preflight passes with calibrated gain ───────────────────────────


def test_preflight_passes_with_calibrated_gain(mixed_ei_connectome):
    """If calibration succeeds, preflight should pass with the found gain."""
    cal = calibrate_gain(
        mixed_ei_connectome,
        target_rate_hz=(1.0, 10.0),
        duration_ms=500.0,
        input_amplitude=10.0,
    )
    if not cal.is_healthy:
        pytest.skip("Calibration didn't find healthy gain on this network")

    result = preflight_gate(
        mixed_ei_connectome,
        gain=cal.gain,
        duration_ms=500.0,
        input_amplitude=10.0,
    )
    assert result.gate1_pass, (
        f"Gate 1 failed with calibrated gain={cal.gain:.6f}: "
        f"rate={result.diagnostics.get('gate1_zero_rate_hz')}"
    )
    assert result.gate2_pass, (
        f"Gate 2 failed with calibrated gain={cal.gain:.6f}: "
        f"rate={result.diagnostics.get('gate2_mod_rate_hz')}"
    )


# ── Test: PreflightResult.all_pass ────────────────────────────────────────


def test_preflight_result_all_pass():
    """PreflightResult.all_pass is True only when all 3 gates pass."""
    assert PreflightResult(True, True, True).all_pass
    assert not PreflightResult(False, True, True).all_pass
    assert not PreflightResult(True, False, True).all_pass
    assert not PreflightResult(True, True, False).all_pass
    assert not PreflightResult(False, False, False).all_pass

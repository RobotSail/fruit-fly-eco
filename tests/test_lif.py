"""Tests for the LIF simulation core.

Validates exact/exponential integration, refractory mechanics,
spike behaviour, and batched simulation equivalence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from flyecon.etl.controls import random_sparse
from flyecon.sim.lif import C_M, LIFNetwork, _DECAY, _ONE_MINUS_DECAY
from flyecon.state.constants import (
    DT_MS,
    REFRACTORY_MS,
    TAU_MS,
    V_REST_MV,
    V_THRESH_MV,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def small_connectome():
    """10-neuron random sparse connectome for unit tests."""
    return random_sparse(n_neurons=10, density=0.1, seed=42)


@pytest.fixture
def single_neuron_connectome():
    """1-neuron connectome with no synapses (isolated neuron)."""
    return random_sparse(n_neurons=1, density=0.0, seed=0)


@pytest.fixture
def medium_connectome():
    """50-neuron random sparse connectome."""
    return random_sparse(n_neurons=50, density=0.05, seed=99)


# ── Test: zero input settles at V_REST, no spikes ──────────────────────────


def test_zero_input_no_spikes(small_connectome):
    """Zero input → V stays at V_REST, no spikes ever."""
    net = LIFNetwork(small_connectome, gain=0.001)
    zero_input = torch.zeros(10, dtype=torch.float32)

    for _ in range(200):  # 200 steps = 1000ms
        spikes = net.step(zero_input)
        assert not spikes.any(), "Spikes occurred with zero input"

    # V should be exactly at V_REST
    assert torch.allclose(
        net.V, torch.full_like(net.V, V_REST_MV), atol=1e-5,
    ), f"V should be V_REST={V_REST_MV}, got {net.V}"


def test_zero_input_simulate(small_connectome):
    """simulate() with zero input → zero spike counts."""
    net = LIFNetwork(small_connectome, gain=0.001)
    zero_input = torch.zeros(10, dtype=torch.float32)
    counts = net.simulate(zero_input, duration_ms=1000.0)
    assert (counts == 0).all(), f"Expected zero spikes, got {counts}"


# ── Test: single neuron f-I curve matches analytical LIF ───────────────────


def test_single_neuron_fi_curve(single_neuron_connectome):
    """Single isolated neuron: firing rate matches exact LIF analytical f-I curve.

    Analytical f-I for exact LIF (with C_m = tau, refractory t_ref):
        ISI = tau * ln(I / (I - I_thresh)) + t_ref
        rate = 1000 / ISI  (Hz, with ISI in ms)

    At dt=5ms the spike-time discretization error is at most dt/ISI,
    so we use moderate-I values where ISI >> dt to keep relative error
    low.  We allow up to 20% tolerance for the coarse time step.
    """
    # With C_m = tau, V_ss = V_rest + I, so I_thresh = V_thresh - V_rest
    I_thresh = V_THRESH_MV - V_REST_MV  # 7.0 mV-equivalent

    # Use currents where ISI is well above dt=5ms to minimise
    # discrete-time quantisation error:
    #   I=7.5 → ISI ≈ 20*ln(15) + 2 ≈ 56ms → rate ≈ 17.8 Hz
    #   I=8.0 → ISI ≈ 20*ln(8) + 2 ≈ 43.6ms → rate ≈ 22.9 Hz
    #   I=9.0 → ISI ≈ 20*ln(4.5) + 2 ≈ 32.1ms → rate ≈ 31.2 Hz
    #   I=10.0 → ISI ≈ 20*ln(10/3) + 2 ≈ 26.1ms → rate ≈ 38.3 Hz
    for I_input in [7.5, 8.0, 9.0, 10.0]:
        assert I_input > I_thresh, "Test requires suprathreshold input"

        net = LIFNetwork(single_neuron_connectome, gain=0.0)
        inp = torch.tensor([I_input], dtype=torch.float32)

        # Analytical ISI for exact LIF (continuous-time):
        analytical_isi_ms = (
            TAU_MS * math.log(I_input / (I_input - I_thresh))
            + REFRACTORY_MS
        )
        analytical_rate_hz = 1000.0 / analytical_isi_ms

        # Simulate long enough to get stable rate
        duration_ms = 10000.0
        counts = net.simulate(inp, duration_ms)
        simulated_rate_hz = float(counts[0].item()) / (duration_ms / 1000.0)

        # Allow 20% tolerance for discrete-time effects at dt=5ms.
        # The main source of error is spike-time quantisation:
        # true ISI gets rounded to nearest multiple of dt.
        rel_error = abs(simulated_rate_hz - analytical_rate_hz) / analytical_rate_hz
        assert rel_error < 0.20, (
            f"I={I_input}: simulated {simulated_rate_hz:.2f} Hz vs "
            f"analytical {analytical_rate_hz:.2f} Hz (error {rel_error:.1%})"
        )


# ── Test: exact integration matches forward Euler (dt=0.01ms) to <1% ──────


def test_exact_vs_forward_euler(small_connectome):
    """Exact integration at dt=5ms matches forward Euler at dt=0.01ms to <1%.

    We compare the final membrane potential after a fixed integration
    period with constant subthreshold input (no spikes).
    """
    n = small_connectome.n_neurons
    # Use subthreshold input so no spikes occur (clean voltage comparison)
    I_input = 3.0  # below threshold for isolated neurons
    inp = torch.full((n,), I_input, dtype=torch.float32)
    gain = 0.0  # disable synaptic current for clean comparison

    # ── Exact integration (dt=5ms) ──
    net_exact = LIFNetwork(small_connectome, gain=gain)
    duration_ms = 100.0
    n_steps_exact = int(duration_ms / DT_MS)
    for _ in range(n_steps_exact):
        net_exact.step(inp)
    V_exact = net_exact.V.clone()

    # ── Forward Euler (dt=0.01ms) ──
    dt_euler = 0.01
    n_steps_euler = int(duration_ms / dt_euler)
    V_euler = torch.full((n,), V_REST_MV, dtype=torch.float32)

    for _ in range(n_steps_euler):
        dV = (-( V_euler - V_REST_MV) / TAU_MS + I_input / C_M) * dt_euler
        V_euler = V_euler + dV

    # Compare (should match to <1%)
    # Both should reach V_rest + I_input * tau/C_m = V_rest + I_input = -49 mV
    rel_error = float(
        torch.max(torch.abs(V_exact - V_euler) / (torch.abs(V_euler - V_REST_MV) + 1e-8)).item()
    )
    assert rel_error < 0.01, (
        f"Exact vs Euler relative error: {rel_error:.4f} "
        f"(V_exact={V_exact[0]:.4f}, V_euler={V_euler[0]:.4f})"
    )


# ── Test: batched simulate == sequential simulate ──────────────────────────


def test_batched_equals_sequential(small_connectome):
    """batched_simulate produces identical results to sequential simulate."""
    n = small_connectome.n_neurons
    batch_size = 4
    duration_ms = 200.0

    # Create varied inputs
    rng = torch.Generator().manual_seed(42)
    input_currents = torch.rand(batch_size, n, generator=rng) * 10.0

    # Batched
    net = LIFNetwork(small_connectome, gain=0.01)
    batched_counts = net.batched_simulate(input_currents, duration_ms)

    # Sequential
    sequential_counts = torch.zeros_like(batched_counts)
    for b in range(batch_size):
        sequential_counts[b] = net.simulate(input_currents[b], duration_ms)

    assert torch.allclose(batched_counts, sequential_counts, atol=1e-5), (
        f"Batched vs sequential mismatch: "
        f"max diff = {(batched_counts - sequential_counts).abs().max():.6f}"
    )


# ── Test: refractory period prevents double-spike within 2ms ───────────────


def test_refractory_no_double_spike(single_neuron_connectome):
    """A neuron that just spiked cannot spike again within REFRACTORY_MS."""
    # Strong suprathreshold current that would spike every step without refractory
    I_strong = 50.0
    inp = torch.tensor([I_strong], dtype=torch.float32)

    net = LIFNetwork(single_neuron_connectome, gain=0.0)

    spike_times: list[float] = []
    n_steps = 200
    for t in range(n_steps):
        spikes = net.step(inp)
        if spikes[0]:
            spike_times.append(t * DT_MS)

    # Check minimum ISI >= DT_MS + REFRACTORY_MS
    # Since dt=5ms and refractory=2ms, the neuron is held at V_REST for
    # ceil(2/5)=1 additional step after spiking, so minimum ISI = 2*DT_MS = 10ms
    assert len(spike_times) >= 2, "Need at least 2 spikes to check ISI"
    for i in range(1, len(spike_times)):
        isi = spike_times[i] - spike_times[i - 1]
        # Refractory guarantees at least one skipped step
        assert isi >= DT_MS + DT_MS, (
            f"ISI {isi}ms is shorter than min refractory ISI "
            f"({DT_MS + DT_MS}ms) between spike {i-1} and {i}"
        )


# ── Test: reset clears all state ───────────────────────────────────────────


def test_reset_clears_state(small_connectome):
    """reset() restores V to V_REST, zeroes spike_counts and refractory."""
    net = LIFNetwork(small_connectome, gain=0.01)
    inp = torch.full((10,), 15.0, dtype=torch.float32)

    # Simulate a bit to dirty state
    net.simulate(inp, duration_ms=100.0)
    assert net.spike_counts.sum() > 0 or True  # may or may not spike

    # Reset
    net.reset()
    assert torch.allclose(net.V, torch.full_like(net.V, V_REST_MV))
    assert (net.spike_counts == 0).all()
    assert (net.refractory == 0).all()
    assert (net._prev_spikes == 0).all()


# ── Test: spike produces reset to V_REST ───────────────────────────────────


def test_spike_resets_voltage(single_neuron_connectome):
    """When a neuron spikes, its voltage is reset to V_REST."""
    I_supra = 20.0  # suprathreshold
    inp = torch.tensor([I_supra], dtype=torch.float32)

    net = LIFNetwork(single_neuron_connectome, gain=0.0)

    spiked_once = False
    for _ in range(100):
        spikes = net.step(inp)
        if spikes[0]:
            # After spike, V should be V_REST
            assert abs(net.V[0].item() - V_REST_MV) < 1e-5, (
                f"Post-spike V should be {V_REST_MV}, got {net.V[0].item()}"
            )
            spiked_once = True
            break

    assert spiked_once, "Neuron never spiked with suprathreshold input"


# ── Test: float32 precision ────────────────────────────────────────────────


def test_float32_tensors(small_connectome):
    """All state tensors use float32."""
    net = LIFNetwork(small_connectome, gain=0.01)
    assert net.V.dtype == torch.float32
    assert net.refractory.dtype == torch.float32
    assert net.spike_counts.dtype == torch.float32
    assert net._prev_spikes.dtype == torch.float32

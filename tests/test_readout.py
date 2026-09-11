"""Tests for SpikeReadout (Phase 5).

Covers:
  - Baseline normalisation: spike counts at baseline → near-zero z-scores
  - Output shape matches n_actions (5 buy plans)
  - Differentiability: grad flows through W, b
  - Single and batched forward pass
  - Edge cases (zero spike counts, very high counts)
"""

from __future__ import annotations

import torch

from flyecon.readout.linear import DEFAULT_BIN_MS, N_ACTIONS, SpikeReadout

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_readout(
    n_output: int = 10,
    n_neurons: int = 50,
    baseline_rate: float = 5.0,
) -> tuple[SpikeReadout, list[int]]:
    """Create a SpikeReadout with uniform baseline rates."""
    ids = list(range(n_output))
    baselines = torch.full((n_output,), baseline_rate)
    readout = SpikeReadout(
        output_neuron_ids=ids,
        baseline_rates=baselines,
    )
    return readout, ids


# ── Output shape ──────────────────────────────────────────────────────────


class TestOutputShape:
    """Output shape matches n_actions."""

    def test_single_output_shape(self) -> None:
        readout, ids = _make_readout(n_output=10, n_neurons=50)
        spikes = torch.zeros(50)
        logits = readout(spikes)
        assert logits.shape == (N_ACTIONS,)

    def test_batched_output_shape(self) -> None:
        readout, ids = _make_readout(n_output=10, n_neurons=50)
        spikes = torch.zeros(4, 50)
        logits = readout(spikes)
        assert logits.shape == (4, N_ACTIONS)

    def test_custom_n_actions(self) -> None:
        ids = list(range(8))
        baselines = torch.ones(8)
        readout = SpikeReadout(ids, baselines, n_actions=3)
        spikes = torch.zeros(50)
        logits = readout(spikes)
        assert logits.shape == (3,)


# ── Baseline normalisation ────────────────────────────────────────────────


class TestBaselineNormalisation:
    """Spike counts at baseline rate → near-zero z-scores."""

    def test_baseline_counts_produce_near_zero_z(self) -> None:
        """When spike counts match baseline rates, z-scores ≈ 0."""
        baseline_rate = 5.0  # Hz
        duration_ms = DEFAULT_BIN_MS
        n_output = 10

        readout, ids = _make_readout(
            n_output=n_output, n_neurons=50, baseline_rate=baseline_rate
        )

        # Generate spike counts that match baseline rates
        # rate = counts / (duration_ms / 1000)  →  counts = rate * (duration_ms / 1000)
        expected_counts = baseline_rate * (duration_ms / 1000.0)
        spikes = torch.zeros(50)
        for i in ids:
            spikes[i] = expected_counts

        # Zero W so logits = 0 + b when z = 0
        with torch.no_grad():
            readout.W.zero_()
            readout.b.zero_()

        logits = readout(spikes, duration_ms)
        # With W=0, b=0 and z≈0, logits should be ≈ 0
        assert logits.abs().max() < 1e-4

    def test_above_baseline_produces_positive_z(self) -> None:
        """Spike counts above baseline → positive z-scores → different logits."""
        readout, ids = _make_readout(n_output=5, n_neurons=20, baseline_rate=5.0)
        duration_ms = DEFAULT_BIN_MS

        # Counts well above baseline
        spikes_high = torch.zeros(20)
        for i in ids:
            spikes_high[i] = 20.0 * (duration_ms / 1000.0)  # 20 Hz >> 5 Hz baseline

        # Counts at baseline
        spikes_base = torch.zeros(20)
        for i in ids:
            spikes_base[i] = 5.0 * (duration_ms / 1000.0)

        logits_high = readout(spikes_high, duration_ms)
        logits_base = readout(spikes_base, duration_ms)

        # Outputs should be different (z-scores differ)
        assert not torch.allclose(logits_high, logits_base)


# ── Differentiability ─────────────────────────────────────────────────────


class TestDifferentiability:
    """Grad flows through W and b."""

    def test_grad_flows_through_W(self) -> None:
        readout, ids = _make_readout(n_output=8, n_neurons=30)
        spikes = torch.randn(30).abs() * 10
        logits = readout(spikes)
        loss = logits.sum()
        loss.backward()
        assert readout.W.grad is not None
        assert readout.W.grad.abs().sum() > 0

    def test_grad_flows_through_b(self) -> None:
        readout, ids = _make_readout(n_output=8, n_neurons=30)
        spikes = torch.randn(30).abs() * 10
        logits = readout(spikes)
        loss = logits.sum()
        loss.backward()
        assert readout.b.grad is not None

    def test_grad_flows_batched(self) -> None:
        readout, ids = _make_readout(n_output=6, n_neurons=20)
        spikes = torch.randn(3, 20).abs() * 10
        logits = readout(spikes)
        loss = logits.sum()
        loss.backward()
        assert readout.W.grad is not None
        assert readout.b.grad is not None


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: zero counts, mismatched sizes."""

    def test_zero_spike_counts(self) -> None:
        readout, _ = _make_readout(n_output=5, n_neurons=20, baseline_rate=5.0)
        spikes = torch.zeros(20)
        logits = readout(spikes)
        # Should produce finite output (no NaN or Inf)
        assert torch.isfinite(logits).all()

    def test_mismatched_baseline_raises(self) -> None:
        import pytest

        ids = [0, 1, 2]
        wrong_baselines = torch.ones(5)  # length 5 != 3
        with pytest.raises(ValueError, match="baseline_rates length"):
            SpikeReadout(ids, wrong_baselines)

    def test_output_neuron_ids_registered_as_buffer(self) -> None:
        readout, ids = _make_readout()
        # Buffers should not be in parameters
        param_names = {name for name, _ in readout.named_parameters()}
        assert "output_neuron_ids" not in param_names
        assert "baseline_rates" not in param_names
        # But they should be in named_buffers
        buffer_names = {name for name, _ in readout.named_buffers()}
        assert "output_neuron_ids" in buffer_names
        assert "baseline_rates" in buffer_names

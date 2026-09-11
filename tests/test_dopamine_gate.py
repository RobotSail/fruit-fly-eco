"""Integration tests for dopamine-modulated reward."""
import torch
import pytest
from flyecon.policy.ppo import DopamineGate


def test_dopamine_gate_output_range():
    """Gate output should be in [0.5, 1.5]."""
    gate = DopamineGate()
    for val in [0.0, 0.5, 1.0, 5.0, 100.0]:
        out = gate(torch.tensor(val))
        assert 0.5 <= float(out) <= 1.5, (
            f"Gate output {out} out of range for input {val}"
        )


def test_dopamine_gate_identity_init():
    """Fresh gate should output ~1.0 (no modulation)."""
    gate = DopamineGate()
    out = gate(torch.tensor(0.0))
    assert abs(float(out) - 1.0) < 0.1, (
        f"Fresh gate should be near 1.0, got {out}"
    )


def test_dopamine_gate_is_differentiable():
    """Gate must be trainable via backprop."""
    gate = DopamineGate()
    x = torch.tensor(1.0, requires_grad=True)
    out = gate(x)
    out.backward()
    assert x.grad is not None


def test_ppl_activation_synthetic_tagged():
    """PPL activation should be readable on MB-tagged synthetic connectome."""
    from flyecon.etl.controls import random_sparse
    from flyecon.sim.lif import LIFNetwork

    conn = random_sparse(n_neurons=200, density=0.05, seed=42, with_mb_tags=True)
    lif = LIFNetwork(conn, gain=1.0)

    # Verify PPL indices were found
    assert len(lif._ppl_indices) > 0, "No PPL neurons found in MB-tagged connectome"
    assert len(lif._kc_indices) > 0, "No KC neurons found"
    assert len(lif._mbon_indices) > 0, "No MBON neurons found"

    # Run simulation
    input_current = torch.full((200,), 10.0)
    lif.simulate(input_current, duration_ms=100.0)

    # PPL activation should be readable
    ppl_act = lif.get_ppl_activation()
    assert isinstance(ppl_act, float)

    pop_rates = lif.get_population_rates()
    assert "PPL" in pop_rates
    assert "KC" in pop_rates
    assert "MBON" in pop_rates


def test_ppl_activation_untagged_returns_zero():
    """PPL activation should return 0.0 on untagged synthetic connectome."""
    from flyecon.etl.controls import random_sparse
    from flyecon.sim.lif import LIFNetwork

    conn = random_sparse(n_neurons=50, density=0.1, seed=42, with_mb_tags=False)
    lif = LIFNetwork(conn, gain=0.0)

    assert len(lif._ppl_indices) == 0
    assert lif.get_ppl_activation() == 0.0

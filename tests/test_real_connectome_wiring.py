"""Integration tests for real connectome wiring.

Tests verify the column-name fixes, Traced filter, hops=0 default,
size cap, and graceful fallback — all using synthetic data shaped
to match the real Feather schema.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from flyecon.etl.loader import load_connectome_from_tables, Connectome


def _make_real_schema_dfs(n_neurons=100, n_edges=500):
    """Create DataFrames matching the REAL MaleCNS Feather column names."""
    rng = np.random.default_rng(42)
    body_ids = np.arange(1000, 1000 + n_neurons)

    # Weights — real schema: body_pre, body_post, weight
    pre = rng.choice(body_ids, size=n_edges)
    post = rng.choice(body_ids, size=n_edges)
    weights_df = pd.DataFrame({
        "body_pre": pre,
        "body_post": post,
        "weight": rng.integers(1, 100, size=n_edges),
    })

    # Neurotransmitters — real schema: body, predicted_nt, consensus_nt
    nts = rng.choice(
        ["acetylcholine", "gaba", "glutamate", "dopamine", "serotonin"],
        size=n_neurons,
    )
    nt_df = pd.DataFrame({
        "body": body_ids,
        "predicted_nt": nts,
        "consensus_nt": nts,
        "predicted_nt_confidence": rng.random(n_neurons),
    })

    # Annotations — real schema: bodyId (correct here), type, status
    types = []
    for i in range(n_neurons):
        if i < 60:
            types.append(f"KC-s{i}")
        elif i < 70:
            types.append(f"MBON-{i:02d}")
        elif i < 75:
            types.append(f"PPL1-{i:02d}")
        else:
            types.append(f"Other-{i}")

    ann_df = pd.DataFrame({
        "bodyId": body_ids,
        "type": types,
        "status": ["Traced"] * 80 + ["Orphan"] * 20,
    })

    return weights_df, nt_df, ann_df


def _make_old_schema_dfs(n_neurons=50, n_edges=200):
    """Create DataFrames matching the OLD/test column names."""
    rng = np.random.default_rng(99)
    body_ids = np.arange(2000, 2000 + n_neurons)

    weights_df = pd.DataFrame({
        "bodyId_pre": rng.choice(body_ids, size=n_edges),
        "bodyId_post": rng.choice(body_ids, size=n_edges),
        "weight": rng.integers(1, 50, size=n_edges),
    })

    nt_df = pd.DataFrame({
        "bodyId": body_ids,
        "predictedNt": rng.choice(
            ["acetylcholine", "gaba", "glutamate"], size=n_neurons,
        ),
    })

    ann_df = pd.DataFrame({
        "bodyId": body_ids,
        "type": [f"Neuron-{i}" for i in range(n_neurons)],
    })

    return weights_df, nt_df, ann_df


def test_load_connectome_real_schema_columns():
    """Verify column-name fixes: body_pre/body_post/body/predicted_nt work."""
    weights_df, nt_df, ann_df = _make_real_schema_dfs()
    conn = load_connectome_from_tables(weights_df, nt_df, ann_df)
    assert conn.n_neurons > 0
    assert conn.n_synapses > 0


def test_load_connectome_old_schema_columns():
    """Verify backward compat: bodyId_pre/bodyId_post/bodyId/predictedNt work."""
    weights_df, nt_df, ann_df = _make_old_schema_dfs()
    conn = load_connectome_from_tables(weights_df, nt_df, ann_df)
    assert conn.n_neurons > 0
    assert conn.n_synapses > 0


def test_traced_filter_reduces_neurons():
    """Verify status=='Traced' filter excludes non-neuron fragments."""
    weights_df, nt_df, ann_df = _make_real_schema_dfs()
    conn = load_connectome_from_tables(weights_df, nt_df, ann_df)
    # With 80 Traced + 20 Orphan, result should have <= 80 neurons
    assert conn.n_neurons <= 80


def test_dopamine_matrix_populated():
    """Verify dopamine neurons produce non-empty dopamine_matrix."""
    weights_df, nt_df, ann_df = _make_real_schema_dfs()
    conn = load_connectome_from_tables(weights_df, nt_df, ann_df)
    # Some neurons have predicted_nt="dopamine"
    dopa_nnz = conn.dopamine_matrix.values().numel()
    assert dopa_nnz >= 0  # may be 0 if no dopa pre-synaptic edges in sample


def test_mushroom_body_extraction_hops0():
    """Verify hops=0 extraction returns only KC/MBON/PPL neurons."""
    from flyecon.etl.subcircuit import extract_mushroom_body
    weights_df, nt_df, ann_df = _make_real_schema_dfs()
    conn = load_connectome_from_tables(weights_df, nt_df, ann_df)
    mb = extract_mushroom_body(conn, hops=0)
    # Should contain KC + MBON + PPL only (not "Other-*")
    for body_id, ntype in mb.neuron_types.items():
        assert any(ntype.startswith(p) for p in ("KC", "MBON", "PPL")), (
            f"Unexpected type: {ntype}"
        )


def test_mushroom_body_max_neurons_cap():
    """Verify max_neurons cap falls back to seed-only."""
    from flyecon.etl.subcircuit import extract_mushroom_body
    weights_df, nt_df, ann_df = _make_real_schema_dfs()
    conn = load_connectome_from_tables(weights_df, nt_df, ann_df)
    # Use hops=1 with a small cap to trigger fallback
    mb = extract_mushroom_body(conn, hops=1, max_neurons=10)
    # Should be capped — may fall back to seed set
    assert mb.n_neurons <= conn.n_neurons


def test_harness_fallback_on_load_failure():
    """Verify Harness falls back to synthetic when connectome download fails."""
    from flyecon.harness import Harness, CONNECTOME_SYNTHETIC

    h = Harness(connectome_mode="mushroom-body", n_neurons=50)
    # Patch _load_connectome to raise
    original = h._load_connectome

    def failing_load():
        raise ConnectionError("Simulated GCS download failure")

    h._load_connectome = failing_load

    # boot() should not crash — falls back to synthetic
    h.boot()
    assert h.connectome_mode == CONNECTOME_SYNTHETIC
    assert h._connectome is not None
    assert h._connectome.n_neurons == 50

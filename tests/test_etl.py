"""Tests for ETL pipeline — all using SYNTHETIC mock data.

No real connectome downloads.  Every test builds small DataFrames
that exercise the same code paths as the real Feather loader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from flyecon.etl.controls import random_sparse, rewire_degree_preserving
from flyecon.etl.loader import Connectome, build_csr_tensor, load_connectome_from_tables
from flyecon.etl.subcircuit import extract_mushroom_body

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_weights() -> pd.DataFrame:
    """6-edge synthetic edge list.

    Neuron 1 (ACh)  → 2, 3
    Neuron 2 (GABA) → 3
    Neuron 3 (Glu)  → 4
    Neuron 4 (Dopa) → 5   ← should go to dopamine matrix
    Neuron 5 (5-HT) → 1   ← should be zero-current
    """
    return pd.DataFrame(
        {
            "bodyId_pre": [1, 1, 2, 3, 4, 5],
            "bodyId_post": [2, 3, 3, 4, 5, 1],
            "weight": [10, 5, 8, 3, 7, 4],
        }
    )


@pytest.fixture
def mock_nt() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5],
            "predictedNt": [
                "acetylcholine",
                "gaba",
                "glutamate",
                "dopamine",
                "serotonin",
            ],
        }
    )


@pytest.fixture
def mock_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5],
            "type": ["KC-alpha", "MBON-01", "PPL-01", "KC-beta", "visual-PN"],
        }
    )


@pytest.fixture
def mock_connectome(
    mock_weights: pd.DataFrame,
    mock_nt: pd.DataFrame,
    mock_annotations: pd.DataFrame,
) -> Connectome:
    """Pre-built connectome from mock data."""
    return load_connectome_from_tables(mock_weights, mock_nt, mock_annotations)


# ── Sign convention tests ───────────────────────────────────────────────────


class TestSignConvention:
    """Verify NT → sign mapping is applied correctly."""

    def test_ach_positive(self, mock_connectome: Connectome) -> None:
        """Acetylcholine → +1 × weight."""
        dense = mock_connectome.weight_matrix.to_dense()
        # Neuron 1 (ACh) → neuron 2: +10
        assert dense[0, 1].item() == pytest.approx(10.0)
        # Neuron 1 (ACh) → neuron 3: +5
        assert dense[0, 2].item() == pytest.approx(5.0)

    def test_gaba_negative(self, mock_connectome: Connectome) -> None:
        """GABA → −1 × weight."""
        dense = mock_connectome.weight_matrix.to_dense()
        # Neuron 2 (GABA) → neuron 3: −8
        assert dense[1, 2].item() == pytest.approx(-8.0)

    def test_glutamate_negative(self, mock_connectome: Connectome) -> None:
        """Glutamate → −1 × weight."""
        dense = mock_connectome.weight_matrix.to_dense()
        # Neuron 3 (Glu) → neuron 4: −3
        assert dense[2, 3].item() == pytest.approx(-3.0)

    def test_serotonin_zero_current(self, mock_connectome: Connectome) -> None:
        """Serotonin → zero-current (edge retained with weight 0)."""
        dense = mock_connectome.weight_matrix.to_dense()
        # Neuron 5 (5-HT) → neuron 1: 0.0
        assert dense[4, 0].item() == pytest.approx(0.0)

    def test_unknown_nt_zero_current(self) -> None:
        """Unknown NT → zero-current."""
        weights = pd.DataFrame(
            {"bodyId_pre": [10, 20], "bodyId_post": [20, 10], "weight": [5, 3]}
        )
        nt = pd.DataFrame({"bodyId": [10], "predictedNt": ["acetylcholine"]})
        ann = pd.DataFrame({"bodyId": [10, 20], "type": ["n1", "n2"]})

        conn = load_connectome_from_tables(weights, nt, ann)
        dense = conn.weight_matrix.to_dense()

        # Neuron 10 (ACh) → 20: +5
        assert dense[0, 1].item() == pytest.approx(5.0)
        # Neuron 20 (unknown) → 10: 0
        assert dense[1, 0].item() == pytest.approx(0.0)


# ── Dopamine separation ────────────────────────────────────────────────────


class TestDopamineSeparation:
    """Dopamine edges must NOT appear in the fast-synaptic matrix."""

    def test_dopamine_not_in_weight_matrix(self, mock_connectome: Connectome) -> None:
        dense_w = mock_connectome.weight_matrix.to_dense()
        # Neuron 4 (dopamine) → neuron 5 should NOT be in weight matrix
        assert dense_w[3, 4].item() == pytest.approx(0.0)

    def test_dopamine_in_dopamine_matrix(self, mock_connectome: Connectome) -> None:
        dense_d = mock_connectome.dopamine_matrix.to_dense()
        # Neuron 4 (dopamine) → neuron 5 should be in dopamine matrix
        assert dense_d[3, 4].item() == pytest.approx(7.0)

    def test_dopamine_matrix_only_dopamine(self, mock_connectome: Connectome) -> None:
        """Only dopamine edges in the dopamine matrix."""
        dense_d = mock_connectome.dopamine_matrix.to_dense()
        # Only one non-zero entry: (3, 4)
        assert dense_d.count_nonzero().item() == 1


# ── Confidence filter ───────────────────────────────────────────────────────


class TestConfidenceFilter:
    """Edges below MIN_CONFIDENCE (0.5) are excluded."""

    def test_low_confidence_excluded(self) -> None:
        weights = pd.DataFrame(
            {
                "bodyId_pre": [1, 1, 2],
                "bodyId_post": [2, 3, 3],
                "weight": [10, 5, 8],
                "confidence": [0.6, 0.3, 0.8],
            }
        )
        nt = pd.DataFrame(
            {
                "bodyId": [1, 2, 3],
                "predictedNt": ["acetylcholine", "acetylcholine", "gaba"],
            }
        )
        ann = pd.DataFrame({"bodyId": [1, 2, 3], "type": ["n1", "n2", "n3"]})

        conn = load_connectome_from_tables(weights, nt, ann)
        dense = conn.weight_matrix.to_dense()

        # Edge 1→2 (conf 0.6 ≥ 0.5): included, ACh → +10
        assert dense[0, 1].item() == pytest.approx(10.0)
        # Edge 1→3 (conf 0.3 < 0.5): excluded
        assert dense[0, 2].item() == pytest.approx(0.0)
        # Edge 2→3 (conf 0.8 ≥ 0.5): included, ACh → +8
        assert dense[1, 2].item() == pytest.approx(8.0)

    def test_no_confidence_column_keeps_all(self) -> None:
        """If no confidence column, all edges are kept."""
        weights = pd.DataFrame(
            {"bodyId_pre": [1, 2], "bodyId_post": [2, 1], "weight": [5, 3]}
        )
        nt = pd.DataFrame(
            {"bodyId": [1, 2], "predictedNt": ["acetylcholine", "gaba"]}
        )
        ann = pd.DataFrame({"bodyId": [1, 2], "type": ["n1", "n2"]})

        conn = load_connectome_from_tables(weights, nt, ann)
        assert conn.weight_matrix.values().numel() == 2


# ── int32 indices ───────────────────────────────────────────────────────────


class TestInt32Indices:
    """CSR tensors MUST use int32 indices (saves 1 GB at full scale)."""

    def test_weight_matrix_int32(self, mock_connectome: Connectome) -> None:
        assert mock_connectome.weight_matrix.crow_indices().dtype == torch.int32
        assert mock_connectome.weight_matrix.col_indices().dtype == torch.int32

    def test_dopamine_matrix_int32(self, mock_connectome: Connectome) -> None:
        assert mock_connectome.dopamine_matrix.crow_indices().dtype == torch.int32
        assert mock_connectome.dopamine_matrix.col_indices().dtype == torch.int32

    def test_values_float32(self, mock_connectome: Connectome) -> None:
        assert mock_connectome.weight_matrix.values().dtype == torch.float32
        assert mock_connectome.dopamine_matrix.values().dtype == torch.float32


# ── Subcircuit extraction ───────────────────────────────────────────────────


class TestSubcircuitExtraction:
    """Mushroom body extraction with mock labelled neurons."""

    def test_mb_seed_neurons_found(self, mock_connectome: Connectome) -> None:
        """KC, MBON, PPL neurons should be identified."""
        # All 4 MB neurons: KC-alpha(1), MBON-01(2), PPL-01(3), KC-beta(4)
        sub = extract_mushroom_body(mock_connectome, hops=0)
        assert sub.n_neurons == 4

    def test_one_hop_includes_neighbours(self, mock_connectome: Connectome) -> None:
        """1-hop neighbours of MB neurons include neuron 5 (visual-PN)."""
        sub = extract_mushroom_body(mock_connectome, hops=1)
        # All 5 neurons should be included via 1-hop connections
        assert sub.n_neurons == 5

    def test_subgraph_reindexed(self, mock_connectome: Connectome) -> None:
        """Subgraph body IDs and matrices are re-indexed from 0."""
        sub = extract_mushroom_body(mock_connectome, hops=0)
        assert sub.weight_matrix.shape[0] == sub.n_neurons
        assert sub.weight_matrix.shape[1] == sub.n_neurons

    def test_subgraph_int32_indices(self, mock_connectome: Connectome) -> None:
        """Subgraph CSR tensors also use int32 indices."""
        sub = extract_mushroom_body(mock_connectome, hops=1)
        assert sub.weight_matrix.crow_indices().dtype == torch.int32
        assert sub.weight_matrix.col_indices().dtype == torch.int32

    def test_no_mb_neurons_raises(self) -> None:
        """Raise ValueError if no MB neurons in annotations."""
        weights = pd.DataFrame(
            {"bodyId_pre": [1], "bodyId_post": [2], "weight": [5]}
        )
        nt = pd.DataFrame({"bodyId": [1, 2], "predictedNt": ["acetylcholine", "gaba"]})
        ann = pd.DataFrame({"bodyId": [1, 2], "type": ["visual-1", "auditory-2"]})

        conn = load_connectome_from_tables(weights, nt, ann)
        with pytest.raises(ValueError, match="No mushroom body neurons"):
            extract_mushroom_body(conn, hops=1)


# ── Rewire control ──────────────────────────────────────────────────────────


class TestRewireControl:
    """Degree-preserving rewire preserves degree distribution."""

    def test_degree_distribution_preserved(self) -> None:
        """Out-degree and in-degree distributions must match."""
        conn = random_sparse(50, density=0.05, seed=42)
        rewired = rewire_degree_preserving(conn, seed=123)

        orig_d = conn.weight_matrix.to_dense()
        rew_d = rewired.weight_matrix.to_dense()

        # Per-neuron out-degree (number of nonzero entries per row)
        orig_out = (orig_d != 0).sum(dim=1).sort()[0]
        rew_out = (rew_d != 0).sum(dim=1).sort()[0]
        assert torch.equal(orig_out, rew_out)

        # Per-neuron in-degree
        orig_in = (orig_d != 0).sum(dim=0).sort()[0]
        rew_in = (rew_d != 0).sum(dim=0).sort()[0]
        assert torch.equal(orig_in, rew_in)

    def test_topology_differs(self) -> None:
        """Rewired graph should have different edge placement."""
        conn = random_sparse(50, density=0.05, seed=42)
        rewired = rewire_degree_preserving(conn, seed=999)

        orig_d = conn.weight_matrix.to_dense()
        rew_d = rewired.weight_matrix.to_dense()

        # Not identical (with high probability for density=0.05, 50 neurons)
        assert not torch.equal(orig_d, rew_d)

    def test_sign_preserved(self) -> None:
        """Sign distribution must be preserved per neuron."""
        # Build a connectome with mixed signs
        rows = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
        cols = np.array([1, 2, 2, 3, 3, 0], dtype=np.int64)
        vals = np.array([5.0, -3.0, -2.0, 4.0, -1.0, 6.0], dtype=np.float32)

        n = 5
        wm = build_csr_tensor(rows, cols, vals, n)
        empty = build_csr_tensor(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            n,
        )

        conn = Connectome(
            weight_matrix=wm,
            dopamine_matrix=empty,
            body_ids=np.arange(n, dtype=np.int64),
            neuron_types={},
            n_neurons=n,
            n_synapses=6,
        )

        rewired = rewire_degree_preserving(conn, seed=42)

        orig_d = conn.weight_matrix.to_dense()
        rew_d = rewired.weight_matrix.to_dense()

        # Count of positive and negative edges should be the same
        assert (orig_d > 0).sum() == (rew_d > 0).sum()
        assert (orig_d < 0).sum() == (rew_d < 0).sum()

    def test_rewired_int32_indices(self) -> None:
        conn = random_sparse(20, density=0.1, seed=7)
        rewired = rewire_degree_preserving(conn, seed=42)
        assert rewired.weight_matrix.crow_indices().dtype == torch.int32
        assert rewired.weight_matrix.col_indices().dtype == torch.int32


# ── Random sparse control ──────────────────────────────────────────────────


class TestRandomSparse:
    """Random sparse graph as baseline control."""

    def test_correct_shape(self) -> None:
        conn = random_sparse(100, density=0.02, seed=42)
        assert conn.n_neurons == 100
        assert conn.weight_matrix.shape == (100, 100)

    def test_approximate_density(self) -> None:
        conn = random_sparse(100, density=0.05, seed=42)
        n_possible = 100 * 99
        expected = int(n_possible * 0.05)
        actual = conn.weight_matrix.values().numel()
        # Allow 1% tolerance (dedup may reduce count slightly)
        assert abs(actual - expected) <= max(expected * 0.01, 5)

    def test_no_self_loops(self) -> None:
        conn = random_sparse(50, density=0.1, seed=42)
        dense = conn.weight_matrix.to_dense()
        assert torch.diagonal(dense).sum().item() == pytest.approx(0.0)

    def test_int32_indices(self) -> None:
        conn = random_sparse(30, density=0.05, seed=42)
        assert conn.weight_matrix.crow_indices().dtype == torch.int32


# ── build_csr_tensor edge cases ─────────────────────────────────────────────


class TestBuildCSR:
    """Edge cases for the CSR builder."""

    def test_empty_matrix(self) -> None:
        m = build_csr_tensor(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            5,
        )
        assert m.shape == (5, 5)
        assert m.values().numel() == 0

    def test_duplicate_edges_summed(self) -> None:
        """Duplicate (row, col) pairs should be summed."""
        rows = np.array([0, 0, 1], dtype=np.int64)
        cols = np.array([1, 1, 2], dtype=np.int64)
        vals = np.array([3.0, 7.0, 5.0], dtype=np.float32)

        m = build_csr_tensor(rows, cols, vals, 3)
        dense = m.to_dense()
        assert dense[0, 1].item() == pytest.approx(10.0)
        assert dense[1, 2].item() == pytest.approx(5.0)

    def test_single_edge(self) -> None:
        m = build_csr_tensor(
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([42.0], dtype=np.float32),
            3,
        )
        assert m.to_dense()[0, 1].item() == pytest.approx(42.0)

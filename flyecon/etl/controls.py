"""Scientific control connectomes for topology ablation.

Provides degree-preserving rewiring and matched-density random graphs
as methodological controls (arXiv:2604.04033, arXiv:2606.17745).
Designed for subcircuit scale (~2K–10K neurons).
"""

from __future__ import annotations

import numpy as np
import structlog
import torch

from flyecon.etl.loader import Connectome, build_csr_tensor

log = structlog.get_logger()


def _coo_from_csr(matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand CSR tensor to COO arrays."""
    crow = matrix.crow_indices().numpy()
    cols = matrix.col_indices().numpy()
    vals = matrix.values().numpy()
    counts = np.diff(crow)
    rows = np.repeat(np.arange(len(crow) - 1, dtype=np.int64), counts)
    return rows, cols.astype(np.int64), vals.astype(np.float32)


def _rewire_edges(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    n_neurons: int,
    rng: np.random.Generator,
    n_swap_factor: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly swap edge endpoints while preserving per-neuron degree.

    For each swap attempt: pick two edges (A→X) and (B→Y), swap targets
    to (A→Y) and (B→X).  Reject if it creates a self-loop or duplicate.
    Both in-degree and out-degree are preserved per neuron.
    """
    n_edges = len(rows)
    if n_edges < 2:
        return rows.copy(), cols.copy(), vals.copy()

    rows = rows.copy()
    cols = cols.copy()
    vals = vals.copy()

    n_swaps = n_edges * n_swap_factor

    # Build a set of existing edges for duplicate detection
    edge_set: set[tuple[int, int]] = set()
    for k in range(n_edges):
        edge_set.add((int(rows[k]), int(cols[k])))

    # Pre-generate random pairs
    idx_i = rng.integers(0, n_edges, size=n_swaps)
    idx_j = rng.integers(0, n_edges, size=n_swaps)

    for s in range(n_swaps):
        i, j = int(idx_i[s]), int(idx_j[s])
        if i == j:
            continue

        ri, ci = int(rows[i]), int(cols[i])
        rj, cj = int(rows[j]), int(cols[j])

        # New edges after swap: (ri, cj) and (rj, ci)
        if ri == cj or rj == ci:
            continue  # would create self-loop

        new_i = (ri, cj)
        new_j = (rj, ci)

        if new_i in edge_set or new_j in edge_set:
            continue  # would create duplicate

        # Execute swap
        edge_set.discard((ri, ci))
        edge_set.discard((rj, cj))
        edge_set.add(new_i)
        edge_set.add(new_j)

        cols[i] = cj
        cols[j] = ci

    return rows, cols, vals


def rewire_degree_preserving(
    connectome: Connectome,
    seed: int,
) -> Connectome:
    """Generate a degree-and-sign-preserving rewired null connectome.

    Separately rewires positive, negative, and zero-valued edges so
    that each neuron's signed in/out degree is preserved.

    Designed for subcircuit scale (~5K neurons); at full connectome
    scale (125M edges) the Python-loop swap would be slow.
    """
    rng = np.random.default_rng(seed)
    n = connectome.n_neurons

    def _rewire_matrix(matrix: torch.Tensor) -> torch.Tensor:
        rows, cols, vals = _coo_from_csr(matrix)
        if len(rows) == 0:
            return build_csr_tensor(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                n,
            )

        # Split by sign class and rewire each independently
        all_rows, all_cols, all_vals = [], [], []
        for sign_mask in [vals > 0, vals < 0, vals == 0]:
            if not sign_mask.any():
                continue
            sr, sc, sv = _rewire_edges(
                rows[sign_mask], cols[sign_mask], vals[sign_mask],
                n, rng,
            )
            all_rows.append(sr)
            all_cols.append(sc)
            all_vals.append(sv)

        return build_csr_tensor(
            np.concatenate(all_rows) if all_rows else np.array([], dtype=np.int64),
            np.concatenate(all_cols) if all_cols else np.array([], dtype=np.int64),
            np.concatenate(all_vals) if all_vals else np.array([], dtype=np.float32),
            n,
        )

    weight_matrix = _rewire_matrix(connectome.weight_matrix)
    dopamine_matrix = _rewire_matrix(connectome.dopamine_matrix)

    log.info(
        "rewire_complete",
        n_neurons=n,
        seed=seed,
        w_nnz=int(weight_matrix.values().numel()),
        d_nnz=int(dopamine_matrix.values().numel()),
    )

    return Connectome(
        weight_matrix=weight_matrix,
        dopamine_matrix=dopamine_matrix,
        body_ids=connectome.body_ids.copy(),
        neuron_types=dict(connectome.neuron_types),
        n_neurons=n,
        n_synapses=connectome.n_synapses,
        metadata={"control": "rewired", "seed": seed},
    )


def random_sparse(
    n_neurons: int,
    density: float,
    seed: int,
) -> Connectome:
    """Generate a sparse random graph of given size and density.

    All edges are excitatory (weight = +1.0).  Dopamine matrix is empty.
    Used as a baseline control for topology ablation.
    """
    rng = np.random.default_rng(seed)
    n_possible = n_neurons * (n_neurons - 1)  # no self-loops
    n_edges = int(n_possible * density)

    if n_edges == 0:
        empty = build_csr_tensor(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            n_neurons,
        )
        body_ids = np.arange(n_neurons, dtype=np.int64)
        return Connectome(
            weight_matrix=empty,
            dopamine_matrix=empty,
            body_ids=body_ids,
            neuron_types={},
            n_neurons=n_neurons,
            n_synapses=0,
            metadata={"control": "random", "density": density, "seed": seed},
        )

    # Sample edges without replacement (no self-loops)
    flat_indices = rng.choice(n_possible, size=n_edges, replace=False)
    rows = flat_indices // (n_neurons - 1)
    cols = flat_indices % (n_neurons - 1)
    # Shift cols to avoid self-loops: if col >= row, col += 1
    cols = np.where(cols >= rows, cols + 1, cols)

    vals = np.ones(n_edges, dtype=np.float32)

    weight_matrix = build_csr_tensor(
        rows.astype(np.int64), cols.astype(np.int64), vals, n_neurons,
    )
    empty_dopa = build_csr_tensor(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float32),
        n_neurons,
    )

    body_ids = np.arange(n_neurons, dtype=np.int64)

    log.info(
        "random_sparse_created",
        n_neurons=n_neurons,
        n_edges=n_edges,
        density=density,
    )

    return Connectome(
        weight_matrix=weight_matrix,
        dopamine_matrix=empty_dopa,
        body_ids=body_ids,
        neuron_types={},
        n_neurons=n_neurons,
        n_synapses=n_edges,
        metadata={"control": "random", "density": density, "seed": seed},
    )

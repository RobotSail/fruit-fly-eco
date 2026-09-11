"""Subcircuit extraction from the full connectome.

Extracts the mushroom-body-adjacent subgraph (KC, MBON, PPL neurons
plus 1-hop neighbours) or an arbitrary ROI subgraph, and re-indexes
to produce a self-contained Connectome.
"""

from __future__ import annotations

import numpy as np
import structlog
import torch

from flyecon.etl.loader import Connectome, build_csr_tensor

log = structlog.get_logger()

# Neuron-type prefixes that define the mushroom body circuit
_MB_PREFIXES = ("KC", "MBON", "PPL")


def _csr_to_coo(matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a CSR tensor into (rows, cols, vals) COO arrays."""
    crow = matrix.crow_indices().numpy()
    cols = matrix.col_indices().numpy()
    vals = matrix.values().numpy()
    row_counts = np.diff(crow)
    rows = np.repeat(np.arange(len(crow) - 1, dtype=np.int64), row_counts)
    return rows, cols.astype(np.int64), vals.astype(np.float32)


def _extract_subgraph(
    connectome: Connectome,
    indices: np.ndarray,
) -> Connectome:
    """Build an induced subgraph on the given dense neuron indices.

    Re-indexes body IDs and both weight/dopamine matrices.
    """
    n_sub = len(indices)

    # old-index → new-index mapping (-1 = not included)
    old_to_new = np.full(connectome.n_neurons, -1, dtype=np.int64)
    old_to_new[indices] = np.arange(n_sub, dtype=np.int64)

    def _filter_matrix(matrix: torch.Tensor) -> torch.Tensor:
        rows, cols, vals = _csr_to_coo(matrix)
        if len(rows) == 0:
            return build_csr_tensor(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                n_sub,
            )
        new_rows = old_to_new[rows]
        new_cols = old_to_new[cols]
        keep = (new_rows >= 0) & (new_cols >= 0)
        return build_csr_tensor(
            new_rows[keep], new_cols[keep], vals[keep], n_sub,
        )

    weight_matrix = _filter_matrix(connectome.weight_matrix)
    dopamine_matrix = _filter_matrix(connectome.dopamine_matrix)

    new_body_ids = connectome.body_ids[indices]
    new_types: dict[int, str] = {}
    for bid in new_body_ids:
        bid_int = int(bid)
        if bid_int in connectome.neuron_types:
            new_types[bid_int] = connectome.neuron_types[bid_int]

    n_syn = (
        int(weight_matrix.values().numel())
        + int(dopamine_matrix.values().numel())
    )

    return Connectome(
        weight_matrix=weight_matrix,
        dopamine_matrix=dopamine_matrix,
        body_ids=new_body_ids,
        neuron_types=new_types,
        n_neurons=n_sub,
        n_synapses=n_syn,
        metadata={
            "parent_neurons": connectome.n_neurons,
            "extraction": "subgraph",
        },
    )


def extract_mushroom_body(
    connectome: Connectome,
    hops: int = 1,
) -> Connectome:
    """Extract the mushroom-body-adjacent subgraph.

    1. Identify Kenyon cells (KC), MBONs, and PPL dopaminergic neurons
       from neuron_types annotations (prefix match).
    2. Optionally include *hops*-hop neighbours (neurons with direct
       synaptic connections to/from the seed set).
    3. Re-index and return the induced subgraph.

    Expected size for MaleCNS v1.0: 2,000–10,000 neurons.
    """
    # ── 1. Seed set: MB neurons ──
    seed_indices: set[int] = set()
    body_id_list = connectome.body_ids.tolist()
    body_set = set(body_id_list)

    for body_id, ntype in connectome.neuron_types.items():
        if any(ntype.startswith(p) for p in _MB_PREFIXES):
            if body_id in body_set:
                idx = int(np.searchsorted(connectome.body_ids, body_id))
                if idx < connectome.n_neurons and connectome.body_ids[idx] == body_id:
                    seed_indices.add(idx)

    if not seed_indices:
        raise ValueError(
            "No mushroom body neurons found. "
            "Annotations must contain types starting with: "
            + ", ".join(_MB_PREFIXES)
        )

    log.info("mb_seed_neurons", count=len(seed_indices))

    # ── 2. Expand by hops ──
    included = set(seed_indices)

    if hops >= 1:
        rows, cols, _ = _csr_to_coo(connectome.weight_matrix)
        seed_arr = np.array(sorted(seed_indices), dtype=np.int64)

        # Outgoing neighbours: rows in seed → cols are neighbours
        out_mask = np.isin(rows, seed_arr)
        included.update(cols[out_mask].tolist())

        # Incoming neighbours: cols in seed → rows are neighbours
        in_mask = np.isin(cols, seed_arr)
        included.update(rows[in_mask].tolist())

        # Also check dopamine matrix for hop expansion
        d_rows, d_cols, _ = _csr_to_coo(connectome.dopamine_matrix)
        if len(d_rows) > 0:
            d_out = np.isin(d_rows, seed_arr)
            included.update(d_cols[d_out].tolist())
            d_in = np.isin(d_cols, seed_arr)
            included.update(d_rows[d_in].tolist())

    included_arr = np.array(sorted(included), dtype=np.int64)

    log.info(
        "mb_subgraph",
        seed=len(seed_indices),
        total=len(included_arr),
        hops=hops,
    )

    return _extract_subgraph(connectome, included_arr)


def extract_by_roi(
    connectome: Connectome,
    roi_name: str,
) -> Connectome:
    """Extract subgraph for neurons whose type contains *roi_name*.

    This is a flexible alternative to mushroom-body extraction: pass
    any ROI name (e.g. "AL" for antennal lobe, "CX" for central
    complex) and get the induced subgraph of matching neurons plus
    1-hop neighbours.
    """
    matched: set[int] = set()
    for body_id, ntype in connectome.neuron_types.items():
        if roi_name in ntype:
            idx = int(np.searchsorted(connectome.body_ids, body_id))
            if idx < connectome.n_neurons and connectome.body_ids[idx] == body_id:
                matched.add(idx)

    if not matched:
        raise ValueError(
            f"No neurons found with ROI label containing '{roi_name}'"
        )

    # 1-hop expansion (same logic as mushroom body)
    included = set(matched)
    rows, cols, _ = _csr_to_coo(connectome.weight_matrix)
    seed_arr = np.array(sorted(matched), dtype=np.int64)

    out_mask = np.isin(rows, seed_arr)
    included.update(cols[out_mask].tolist())
    in_mask = np.isin(cols, seed_arr)
    included.update(rows[in_mask].tolist())

    included_arr = np.array(sorted(included), dtype=np.int64)
    log.info("roi_subgraph", roi=roi_name, seed=len(matched), total=len(included_arr))

    return _extract_subgraph(connectome, included_arr)

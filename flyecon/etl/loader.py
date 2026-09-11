"""Connectome ETL: Feather table ingestion → signed sparse CSR tensors.

Downloads MaleCNS v1.0 Feather tables from GCS, applies the NT sign
convention (ACh +1, GABA −1, Glu −1, Histamine −1), separates dopamine
into its own modulation tensor, and builds torch.sparse_csr_tensor with
int32 indices.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch

from flyecon.state.constants import (
    AMINERGIC_ZERO_CURRENT,
    DOPAMINE_NTS,
    MIN_CONFIDENCE,
    NT_SIGN,
)

log = structlog.get_logger()

# ── GCS download URLs ───────────────────────────────────────────────────────
_BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/"
    "v1.0/connectome-data/flat-connectome"
)

_FILES: dict[str, str] = {
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
}


# ── Connectome dataclass ────────────────────────────────────────────────────


@dataclass
class Connectome:
    """Signed sparse connectome with separate dopamine modulation tensor.

    weight_matrix: fast-synaptic CSR (int32 indices, float32 values).
    dopamine_matrix: reward-modulation CSR (int32 indices, float32 values).
    body_ids: sorted array of unique body IDs (maps dense index → body ID).
    neuron_types: body_id → type label from annotations.
    """

    weight_matrix: torch.Tensor
    dopamine_matrix: torch.Tensor
    body_ids: np.ndarray
    neuron_types: dict[int, str]
    n_neurons: int
    n_synapses: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ── CSR builder ─────────────────────────────────────────────────────────────


def build_csr_tensor(
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    values: np.ndarray,
    n: int,
) -> torch.Tensor:
    """Build a sparse CSR tensor with **int32** indices from COO triplets.

    Duplicate (row, col) entries are summed.  The resulting tensor has
    shape (n, n) with crow_indices/col_indices in int32.
    """
    if len(row_indices) == 0:
        crow = np.zeros(n + 1, dtype=np.int32)
        col = np.array([], dtype=np.int32)
        val = np.array([], dtype=np.float32)
        return torch.sparse_csr_tensor(
            torch.from_numpy(crow),
            torch.from_numpy(col),
            torch.from_numpy(val),
            size=(n, n),
        )

    rows = row_indices.astype(np.int64)
    cols = col_indices.astype(np.int64)
    vals = values.astype(np.float32)

    # ── Deduplicate by summing values for identical (row, col) pairs ──
    keys = rows * n + cols
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    vals_sorted = vals[order]

    # Find boundaries between unique keys
    mask = np.empty(len(keys_sorted), dtype=np.bool_)
    mask[0] = True
    mask[1:] = keys_sorted[1:] != keys_sorted[:-1]

    unique_keys = keys_sorted[mask]
    # Sum values within each group
    group_ids = np.cumsum(mask) - 1
    unique_vals = np.zeros(len(unique_keys), dtype=np.float32)
    np.add.at(unique_vals, group_ids, vals_sorted)

    unique_rows = (unique_keys // n).astype(np.int64)
    unique_cols = (unique_keys % n).astype(np.int64)

    # ── Build CSR crow_indices ──
    crow = np.zeros(n + 1, dtype=np.int32)
    np.add.at(crow, unique_rows.astype(np.int32) + 1, 1)
    np.cumsum(crow, out=crow)

    return torch.sparse_csr_tensor(
        torch.from_numpy(crow),
        torch.from_numpy(unique_cols.astype(np.int32)),
        torch.from_numpy(unique_vals),
        size=(n, n),
    )


# ── Download ────────────────────────────────────────────────────────────────


def download_feather_tables(
    cache_dir: str = "cache/connectome/",
) -> dict[str, Path]:
    """Download (or load from cache) the 3 core Feather files from GCS.

    Returns a dict mapping logical name → local Path.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    for key, filename in _FILES.items():
        local_path = cache / filename
        if local_path.exists():
            log.info("feather_cached", file=key, path=str(local_path))
        else:
            url = f"{_BASE_URL}/{filename}"
            log.info("feather_downloading", file=key, url=url)
            urllib.request.urlretrieve(url, str(local_path))
            log.info("feather_downloaded", file=key, path=str(local_path),
                     size_mb=round(local_path.stat().st_size / 1e6, 1))
        result[key] = local_path

    return result


# ── Load connectome from on-disk Feather tables ────────────────────────────


def load_connectome(cache_dir: str = "cache/connectome/") -> Connectome:
    """Read cached Feather files and build signed sparse CSR connectome.

    Raises FileNotFoundError if files haven't been downloaded yet.
    """
    import pyarrow.feather as pf

    paths = {k: Path(cache_dir) / v for k, v in _FILES.items()}

    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Feather file: {path}. "
                "Run 'python -m flyecon etl download' first."
            )

    log.info("loading_connectome", cache_dir=cache_dir)

    weights_tbl = pf.read_table(str(paths["weights"]))
    nt_tbl = pf.read_table(str(paths["neurotransmitters"]))
    ann_tbl = pf.read_table(str(paths["annotations"]))

    return load_connectome_from_tables(
        weights_tbl.to_pandas(),
        nt_tbl.to_pandas(),
        ann_tbl.to_pandas(),
    )


# ── Core builder: DataFrames → Connectome ───────────────────────────────────


def load_connectome_from_tables(
    weights_df: Any,
    nt_df: Any,
    ann_df: Any,
) -> Connectome:
    """Build a Connectome from pandas-like DataFrames.

    This is the shared path used by both the real Feather loader and
    synthetic test fixtures.

    Sign convention (from constants.py):
      ACh → +1, GABA → −1, Glutamate → −1, Histamine → −1.
      Dopamine → separate tensor.
      Serotonin / octopamine / tyramine → zero-current.
      Unknown / low-confidence NT → zero-current (retain, don't drop).
    """
    # ── NT lookup ──
    nt_col = "predictedNt" if "predictedNt" in nt_df.columns else "consensusNt"
    nt_lookup: dict[int, str] = {}
    for i in range(len(nt_df)):
        bid = int(nt_df["bodyId"].iloc[i])
        nt_val = nt_df[nt_col].iloc[i]
        nt_lookup[bid] = str(nt_val).lower() if nt_val is not None else "unknown"

    # ── Neuron-type lookup from annotations ──
    neuron_types: dict[int, str] = {}
    if "type" in ann_df.columns:
        for i in range(len(ann_df)):
            t = ann_df["type"].iloc[i]
            if t is not None and str(t) != "nan" and str(t) != "None":
                neuron_types[int(ann_df["bodyId"].iloc[i])] = str(t)

    # ── Confidence filter ──
    if "confidence" in weights_df.columns:
        keep = weights_df["confidence"] >= MIN_CONFIDENCE
        weights_df = weights_df.loc[keep].reset_index(drop=True)

    pre_ids = weights_df["bodyId_pre"].values
    post_ids = weights_df["bodyId_post"].values
    raw_weights = weights_df["weight"].values.astype(np.float32)

    # ── Unique body IDs + dense-index mapping ──
    all_body_ids = np.union1d(pre_ids, post_ids)
    all_body_ids = np.sort(all_body_ids)
    n = len(all_body_ids)

    row_idx = np.searchsorted(all_body_ids, pre_ids).astype(np.int64)
    col_idx = np.searchsorted(all_body_ids, post_ids).astype(np.int64)

    # ── Classify each edge by its presynaptic NT ──
    dopamine_set = {d.lower() for d in DOPAMINE_NTS}
    aminergic_set = {a.lower() for a in AMINERGIC_ZERO_CURRENT}
    sign_map = {k.lower(): float(v) for k, v in NT_SIGN.items()}

    nts = np.array(
        [nt_lookup.get(int(pid), "unknown") for pid in pre_ids],
        dtype=object,
    )

    is_dopamine = np.array([nt in dopamine_set for nt in nts])
    is_aminergic = np.array([nt in aminergic_set for nt in nts])
    signs = np.array(
        [sign_map.get(str(nt), 0.0) for nt in nts], dtype=np.float32,
    )
    # Aminergic NTs explicitly → zero-current
    signs[is_aminergic] = 0.0

    # ── Fast-synaptic edges (everything except dopamine) ──
    fast_mask = ~is_dopamine
    fast_vals = (signs[fast_mask] * raw_weights[fast_mask]).astype(np.float32)

    weight_matrix = build_csr_tensor(
        row_idx[fast_mask], col_idx[fast_mask], fast_vals, n,
    )

    # ── Dopamine edges → separate tensor ──
    dopa_vals = raw_weights[is_dopamine].astype(np.float32)
    dopamine_matrix = build_csr_tensor(
        row_idx[is_dopamine], col_idx[is_dopamine], dopa_vals, n,
    )

    n_synapses = len(pre_ids)

    log.info(
        "connectome_built",
        n_neurons=n,
        n_synapses=n_synapses,
        fast_edges=int(fast_mask.sum()),
        dopamine_edges=int(is_dopamine.sum()),
    )

    return Connectome(
        weight_matrix=weight_matrix,
        dopamine_matrix=dopamine_matrix,
        body_ids=all_body_ids,
        neuron_types=neuron_types,
        n_neurons=n,
        n_synapses=n_synapses,
        metadata={
            "fast_edges": int(fast_mask.sum()),
            "dopamine_edges": int(is_dopamine.sum()),
        },
    )

"""Calibrate LIF parameters from rate-model activations.

Runs the rate-based reservoir on sampled economy states, records per-neuron
mean absolute activations (tanh output), then grid-searches LIF parameters
(gain, mbon_hold_frac) to produce proportional firing rates on the
mushroom-body subcircuit.

Usage::

    python -m flyecon.calibrate_from_rate [--cache-dir cache/connectome/]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import structlog
import torch
from scipy import sparse

from flyecon.etl.loader import Connectome, load_connectome
from flyecon.etl.subcircuit import extract_mushroom_body
from flyecon.sim.lif import LIFNetwork
from flyecon.state.constants import (
    DT_MS,
    MAX_LOSS_STREAK,
    MAX_MONEY,
    MIN_MONEY,
    MONEY_STEP,
    ROUNDS_PER_HALF,
    TAU_MS,
    V_REST_MV,
    V_THRESH_MV,
)
from flyecon.state.economy import EconomyState

log = structlog.get_logger()

GAIN_VALUES: list[float] = [0.5, 1.0, 2.0, 5.0]
MBON_HOLD_VALUES: list[float] = [0.3, 0.5, 0.7, 0.85]
_TYPE_PREFIXES: dict[str, str] = {"KC": "KC", "MBON": "MBON", "PPL": "PPL"}


@dataclass
class GroupStats:
    """Per-group rate-model activation statistics."""

    group: str
    count: int
    mean_abs_activation: float
    std_abs_activation: float


@dataclass
class GridSearchEntry:
    """One row in the grid search result table."""

    gain: float
    mbon_hold_frac: float
    correlation: float
    mean_lif_rate_hz: float


@dataclass
class CalibrationFromRateResult:
    """Full output of ``calibrate_from_rate``."""

    group_stats: list[GroupStats]
    grid_results: list[GridSearchEntry]
    best_gain: float
    best_mbon_hold_frac: float
    best_correlation: float
    n_economy_states: int
    n_subcircuit_neurons: int


# ── Helpers ────────────────────────────────────────────────────────────────


def _sample_economy_states(n: int = 100, seed: int = 42) -> list[EconomyState]:
    """Generate *n* diverse economy states by uniform random sampling."""
    rng = np.random.default_rng(seed)
    states: list[EconomyState] = []
    for _ in range(n):
        money = int(rng.integers(MIN_MONEY, MAX_MONEY + 1))
        money = max(MIN_MONEY, min((money // MONEY_STEP) * MONEY_STEP, MAX_MONEY))
        states.append(EconomyState(
            money=money,
            loss_streak=int(rng.integers(0, MAX_LOSS_STREAK + 1)),
            round_number=int(rng.integers(1, ROUNDS_PER_HALF + 1)),
            half=int(rng.integers(0, 2)),
            opponent_loss_streak=int(rng.integers(0, MAX_LOSS_STREAK + 1)),
        ))
    return states


def _collect_rate_activations(
    connectome: Connectome,
    states: list[EconomyState],
    device: str = "cpu",
) -> torch.Tensor:
    """Run the rate-model reservoir on each state; return mean |activation|."""
    from flyecon.reservoir import FlyReservoir

    W_torch = connectome.weight_matrix
    crow = W_torch.crow_indices().numpy()
    col = W_torch.col_indices().numpy()
    vals = W_torch.values().numpy().copy()
    n = W_torch.shape[0]
    W_scipy = sparse.csr_matrix((vals, col, crow), shape=(n, n))

    # Fan-in normalization (mirrors reservoir.py _build_weight_matrix)
    abs_row_sums = np.maximum(
        np.array(np.abs(W_scipy).sum(axis=1)).flatten(), 1.0,
    )
    W_norm = (
        sparse.diags(1.0 / abs_row_sums, format="csr") @ W_scipy
    ).astype(np.float32)

    reservoir = FlyReservoir(n, W_norm, device=device)
    abs_sum = torch.zeros(n, dtype=torch.float32, device=device)
    for state in states:
        reservoir.reset()
        for _ in range(3):  # 3 recurrence steps (same as reservoir.forward)
            reservoir.step(state)
        abs_sum += reservoir.state.abs()
    return (abs_sum / len(states)).cpu()


def _classify_neurons(connectome: Connectome) -> dict[str, list[int]]:
    """Group dense indices by neuron type prefix (KC/MBON/PPL/Other)."""
    groups: dict[str, list[int]] = {g: [] for g in _TYPE_PREFIXES}
    groups["Other"] = []
    for idx in range(connectome.n_neurons):
        ntype = connectome.neuron_types.get(int(connectome.body_ids[idx]), "")
        matched = False
        for gname, prefix in _TYPE_PREFIXES.items():
            if ntype.startswith(prefix):
                groups[gname].append(idx)
                matched = True
                break
        if not matched:
            groups["Other"].append(idx)
    return groups


def _compute_group_stats(
    activations: torch.Tensor,
    groups: dict[str, list[int]],
) -> list[GroupStats]:
    """Compute per-group activation mean and std."""
    stats: list[GroupStats] = []
    for name, indices in groups.items():
        if not indices:
            stats.append(GroupStats(name, 0, 0.0, 0.0))
            continue
        g = activations[torch.tensor(indices, dtype=torch.long)]
        stats.append(GroupStats(
            name,
            len(indices),
            float(g.mean().item()),
            float(g.std().item()) if len(indices) > 1 else 0.0,
        ))
    return stats


def _build_lif_with_hold(
    connectome: Connectome,
    gain: float,
    mbon_hold_frac: float,
    device: str = "cpu",
) -> LIFNetwork:
    """Build a LIFNetwork with a custom *mbon_hold_frac*."""
    net = LIFNetwork(connectome, gain=gain, device=device)
    decay = math.exp(-DT_MS / TAU_MS)
    net._tonic_current.zero_()
    if len(net._mbon_indices) > 0:
        tonic_mv = mbon_hold_frac * (V_THRESH_MV - V_REST_MV) / (1.0 - decay)
        net._tonic_current[net._mbon_idx_t] = tonic_mv
    return net


def _encode_economy_state(state: EconomyState) -> np.ndarray:
    """Encode economy state to 6-d vector (same as reservoir._encode_state)."""
    return np.array([
        state.money / 16000.0,
        state.round_number / 12.0,
        state.half,
        state.loss_streak / 4.0,
        state.opponent_loss_streak / 4.0,
        1.0,
    ], dtype=np.float32)


def _run_lif_on_states(
    connectome: Connectome,
    states: list[EconomyState],
    gain: float,
    mbon_hold_frac: float,
    duration_ms: float = 500.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Run LIF per economy state; return mean firing rate (Hz) per neuron."""
    n = connectome.n_neurons
    rng = np.random.default_rng(42)
    input_bins = rng.integers(0, 6, n)
    input_sign = rng.choice([-1.0, 1.0], n).astype(np.float32)
    rate_sum = torch.zeros(n, dtype=torch.float32)
    duration_s = duration_ms / 1000.0

    for state in states:
        encoded = _encode_economy_state(state)
        # Same random projection as reservoir; scale to LIF mV range
        inp = torch.tensor(
            encoded[input_bins] * input_sign * 10.0,
            dtype=torch.float32,
        )
        net = _build_lif_with_hold(connectome, gain, mbon_hold_frac, device)
        counts = net.simulate(inp.to(device), duration_ms, jitter=True)
        rate_sum += counts.cpu() / duration_s
    return rate_sum / len(states)


def _pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson r between two 1-D tensors. Returns 0.0 on zero variance."""
    xc = (x - x.mean()).float()
    yc = (y - y.mean()).float()
    denom = torch.sqrt((xc**2).sum() * (yc**2).sum())
    if denom.item() < 1e-12:
        return 0.0
    return float(((xc * yc).sum() / denom).item())


def _map_full_to_sub(full: Connectome, sub: Connectome) -> np.ndarray:
    """Return array[sub.n_neurons] of full dense indices (-1 if missing)."""
    idx = np.searchsorted(full.body_ids, sub.body_ids)
    valid = (idx < len(full.body_ids)) & (full.body_ids[idx] == sub.body_ids)
    return np.where(valid, idx, -1).astype(np.int64)


# ── Main calibration ──────────────────────────────────────────────────────


def calibrate_from_rate(
    cache_dir: str = "cache/connectome/",
    n_states: int = 100,
    duration_ms: float = 500.0,
    gain_values: list[float] | None = None,
    mbon_hold_values: list[float] | None = None,
    max_subcircuit_neurons: int = 5000,
    device: str = "cpu",
) -> CalibrationFromRateResult:
    """Run full calibration: rate-model activations -> LIF parameter search.

    1. Load full connectome, run rate model on *n_states* economy states.
    2. Extract mushroom-body subcircuit, group neurons by type.
    3. Grid search (gain x mbon_hold_frac), measuring Pearson r between
       rate-model |activation| and LIF firing rate per neuron.
    4. Return best parameters and the correlation achieved.
    """
    gains = gain_values or GAIN_VALUES
    holds = mbon_hold_values or MBON_HOLD_VALUES

    # 1. Load connectome
    log.info("calibrate.loading_connectome", cache_dir=cache_dir)
    full_conn = load_connectome(cache_dir)
    log.info("calibrate.connectome_loaded", n=full_conn.n_neurons)

    # 2. Sample states & run rate model
    states = _sample_economy_states(n_states)
    log.info("calibrate.running_rate_model", n_states=len(states))
    full_act = _collect_rate_activations(full_conn, states, device)
    log.info(
        "calibrate.rate_model_done",
        mean=float(full_act.mean()),
        max=float(full_act.max()),
    )

    # 3. Group stats on full connectome
    full_groups = _classify_neurons(full_conn)
    group_stats = _compute_group_stats(full_act, full_groups)
    for gs in group_stats:
        log.info(
            "calibrate.group",
            group=gs.group,
            n=gs.count,
            mean=round(gs.mean_abs_activation, 6),
        )

    # 4. Extract mushroom-body subcircuit
    sub_conn = extract_mushroom_body(
        full_conn, hops=1, max_neurons=max_subcircuit_neurons,
    )
    log.info("calibrate.subcircuit", n=sub_conn.n_neurons)

    sub_to_full = _map_full_to_sub(full_conn, sub_conn)
    valid_mask = sub_to_full >= 0
    sub_act = full_act[sub_to_full[valid_mask]]

    # 5. Grid search
    grid_results: list[GridSearchEntry] = []
    best_corr, best_gain, best_hold = -2.0, gains[0], holds[0]

    for gain in gains:
        for hold in holds:
            log.info("calibrate.grid", gain=gain, hold=hold)
            lif_rates = _run_lif_on_states(
                sub_conn, states, gain, hold, duration_ms, device,
            )
            lif_sub = lif_rates[torch.tensor(valid_mask)]
            corr = _pearson_correlation(sub_act, lif_sub)
            entry = GridSearchEntry(gain, hold, corr, float(lif_rates.mean()))
            grid_results.append(entry)
            log.info("calibrate.result", gain=gain, hold=hold, corr=round(corr, 4))
            if corr > best_corr:
                best_corr, best_gain, best_hold = corr, gain, hold

    log.info(
        "calibrate.best", gain=best_gain, hold=best_hold, corr=round(best_corr, 4),
    )
    return CalibrationFromRateResult(
        group_stats=group_stats,
        grid_results=grid_results,
        best_gain=best_gain,
        best_mbon_hold_frac=best_hold,
        best_correlation=best_corr,
        n_economy_states=n_states,
        n_subcircuit_neurons=sub_conn.n_neurons,
    )


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run calibration from command line."""
    import argparse

    p = argparse.ArgumentParser(
        description="Calibrate LIF from rate-model activations",
    )
    p.add_argument("--cache-dir", default="cache/connectome/")
    p.add_argument("--n-states", type=int, default=100)
    p.add_argument("--duration-ms", type=float, default=500.0)
    args = p.parse_args()

    result = calibrate_from_rate(args.cache_dir, args.n_states, args.duration_ms)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"\nEconomy states: {result.n_economy_states}")
    print(f"Subcircuit neurons: {result.n_subcircuit_neurons}")
    print("\n--- Rate-Model Activation by Neuron Group ---")
    for gs in result.group_stats:
        print(
            f"  {gs.group:6s}: n={gs.count:6d}"
            f"  mean|act|={gs.mean_abs_activation:.6f}"
        )
    print("\n--- Grid Search ---")
    print(f"{'gain':>6s}  {'hold':>6s}  {'corr':>8s}  {'rate_hz':>10s}")
    for e in result.grid_results:
        print(
            f"{e.gain:6.2f}  {e.mbon_hold_frac:6.2f}"
            f"  {e.correlation:8.4f}  {e.mean_lif_rate_hz:10.2f}"
        )
    print(
        f"\nBest: gain={result.best_gain}  hold={result.best_mbon_hold_frac}"
        f"  corr={result.best_correlation:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()

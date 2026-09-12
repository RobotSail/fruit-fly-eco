"""Calibrate LIF spiking parameters from rate-model activations.

Grid-searches gain and MBON tonic hold to match LIF firing rates
to rate-model activation magnitudes (Pearson r per neuron type).

Usage: python -m flyecon.calibrate_from_rate
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog
import torch

from flyecon.etl.loader import Connectome, load_connectome
from flyecon.etl.subcircuit import extract_mushroom_body
from flyecon.policy.imitation import df_to_states, load_pro_rounds
from flyecon.reservoir import build_reservoir
from flyecon.sim.lif import C_M, LIFNetwork, _ONE_MINUS_DECAY
from flyecon.state.constants import TAU_MS, V_REST_MV, V_THRESH_MV

log = structlog.get_logger()

N_CALIBRATION_STATES: int = 200
DURATION_MS: float = 400.0
GAIN_GRID: list[float] = [0.5, 1.0, 2.0, 5.0, 10.0]
MBON_HOLD_GRID: list[float] = [0.0, 0.3, 0.5, 0.7, 0.85]
_NEURON_GROUPS = ("KC", "MBON", "PPL")

@dataclass
class CalibrationFromRateResult:
    """Output of the rate-to-LIF calibration sweep."""
    best_gain: float
    best_mbon_hold: float
    best_correlation: float
    per_type_correlation: dict[str, float]
    grid_results: list[dict[str, float]] = field(default_factory=list)


def _classify_neuron(ntype: str) -> str:
    """Map a neuron-type label to KC / MBON / PPL / Other."""
    for prefix in _NEURON_GROUPS:
        if ntype.startswith(prefix):
            return prefix
    return "Other"


def _build_type_masks(connectome: Connectome) -> dict[str, np.ndarray]:
    """Return boolean masks for each neuron-type group."""
    n = connectome.n_neurons
    masks: dict[str, list[bool]] = {g: [False] * n for g in (*_NEURON_GROUPS, "Other")}
    for idx in range(n):
        body_id = int(connectome.body_ids[idx])
        ntype = connectome.neuron_types.get(body_id, "")
        masks[_classify_neuron(ntype)][idx] = True
    return {k: np.array(v) for k, v in masks.items()}


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between *x* and *y*.  Returns 0.0 if degenerate."""
    if len(x) < 2:
        return 0.0
    xd, yd = x - x.mean(), y - y.mean()
    denom = np.sqrt((xd ** 2).sum() * (yd ** 2).sum())
    return float((xd * yd).sum() / denom) if denom > 1e-12 else 0.0


def collect_rate_activations(
    states: list, cache_dir: str = "cache/connectome/",
) -> tuple[np.ndarray, Connectome]:
    """Run rate model on *states*, return mean |activation| per neuron."""
    reservoir = build_reservoir(cache_dir=cache_dir, device="cpu")
    conn_full = load_connectome(cache_dir)
    n = reservoir.n
    all_acts = np.zeros(n, dtype=np.float64)
    for s in states:
        reservoir.reset()
        for _ in range(3):  # 3 recurrence steps (same as FlyReservoir.forward)
            reservoir.step(s)
        all_acts += np.abs(reservoir.state.detach().cpu().numpy())
    mean_abs_act = (all_acts / len(states)).astype(np.float32)
    log.info("rate_activations_collected", n_states=len(states), n_neurons=n,
             mean_act=round(float(np.mean(mean_abs_act)), 6))
    return mean_abs_act, conn_full


def _map_activations_to_subcircuit(
    full_activations: np.ndarray, full_conn: Connectome, mb_conn: Connectome,
) -> np.ndarray:
    """Map full-connectome activations to MB subcircuit by body_id."""
    full_bid_to_idx = {int(bid): i for i, bid in enumerate(full_conn.body_ids)}
    mb_acts = np.zeros(mb_conn.n_neurons, dtype=np.float32)
    for j in range(mb_conn.n_neurons):
        bid = int(mb_conn.body_ids[j])
        if bid in full_bid_to_idx:
            mb_acts[j] = full_activations[full_bid_to_idx[bid]]
    return mb_acts


def _simulate_lif_rates(
    mb_conn: Connectome, states: list, gain: float,
    mbon_hold_frac: float, duration_ms: float = DURATION_MS,
) -> np.ndarray:
    """Simulate LIF on *states*, return mean firing rate (Hz) per neuron."""
    from flyecon.encoding.population import (
        N_INPUT_NEURONS, PopulationEncoder, map_to_connectome_inputs,
    )
    n = mb_conn.n_neurons
    lif = LIFNetwork(mb_conn, gain=gain, device="cpu")

    # Override tonic current: mbon_hold_frac * (V_THRESH-V_REST) / (TAU*(1-decay)/C_M)
    lif._tonic_current.zero_()
    if len(lif._mbon_indices) > 0 and mbon_hold_frac > 0:
        tonic_mv = mbon_hold_frac * (V_THRESH_MV - V_REST_MV) / (
            TAU_MS * _ONE_MINUS_DECAY / C_M
        )
        lif._tonic_current[lif._mbon_idx_t] = tonic_mv

    encoder = PopulationEncoder()
    # Use KC neurons as input layer (same logic as build_fly_policy)
    kc_ids = [
        idx for idx in range(n)
        if mb_conn.neuron_types.get(int(mb_conn.body_ids[idx]), "").startswith("KC")
    ]
    if len(kc_ids) >= N_INPUT_NEURONS:
        import random as _rng
        _rng.seed(42)
        _rng.shuffle(kc_ids)
        input_ids = kc_ids
    else:
        input_ids = list(range(min(N_INPUT_NEURONS, n)))

    total_counts = np.zeros(n, dtype=np.float64)
    duration_s = duration_ms / 1000.0
    for s in states:
        encoded = encoder.encode(s).unsqueeze(0)
        n_in = len(input_ids)
        if n_in < encoded.shape[-1]:
            encoded = encoded[:, :n_in]
        full_input = map_to_connectome_inputs(encoded, input_ids, n).squeeze(0)
        counts = lif.simulate(full_input, duration_ms, jitter=True)
        total_counts += counts.numpy()
    return (total_counts / (len(states) * duration_s)).astype(np.float32)


def calibrate_from_rate(
    cache_dir: str = "cache/connectome/",
    train_path: str = "data/cs2_pro_train.parquet",
    n_states: int = N_CALIBRATION_STATES,
) -> CalibrationFromRateResult:
    """Run the full rate-to-LIF calibration pipeline and return best params."""
    # 1. Load 200 economy states from training data
    log.info("calibration.loading_states", path=train_path, n=n_states)
    df = load_pro_rounds(train_path)
    df_sample = df.sample(n=min(n_states, len(df)), random_state=42)
    states = df_to_states(df_sample)

    # 2. Collect rate-model activations
    full_activations, full_conn = collect_rate_activations(states, cache_dir)

    # 3. Extract mushroom body and map activations
    mb_conn = extract_mushroom_body(full_conn, max_neurons=5000)
    type_masks = _build_type_masks(mb_conn)
    mb_activations = _map_activations_to_subcircuit(full_activations, full_conn, mb_conn)
    log.info("calibration.mb_activations", n_mb=mb_conn.n_neurons,
             kc_mean=round(float(np.mean(mb_activations[type_masks["KC"]])), 4),
             mbon_mean=round(float(np.mean(mb_activations[type_masks["MBON"]])), 4),
             ppl_mean=round(float(np.mean(mb_activations[type_masks["PPL"]])), 4))

    # 4. Grid search over gain x mbon_hold_frac
    log.info("calibration.grid_search", gains=GAIN_GRID, holds=MBON_HOLD_GRID)
    best_corr, best_gain, best_hold = -1.0, GAIN_GRID[0], MBON_HOLD_GRID[0]
    best_per_type: dict[str, float] = {}
    grid_results: list[dict[str, float]] = []

    for gain in GAIN_GRID:
        for mbon_hold in MBON_HOLD_GRID:
            log.info("calibration.trial", gain=gain, mbon_hold=mbon_hold)
            lif_rates = _simulate_lif_rates(mb_conn, states, gain, mbon_hold)

            per_type_corr: dict[str, float] = {}
            for group in _NEURON_GROUPS:
                mask = type_masks[group]
                if mask.sum() < 2:
                    per_type_corr[group] = 0.0
                    continue
                per_type_corr[group] = _pearson_correlation(
                    mb_activations[mask], lif_rates[mask])

            overall_corr = _pearson_correlation(mb_activations, lif_rates)
            typed_mean = float(np.mean([per_type_corr[g] for g in _NEURON_GROUPS]))

            grid_results.append({
                "gain": gain, "mbon_hold": mbon_hold,
                "overall_corr": overall_corr, "typed_mean_corr": typed_mean,
                "kc_corr": per_type_corr["KC"], "mbon_corr": per_type_corr["MBON"],
                "ppl_corr": per_type_corr["PPL"],
                "mean_rate_hz": float(np.mean(lif_rates)),
                "kc_rate_hz": float(np.mean(lif_rates[type_masks["KC"]])),
                "mbon_rate_hz": float(np.mean(lif_rates[type_masks["MBON"]])),
                "ppl_rate_hz": float(np.mean(lif_rates[type_masks["PPL"]])),
            })
            log.info("calibration.trial_result", gain=gain, mbon_hold=mbon_hold,
                     overall=round(overall_corr, 4), typed=round(typed_mean, 4),
                     rate_hz=round(float(np.mean(lif_rates)), 2))

            if typed_mean > best_corr:
                best_corr = typed_mean
                best_gain, best_hold = gain, mbon_hold
                best_per_type = dict(per_type_corr)

    log.info("calibration.best", gain=best_gain, hold=best_hold,
             corr=round(best_corr, 4), per_type=best_per_type)
    return CalibrationFromRateResult(
        best_gain=best_gain, best_mbon_hold=best_hold,
        best_correlation=best_corr, per_type_correlation=best_per_type,
        grid_results=grid_results)


def train_readout_at_best(
    result: CalibrationFromRateResult,
    cache_dir: str = "cache/connectome/",
    train_path: str = "data/cs2_pro_train.parquet",
    eval_path: str = "data/cs2_pro_eval.parquet",
    n_epochs: int = 10,
) -> dict[str, float]:
    """Train imitation-learning readout at best params and return eval accuracy."""
    from flyecon.policy.imitation import ImitationTrainer
    from flyecon.policy.ppo import build_fly_policy

    log.info("readout.building_policy", gain=result.best_gain, hold=result.best_mbon_hold)
    full_conn = load_connectome(cache_dir)
    mb_conn = extract_mushroom_body(full_conn, max_neurons=5000)

    policy = build_fly_policy(mb_conn, gain=result.best_gain, duration_ms=DURATION_MS)
    # Override MBON tonic current at best mbon_hold
    lif = policy._lif
    lif._tonic_current.zero_()
    if len(lif._mbon_indices) > 0 and result.best_mbon_hold > 0:
        tonic_mv = result.best_mbon_hold * (V_THRESH_MV - V_REST_MV) / (
            TAU_MS * _ONE_MINUS_DECAY / C_M)
        lif._tonic_current[lif._mbon_idx_t] = tonic_mv

    trainer = ImitationTrainer(policy=policy, train_path=train_path,
                               eval_path=eval_path, lr=1e-3, batch_size=64)
    train_metrics: dict[str, float] = {}
    for epoch in range(n_epochs):
        train_metrics = trainer.train_epoch()
        log.info("readout.epoch", epoch=epoch + 1,
                 loss=round(train_metrics["loss"], 4),
                 acc=round(train_metrics["accuracy"], 4))

    eval_metrics = trainer.evaluate()
    log.info("readout.eval", acc=round(eval_metrics["accuracy"], 4),
             loss=round(eval_metrics["loss"], 4))
    return {
        "eval_accuracy": eval_metrics["accuracy"],
        "eval_loss": eval_metrics["loss"],
        "train_accuracy": train_metrics.get("accuracy", 0.0),
    }


def main() -> None:
    """Run the full calibration pipeline and print results."""
    print("=" * 70)
    print("Rate -> LIF Calibration")
    print("=" * 70)

    result = calibrate_from_rate()

    print()
    print("Best gain:        " + str(result.best_gain))
    print("Best MBON hold:   " + str(result.best_mbon_hold))
    print("Best correlation: " + f"{result.best_correlation:.4f}")
    print("Per-type correlation:")
    for k, v in result.per_type_correlation.items():
        print(f"  {k:>6s}: {v:.4f}")

    hdr = "  gain    hold  overall    typed       KC     MBON      PPL  rate_hz"
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in result.grid_results:
        print(f"  {r['gain']:4.1f}    {r['mbon_hold']:4.2f}  "
              f"{r['overall_corr']:7.4f}  {r['typed_mean_corr']:7.4f}  "
              f"{r['kc_corr']:7.4f}  {r['mbon_corr']:7.4f}  "
              f"{r['ppl_corr']:7.4f}  {r['mean_rate_hz']:7.2f}")

    print()
    print("=" * 70)
    print("READOUT TRAINING")
    print("=" * 70)
    readout = train_readout_at_best(result)
    print("Eval accuracy:  " + f"{readout['eval_accuracy']:.4f}")
    print("Eval loss:      " + f"{readout['eval_loss']:.4f}")
    print("Train accuracy: " + f"{readout['train_accuracy']:.4f}")


if __name__ == "__main__":
    main()

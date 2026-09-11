"""Gain calibration and pre-flight gate for LIF simulation.

The calibration sweep finds a gain where the network produces a healthy
firing rate under moderate input.  The pre-flight gate validates three
necessary conditions before encoding/readout work begins:

  Gate 1: zero-input stability (no spontaneous spiking)
  Gate 2: non-degenerate response (1–10 Hz under moderate input)
  Gate 3: state discrimination (different inputs → different outputs,
           Cohen's d > 0.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import structlog
import torch

from flyecon.etl.loader import Connectome
from flyecon.sim.lif import LIFNetwork
from flyecon.state.constants import DT_MS

log = structlog.get_logger()


@dataclass
class CalibrationResult:
    """Output of ``calibrate_gain``."""

    gain: float
    mean_rate_hz: float
    per_neuron_rates: torch.Tensor
    is_healthy: bool
    trials: list[dict[str, float]] = field(default_factory=list)


@dataclass
class PreflightResult:
    """Output of ``preflight_gate``."""

    gate1_pass: bool
    gate2_pass: bool
    gate3_pass: bool
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def all_pass(self) -> bool:
        return self.gate1_pass and self.gate2_pass and self.gate3_pass


def _mean_rate_hz(
    connectome: Connectome,
    gain: float,
    input_current: torch.Tensor,
    duration_ms: float,
    jitter: bool = True,
) -> tuple[float, torch.Tensor]:
    """Run a single simulation and return (mean_rate_hz, per_neuron_rates).

    Uses jittered initial membrane potentials by default to break
    artificial synchrony (all neurons at identical V_REST receiving
    identical input would spike in lockstep, preventing recurrent
    dynamics from ever affecting firing rates).
    """
    net = LIFNetwork(connectome, gain=gain)
    counts = net.simulate(input_current, duration_ms, jitter=jitter)
    duration_s = duration_ms / 1000.0
    per_neuron = counts / duration_s
    return float(per_neuron.mean().item()), per_neuron


def calibrate_gain(
    connectome: Connectome,
    target_rate_hz: tuple[float, float] = (1.0, 10.0),
    duration_ms: float = 1000.0,
    n_trials: int = 5,
    input_amplitude: float = 10.0,
) -> CalibrationResult:
    """Binary search for a gain that produces healthy firing rates.

    Uses a suprathreshold uniform input current (``input_amplitude`` mV-eq,
    must exceed ``V_THRESH - V_REST = 7 mV``) and searches for a gain
    where mean firing rate falls in ``target_rate_hz``.

    Handles both excitation-dominated networks (rate increases with gain)
    and inhibition-dominated networks (rate decreases with gain at high
    gain due to recurrent inhibitory feedback).

    Parameters
    ----------
    connectome : Connectome
        Signed sparse connectome.
    target_rate_hz : tuple[float, float]
        Target mean firing rate range (inclusive).
    duration_ms : float
        Simulation duration per trial.
    n_trials : int
        Max binary-search refinement steps after bracketing.
    input_amplitude : float
        Amplitude of uniform input current for calibration.

    Returns
    -------
    CalibrationResult
        Best gain, mean rate, per-neuron rates, health flag, trial log.
    """
    n = connectome.n_neurons
    inp = torch.full((n,), input_amplitude, dtype=torch.float32)
    lo, hi = target_rate_hz

    trials: list[dict[str, float]] = []

    # Phase 1: scan gain decades to map the rate landscape.
    # Extended range covers both small-weight test networks (need gain ~1-10)
    # and real connectomes (need gain ~1e-4 to 1e-2).
    scan_exponents = list(range(-6, 4))  # 1e-6 to 1e3
    scan_gains: list[float] = []
    scan_rates: list[float] = []
    scan_per_neuron: list[torch.Tensor] = []

    for exp in scan_exponents:
        g = 10.0 ** exp
        rate, per_neuron = _mean_rate_hz(connectome, g, inp, duration_ms)
        trials.append({"gain": g, "rate_hz": rate})
        scan_gains.append(g)
        scan_rates.append(rate)
        scan_per_neuron.append(per_neuron)
        log.info("calibration_scan", gain=g, rate_hz=rate)

        # Early return on direct hit
        if lo <= rate <= hi:
            return CalibrationResult(
                gain=g,
                mean_rate_hz=rate,
                per_neuron_rates=per_neuron,
                is_healthy=True,
                trials=trials,
            )

    # Phase 2: find the bracket where the target range is reachable.
    # Look for adjacent scan points where the rate crosses the target range.
    bracket: tuple[int, int] | None = None
    for i in range(len(scan_gains) - 1):
        r1, r2 = scan_rates[i], scan_rates[i + 1]
        rmin, rmax = min(r1, r2), max(r1, r2)
        if rmin <= hi and rmax >= lo:
            bracket = (i, i + 1)
            break

    if bracket is None:
        # No bracket found — return the closest result to target midpoint
        target_mid = (lo + hi) / 2.0
        closest_idx = min(
            range(len(scan_rates)),
            key=lambda j: abs(scan_rates[j] - target_mid),
        )
        return CalibrationResult(
            gain=scan_gains[closest_idx],
            mean_rate_hz=scan_rates[closest_idx],
            per_neuron_rates=scan_per_neuron[closest_idx],
            is_healthy=lo <= scan_rates[closest_idx] <= hi,
            trials=trials,
        )

    # Phase 3: binary search within the bracket
    idx_lo, idx_hi = bracket
    g_lo = scan_gains[idx_lo]
    g_hi = scan_gains[idx_hi]

    # Determine if rate increases or decreases with gain in this bracket
    rate_increases = scan_rates[idx_hi] >= scan_rates[idx_lo]

    best_gain = g_lo
    best_rate = scan_rates[idx_lo]
    best_per_neuron = scan_per_neuron[idx_lo]
    target_mid = (lo + hi) / 2.0

    for _ in range(n_trials + 10):
        mid = math.sqrt(g_lo * g_hi)  # geometric midpoint
        rate, per_neuron = _mean_rate_hz(connectome, mid, inp, duration_ms)
        trials.append({"gain": mid, "rate_hz": rate})
        log.info("calibration_search", gain=mid, rate_hz=rate)

        if lo <= rate <= hi:
            return CalibrationResult(
                gain=mid,
                mean_rate_hz=rate,
                per_neuron_rates=per_neuron,
                is_healthy=True,
                trials=trials,
            )

        # Update bracket based on rate direction
        if rate_increases:
            # Higher gain → higher rate: search lower if too high
            if rate < lo:
                g_lo = mid
            else:
                g_hi = mid
        else:
            # Higher gain → lower rate: search higher if too high
            if rate > hi:
                g_lo = mid
            else:
                g_hi = mid

        # Track closest to target
        if abs(rate - target_mid) < abs(best_rate - target_mid):
            best_gain = mid
            best_rate = rate
            best_per_neuron = per_neuron

    # Return closest result
    return CalibrationResult(
        gain=best_gain,
        mean_rate_hz=best_rate,
        per_neuron_rates=best_per_neuron,
        is_healthy=lo <= best_rate <= hi,
        trials=trials,
    )


def preflight_gate(
    connectome: Connectome,
    gain: float,
    duration_ms: float = 500.0,
    input_amplitude: float = 10.0,
) -> PreflightResult:
    """Run the three pre-flight gates.

    Gate 1 — zero-input stability:
        No spontaneous spikes (rate < 0.1 Hz) under zero input.

    Gate 2 — non-degenerate response:
        Mean rate in [1, 10] Hz under moderate uniform input.

    Gate 3 — state discrimination:
        Two different input amplitudes produce measurably different
        spike counts (Cohen's d > 0.5).

    Parameters
    ----------
    connectome : Connectome
        Signed sparse connectome.
    gain : float
        Synaptic gain to test.
    duration_ms : float
        Simulation duration per gate test.
    input_amplitude : float
        Base amplitude for gate 2/3 input current (must be suprathreshold,
        i.e. > V_THRESH - V_REST = 7 mV).

    Returns
    -------
    PreflightResult
        Pass/fail for each gate plus diagnostics.
    """
    n = connectome.n_neurons
    diagnostics: dict[str, object] = {}

    # ── Gate 1: zero-input stability ──
    # No jitter: test from exact V_REST — any spontaneous spike is a fail.
    zero_input = torch.zeros(n, dtype=torch.float32)
    rate_zero, _ = _mean_rate_hz(connectome, gain, zero_input, duration_ms, jitter=False)
    gate1 = rate_zero < 0.1
    diagnostics["gate1_zero_rate_hz"] = rate_zero
    log.info("preflight_gate1", rate_hz=rate_zero, passed=gate1)

    # ── Gate 2: non-degenerate response ──
    # Jitter initial V to break synchrony (same as calibration).
    mod_input = torch.full((n,), input_amplitude, dtype=torch.float32)
    rate_mod, _ = _mean_rate_hz(connectome, gain, mod_input, duration_ms, jitter=True)
    gate2 = 1.0 <= rate_mod <= 10.0
    diagnostics["gate2_mod_rate_hz"] = rate_mod
    log.info("preflight_gate2", rate_hz=rate_mod, passed=gate2)

    # ── Gate 3: state discrimination ──
    # Jitter to enable recurrent dynamics (same seed via torch global state).
    low_input = torch.full((n,), input_amplitude * 0.5, dtype=torch.float32)
    high_input = torch.full((n,), input_amplitude * 1.5, dtype=torch.float32)

    net_low = LIFNetwork(connectome, gain=gain)
    counts_low = net_low.simulate(low_input, duration_ms, jitter=True)

    net_high = LIFNetwork(connectome, gain=gain)
    counts_high = net_high.simulate(high_input, duration_ms, jitter=True)

    # Cohen's d: (mean_high - mean_low) / pooled_std
    diff = counts_high - counts_low
    mean_diff = float(diff.mean().item())
    pooled_std = float(
        torch.sqrt(
            (counts_low.var() + counts_high.var()) / 2.0 + 1e-8
        ).item()
    )
    cohens_d = abs(mean_diff) / pooled_std if pooled_std > 0 else 0.0
    gate3 = cohens_d > 0.5
    diagnostics["gate3_cohens_d"] = cohens_d
    diagnostics["gate3_mean_diff"] = mean_diff
    log.info("preflight_gate3", cohens_d=cohens_d, passed=gate3)

    return PreflightResult(
        gate1_pass=gate1,
        gate2_pass=gate2,
        gate3_pass=gate3,
        diagnostics=diagnostics,
    )

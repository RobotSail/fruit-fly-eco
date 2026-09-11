"""Learned linear spike readout: spike counts → action logits.

Reads output-neuron spike counts from the full spike-count vector,
converts to firing rates, baseline-normalises to isolate state-driven
modulation from structural firing-rate bias, then projects through a
trainable linear layer to produce action logits.

Trainable parameters: W [n_actions, n_output_neurons], b [n_actions].
This is the small, fast-to-train layer that makes the fixed-reservoir
approach work.

Default readout bin length: 400 ms (upper end of 100–500 ms range —
longer bins reduce spike-count noise during early training).
"""

from __future__ import annotations

from typing import List

import structlog
import torch
import torch.nn as nn

log = structlog.get_logger()

# ── Constants ──────────────────────────────────────────────────────────────

N_ACTIONS: int = 5
"""Number of buy-plan actions: FULL_BUY, FORCE_BUY, HALF_BUY, ECO, SAVE."""

DEFAULT_BIN_MS: float = 400.0
"""Default readout bin length (ms)."""

_EPS: float = 1e-6
"""Epsilon for numerical stability in baseline normalisation."""


class SpikeReadout(nn.Module):
    """Learned linear readout from output neuron spike counts to action logits.

    Parameters
    ----------
    output_neuron_ids : list[int]
        Indices of MBON/DN neurons in the connectome to read from.
    baseline_rates : Tensor
        Per-neuron mean firing rate (Hz) under neutral input, from
        gain calibration.  Used to normalise spike rates.
    n_actions : int
        Number of output actions (default 5 buy plans).
    """

    def __init__(
        self,
        output_neuron_ids: List[int],
        baseline_rates: torch.Tensor,
        n_actions: int = N_ACTIONS,
    ) -> None:
        super().__init__()

        self.n_output_neurons = len(output_neuron_ids)
        self.n_actions = n_actions

        # Store output neuron IDs as a buffer (not a parameter)
        self.register_buffer(
            "output_neuron_ids",
            torch.tensor(output_neuron_ids, dtype=torch.long),
        )

        # Store baseline rates as a buffer (not trainable)
        if baseline_rates.shape[0] != self.n_output_neurons:
            raise ValueError(
                f"baseline_rates length ({baseline_rates.shape[0]}) != "
                f"output_neuron_ids length ({self.n_output_neurons})"
            )
        self.register_buffer(
            "baseline_rates",
            baseline_rates.float(),
        )

        # ── Trainable parameters: linear projection ──
        self.W = nn.Parameter(
            torch.zeros(n_actions, self.n_output_neurons, dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.zeros(n_actions, dtype=torch.float32)
        )

        # Initialize W with small random values (Xavier-style)
        nn.init.xavier_uniform_(self.W)

    def forward(
        self,
        spike_counts: torch.Tensor,
        duration_ms: float = DEFAULT_BIN_MS,
    ) -> torch.Tensor:
        """Convert spike counts to action logits.

        Parameters
        ----------
        spike_counts : Tensor
            Full spike-count vector, shape ``[n_neurons]`` or
            ``[batch, n_neurons]``.
        duration_ms : float
            Simulation duration that produced these counts (ms).

        Returns
        -------
        logits : Tensor, shape ``[n_actions]`` or ``[batch, n_actions]``
            Raw logits for each action.  Softmax is applied downstream
            by PPO's Categorical distribution.
        """
        # 1. Extract output neuron spike counts
        ids = self.output_neuron_ids
        if spike_counts.dim() == 1:
            output_counts = spike_counts[ids]  # [n_output_neurons]
        else:
            output_counts = spike_counts[:, ids]  # [batch, n_output_neurons]

        # 2. Convert counts to firing rates (Hz)
        duration_s = duration_ms / 1000.0
        rates = output_counts / duration_s

        # 3. Baseline-normalise: z = (rates - baseline) / (baseline + eps)
        # Isolates state-driven modulation from structural bias
        z = (rates - self.baseline_rates) / (self.baseline_rates + _EPS)

        # 4. Linear projection: logits = z @ W^T + b
        logits = torch.nn.functional.linear(z, self.W, self.b)

        return logits

"""Population-coded state encoding: EconomyState → input current vector.

Uses the PopSAN pattern (arXiv:2010.09635) — population coding with
overlapping Gaussian receptive fields for the continuous money dimension,
one-hot for discrete dimensions (loss streak, opponent loss streak),
and thermometer code for ordinal dimensions (round in half).

All encoder parameters (gains, centers, widths) are ``torch.nn.Parameter``
so they participate in PPO gradient updates.  Connectome weights are
NOT included — only the encoding layer is trainable.

The output is a **sustained DC drive**: economy state is static during
the buy phase, so the encoded current is constant for the full readout
window (no per-ms variation needed).
"""

from __future__ import annotations

from typing import List

import structlog
import torch
import torch.nn as nn

from flyecon.state.constants import MAX_MONEY, MIN_MONEY
from flyecon.state.economy import EconomyState

log = structlog.get_logger()

# ── Encoding dimensions ────────────────────────────────────────────────────

N_MONEY_NEURONS: int = 20
"""Number of neurons in the money population code."""

N_LOSS_STREAK_NEURONS: int = 5
"""One-hot encoding of own loss streak (rungs 0–4)."""

N_ROUND_NEURONS: int = 12
"""Thermometer code for round-in-half (1–12)."""

N_HALF_NEURONS: int = 2
"""One-hot encoding of half (0 or 1)."""

N_OPP_LOSS_STREAK_NEURONS: int = 5
"""One-hot encoding of opponent loss streak (rungs 0–4)."""

N_INPUT_NEURONS: int = (
    N_MONEY_NEURONS
    + N_LOSS_STREAK_NEURONS
    + N_ROUND_NEURONS
    + N_HALF_NEURONS
    + N_OPP_LOSS_STREAK_NEURONS
)
"""Total number of input neurons (~44)."""

# Default gain initial values
_DEFAULT_GAIN_MONEY: float = 1.0
_DEFAULT_GAIN_STREAK: float = 1.0
_DEFAULT_GAIN_ROUND: float = 1.0


class PopulationEncoder(nn.Module):
    """Encode an ``EconomyState`` into a fixed-length input current vector.

    Trainable parameters
    --------------------
    gain_money : scalar
        Global gain for money population code neurons.
    gain_streak : scalar
        Global gain for loss-streak one-hot neurons.
    gain_round : scalar
        Global gain for round/half neurons.
    centers : Tensor[N_MONEY_NEURONS]
        Gaussian receptive field centres for money encoding.
    log_widths : Tensor[N_MONEY_NEURONS]
        Log of Gaussian widths (σ).  Stored as log to ensure σ > 0.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Trainable gains ──
        self.gain_money = nn.Parameter(torch.tensor(_DEFAULT_GAIN_MONEY))
        self.gain_streak = nn.Parameter(torch.tensor(_DEFAULT_GAIN_STREAK))
        self.gain_round = nn.Parameter(torch.tensor(_DEFAULT_GAIN_ROUND))

        # ── Money population code: centres evenly tile [MIN_MONEY, MAX_MONEY] ──
        centers_init = torch.linspace(
            float(MIN_MONEY), float(MAX_MONEY), N_MONEY_NEURONS
        )
        self.centers = nn.Parameter(centers_init)

        # Width σ ≈ spacing between centres (overlapping Gaussian coverage)
        spacing = float(MAX_MONEY - MIN_MONEY) / max(N_MONEY_NEURONS - 1, 1)
        log_sigma_init = torch.full((N_MONEY_NEURONS,), float(torch.tensor(spacing).log()))
        self.log_widths = nn.Parameter(log_sigma_init)

    @property
    def n_input_neurons(self) -> int:
        """Total encoded vector dimensionality."""
        return N_INPUT_NEURONS

    def _encode_money(self, money: float) -> torch.Tensor:
        """Population code for money via overlapping Gaussians.

        I_i = gain_money * exp(-(m - c_i)^2 / (2 * sigma_i^2))
        """
        sigma = self.log_widths.exp()  # ensure σ > 0
        diff = money - self.centers
        activations = self.gain_money * torch.exp(
            -(diff ** 2) / (2.0 * sigma ** 2)
        )
        return activations

    @staticmethod
    def _encode_one_hot(index: int, n: int) -> torch.Tensor:
        """One-hot encoding: exactly one neuron active."""
        vec = torch.zeros(n, dtype=torch.float32)
        clamped = max(0, min(index, n - 1))
        vec[clamped] = 1.0
        return vec

    @staticmethod
    def _encode_thermometer(value: int, n: int) -> torch.Tensor:
        """Thermometer code: neurons 0..value-1 are active.

        For round k (1-indexed), neurons 0 through k-1 are 1.0.
        Preserves ordinality: round 6 > round 3 is visible in the
        activation pattern.
        """
        vec = torch.zeros(n, dtype=torch.float32)
        clamped = max(0, min(value, n))
        vec[:clamped] = 1.0
        return vec

    def encode(self, state: EconomyState) -> torch.Tensor:
        """Encode an economy state into a current vector.

        Parameters
        ----------
        state : EconomyState
            The current economy state to encode.

        Returns
        -------
        Tensor, shape ``[N_INPUT_NEURONS]``
            Sustained DC input current vector (~44 neurons).
        """
        parts: list[torch.Tensor] = []

        # 1. Money population code (N_MONEY_NEURONS neurons)
        parts.append(self._encode_money(float(state.money)))

        # 2. Own loss streak one-hot (N_LOSS_STREAK_NEURONS neurons)
        streak_onehot = self._encode_one_hot(state.loss_streak, N_LOSS_STREAK_NEURONS)
        parts.append(self.gain_streak * streak_onehot)

        # 3. Round thermometer + half one-hot
        # Round-in-half: thermometer code (N_ROUND_NEURONS neurons)
        round_thermo = self._encode_thermometer(state.round_number, N_ROUND_NEURONS)
        parts.append(self.gain_round * round_thermo)

        # Half: one-hot (N_HALF_NEURONS neurons)
        half_onehot = self._encode_one_hot(state.half, N_HALF_NEURONS)
        parts.append(self.gain_round * half_onehot)

        # 4. Opponent loss streak one-hot (N_OPP_LOSS_STREAK_NEURONS neurons)
        opp_onehot = self._encode_one_hot(
            state.opponent_loss_streak, N_OPP_LOSS_STREAK_NEURONS
        )
        parts.append(self.gain_streak * opp_onehot)

        return torch.cat(parts)

    def encode_batch(self, states: list[EconomyState]) -> torch.Tensor:
        """Encode a batch of states.

        Parameters
        ----------
        states : list[EconomyState]
            Batch of economy states.

        Returns
        -------
        Tensor, shape ``[batch, N_INPUT_NEURONS]``
        """
        return torch.stack([self.encode(s) for s in states])


def map_to_connectome_inputs(
    encoded: torch.Tensor,
    input_neuron_ids: List[int],
    n_neurons: int,
) -> torch.Tensor:
    """Map encoded vector onto the connectome's full neuron dimension.

    Injects the ~44-dimensional encoded currents at designated input
    neuron positions (e.g. visual/ascending neuron body IDs from the
    subcircuit).  All other neurons receive zero external current.

    Parameters
    ----------
    encoded : Tensor
        Encoded state vector, shape ``[n_input_neurons]`` or
        ``[batch, n_input_neurons]``.
    input_neuron_ids : list[int]
        Indices of input neurons in the connectome (dense indexing).
        Length must equal ``encoded.shape[-1]``.
    n_neurons : int
        Total number of neurons in the connectome.

    Returns
    -------
    Tensor, shape ``[n_neurons]`` or ``[batch, n_neurons]``
        Full-dimension current vector with encoded values at the
        designated input positions.

    Raises
    ------
    ValueError
        If ``len(input_neuron_ids)`` does not match the encoded
        dimensionality.
    """
    n_inputs = encoded.shape[-1]
    if len(input_neuron_ids) != n_inputs:
        raise ValueError(
            f"input_neuron_ids length ({len(input_neuron_ids)}) != "
            f"encoded dim ({n_inputs})"
        )

    ids_tensor = torch.tensor(input_neuron_ids, dtype=torch.long)

    if encoded.dim() == 1:
        full = torch.zeros(n_neurons, dtype=encoded.dtype, device=encoded.device)
        full[ids_tensor] = encoded
        return full

    # Batched: encoded shape [batch, n_input_neurons]
    batch_size = encoded.shape[0]
    full = torch.zeros(
        batch_size, n_neurons, dtype=encoded.dtype, device=encoded.device
    )
    full[:, ids_tensor] = encoded
    return full

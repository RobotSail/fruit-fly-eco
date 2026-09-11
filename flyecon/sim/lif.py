"""LIF (Leaky Integrate-and-Fire) network with exact/exponential integration.

Uses sparse CSR matmul for synaptic current computation. The exact
integration formula is unconditionally stable at dt=5ms:

    decay = exp(-dt / tau)
    V_new = V_rest + (V - V_rest) * decay + I_total * tau * (1 - decay) / C_m

Forward Euler is FORBIDDEN — it requires dt ≤ 1–2ms for the 7mV dynamic
range between V_rest and V_thresh.
"""

from __future__ import annotations

import math

import structlog
import torch

from flyecon.etl.loader import Connectome
from flyecon.state.constants import (
    DT_MS,
    REFRACTORY_MS,
    TAU_MS,
    V_REST_MV,
    V_THRESH_MV,
)

log = structlog.get_logger()

# Membrane capacitance — set to TAU_MS so that I in "mV-equivalent current"
# units directly drives voltage.  With C_m = tau the steady-state response
# to constant current I is V_rest + I (mV), making gain intuition simple.
C_M: float = TAU_MS

# Pre-compute the decay factor (constant across all steps)
_DECAY: float = math.exp(-DT_MS / TAU_MS)
_ONE_MINUS_DECAY: float = 1.0 - _DECAY


class LIFNetwork:
    """Sparse-CSR LIF network with exact/exponential integration.

    Parameters
    ----------
    connectome : Connectome
        Signed sparse CSR weight matrix from ETL.
    gain : float
        Global synaptic gain scalar applied multiplicatively to all weights.
    device : str
        Torch device for state tensors (default ``"cpu"``).
    """

    def __init__(
        self,
        connectome: Connectome,
        gain: float,
        device: str = "cpu",
    ) -> None:
        self.n_neurons: int = connectome.n_neurons
        self.gain: float = gain
        self.device: str = device

        # Store weight matrix as CSR on the target device
        self._W_csr: torch.Tensor = connectome.weight_matrix.to(device)

        # Store dopamine matrix and neuron-type indices
        self._dopamine_csr: torch.Tensor = connectome.dopamine_matrix.to(device)
        self._neuron_types: dict[int, str] = connectome.neuron_types

        # Pre-compute PPL, KC, MBON index sets from neuron_types
        self._ppl_indices: list[int] = []
        self._kc_indices: list[int] = []
        self._mbon_indices: list[int] = []
        for dense_idx in range(connectome.n_neurons):
            body_id = int(connectome.body_ids[dense_idx])
            ntype = connectome.neuron_types.get(body_id, "")
            if ntype.startswith("PPL"):
                self._ppl_indices.append(dense_idx)
            elif ntype.startswith("KC"):
                self._kc_indices.append(dense_idx)
            elif ntype.startswith("MBON"):
                self._mbon_indices.append(dense_idx)

        # Convert to tensors for efficient indexing
        self._ppl_idx_t = torch.tensor(
            self._ppl_indices, dtype=torch.long, device=device,
        )
        self._kc_idx_t = torch.tensor(
            self._kc_indices, dtype=torch.long, device=device,
        )
        self._mbon_idx_t = torch.tensor(
            self._mbon_indices, dtype=torch.long, device=device,
        )

        # ── State tensors ──
        self.V: torch.Tensor = torch.full(
            (self.n_neurons,), V_REST_MV, dtype=torch.float32, device=device,
        )
        self.refractory: torch.Tensor = torch.zeros(
            self.n_neurons, dtype=torch.float32, device=device,
        )
        self.spike_counts: torch.Tensor = torch.zeros(
            self.n_neurons, dtype=torch.float32, device=device,
        )

        # Track previous-step spikes for synaptic current computation
        self._prev_spikes: torch.Tensor = torch.zeros(
            self.n_neurons, dtype=torch.float32, device=device,
        )

    def reset(self, jitter: bool = False) -> None:
        """Reset membrane potentials, refractory timers, and spike counts.

        Parameters
        ----------
        jitter : bool
            If True, initialize membrane potentials uniformly between
            ``V_REST`` and ``V_THRESH`` instead of exactly at ``V_REST``.
            This breaks artificial synchrony that occurs when all neurons
            start at the same potential and receive the same input current.
            Standard practice in computational neuroscience to avoid
            synchronization artifacts with coarse time steps.
        """
        if jitter:
            self.V = torch.empty(
                self.n_neurons, dtype=torch.float32, device=self.device,
            ).uniform_(V_REST_MV, V_THRESH_MV)
        else:
            self.V.fill_(V_REST_MV)
        self.refractory.zero_()
        self.spike_counts.zero_()
        self._prev_spikes.zero_()

    def step(self, input_current: torch.Tensor) -> torch.Tensor:
        """One LIF integration step using EXACT/EXPONENTIAL integration.

        Parameters
        ----------
        input_current : Tensor, shape ``[n_neurons]``
            External input current (mV-equivalent).

        Returns
        -------
        spikes : Tensor[bool], shape ``[n_neurons]``
            Boolean mask of neurons that spiked this step.
        """
        # 1. Synaptic current via sparse CSR matmul
        I_syn = self.gain * torch.mv(self._W_csr, self._prev_spikes)

        # 2. Total current
        I_total = I_syn + input_current

        # 3. Exact/exponential integration (NOT forward Euler)
        #    decay = exp(-dt/tau)  [pre-computed as _DECAY]
        #    V_new = V_rest + (V - V_rest) * decay + I_total * tau * (1 - decay) / C_m
        V_new = (
            V_REST_MV
            + (self.V - V_REST_MV) * _DECAY
            + I_total * TAU_MS * _ONE_MINUS_DECAY / C_M
        )

        # 4. Refractory: neurons in refractory period keep V_REST
        in_refractory = self.refractory > 0
        V_new = torch.where(in_refractory, torch.tensor(V_REST_MV, device=self.device), V_new)

        # Decrease refractory timers
        self.refractory = torch.clamp(self.refractory - DT_MS, min=0.0)

        # 5. Spike detection
        spikes = V_new >= V_THRESH_MV

        # 6. Reset spiking neurons to V_REST
        V_new = torch.where(spikes, torch.tensor(V_REST_MV, device=self.device), V_new)

        # 7. Set refractory timers for neurons that just spiked
        self.refractory = torch.where(
            spikes,
            torch.tensor(REFRACTORY_MS, dtype=torch.float32, device=self.device),
            self.refractory,
        )

        # 8. Update state
        self.V = V_new
        self._prev_spikes = spikes.float()
        self.spike_counts += self._prev_spikes

        return spikes

    def get_ppl_activation(self) -> float:
        """Return mean PPL dopaminergic neuron firing rate from last simulation.

        Returns 0.0 if no PPL neurons are indexed (e.g., synthetic connectome).
        """
        if len(self._ppl_indices) == 0:
            return 0.0
        ppl_counts = self.spike_counts[self._ppl_idx_t]
        return float(ppl_counts.mean().item())

    def get_population_rates(self) -> dict[str, float]:
        """Return mean spike counts per population (KC, MBON, PPL).

        Used by dashboard telemetry for the Dopamine Ledger panel.
        """
        result: dict[str, float] = {}
        for name, idx_t in [
            ("KC", self._kc_idx_t),
            ("MBON", self._mbon_idx_t),
            ("PPL", self._ppl_idx_t),
        ]:
            if len(idx_t) > 0:
                result[name] = float(self.spike_counts[idx_t].mean().item())
            else:
                result[name] = 0.0
        return result

    def simulate(
        self,
        input_current: torch.Tensor,
        duration_ms: float,
        jitter: bool = False,
    ) -> torch.Tensor:
        """Run simulation with constant input for ``duration_ms``.

        Parameters
        ----------
        input_current : Tensor, shape ``[n_neurons]``
            Constant external input current.
        duration_ms : float
            Simulation duration in milliseconds.
        jitter : bool
            If True, jitter initial membrane potentials to break
            artificial synchrony (see ``reset``).

        Returns
        -------
        spike_counts : Tensor, shape ``[n_neurons]``
            Total spikes per neuron over the simulation window.
        """
        self.reset(jitter=jitter)
        n_steps = int(duration_ms / DT_MS)
        for _ in range(n_steps):
            self.step(input_current)
        return self.spike_counts.clone()

    def batched_simulate(
        self,
        input_currents: torch.Tensor,
        duration_ms: float,
    ) -> torch.Tensor:
        """Run batched simulation for PPO rollout collection.

        Each row of ``input_currents`` is an independent input-current
        vector. The weight matrix is shared across the batch — we iterate
        over batch items sequentially but each simulation is independent.

        Parameters
        ----------
        input_currents : Tensor, shape ``[batch, n_neurons]``
            Batch of input current vectors.
        duration_ms : float
            Simulation duration in milliseconds.

        Returns
        -------
        spike_counts : Tensor, shape ``[batch, n_neurons]``
            Per-neuron spike counts for each batch item.
        """
        batch_size = input_currents.shape[0]
        results = torch.zeros(
            batch_size, self.n_neurons, dtype=torch.float32, device=self.device,
        )
        for b in range(batch_size):
            results[b] = self.simulate(input_currents[b], duration_ms)
        return results

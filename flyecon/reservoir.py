"""Rate-based connectome reservoir for CS economy decisions.

Same approach as FLM (nftechie/flm): frozen connectome as a recurrent
reservoir with tanh nonlinearity and fan-in normalized weights.
No spikes, no gain calibration, no tonic hold. Just:

    x_t = tanh(W @ (0.6 * x_{t-1} + 0.4 * input_t))

The full 166,700-neuron MaleCNS connectome runs in one sparse matmul.
"""

from __future__ import annotations

import numpy as np
import structlog
import torch
import torch.nn as nn
from scipy import sparse

from flyecon.etl.loader import load_connectome
from flyecon.state.economy import BuyPlan, EconomyState

log = structlog.get_logger()

N_BUY_PLANS = len(BuyPlan)


def _build_weight_matrix(cache_dir: str = "cache/connectome/") -> sparse.csr_matrix:
    """Load connectome and normalize by fan-in, staying sparse throughout."""
    conn = load_connectome(cache_dir)
    # Convert torch sparse CSR → scipy sparse CSR without densifying
    W_torch = conn.weight_matrix
    crow = W_torch.crow_indices().numpy()
    col = W_torch.col_indices().numpy()
    vals = W_torch.values().numpy().copy()
    n = W_torch.shape[0]

    W_scipy = sparse.csr_matrix((vals, col, crow), shape=(n, n))

    # Fan-in normalization: divide each row by sum of absolute incoming weights
    abs_row_sums = np.array(np.abs(W_scipy).sum(axis=1)).flatten()
    abs_row_sums = np.maximum(abs_row_sums, 1.0)
    # Scale each row in-place using sparse diagonal multiply
    inv_sums = sparse.diags(1.0 / abs_row_sums, format='csr')
    W_norm = inv_sums @ W_scipy
    W_norm = W_norm.astype(np.float32)

    log.info("reservoir.weights_built",
             n_neurons=n, nnz=W_norm.nnz,
             density=f"{W_norm.nnz / (n * n):.6f}")

    return W_norm, n, conn


class FlyReservoir(nn.Module):
    """Frozen connectome reservoir with trainable input/output projections."""

    def __init__(
        self,
        n_neurons: int,
        W_sparse: sparse.csr_matrix,
        input_dim: int = 6,
        output_dim: int = N_BUY_PLANS,
        hidden_dim: int = 128,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.n = n_neurons
        self.device = device
        self.hidden_dim = hidden_dim

        # Sparse weight matrix → torch sparse CSR on device
        W_csr = W_sparse.tocsr()
        self.W = torch.sparse_csr_tensor(
            torch.tensor(W_csr.indptr, dtype=torch.int32),
            torch.tensor(W_csr.indices, dtype=torch.int32),
            torch.tensor(W_csr.data, dtype=torch.float32),
            size=(n_neurons, n_neurons),
        ).to(device)

        # Random seeded input projection: input_dim → n_neurons
        rng = np.random.default_rng(seed)
        # Sparse random projection: each neuron gets input from one input channel
        self.input_bins = torch.tensor(
            rng.integers(0, input_dim, n_neurons), dtype=torch.long, device=device
        )
        self.input_sign = torch.tensor(
            rng.choice([-1.0, 1.0], n_neurons), dtype=torch.float32, device=device
        )

        # Random seeded output projection: n_neurons → hidden_dim
        self.output_bins = torch.tensor(
            rng.integers(0, hidden_dim, n_neurons), dtype=torch.long, device=device
        )
        self.output_sign = torch.tensor(
            rng.choice([-1.0, 1.0], n_neurons), dtype=torch.float32, device=device
        )
        output_counts = torch.bincount(self.output_bins, minlength=hidden_dim).float()
        self.output_scale = torch.sqrt(torch.clamp(output_counts, min=1.0)).to(device)

        # Trainable readout: hidden_dim → output_dim
        self.readout = nn.Linear(hidden_dim, output_dim).to(device)
        nn.init.orthogonal_(self.readout.weight, gain=0.1)
        nn.init.zeros_(self.readout.bias)

        # State
        self.state = torch.zeros(n_neurons, dtype=torch.float32, device=device)

    def reset(self) -> None:
        self.state.zero_()

    def _encode_state(self, economy: EconomyState) -> torch.Tensor:
        """Encode economy state to a small vector."""
        return torch.tensor([
            economy.money / 16000.0,
            economy.round_number / 12.0,
            economy.half,
            economy.loss_streak / 4.0,
            economy.opponent_loss_streak / 4.0,
            1.0,  # bias
        ], dtype=torch.float32)

    def _project_input(self, encoded: torch.Tensor) -> torch.Tensor:
        """Project encoded state onto the full neuron population."""
        return encoded[self.input_bins] * self.input_sign

    def _pool_output(self) -> torch.Tensor:
        """Pool neuron states into a fixed-dim feature vector."""
        features = torch.zeros(self.hidden_dim, dtype=torch.float32, device=self.device)
        features.scatter_add_(0, self.output_bins, self.state * self.output_sign)
        features = features / self.output_scale
        # Normalize
        features = features / (torch.sqrt(torch.mean(features ** 2)) + 1e-6)
        return features

    def step(self, economy: EconomyState) -> torch.Tensor:
        """One reservoir step: economy state → buy plan logits."""
        encoded = self._encode_state(economy).to(self.device)
        drive = 0.6 * self.state + 0.4 * self._project_input(encoded)
        self.state = torch.tanh(torch.mv(self.W, drive))
        features = self._pool_output()
        return self.readout(features)

    def forward(self, states: list[EconomyState]) -> torch.Tensor:
        """Process a batch of INDEPENDENT states (no sequential memory).

        For sequential processing, call step() repeatedly.
        """
        batch_logits = []
        for s in states:
            self.reset()
            # Run a few recurrence steps to let the state settle
            for _ in range(3):
                logits = self.step(s)
            batch_logits.append(logits)
        return torch.stack(batch_logits)

    def forward_sequence(self, states: list[EconomyState]) -> list[torch.Tensor]:
        """Process a SEQUENCE of rounds, carrying state between them."""
        self.reset()
        logits = []
        for s in states:
            logits.append(self.step(s))
        return logits

    def telemetry(self) -> dict:
        """Return sampled neuron states for visualization."""
        indices = torch.linspace(0, self.n - 1, 96).long().to(self.device)
        return {
            "state_rms": float(torch.sqrt(torch.mean(self.state ** 2)).item()),
            "sampled_states": self.state[indices].cpu().tolist(),
            "n_active": int((self.state.abs() > 0.1).sum().item()),
            "n_neurons": self.n,
        }


def build_reservoir(
    cache_dir: str = "cache/connectome/",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> FlyReservoir:
    """Build a FlyReservoir from the real MaleCNS connectome."""
    W_sparse, n, conn = _build_weight_matrix(cache_dir)
    log.info("reservoir.building", n_neurons=n, device=device)
    reservoir = FlyReservoir(n, W_sparse, device=device)
    log.info("reservoir.ready", n_neurons=n,
             params=sum(p.numel() for p in reservoir.parameters()))
    return reservoir

"""Hybrid rate + spiking architecture for CS economy decisions.

Economy state → Rate model (164K neurons, tanh reservoir) → extract KC
activations → Spiking LIF mushroom body (4,185 neurons) → MBON spike
readout → linear layer → buy decision logits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.feather as pf
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from flyecon.etl.loader import Connectome, load_connectome_from_tables
from flyecon.etl.subcircuit import extract_mushroom_body
from flyecon.sim.lif import LIFNetwork
from flyecon.state.constants import DT_MS
from flyecon.state.economy import BuyPlan, EconomyState

log = structlog.get_logger()

N_BUY_PLANS = len(BuyPlan)
INPUT_SCALE: float = 30.0
SIM_DURATION_MS: float = 200.0
RESERVOIR_STEPS: int = 3



def _load_connectome_adapted(cache_dir: str = "cache/connectome/") -> Connectome:
    """Load connectome, adapting on-disk column names to ETL expectations."""
    cache = Path(cache_dir)

    # Try both naming conventions for weights file
    weights_names = [
        "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    ]
    weights_path = None
    for name in weights_names:
        p = cache / name
        if p.exists():
            weights_path = p
            break
    if weights_path is None:
        raise FileNotFoundError(
            f"No weights Feather file found in {cache_dir}. "
            "Run 'python -m flyecon etl download' first."
        )

    nt_path = cache / "body-neurotransmitters-male-cns-v1.0.feather"
    ann_path = cache / "body-annotations-male-cns-v1.0-minconf-0.5.feather"

    for p in [nt_path, ann_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing Feather file: {p}")

    weights_df = pf.read_table(str(weights_path)).to_pandas()
    nt_df = pf.read_table(str(nt_path)).to_pandas()
    ann_df = pf.read_table(str(ann_path)).to_pandas()

    # Adapt column names to match ETL builder expectations
    weights_df = weights_df.rename(columns={
        "body_pre": "bodyId_pre", "body_post": "bodyId_post",
    })
    nt_df = nt_df.rename(columns={
        "body": "bodyId", "predicted_nt": "predictedNt",
        "consensus_nt": "consensusNt",
    })

    return load_connectome_from_tables(weights_df, nt_df, ann_df)



def _find_indices_by_prefix(
    connectome: Connectome,
    prefix: str,
) -> list[int]:
    """Return sorted dense indices of neurons whose type starts with *prefix*."""
    indices: list[int] = []
    for body_id, ntype in connectome.neuron_types.items():
        if ntype.startswith(prefix):
            idx = int(np.searchsorted(connectome.body_ids, body_id))
            if idx < connectome.n_neurons and connectome.body_ids[idx] == body_id:
                indices.append(idx)
    return sorted(indices)


def _map_kc_full_to_mb(
    full_conn: Connectome,
    mb_conn: Connectome,
) -> tuple[list[int], list[int]]:
    """Map KC indices between full connectome and MB subcircuit.

    Returns (kc_full_indices, kc_mb_indices) in corresponding order.
    """
    full_bid_to_idx: dict[int, int] = {}
    for body_id, ntype in full_conn.neuron_types.items():
        if ntype.startswith("KC"):
            idx = int(np.searchsorted(full_conn.body_ids, body_id))
            if idx < full_conn.n_neurons and full_conn.body_ids[idx] == body_id:
                full_bid_to_idx[body_id] = idx

    mb_bid_to_idx: dict[int, int] = {}
    for body_id, ntype in mb_conn.neuron_types.items():
        if ntype.startswith("KC"):
            idx = int(np.searchsorted(mb_conn.body_ids, body_id))
            if idx < mb_conn.n_neurons and mb_conn.body_ids[idx] == body_id:
                mb_bid_to_idx[body_id] = idx

    # Intersect — only KCs present in both
    common_bids = sorted(set(full_bid_to_idx) & set(mb_bid_to_idx))
    kc_full = [full_bid_to_idx[bid] for bid in common_bids]
    kc_mb = [mb_bid_to_idx[bid] for bid in common_bids]

    return kc_full, kc_mb



class _RateReservoir:
    """Rate-model reservoir: r_new = tanh(gain * W @ r + I_ext) with leak."""

    def __init__(
        self,
        connectome: Connectome,
        gain: float = 0.05,
        leak: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.n = connectome.n_neurons
        self.gain = gain
        self.leak = leak
        self.device = device
        self._W = connectome.weight_matrix.to(device)
        self.rates = torch.zeros(self.n, dtype=torch.float32, device=device)

    def reset(self) -> None:
        self.rates.zero_()

    def step(self, input_current: torch.Tensor) -> None:
        recurrent = self.gain * torch.mv(self._W, self.rates)
        new = torch.tanh(recurrent + input_current)
        self.rates = (1.0 - self.leak) * self.rates + self.leak * new



class HybridModel(nn.Module):
    """Rate-model full brain + spiking LIF mushroom body hybrid."""

    def __init__(
        self,
        full_conn: Connectome,
        mb_conn: Connectome,
        reservoir_gain: float = 0.05,
        lif_gain: float = 0.5,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device

        # Rate-model reservoir (NOT an nn.Module — frozen)
        self._reservoir = _RateReservoir(
            full_conn, gain=reservoir_gain, device=device,
        )

        # Spiking LIF mushroom body (NOT an nn.Module — frozen)
        self._lif = LIFNetwork(mb_conn, gain=lif_gain, device=device)

        # Index mappings
        self._kc_full_idx, self._kc_mb_idx = _map_kc_full_to_mb(
            full_conn, mb_conn,
        )
        self._mbon_mb_idx = _find_indices_by_prefix(mb_conn, "MBON")

        # Random input projection (seeded for reproducibility)
        self._n_full = full_conn.n_neurons
        self._n_mb = mb_conn.n_neurons
        self._n_kc = len(self._kc_full_idx)
        self._n_mbon = len(self._mbon_mb_idx)

        rng = np.random.default_rng(42)
        input_dim = 6  # economy encoding dimension
        self._input_bins = torch.tensor(
            rng.integers(0, input_dim, self._n_full),
            dtype=torch.long, device=device,
        )
        self._input_sign = torch.tensor(
            rng.choice([-1.0, 1.0], self._n_full),
            dtype=torch.float32, device=device,
        )

        # Index tensors for fast extraction
        self._kc_full_t = torch.tensor(
            self._kc_full_idx, dtype=torch.long, device=device,
        )
        self._kc_mb_t = torch.tensor(
            self._kc_mb_idx, dtype=torch.long, device=device,
        )
        self._mbon_mb_t = torch.tensor(
            self._mbon_mb_idx, dtype=torch.long, device=device,
        )

        # Trainable readout: MBON spike rates → buy plan logits
        self.readout = nn.Linear(self._n_mbon, N_BUY_PLANS)
        nn.init.orthogonal_(self.readout.weight, gain=0.1)
        nn.init.zeros_(self.readout.bias)

        log.info(
            "hybrid.init",
            n_full=self._n_full,
            n_mb=self._n_mb,
            n_kc=self._n_kc,
            n_mbon=self._n_mbon,
        )

    def _encode_state(self, economy: EconomyState) -> torch.Tensor:
        """Encode economy state to a 6-dim vector."""
        return torch.tensor([
            economy.money / 16000.0,
            economy.round_number / 12.0,
            float(economy.half),
            economy.loss_streak / 4.0,
            economy.opponent_loss_streak / 4.0,
            1.0,  # bias
        ], dtype=torch.float32, device=self.device)

    def _project_input(self, encoded: torch.Tensor) -> torch.Tensor:
        """Project encoded economy state onto full neuron population."""
        return encoded[self._input_bins] * self._input_sign

    def _forward_single(self, economy: EconomyState) -> torch.Tensor:
        """Single economy state → logits [N_BUY_PLANS]."""
        # 1. Encode and project to full neuron population
        encoded = self._encode_state(economy)
        drive = self._project_input(encoded)

        # 2. Run rate-model reservoir
        self._reservoir.reset()
        for _ in range(RESERVOIR_STEPS):
            self._reservoir.step(drive)

        # 3. Extract KC activations from reservoir
        kc_activations = self._reservoir.rates[self._kc_full_t]

        # 4. Build input current for spiking MB
        input_current = torch.zeros(
            self._n_mb, dtype=torch.float32, device=self.device,
        )
        input_current[self._kc_mb_t] = kc_activations * INPUT_SCALE

        # 5. Simulate spiking LIF mushroom body
        spike_counts = self._lif.simulate(
            input_current, duration_ms=SIM_DURATION_MS, jitter=True,
        )

        # 6. Compute MBON spike rates (Hz)
        mbon_spikes = spike_counts[self._mbon_mb_t]
        mbon_rates = mbon_spikes / (SIM_DURATION_MS / 1000.0)

        # 7. Trainable linear readout
        logits = self.readout(mbon_rates)
        return logits

    def forward(self, states: list[EconomyState]) -> torch.Tensor:
        """Batch of economy states → logits [batch, N_BUY_PLANS]."""
        batch_logits = []
        for s in states:
            logits = self._forward_single(s)
            batch_logits.append(logits)
        return torch.stack(batch_logits)



BUY_PLAN_NAMES = [p.value for p in BuyPlan]
BUY_PLAN_TO_IDX = {name: i for i, name in enumerate(BUY_PLAN_NAMES)}
SIDE_MAP = {"t": 0, "ct": 1}


def _load_pro_rounds(path: str | Path) -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_parquet(path)
    df = df[df["n_players"] == 5].copy()
    df["label"] = df["buy_decision"].map(BUY_PLAN_TO_IDX)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def _df_to_states(df: "pd.DataFrame") -> list[EconomyState]:
    states: list[EconomyState] = []
    for _, row in df.iterrows():
        money = int(row["avg_balance"])
        rnd = int(row.get("round_in_half", row["round_num"]))
        rnd = max(1, min(12, rnd))
        half = int(row.get("half", 0))
        loss_streak = int(row.get("loss_streak", 0))
        opp_streak = int(row.get("opponent_loss_streak", 0))
        states.append(EconomyState(
            money=max(0, min(16000, money)),
            round_number=rnd,
            loss_streak=min(loss_streak, 4),
            half=half,
            opponent_loss_streak=min(opp_streak, 4),
        ))
    return states



def train_and_eval(
    cache_dir: str = "cache/connectome/",
    train_path: str = "data/cs2_pro_train.parquet",
    eval_path: str = "data/cs2_pro_eval.parquet",
    n_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    reservoir_gain: float = 0.05,
    lif_gain: float = 0.5,
    max_train_samples: int = 2048,
    max_eval_samples: int = 512,
) -> dict[str, float]:
    """Build hybrid model, train readout on pro data, evaluate. Print accuracy.

    The full-brain rate model + spiking MB simulation is expensive per
    sample (~0.5s on CPU), so *max_train_samples* and *max_eval_samples*
    cap the data used.  Set to 0 to use the full dataset.
    """

    # 1. Load full connectome
    log.info("hybrid.loading_connectome")
    full_conn = _load_connectome_adapted(cache_dir)
    log.info("hybrid.full_connectome", n=full_conn.n_neurons)

    # 2. Extract mushroom body subcircuit (0 hops — just KC/MBON/PPL)
    log.info("hybrid.extracting_mb")
    mb_conn = extract_mushroom_body(full_conn, hops=0)
    log.info("hybrid.mb_extracted", n=mb_conn.n_neurons)

    # 3. Build hybrid model
    model = HybridModel(
        full_conn, mb_conn,
        reservoir_gain=reservoir_gain,
        lif_gain=lif_gain,
    )

    # 4. Load data (cap to max samples for CPU feasibility)
    log.info("hybrid.loading_data")
    train_df = _load_pro_rounds(train_path)
    eval_df = _load_pro_rounds(eval_path)
    if max_train_samples > 0 and len(train_df) > max_train_samples:
        train_df = train_df.sample(n=max_train_samples, random_state=42)
    if max_eval_samples > 0 and len(eval_df) > max_eval_samples:
        eval_df = eval_df.sample(n=max_eval_samples, random_state=42)
    log.info("hybrid.data_loaded", train=len(train_df), eval=len(eval_df))

    # 5. Train readout
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        shuffled = train_df.sample(frac=1.0, random_state=epoch)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(shuffled), batch_size):
            batch_df = shuffled.iloc[start:start + batch_size]
            if len(batch_df) < 2:
                continue

            states = _df_to_states(batch_df)
            labels = torch.tensor(
                batch_df["label"].values, dtype=torch.long,
            )

            logits = model.forward(states)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(batch_df)
            total_samples += len(batch_df)

        train_acc = total_correct / max(total_samples, 1)
        train_loss = total_loss / max(total_samples, 1)
        log.info(
            "hybrid.train_epoch",
            epoch=epoch,
            loss=round(train_loss, 4),
            accuracy=round(train_acc, 4),
            samples=total_samples,
        )

    # 6. Evaluate
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for start in range(0, len(eval_df), batch_size):
            batch_df = eval_df.iloc[start:start + batch_size]
            if len(batch_df) < 2:
                continue

            states = _df_to_states(batch_df)
            labels = torch.tensor(
                batch_df["label"].values, dtype=torch.long,
            )

            logits = model.forward(states)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)

            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(batch_df)
            total_samples += len(batch_df)

    eval_acc = total_correct / max(total_samples, 1)
    eval_loss = total_loss / max(total_samples, 1)

    log.info(
        "hybrid.eval",
        loss=round(eval_loss, 4),
        accuracy=round(eval_acc, 4),
        samples=total_samples,
    )
    print(f"\n{'='*60}")
    print(f"Hybrid Model Evaluation")
    print(f"{'='*60}")
    print(f"  Eval accuracy: {eval_acc:.4f}")
    print(f"  Eval loss:     {eval_loss:.4f}")
    print(f"  Eval samples:  {total_samples}")
    print(f"{'='*60}\n")

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
    }



def main() -> None:
    """Run hybrid model training and evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid rate + spiking architecture for CS economy",
    )
    parser.add_argument(
        "--cache-dir", default="cache/connectome/",
        help="Path to cached connectome Feather files",
    )
    parser.add_argument(
        "--train-path", default="data/cs2_pro_train.parquet",
        help="Path to training parquet file",
    )
    parser.add_argument(
        "--eval-path", default="data/cs2_pro_eval.parquet",
        help="Path to evaluation parquet file",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reservoir-gain", type=float, default=0.05)
    parser.add_argument("--lif-gain", type=float, default=0.5)
    parser.add_argument("--max-train", type=int, default=2048)
    parser.add_argument("--max-eval", type=int, default=512)
    args = parser.parse_args()

    train_and_eval(
        cache_dir=args.cache_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reservoir_gain=args.reservoir_gain,
        lif_gain=args.lif_gain,
        max_train_samples=args.max_train,
        max_eval_samples=args.max_eval,
    )


if __name__ == "__main__":
    main()

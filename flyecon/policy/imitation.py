"""Imitation learning from pro CS2 match data.

Trains the fly's readout layer to predict what pro teams chose,
given the economy state routed through the real mushroom body connectome.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog
import torch
import torch.nn.functional as F

from flyecon.policy.ppo import FlyPolicy
from flyecon.state.economy import BuyPlan, EconomyState

log = structlog.get_logger()

BUY_PLAN_NAMES = [p.value for p in BuyPlan]
BUY_PLAN_TO_IDX = {name: i for i, name in enumerate(BUY_PLAN_NAMES)}

SIDE_MAP = {"t": 0, "ct": 1}


def load_pro_rounds(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["n_players"] == 5].copy()
    df["label"] = df["buy_decision"].map(BUY_PLAN_TO_IDX)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["side_num"] = df["side"].map(SIDE_MAP).fillna(0).astype(int)
    return df


def df_to_states(df: pd.DataFrame) -> list[EconomyState]:
    states = []
    for _, row in df.iterrows():
        money = int(row["avg_balance"])
        rnd = int(row.get("round_in_half", row["round_num"]))
        rnd = max(1, min(12, rnd))
        half = int(row.get("half", 0))
        loss_streak = int(row.get("loss_streak", 0))
        opp_streak = int(row.get("opponent_loss_streak", 0))
        states.append(EconomyState(
            money=money,
            round_number=rnd,
            loss_streak=min(loss_streak, 4),
            half=half,
            opponent_loss_streak=min(opp_streak, 4),
        ))
    return states


class ImitationTrainer:
    """Train the fly to imitate pro CS2 economy decisions."""

    def __init__(
        self,
        policy: FlyPolicy,
        train_path: str | Path = "data/cs2_pro_train.parquet",
        eval_path: str | Path = "data/cs2_pro_eval.parquet",
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> None:
        self.policy = policy
        self.batch_size = batch_size

        self.train_df = load_pro_rounds(train_path)
        self.eval_df = load_pro_rounds(eval_path)

        log.info("imitation.loaded",
                 train_rows=len(self.train_df),
                 eval_rows=len(self.eval_df))

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self._step = 0

    def train_epoch(self) -> dict[str, float]:
        self.policy.train()
        shuffled = self.train_df.sample(frac=1.0, random_state=self._step)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = 0

        for start in range(0, len(shuffled), self.batch_size):
            batch_df = shuffled.iloc[start:start + self.batch_size]
            if len(batch_df) < 2:
                continue

            states = df_to_states(batch_df)
            labels = torch.tensor(batch_df["label"].values, dtype=torch.long)

            dist, _, _ = self.policy.forward(states)
            logits = dist.logits

            labels = labels.to(logits.device)
            loss = F.cross_entropy(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(batch_df)
            total_samples += len(batch_df)
            n_batches += 1

        self._step += 1
        acc = total_correct / max(total_samples, 1)
        avg_loss = total_loss / max(total_samples, 1)

        log.info("imitation.train_epoch",
                 epoch=self._step, loss=round(avg_loss, 4),
                 accuracy=round(acc, 4), batches=n_batches)

        return {"loss": avg_loss, "accuracy": acc, "epoch": self._step}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.policy.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        per_class_correct: dict[str, int] = {n: 0 for n in BUY_PLAN_NAMES}
        per_class_total: dict[str, int] = {n: 0 for n in BUY_PLAN_NAMES}

        for start in range(0, len(self.eval_df), self.batch_size):
            batch_df = self.eval_df.iloc[start:start + self.batch_size]
            if len(batch_df) < 2:
                continue

            states = df_to_states(batch_df)
            labels = torch.tensor(batch_df["label"].values, dtype=torch.long)

            dist, _, _ = self.policy.forward(states)
            logits = dist.logits

            labels = labels.to(logits.device)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)

            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(batch_df)
            total_samples += len(batch_df)

            for pred, label in zip(preds.tolist(), labels.tolist()):
                name = BUY_PLAN_NAMES[label]
                per_class_total[name] += 1
                if pred == label:
                    per_class_correct[name] += 1

        self.policy.train()
        acc = total_correct / max(total_samples, 1)
        avg_loss = total_loss / max(total_samples, 1)

        per_class_acc = {}
        for name in BUY_PLAN_NAMES:
            if per_class_total[name] > 0:
                per_class_acc[name] = round(
                    per_class_correct[name] / per_class_total[name], 3
                )

        log.info("imitation.eval",
                 loss=round(avg_loss, 4), accuracy=round(acc, 4),
                 per_class=per_class_acc)

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "per_class_accuracy": per_class_acc,
        }
# test write

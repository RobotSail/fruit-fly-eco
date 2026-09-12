"""Knowledge distillation: rate-model teacher → spiking-model student.

The rate model (FlyReservoir) hits ~76.7% on CS2 pro economy data.
The spiking model (FlyPolicy + LIFNetwork) only gets ~67.5%.
This module bridges the gap:
1. Train rate model (teacher) on pro match data (5 epochs).
2. Generate soft labels (teacher probability distributions over 5 buy plans).
3. Train spiking model readout with KL divergence loss against teacher.
4. Use mushroom body subcircuit (hops=0) for the student network.
5. Print comparison: teacher accuracy vs student accuracy vs majority baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from flyecon.etl.loader import load_connectome
from flyecon.etl.subcircuit import extract_mushroom_body
from flyecon.policy.imitation import (
    BUY_PLAN_NAMES,
    BUY_PLAN_TO_IDX,
    df_to_states,
    load_pro_rounds,
)
from flyecon.policy.ppo import FlyPolicy, build_fly_policy
from flyecon.reservoir import FlyReservoir, build_reservoir
from flyecon.sim.calibration import calibrate_gain
from flyecon.state.economy import BuyPlan, EconomyState

log = structlog.get_logger()

N_BUY_PLANS = len(BuyPlan)


# ── Teacher training ─────────────────────────────────────────────────────────


def train_teacher(
    reservoir: FlyReservoir,
    train_df: pd.DataFrame,
    *,
    n_epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> list[dict[str, float]]:
    """Train the rate-model teacher on pro match data.

    Only the readout layer is trained (connectome reservoir is frozen).

    Returns per-epoch metrics (loss, accuracy).
    """
    optimizer = torch.optim.Adam(reservoir.parameters(), lr=lr)
    epoch_logs: list[dict[str, float]] = []

    for epoch in range(1, n_epochs + 1):
        reservoir.train()
        shuffled = train_df.sample(frac=1.0, random_state=epoch)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(shuffled), batch_size):
            batch_df = shuffled.iloc[start : start + batch_size]
            if len(batch_df) < 2:
                continue

            states = df_to_states(batch_df)
            labels = torch.tensor(
                batch_df["label"].values, dtype=torch.long,
            )

            logits = reservoir.forward(states)
            labels = labels.to(logits.device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(reservoir.parameters(), 1.0)
            optimizer.step()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(batch_df)
            total_samples += len(batch_df)

        acc = total_correct / max(total_samples, 1)
        avg_loss = total_loss / max(total_samples, 1)
        epoch_logs.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc})

        log.info(
            "distill.teacher_epoch",
            epoch=epoch,
            loss=round(avg_loss, 4),
            accuracy=round(acc, 4),
        )

    return epoch_logs


# ── Soft label generation ────────────────────────────────────────────────────


@torch.no_grad()
def generate_soft_labels(
    teacher: FlyReservoir,
    states: list[EconomyState],
    *,
    temperature: float = 2.0,
    batch_size: int = 128,
) -> torch.Tensor:
    """Generate soft probability distributions from the teacher.

    Temperature controls softness — higher T = softer distributions
    with more inter-class information (Hinton et al. 2015, default T=2).
    Returns Tensor [n_states, N_BUY_PLANS].
    """
    teacher.eval()
    all_probs: list[torch.Tensor] = []

    for start in range(0, len(states), batch_size):
        batch_states = states[start : start + batch_size]
        logits = teacher.forward(batch_states)
        # Apply temperature scaling before softmax
        soft_probs = F.softmax(logits / temperature, dim=-1)
        all_probs.append(soft_probs)

    return torch.cat(all_probs, dim=0)


# ── Student distillation training ────────────────────────────────────────────


def distill_student(
    student: FlyPolicy,
    train_states: list[EconomyState],
    soft_labels: torch.Tensor,
    hard_labels: torch.Tensor,
    *,
    n_epochs: int = 10,
    lr: float = 5e-4,
    batch_size: int = 64,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> list[dict[str, float]]:
    """Train student readout via KL divergence against teacher soft labels.

    Loss = alpha * KL(teacher || student) * T^2 + (1-alpha) * CE(hard).
    Only readout weights and value head are updated (LIF is frozen).
    Temperature must match the one used to generate soft labels.
    """
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    epoch_logs: list[dict[str, float]] = []
    n = len(train_states)

    for epoch in range(1, n_epochs + 1):
        student.train()
        perm = torch.randperm(n)

        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue

            batch_states = [train_states[i] for i in idx.tolist()]
            batch_soft = soft_labels[idx]
            batch_hard = hard_labels[idx]

            dist, _, features = student.forward(batch_states)
            student_logits = dist.logits
            device = student_logits.device

            # KL divergence loss (soft targets)
            # KL(teacher || student) at temperature T
            student_log_soft = F.log_softmax(
                student_logits / temperature, dim=-1,
            )
            # Move soft labels to student's device
            batch_soft_dev = batch_soft.to(device)
            # KL divergence: sum over classes, mean over batch
            # Multiply by T^2 per Hinton et al. to scale gradients correctly
            kl_loss = (
                F.kl_div(
                    student_log_soft,
                    batch_soft_dev,
                    reduction="batchmean",
                )
                * (temperature ** 2)
            )

            # Hard label cross-entropy loss
            batch_hard = batch_hard.to(device)
            ce_loss = F.cross_entropy(student_logits, batch_hard)

            # Combined loss
            loss = alpha * kl_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            preds = student_logits.argmax(dim=-1)
            total_correct += (preds == batch_hard).sum().item()
            total_kl_loss += kl_loss.item() * len(idx)
            total_ce_loss += ce_loss.item() * len(idx)
            total_samples += len(idx)

        acc = total_correct / max(total_samples, 1)
        avg_kl = total_kl_loss / max(total_samples, 1)
        avg_ce = total_ce_loss / max(total_samples, 1)
        epoch_logs.append({
            "epoch": epoch,
            "kl_loss": avg_kl,
            "ce_loss": avg_ce,
            "accuracy": acc,
        })

        log.info(
            "distill.student_epoch",
            epoch=epoch,
            kl_loss=round(avg_kl, 4),
            ce_loss=round(avg_ce, 4),
            accuracy=round(acc, 4),
        )

    return epoch_logs


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_accuracy(
    model_fn,
    eval_states: list[EconomyState],
    labels: torch.Tensor,
    *,
    batch_size: int = 128,
) -> float:
    """Evaluate top-1 accuracy. model_fn: list[EconomyState] → logits Tensor."""
    total_correct = 0
    total = 0

    for start in range(0, len(eval_states), batch_size):
        batch_states = eval_states[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]
        if len(batch_states) < 1:
            continue

        logits = model_fn(batch_states)
        logits = logits.to(batch_labels.device)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch_labels).sum().item()
        total += len(batch_labels)

    return total_correct / max(total, 1)


def majority_baseline(labels: torch.Tensor) -> float:
    """Compute majority-class baseline accuracy.

    Returns the accuracy of always predicting the most frequent class.
    """
    counts = torch.bincount(labels, minlength=N_BUY_PLANS)
    return float(counts.max().item()) / max(len(labels), 1)


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_distillation(
    train_path: str | Path = "data/cs2_pro_train.parquet",
    eval_path: str | Path = "data/cs2_pro_eval.parquet",
    cache_dir: str = "cache/connectome/",
    teacher_epochs: int = 5,
    student_epochs: int = 10,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> dict:
    """Run the full distillation pipeline end-to-end.

    Returns dict with teacher_acc, student_acc, majority_acc, and logs.
    """
    # ── Load data ──
    log.info("distill.loading_data", train=str(train_path), eval=str(eval_path))
    train_df = load_pro_rounds(train_path)
    eval_df = load_pro_rounds(eval_path)

    train_states = df_to_states(train_df)
    eval_states = df_to_states(eval_df)
    train_labels = torch.tensor(train_df["label"].values, dtype=torch.long)
    eval_labels = torch.tensor(eval_df["label"].values, dtype=torch.long)

    log.info(
        "distill.data_loaded",
        train_n=len(train_states),
        eval_n=len(eval_states),
    )

    # ── Build teacher (rate model on full connectome) ──
    log.info("distill.building_teacher")
    teacher = build_reservoir(cache_dir=cache_dir)

    # ── Train teacher ──
    log.info("distill.training_teacher", epochs=teacher_epochs)
    teacher_logs = train_teacher(
        teacher, train_df, n_epochs=teacher_epochs,
    )

    # ── Evaluate teacher ──
    teacher.eval()

    def teacher_model_fn(states: list[EconomyState]) -> torch.Tensor:
        return teacher.forward(states)

    teacher_acc = evaluate_accuracy(
        teacher_model_fn, eval_states, eval_labels,
    )
    log.info("distill.teacher_eval_acc", accuracy=round(teacher_acc, 4))

    # ── Generate soft labels from teacher ──
    log.info("distill.generating_soft_labels", temperature=temperature)
    soft_labels = generate_soft_labels(
        teacher, train_states, temperature=temperature,
    )
    log.info(
        "distill.soft_labels_ready",
        shape=list(soft_labels.shape),
        entropy=round(
            float(
                -(soft_labels * torch.log(soft_labels + 1e-10)).sum(dim=-1).mean()
            ),
            4,
        ),
    )

    # ── Build student (spiking model on MB subcircuit, hops=0) ──
    log.info("distill.building_student")
    connectome = load_connectome(cache_dir)
    mb_connectome = extract_mushroom_body(connectome, hops=0)
    log.info(
        "distill.mb_subcircuit",
        n_neurons=mb_connectome.n_neurons,
        n_synapses=mb_connectome.n_synapses,
    )

    # Calibrate gain for the subcircuit
    cal = calibrate_gain(mb_connectome)
    log.info("distill.gain_calibrated", gain=cal.gain, rate_hz=cal.mean_rate_hz)

    student = build_fly_policy(mb_connectome, gain=cal.gain)

    # ── Distill: train student from teacher's soft labels ──
    log.info(
        "distill.training_student",
        epochs=student_epochs,
        temperature=temperature,
        alpha=alpha,
    )
    student_logs = distill_student(
        student,
        train_states,
        soft_labels,
        train_labels,
        n_epochs=student_epochs,
        temperature=temperature,
        alpha=alpha,
    )

    # ── Evaluate student ──
    student.eval()

    def student_model_fn(states: list[EconomyState]) -> torch.Tensor:
        dist, _, _ = student.forward(states)
        return dist.logits

    student_acc = evaluate_accuracy(
        student_model_fn, eval_states, eval_labels,
    )
    log.info("distill.student_eval_acc", accuracy=round(student_acc, 4))

    # ── Majority baseline ──
    maj_acc = majority_baseline(eval_labels)
    most_common_class = BUY_PLAN_NAMES[int(eval_labels.mode().values.item())]
    log.info(
        "distill.majority_baseline",
        accuracy=round(maj_acc, 4),
        class_name=most_common_class,
    )

    # ── Print comparison ──
    print("\n" + "=" * 60)
    print("  DISTILLATION RESULTS — CS2 Pro Economy (eval set)")
    print("=" * 60)
    print(f"  Teacher (rate model):      {teacher_acc:6.1%}")
    print(f"  Student (spiking, distill):{student_acc:6.1%}")
    print(f"  Majority baseline:         {maj_acc:6.1%}")
    print(
        f"  Majority class:            {most_common_class}"
    )
    print("-" * 60)
    gap = teacher_acc - student_acc
    print(f"  Teacher→Student gap:       {gap:+6.1%}")
    print("=" * 60 + "\n")

    return {
        "teacher_acc": teacher_acc,
        "student_acc": student_acc,
        "majority_acc": maj_acc,
        "majority_class": most_common_class,
        "teacher_logs": teacher_logs,
        "student_logs": student_logs,
        "temperature": temperature,
        "alpha": alpha,
    }


if __name__ == "__main__":
    run_distillation()

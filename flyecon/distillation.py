"""Rate→Spiking distillation: transfer knowledge from FlyReservoir to FlyPolicy.

Pipeline:
  1. Build rate-model teacher (FlyReservoir) from real connectome
  2. Train teacher via imitation learning on pro CS2 economy data
  3. Generate soft labels (teacher probability distributions)
  4. Build spiking student (FlyPolicy with mushroom-body connectome)
  5. Train student readout against teacher soft labels (KL divergence)
  6. Evaluate both on held-out eval data, print comparison table
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from flyecon.policy.imitation import BUY_PLAN_NAMES, df_to_states, load_pro_rounds
from flyecon.reservoir import FlyReservoir, build_reservoir

log = structlog.get_logger()


# ── Step 1 & 2: Train rate-model teacher ─────────────────────────────────────


def train_teacher(
    teacher: FlyReservoir,
    train_df: pd.DataFrame,
    *,
    n_epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> list[dict[str, float]]:
    """Train the rate-model teacher via cross-entropy imitation learning."""
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    epoch_logs: list[dict[str, float]] = []

    for epoch in range(1, n_epochs + 1):
        teacher.train()
        shuffled = train_df.sample(frac=1.0, random_state=epoch)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(shuffled), batch_size):
            batch_df = shuffled.iloc[start : start + batch_size]
            if len(batch_df) < 2:
                continue

            states = df_to_states(batch_df)
            labels = torch.tensor(batch_df["label"].values, dtype=torch.long)
            labels = labels.to(teacher.readout.weight.device)

            logits = teacher.forward(states)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(batch_df)
            total_samples += len(batch_df)

        acc = total_correct / max(total_samples, 1)
        avg_loss = total_loss / max(total_samples, 1)
        log.info("distill.teacher_epoch", epoch=epoch,
                 loss=round(avg_loss, 4), accuracy=round(acc, 4))
        epoch_logs.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc})

    return epoch_logs


# ── Step 3: Generate soft labels ─────────────────────────────────────────────


@torch.no_grad()
def generate_soft_labels(
    teacher: FlyReservoir,
    train_df: pd.DataFrame,
    *,
    batch_size: int = 256,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Run teacher on all training samples, return probability distributions.

    Returns shape [n_samples, n_actions].
    """
    teacher.eval()
    all_probs: list[torch.Tensor] = []

    for start in range(0, len(train_df), batch_size):
        batch_df = train_df.iloc[start : start + batch_size]
        states = df_to_states(batch_df)
        logits = teacher.forward(states)
        probs = F.softmax(logits / temperature, dim=-1)
        all_probs.append(probs.cpu())

    soft_labels = torch.cat(all_probs, dim=0)
    entropy = float(-(soft_labels * soft_labels.clamp(min=1e-8).log()).sum(-1).mean())
    log.info("distill.soft_labels", n_samples=soft_labels.shape[0],
             mean_entropy=round(entropy, 4))
    return soft_labels


# ── Step 4 & 5: Train spiking student with KL divergence ────────────────────


def train_student(
    student: nn.Module,
    train_df: pd.DataFrame,
    soft_labels: torch.Tensor,
    *,
    n_epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 256,
    temperature: float = 1.0,
) -> list[dict[str, float]]:
    """Train student readout against teacher soft labels using KL divergence.

    KL(teacher || student) with temperature scaling. Only readout parameters
    are trained; the LIF network remains frozen.
    """
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    hard_labels = torch.tensor(train_df["label"].values, dtype=torch.long)
    epoch_logs: list[dict[str, float]] = []

    for epoch in range(1, n_epochs + 1):
        student.train()
        rng = np.random.default_rng(epoch + 100)
        indices = rng.permutation(len(train_df))
        total_kl = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            if len(batch_idx) < 2:
                continue

            batch_df = train_df.iloc[batch_idx]
            states = df_to_states(batch_df)
            teacher_probs = soft_labels[batch_idx]

            out = student.forward(states)
            student_logits = out[0].logits if isinstance(out, tuple) else out

            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs.to(student_log_probs.device),
                reduction="batchmean",
            )
            loss = kl_loss * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            preds = student_logits.argmax(dim=-1)
            batch_hard = hard_labels[batch_idx].to(preds.device)
            total_correct += (preds == batch_hard).sum().item()
            total_kl += kl_loss.item() * len(batch_idx)
            total_samples += len(batch_idx)

        acc = total_correct / max(total_samples, 1)
        avg_kl = total_kl / max(total_samples, 1)
        log.info("distill.student_epoch", epoch=epoch,
                 kl_loss=round(avg_kl, 4), hard_accuracy=round(acc, 4))
        epoch_logs.append({"epoch": epoch, "kl_loss": avg_kl, "hard_accuracy": acc})

    return epoch_logs


# ── Step 6: Evaluate and compare ─────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    eval_df: pd.DataFrame,
    model_name: str,
    *,
    batch_size: int = 256,
) -> dict:
    """Evaluate a model on the eval dataset. Returns accuracy, loss, per-class."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    per_class_correct: dict[str, int] = {n: 0 for n in BUY_PLAN_NAMES}
    per_class_total: dict[str, int] = {n: 0 for n in BUY_PLAN_NAMES}

    for start in range(0, len(eval_df), batch_size):
        batch_df = eval_df.iloc[start : start + batch_size]
        if len(batch_df) < 2:
            continue

        states = df_to_states(batch_df)
        labels = torch.tensor(batch_df["label"].values, dtype=torch.long)

        out = model.forward(states)
        logits = out[0].logits if isinstance(out, tuple) else out
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

    acc = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(total_samples, 1)
    per_class_acc = {
        n: round(per_class_correct[n] / per_class_total[n], 3)
        for n in BUY_PLAN_NAMES if per_class_total[n] > 0
    }
    log.info("distill.eval", model=model_name,
             loss=round(avg_loss, 4), accuracy=round(acc, 4))
    return {"accuracy": acc, "loss": avg_loss, "per_class_accuracy": per_class_acc}


def print_comparison_table(teacher_results: dict, student_results: dict) -> None:
    """Print a formatted comparison table of teacher vs student results."""
    print("\n" + "=" * 70)
    print("  DISTILLATION RESULTS: Rate Teacher → Spiking Student")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Teacher (Rate)':<20} {'Student (Spiking)':<20}")
    print("-" * 70)
    print(f"  {'Overall Accuracy':<25} "
          f"{teacher_results['accuracy']:>18.1%}  "
          f"{student_results['accuracy']:>18.1%}")
    print(f"  {'Cross-Entropy Loss':<25} "
          f"{teacher_results['loss']:>18.4f}  "
          f"{student_results['loss']:>18.4f}")
    print("-" * 70)
    print(f"  {'Per-Class Accuracy:':<25}")
    t_pca = teacher_results.get("per_class_accuracy", {})
    s_pca = student_results.get("per_class_accuracy", {})
    for name in BUY_PLAN_NAMES:
        t_v = t_pca.get(name, 0.0)
        s_v = s_pca.get(name, 0.0)
        print(f"    {name:<23} {t_v:>18.1%}  {s_v:>18.1%}")
    print("=" * 70)
    gap = teacher_results["accuracy"] - student_results["accuracy"]
    print(f"\n  Accuracy gap: {gap:+.1%} (teacher - student)")
    if gap > 0.05:
        print("  → Student lags teacher by >5 pp. Consider more epochs or lower T.")
    elif gap < 0.02:
        print("  → Student nearly matches teacher. Distillation successful!")
    print()


# ── Full pipeline ────────────────────────────────────────────────────────────


def run_distillation(
    train_path: str | Path = "data/cs2_pro_train.parquet",
    eval_path: str | Path = "data/cs2_pro_eval.parquet",
    connectome_cache: str = "cache/connectome/",
    teacher_epochs: int = 3,
    student_epochs: int = 5,
    teacher_lr: float = 1e-3,
    student_lr: float = 1e-3,
    batch_size: int = 256,
    temperature: float = 1.0,
    n_neurons_fallback: int = 200,
    density_fallback: float = 0.1,
    seed: int = 42,
) -> dict:
    """Execute the full distillation pipeline end-to-end."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    log.info("distill.loading_data", train=str(train_path), eval=str(eval_path))
    train_df = load_pro_rounds(train_path)
    eval_df = load_pro_rounds(eval_path)
    log.info("distill.data_loaded", train=len(train_df), eval=len(eval_df))

    # Step 1: Build rate-model teacher
    log.info("distill.building_teacher")
    try:
        teacher = build_reservoir(cache_dir=connectome_cache)
    except (FileNotFoundError, Exception) as exc:
        log.warning("distill.teacher_fallback", error=str(exc))
        teacher = _build_synthetic_teacher(n_neurons_fallback, density_fallback, seed)

    # Step 2: Train teacher
    teacher_metrics = train_teacher(
        teacher, train_df,
        n_epochs=teacher_epochs, lr=teacher_lr, batch_size=batch_size,
    )

    # Step 3: Generate soft labels
    soft_labels = generate_soft_labels(
        teacher, train_df, batch_size=batch_size, temperature=temperature,
    )

    # Step 4: Build spiking student
    student = _build_student(connectome_cache, n_neurons_fallback, density_fallback, seed)

    # Step 5: Train student with KL divergence
    student_metrics = train_student(
        student, train_df, soft_labels,
        n_epochs=student_epochs, lr=student_lr,
        batch_size=batch_size, temperature=temperature,
    )

    # Step 6: Evaluate both
    teacher_eval = evaluate_model(teacher, eval_df, "teacher")
    student_eval = evaluate_model(student, eval_df, "student")
    print_comparison_table(teacher_eval, student_eval)

    return {
        "teacher_training": teacher_metrics,
        "student_training": student_metrics,
        "teacher_eval": teacher_eval,
        "student_eval": student_eval,
    }


# ── Internal helpers ─────────────────────────────────────────────────────────


def _build_synthetic_teacher(n: int, density: float, seed: int) -> FlyReservoir:
    """Build a FlyReservoir with a synthetic random connectome (fallback)."""
    from scipy import sparse

    rng = np.random.default_rng(seed)
    n_edges = int(n * n * density)
    rows = rng.integers(0, n, n_edges)
    cols = rng.integers(0, n, n_edges)
    vals = rng.standard_normal(n_edges).astype(np.float32)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    abs_sums = np.maximum(np.array(np.abs(W).sum(axis=1)).flatten(), 1.0)
    W = (sparse.diags(1.0 / abs_sums, format="csr") @ W).astype(np.float32)
    return FlyReservoir(n, W, device="cpu")


def _build_student(cache: str, n_fb: int, d_fb: float, seed: int) -> nn.Module:
    """Build spiking student (mushroom body or synthetic fallback)."""
    from flyecon.etl.controls import random_sparse
    from flyecon.policy.ppo import build_fly_policy
    from flyecon.sim.calibration import calibrate_gain

    try:
        from flyecon.etl.loader import load_connectome
        from flyecon.etl.subcircuit import extract_mushroom_body
        conn = extract_mushroom_body(load_connectome(cache), hops=1)
        log.info("distill.student_connectome", source="mushroom-body", n=conn.n_neurons)
    except (FileNotFoundError, Exception) as exc:
        log.warning("distill.student_fallback", error=str(exc))
        conn = random_sparse(n_neurons=n_fb, density=d_fb, seed=seed)

    cal = calibrate_gain(conn, target_rate_hz=(1.0, 10.0), duration_ms=500.0)
    log.info("distill.student_gain", gain=cal.gain, rate_hz=cal.mean_rate_hz)
    return build_fly_policy(conn, gain=cal.gain, duration_ms=400.0)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_distillation()

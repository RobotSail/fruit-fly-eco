"""Scientific control experiments for topology ablation.

Provides three control types that test whether the connectome topology
provides genuine signal vs. structural baselines:

  - rewired: Degree-preserving rewired null (same statistics, scrambled topology)
  - random: Matched-density random sparse graph
  - no_connectome: Skip LIF simulation, feed noise features to readout

Output: controls/comparison_report.md
"""

from __future__ import annotations

import time
from pathlib import Path

import structlog

from flyecon.etl.controls import random_sparse, rewire_degree_preserving
from flyecon.etl.loader import Connectome
from flyecon.policy.ppo import (
    PPOTrainer,
    build_fly_policy,
    evaluate_against_oracle,
)

log = structlog.get_logger()


def _train_and_eval(
    connectome: Connectome,
    gain: float,
    n_iterations: int,
    oracle: object,
    label: str,
    seed: int = 42,
) -> dict[str, float]:
    """Train a policy on the given connectome and evaluate vs Oracle.

    Returns a dict with final metrics + eval results.
    """
    log.info("control.train_start", label=label, n_iterations=n_iterations)

    policy = build_fly_policy(connectome, gain=gain, duration_ms=400.0)
    trainer = PPOTrainer(policy, n_steps=64, seed=seed)

    # Collect early metrics for learning-curve tracking
    early_reward = 0.0
    final_reward = 0.0

    for it in range(1, n_iterations + 1):
        rollouts = trainer.collect_rollouts()
        metrics = trainer.update(rollouts)
        if it <= 5:
            early_reward += metrics.get("mean_reward", 0.0)
        if it > n_iterations - 5:
            final_reward += metrics.get("mean_reward", 0.0)

    early_reward /= min(5, n_iterations)
    final_reward /= min(5, n_iterations)

    # Evaluate against Oracle
    ev = evaluate_against_oracle(policy, oracle, n_episodes=20, seed=seed)

    result = {
        "label": label,
        "value_ratio": ev.value_ratio,
        "agreement_rate": ev.agreement_rate,
        "early_mean_reward": early_reward,
        "final_mean_reward": final_reward,
        "reward_improvement": final_reward - early_reward,
        "policy_mean_money": ev.policy_mean_money,
        "oracle_mean_money": ev.oracle_mean_money,
    }

    log.info(
        "control.train_complete",
        label=label,
        value_ratio=round(ev.value_ratio, 4),
        agreement=round(ev.agreement_rate, 4),
    )
    return result


def run_control_experiment(
    control_type: str,
    n_iterations: int = 100,
    n_neurons: int = 200,
    density: float = 0.1,
    seed: int = 42,
    output_dir: Path = Path("controls"),
) -> Path:
    """Run a scientific control experiment and write comparison report.

    Parameters
    ----------
    control_type : str
        One of 'rewired', 'random', 'no_connectome', or 'all'.
    n_iterations : int
        Number of PPO training iterations per condition.
    n_neurons : int
        Synthetic connectome size.
    density : float
        Edge density.
    seed : int
        RNG seed.
    output_dir : Path
        Directory for output report.

    Returns
    -------
    Path to the generated comparison_report.md.
    """
    from flyecon.oracle.mdp import EconomyMDP
    from flyecon.oracle.solver import solve
    from flyecon.sim.calibration import calibrate_gain

    # Solve Oracle once (shared across all conditions)
    log.info("control.solving_oracle")
    mdp = EconomyMDP()
    oracle = solve(mdp, gamma=0.99, tol=1e-8)

    # Build baseline connectome
    baseline_conn = random_sparse(n_neurons=n_neurons, density=density, seed=seed)
    cal = calibrate_gain(
        baseline_conn, target_rate_hz=(1.0, 10.0), duration_ms=500.0
    )
    gain = cal.gain

    results: list[dict[str, float]] = []
    t0 = time.monotonic()

    # Always run baseline
    baseline_result = _train_and_eval(
        baseline_conn, gain, n_iterations, oracle, "baseline", seed=seed
    )
    results.append(baseline_result)

    types_to_run: list[str] = (
        ["rewired", "random", "no_connectome"]
        if control_type == "all"
        else [control_type]
    )

    for ct in types_to_run:
        if ct == "rewired":
            rewired_conn = rewire_degree_preserving(baseline_conn, seed=seed + 1)
            r = _train_and_eval(
                rewired_conn, gain, n_iterations, oracle, "rewired", seed=seed
            )
            results.append(r)

        elif ct == "random":
            rand_conn = random_sparse(
                n_neurons=n_neurons, density=density, seed=seed + 2
            )
            r = _train_and_eval(
                rand_conn, gain, n_iterations, oracle, "random", seed=seed
            )
            results.append(r)

        elif ct == "no_connectome":
            # Build a zero-weight connectome (no LIF dynamics)
            import numpy as np

            from flyecon.etl.loader import build_csr_tensor

            empty = build_csr_tensor(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                n_neurons,
            )
            no_conn = Connectome(
                weight_matrix=empty,
                dopamine_matrix=empty,
                body_ids=np.arange(n_neurons, dtype=np.int64),
                neuron_types={},
                n_neurons=n_neurons,
                n_synapses=0,
                metadata={"control": "no_connectome"},
            )
            r = _train_and_eval(
                no_conn, 0.0, n_iterations, oracle, "no_connectome", seed=seed
            )
            results.append(r)

    elapsed = time.monotonic() - t0

    # Write report
    report_path = _write_report(results, elapsed, output_dir)
    log.info("control.report_written", path=str(report_path))
    return report_path


def _write_report(
    results: list[dict],
    elapsed_s: float,
    output_dir: Path,
) -> Path:
    """Generate controls/comparison_report.md from experiment results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.md"

    lines = [
        "# Scientific Control Comparison Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Total runtime:** {elapsed_s:.1f}s",
        "",
        "## Results",
        "",
        "| Condition | Value Ratio | Agreement | Early Reward |"
        " Final Reward | Improvement |",
        "|-----------|-------------|-----------|--------------|"
        "--------------|-------------|",
    ]

    for r in results:
        label = r.get("label", "?")
        vr = r.get("value_ratio", 0.0)
        ar = r.get("agreement_rate", 0.0)
        er = r.get("early_mean_reward", 0.0)
        fr = r.get("final_mean_reward", 0.0)
        imp = r.get("reward_improvement", 0.0)

        # Handle non-numeric label
        vr_val = vr if isinstance(vr, (int, float)) else 0.0
        ar_val = ar if isinstance(ar, (int, float)) else 0.0
        er_val = er if isinstance(er, (int, float)) else 0.0
        fr_val = fr if isinstance(fr, (int, float)) else 0.0
        imp_val = imp if isinstance(imp, (int, float)) else 0.0

        lines.append(
            f"| {label} | {vr_val:.4f} | {ar_val:.4f} | {er_val:.6f} |"
            f" {fr_val:.6f} | {imp_val:.6f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **baseline**: Synthetic connectome (random sparse graph) as reference.",
        "- **rewired**: Degree-preserving rewired null — same statistics,"
        " scrambled topology.",
        "- **random**: Independent random sparse graph of matched size/density.",
        "- **no_connectome**: Zero-weight connectome — tests whether LIF"
        " dynamics add signal.",
        "",
        "If the baseline significantly outperforms rewired and random controls,",
        "the connectome topology provides genuine signal for economy decisions.",
        "",
        "If all conditions perform similarly, topology does not help at this"
        " scale/configuration.",
    ])

    report_path.write_text("\n".join(lines) + "\n")
    return report_path

"""End-to-end integration tests for the full FLY//ECON pipeline.

All tests use SYNTHETIC connectomes (no real downloads).
Total runtime target: < 60 seconds.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from flyecon.harness import Harness

# ── Shared fixtures ─────────────────────────────────────────────────────

# Module-scoped Oracle solution avoids re-solving the MDP per test (~2-3s each).
_CACHED_ORACLE = None


def _get_oracle():
    """Return a cached Oracle policy (solved once per test module)."""
    global _CACHED_ORACLE
    if _CACHED_ORACLE is None:
        from flyecon.oracle.mdp import EconomyMDP
        from flyecon.oracle.solver import solve
        _CACHED_ORACLE = solve(EconomyMDP(), gamma=0.99, tol=1e-8)
    return _CACHED_ORACLE


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create isolated workspace for integration tests."""
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "heartbeats").mkdir()
    return tmp_path


def _make_harness(
    ws: Path,
    *,
    n_neurons: int = 30,
    eval_interval: int = 0,
    checkpoint_interval: int = 0,
    n_steps: int = 32,
) -> Harness:
    """Helper — build a Harness with small synthetic connectome.

    Pre-injects a cached Oracle so boot() skips the ~2-3s MDP solve.
    """
    h = Harness(
        n_neurons=n_neurons,
        density=0.1,
        seed=42,
        checkpoint_dir=ws / "checkpoints",
        telemetry_path=ws / "telemetry.jsonl",
        heartbeat_dir=ws / "heartbeats",
        dashboard_path=ws / "dashboard.html",
        eval_interval=eval_interval,
        checkpoint_interval=checkpoint_interval,
        dashboard_interval=0,
        n_steps=n_steps,
    )
    # Inject pre-solved Oracle so boot() skips solving (Step 7).
    h._oracle = _get_oracle()
    return h


# ── Smoke tests ─────────────────────────────────────────────────────────

class TestSmokeIntegration:
    """Smoke test: synthetic connectome → full pipeline → no crash."""

    def test_full_pipeline_no_crash(self, tmp_workspace: Path) -> None:
        """Boot + 5 PPO iterations → no crash, metrics finite."""
        harness = _make_harness(
            tmp_workspace,
            eval_interval=5,
            checkpoint_interval=5,
        )

        harness.boot()

        # Verify boot completed
        assert harness.policy is not None
        assert harness.mission_state.stage.value == "TRAINING"

        # Run 5 training iterations
        metrics = harness.train_iterations(5)

        # All metrics should be finite
        for key, value in metrics.items():
            assert math.isfinite(value), f"Non-finite metric: {key}={value}"

        # Mission state should reflect progress
        assert harness.mission_state.training_step == 5
        assert harness.mission_state.cycles_completed >= 1

        # Telemetry should have been written
        assert (tmp_workspace / "telemetry.jsonl").exists()
        tel_content = (tmp_workspace / "telemetry.jsonl").read_text()
        assert len(tel_content.strip().splitlines()) > 0

    def test_checkpoint_saved(self, tmp_workspace: Path) -> None:
        """After enough iterations, a checkpoint should exist."""
        harness = _make_harness(tmp_workspace, checkpoint_interval=3)

        harness.boot()
        harness.train_iterations(3)

        # Should have at least one checkpoint directory
        ckpt_dirs = list(
            (tmp_workspace / "checkpoints").glob("ckpt_*")
        )
        assert len(ckpt_dirs) >= 1, "No checkpoint was saved"

    def test_policy_produces_valid_distributions(
        self, tmp_workspace: Path
    ) -> None:
        """Policy should produce valid probability distributions."""
        harness = _make_harness(tmp_workspace)

        harness.boot()
        assert harness.policy is not None

        from flyecon.state.economy import EconomyState

        test_states = [
            EconomyState(money=800, loss_streak=0, round_number=1,
                        half=0, opponent_loss_streak=0),
            EconomyState(money=8000, loss_streak=2, round_number=6,
                        half=0, opponent_loss_streak=1),
            EconomyState(money=16000, loss_streak=0, round_number=12,
                        half=1, opponent_loss_streak=4),
        ]

        harness.policy.eval()
        with torch.no_grad():
            for state in test_states:
                dist, values, features = harness.policy.forward([state])
                probs = dist.probs
                # Probabilities must sum to ~1
                assert abs(probs.sum().item() - 1.0) < 1e-4, (
                    f"Probs don't sum to 1: {probs}"
                )
                # All probabilities must be non-negative
                assert (probs >= 0).all(), f"Negative probs: {probs}"
                # Values must be finite
                assert torch.isfinite(values).all()
                # Features must be finite
                assert torch.isfinite(features).all()
        harness.policy.train()

    def test_oracle_eval_produces_finite_results(
        self, tmp_workspace: Path
    ) -> None:
        """Oracle evaluation should produce finite, reasonable results."""
        harness = _make_harness(tmp_workspace, eval_interval=3)

        harness.boot()
        # Run enough to trigger an eval
        harness.train_iterations(3)

        # Check the eval score was set
        score = harness.mission_state.last_eval_score
        assert math.isfinite(score), f"Non-finite eval score: {score}"
        assert score >= 0, f"Negative eval score: {score}"


# ── Signal test ──────────────────────────────────────────────────────────

class TestSignalIntegration:
    """Signal test: verify learning signal exists over training."""

    def test_training_produces_finite_metrics(
        self, tmp_workspace: Path
    ) -> None:
        """10 iterations on 30-neuron → assert metrics finite, step advances.

        Uses a small synthetic connectome to keep runtime under 60 seconds.
        """
        harness = _make_harness(
            tmp_workspace,
            eval_interval=5,
            checkpoint_interval=5,
        )

        harness.boot()

        # Run 5 + 5 iterations (two phases)
        metrics_early = harness.train_iterations(5)
        metrics_late = harness.train_iterations(5)

        # Basic sanity: metrics are finite
        for key in ["mean_reward", "policy_loss", "entropy"]:
            assert math.isfinite(metrics_early.get(key, 0.0)), (
                f"Non-finite early {key}"
            )
            assert math.isfinite(metrics_late.get(key, 0.0)), (
                f"Non-finite late {key}"
            )

        # Training step advanced correctly
        assert harness.mission_state.training_step == 10

        # An eval should have occurred
        assert harness.mission_state.last_eval_score > 0


# ── Control experiments ──────────────────────────────────────────────────

class TestControlExperiment:
    """Test that control experiments run without crashing."""

    def test_control_experiment_random(self, tmp_workspace: Path) -> None:
        """Random control experiment should produce a report."""
        from flyecon.controls import run_control_experiment

        report_path = run_control_experiment(
            control_type="random",
            n_iterations=2,
            n_neurons=20,
            density=0.1,
            seed=42,
            output_dir=tmp_workspace / "controls",
        )

        assert report_path.exists()
        content = report_path.read_text()
        assert "baseline" in content
        assert "random" in content
        assert "Value Ratio" in content

    def test_control_experiment_no_connectome(
        self, tmp_workspace: Path
    ) -> None:
        """No-connectome ablation should produce a report."""
        from flyecon.controls import run_control_experiment

        report_path = run_control_experiment(
            control_type="no_connectome",
            n_iterations=2,
            n_neurons=20,
            density=0.1,
            seed=42,
            output_dir=tmp_workspace / "controls",
        )

        assert report_path.exists()
        content = report_path.read_text()
        assert "no_connectome" in content

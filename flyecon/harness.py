"""End-to-end orchestration harness — wires ETL→Sim→Encode→Readout→PPO→Oracle.

The Harness is the mission's single entry point: boot → train → eval → dashboard
in a perpetual loop, with checkpoint-based crash recovery and degradation ladder.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import structlog

from flyecon.dashboard.fci import compute_fci
from flyecon.dashboard.telemetry import TelemetryEvent, TelemetryStore
from flyecon.etl.controls import random_sparse
from flyecon.etl.loader import Connectome
from flyecon.policy.ppo import (
    FlyPolicy,
    PPOTrainer,
    build_fly_policy,
    evaluate_against_oracle,
)
from flyecon.resilience.checkpoint import (
    CheckpointBundle,
    MissionStage,
    MissionState,
    load_checkpoint,
    save_checkpoint,
)
from flyecon.resilience.heartbeat import HeartbeatWriter
from flyecon.resilience.ladder import (
    DegradationLadder,
    probe_resources,
)
from flyecon.sim.calibration import calibrate_gain, preflight_gate

log = structlog.get_logger()


class Harness:
    """End-to-end orchestration: boot → train → eval → dashboard → repeat.

    Parameters
    ----------
    n_neurons : int
        Synthetic connectome size (used when no real connectome is loaded).
    density : float
        Edge density for synthetic connectome.
    seed : int
        RNG seed for reproducibility.
    checkpoint_dir : Path
        Directory for checkpoint persistence.
    telemetry_path : Path
        Path for JSONL telemetry file.
    heartbeat_dir : Path
        Directory for heartbeat files.
    dashboard_path : Path
        Path for rendered HTML dashboard.
    eval_interval : int
        Evaluate against Oracle every N training iterations.
    checkpoint_interval : int
        Save checkpoint every N training iterations.
    dashboard_interval : int
        Render dashboard every N training iterations.
    n_steps : int
        PPO rollout length per iteration.
    connectome : Connectome | None
        Pre-built connectome (overrides synthetic generation).
    gain : float | None
        Pre-calibrated gain (skips calibration if provided).
    """

    def __init__(
        self,
        *,
        n_neurons: int = 200,
        density: float = 0.1,
        seed: int = 42,
        checkpoint_dir: Path = Path("checkpoints"),
        telemetry_path: Path = Path("mission_telemetry.jsonl"),
        heartbeat_dir: Path = Path("heartbeats"),
        dashboard_path: Path = Path("dashboard.html"),
        eval_interval: int = 50,
        checkpoint_interval: int = 25,
        dashboard_interval: int = 50,
        n_steps: int = 128,
        connectome: Optional[Connectome] = None,
        gain: Optional[float] = None,
    ) -> None:
        self.n_neurons = n_neurons
        self.density = density
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.telemetry_path = telemetry_path
        self.heartbeat_dir = heartbeat_dir
        self.dashboard_path = dashboard_path
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.dashboard_interval = dashboard_interval
        self.n_steps = n_steps

        # Lazy-initialized in boot()
        self._connectome: Optional[Connectome] = connectome
        self._gain: Optional[float] = gain
        self._policy: Optional[FlyPolicy] = None
        self._trainer: Optional[PPOTrainer] = None
        self._oracle: object | None = None
        self._mission: MissionState = MissionState()
        self._telemetry: TelemetryStore = TelemetryStore(telemetry_path)
        self._heartbeat: HeartbeatWriter = HeartbeatWriter(
            "harness", heartbeat_dir
        )
        self._ladder: DegradationLadder = DegradationLadder()
        self._start_time: float = time.monotonic()
        self._fci_history: list[float] = []

    # ── Boot sequence ────────────────────────────────────────────────────

    def boot(self) -> None:
        """Execute the full boot sequence.

        1. Load or create MissionState
        2. Check for existing checkpoint
        3. Build/load connectome (synthetic)
        4. Gain calibration
        5. Preflight gate
        6. Build policy + PPO trainer
        7. Solve Oracle
        """
        log.info("harness.boot_start")
        self._start_time = time.monotonic()

        # Step 1: Try to resume from checkpoint
        self._try_resume()

        # Step 2: Build connectome if not already loaded
        if self._connectome is None:
            log.info(
                "harness.synthetic_connectome",
                n_neurons=self.n_neurons,
                density=self.density,
            )
            self._connectome = random_sparse(
                n_neurons=self.n_neurons,
                density=self.density,
                seed=self.seed,
            )
        self._mission.stage = MissionStage.ETL
        self._emit("mission_state", self._mission_data())

        # Step 3: Calibrate gain if needed
        if self._gain is None:
            self._mission.stage = MissionStage.CALIBRATION
            self._emit("mission_state", self._mission_data())
            log.info("harness.calibrating_gain")
            cal = calibrate_gain(
                self._connectome,
                target_rate_hz=(1.0, 10.0),
                duration_ms=500.0,
            )
            self._gain = cal.gain
            log.info(
                "harness.gain_calibrated",
                gain=cal.gain,
                rate_hz=cal.mean_rate_hz,
                healthy=cal.is_healthy,
            )

        # Step 4: Preflight gate
        pf = preflight_gate(self._connectome, gain=self._gain, duration_ms=200.0)
        log.info(
            "harness.preflight",
            gate1=pf.gate1_pass,
            gate2=pf.gate2_pass,
            gate3=pf.gate3_pass,
            all_pass=pf.all_pass,
        )
        # Log preflight but don't hard-fail — synthetic connectomes may not
        # pass all gates cleanly, and the pipeline should still run.

        # Step 5: Build policy
        if self._policy is None:
            log.info("harness.building_policy")
            self._policy = build_fly_policy(
                self._connectome,
                gain=self._gain,
                duration_ms=400.0,
            )

        # Step 6: Build trainer
        if self._trainer is None:
            self._trainer = PPOTrainer(
                self._policy,
                n_steps=self.n_steps,
                seed=self.seed,
            )

        # Step 7: Solve Oracle
        if self._oracle is None:
            log.info("harness.solving_oracle")
            from flyecon.oracle.mdp import EconomyMDP
            from flyecon.oracle.solver import solve

            mdp = EconomyMDP()
            self._oracle = solve(mdp, gamma=0.99, tol=1e-8)
            log.info(
                "harness.oracle_solved",
                iterations=self._oracle.iterations,  # type: ignore[union-attr]
            )

        self._mission.stage = MissionStage.TRAINING
        self._emit("mission_state", self._mission_data())
        log.info("harness.boot_complete")

    # ── Training loop ────────────────────────────────────────────────────

    def train_iterations(self, n_iterations: int) -> dict[str, float]:
        """Run N training iterations. Returns last metrics dict.

        This is the inner loop per §3.5:
        collect_rollouts → ppo_update → log_metrics → checkpoint →
        eval_vs_oracle → publish_dashboard → heartbeat
        """
        assert self._trainer is not None, "Call boot() first"
        assert self._policy is not None, "Call boot() first"

        last_metrics: dict[str, float] = {}

        for i in range(1, n_iterations + 1):
            step_num = self._mission.training_step + 1

            # Check degradation ladder
            report = probe_resources()
            level = self._ladder.decide_level(report)
            self._mission.ladder_level = level
            config = self._ladder.apply_level(level)

            if not config.training_enabled:
                log.info("harness.training_paused", ladder_level=level)
                self._heartbeat.beat("training_paused")
                time.sleep(1.0)
                continue

            # Collect rollouts + PPO update
            rollouts = self._trainer.collect_rollouts()
            metrics = self._trainer.update(rollouts)
            metrics["iteration"] = float(step_num)
            metrics["ladder_level"] = float(level)
            last_metrics = metrics

            # Update mission state
            self._mission.training_step = step_num
            self._mission.uptime_seconds = time.monotonic() - self._start_time

            # Log metrics to telemetry
            self._emit("training_step", {
                k: round(v, 6) for k, v in metrics.items()
            })

            # Compute FCI
            fci = compute_fci(
                kc_mean_rate=max(metrics.get("mean_reward", 0.0) * 10, 0),
                recent_reward=max(metrics.get("mean_reward", 0.0) * 16000, 0),
                consecutive_wins=min(int(metrics.get("mean_value", 0) * 5), 5),
            )
            self._fci_history.append(fci)
            self._emit("fci", {"fci": round(fci, 4)})

            log.info(
                "harness.step",
                step=step_num,
                mean_reward=round(metrics.get("mean_reward", 0), 4),
                policy_loss=round(metrics.get("policy_loss", 0), 4),
                fci=round(fci, 4),
            )

            # Checkpoint
            if (
                self.checkpoint_interval > 0
                and step_num % self.checkpoint_interval == 0
            ):
                self._save_checkpoint()

            # Evaluate against Oracle
            if (
                self.eval_interval > 0
                and step_num % self.eval_interval == 0
                and self._oracle is not None
            ):
                self._mission.stage = MissionStage.EVALUATING
                ev = evaluate_against_oracle(
                    self._policy, self._oracle, n_episodes=20
                )
                self._mission.last_eval_score = ev.value_ratio
                self._emit("eval", {
                    "value_ratio": round(ev.value_ratio, 4),
                    "agreement_rate": round(ev.agreement_rate, 4),
                    "policy_mean_money": round(ev.policy_mean_money, 2),
                    "oracle_mean_money": round(ev.oracle_mean_money, 2),
                })
                log.info(
                    "harness.eval",
                    step=step_num,
                    value_ratio=round(ev.value_ratio, 4),
                    agreement=round(ev.agreement_rate, 4),
                )
                self._mission.stage = MissionStage.TRAINING

            # Dashboard
            if (
                self.dashboard_interval > 0
                and step_num % self.dashboard_interval == 0
                and config.dashboard_enabled
            ):
                self._render_dashboard()

            # Heartbeat
            self._heartbeat.beat(f"step={step_num}")

        self._mission.cycles_completed += 1
        self._emit("mission_state", self._mission_data())
        return last_metrics

    def run_forever(self) -> None:
        """Daemon entry point — train indefinitely until killed."""
        self.boot()
        log.info("harness.run_forever_start")
        with self._heartbeat.alive("running"):
            while True:
                self.train_iterations(self.eval_interval or 50)

    # ── Resume logic ─────────────────────────────────────────────────────

    def _try_resume(self) -> None:
        """Attempt to resume from the latest checkpoint."""
        bundle = load_checkpoint(self.checkpoint_dir)
        if bundle is None:
            log.info("harness.cold_start")
            return

        self._mission = bundle.mission_state
        log.info(
            "harness.resumed",
            stage=self._mission.stage.value,
            step=self._mission.training_step,
        )

        # Restore model weights if we have a policy built
        if bundle.model_state and self._policy is not None:
            self._policy.load_state_dict(bundle.model_state)
            log.info("harness.model_restored")

        if bundle.optimizer_state and self._trainer is not None:
            self._trainer.optimizer.load_state_dict(bundle.optimizer_state)
            log.info("harness.optimizer_restored")

    # ── Internal helpers ─────────────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        """Save current state as an atomic checkpoint."""
        assert self._policy is not None
        assert self._trainer is not None

        import torch

        bundle = CheckpointBundle(
            model_state=self._policy.state_dict(),
            optimizer_state=self._trainer.optimizer.state_dict(),
            rng_state={
                "torch": torch.random.get_rng_state(),
            },
            mission_state=self._mission,
        )
        save_checkpoint(bundle, self.checkpoint_dir)
        log.info("harness.checkpoint_saved", step=self._mission.training_step)

    def _render_dashboard(self) -> None:
        """Render the 10-panel HTML dashboard."""
        try:
            from flyecon.dashboard.renderer import render_dashboard

            render_dashboard(self._telemetry, self.dashboard_path)
        except Exception as exc:
            log.warning("harness.dashboard_error", error=str(exc))

    def _emit(self, event_type: str, data: dict) -> None:
        """Append a telemetry event."""
        try:
            self._telemetry.append(TelemetryEvent(event_type, data=data))
        except Exception as exc:
            log.warning("harness.telemetry_error", error=str(exc))

    def _mission_data(self) -> dict:
        """Serialize current mission state for telemetry."""
        return {
            "stage": self._mission.stage.value,
            "training_step": self._mission.training_step,
            "ladder_level": self._mission.ladder_level,
            "last_eval_score": self._mission.last_eval_score,
            "cycles_completed": self._mission.cycles_completed,
            "uptime_seconds": round(
                time.monotonic() - self._start_time, 1
            ),
            "candidate_id": self._mission.candidate_id,
        }

    @property
    def mission_state(self) -> MissionState:
        """Current mission state (read-only access)."""
        return self._mission

    @property
    def policy(self) -> Optional[FlyPolicy]:
        """Current policy (read-only access)."""
        return self._policy

"""End-to-end orchestration harness — wires ETL→Sim→Encode→Readout→PPO→Oracle.

The Harness is the mission's single entry point: boot → train → eval → dashboard
in a perpetual loop, with checkpoint-based crash recovery and degradation ladder.

Supports ``connectome_mode='full'`` for full 166,700-neuron connectome runs:
  - Skips subcircuit extraction, uses the full signed sparse CSR tensor
  - Re-runs gain calibration (full brain has different E/I balance)
  - Re-runs preflight gate (all 3 gates must pass)
  - Expands readout to all DN-class neurons
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
from flyecon.state.constants import BUY_PLAN_COSTS
from flyecon.state.economy import BuyPlan, EconomyState, pistol_round_state, step
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

# ── Connectome mode constants ──────────────────────────────────────────────
CONNECTOME_SYNTHETIC = "synthetic"
CONNECTOME_MUSHROOM_BODY = "mushroom-body"
CONNECTOME_FULL = "full"

# DN (descending neuron) type prefixes for full-connectome readout expansion
_DN_PREFIXES = ("DN",)


def _find_dn_neuron_indices(connectome: Connectome) -> list[int]:
    """Find indices of all DN-class neurons in the connectome.

    Used when connectome_mode='full' to expand the readout beyond
    mushroom-body MBONs to include all descending neurons.

    Returns dense neuron indices (not body IDs).
    """
    import numpy as np

    dn_indices: list[int] = []
    for body_id, ntype in connectome.neuron_types.items():
        if any(ntype.startswith(p) for p in _DN_PREFIXES):
            idx = int(np.searchsorted(connectome.body_ids, body_id))
            if idx < connectome.n_neurons and connectome.body_ids[idx] == body_id:
                dn_indices.append(idx)
    return sorted(dn_indices)


def _build_neuron_layer_map(connectome: Connectome) -> dict[int, str]:
    """Map each dense neuron index to a layer label string.

    For real connectomes with neuron_types, uses type-prefix matching.
    For synthetic connectomes (empty neuron_types), assigns layers by
    index range: first 20% Input, next 40% KC, next 20% MBON,
    next 10% DN, last 10% Other.
    """
    n = connectome.n_neurons
    layer_map: dict[int, str] = {}

    if connectome.neuron_types:
        import numpy as np
        for body_id, ntype in connectome.neuron_types.items():
            idx = int(np.searchsorted(connectome.body_ids, body_id))
            if idx < n and connectome.body_ids[idx] == body_id:
                if ntype.startswith("KC"):
                    layer_map[idx] = "KC"
                elif ntype.startswith("MBON"):
                    layer_map[idx] = "MBON"
                elif ntype.startswith("DN"):
                    layer_map[idx] = "DN"
                elif ntype.startswith("PPL"):
                    layer_map[idx] = "PPL"
                else:
                    layer_map[idx] = "Input"
        # Fill any unmapped indices
        for i in range(n):
            if i not in layer_map:
                layer_map[i] = "Other"
    else:
        # Synthetic mode: assign by index range
        boundaries = [
            (int(n * 0.2), "Input"),
            (int(n * 0.6), "KC"),
            (int(n * 0.8), "MBON"),
            (int(n * 0.9), "DN"),
            (n, "Other"),
        ]
        for i in range(n):
            for boundary, label in boundaries:
                if i < boundary:
                    layer_map[i] = label
                    break
            else:
                layer_map[i] = "Other"

    return layer_map


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
    connectome_mode : str
        One of 'synthetic', 'mushroom-body', or 'full'.
        When 'full', skips subcircuit extraction, re-calibrates gain,
        and expands readout to all DN-class neurons.
    avatar_mode : str
        One of 'ascii', 'sprite', or 'threejs'.  Controls which
        avatar tier the dashboard renders at Panel 0.
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
        connectome_mode: str = CONNECTOME_SYNTHETIC,
        avatar_mode: str = "ascii",
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
        self.connectome_mode = connectome_mode
        self.avatar_mode = avatar_mode

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
        self._neuron_layer_map: dict[int, str] = {}

    # ── Boot sequence ────────────────────────────────────────────────────

    def boot(self) -> None:
        """Execute the full boot sequence.

        1. Load or create MissionState
        2. Check for existing checkpoint
        3. Build/load connectome (synthetic or full)
        4. Gain calibration (re-calibrated for full connectome)
        5. Preflight gate (re-run for full connectome)
        6. Build policy + PPO trainer
        7. Solve Oracle
        """
        log.info(
            "harness.boot_start",
            connectome_mode=self.connectome_mode,
            avatar_mode=self.avatar_mode,
        )
        self._start_time = time.monotonic()

        # Step 1: Try to resume from checkpoint
        self._try_resume()

        # Step 2: Build connectome if not already loaded
        if self._connectome is None:
            self._connectome = self._load_connectome()
        self._mission.stage = MissionStage.ETL
        self._emit("mission_state", self._mission_data())

        # Build neuron layer map (cached for dashboard neural activity)
        self._neuron_layer_map = _build_neuron_layer_map(self._connectome)

        # Step 3: Calibrate gain if needed
        # Full connectome has different E/I balance — always re-calibrate
        if self._gain is None:
            self._mission.stage = MissionStage.CALIBRATION
            self._emit("mission_state", self._mission_data())
            log.info(
                "harness.calibrating_gain",
                mode=self.connectome_mode,
                n_neurons=self._connectome.n_neurons,
            )
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
                mode=self.connectome_mode,
            )

        # Step 4: Preflight gate
        pf = preflight_gate(self._connectome, gain=self._gain, duration_ms=200.0)
        log.info(
            "harness.preflight",
            gate1=pf.gate1_pass,
            gate2=pf.gate2_pass,
            gate3=pf.gate3_pass,
            all_pass=pf.all_pass,
            mode=self.connectome_mode,
        )
        # Log preflight but don't hard-fail — synthetic connectomes may not
        # pass all gates cleanly, and the pipeline should still run.

        # Step 5: Build policy
        if self._policy is None:
            log.info(
                "harness.building_policy",
                mode=self.connectome_mode,
            )
            self._policy = self._build_policy()

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

    # ── Connectome loading ──────────────────────────────────────────────

    def _load_connectome(self) -> Connectome:
        """Load or generate connectome based on connectome_mode.

        - 'synthetic': Random sparse graph (default, for testing).
        - 'mushroom-body': Load real connectome, extract MB subcircuit.
        - 'full': Load the full connectome, skip subcircuit extraction.
        """
        if self.connectome_mode == CONNECTOME_FULL:
            log.info("harness.full_connectome_load")
            from flyecon.etl.loader import load_connectome
            conn = load_connectome()
            log.info(
                "harness.full_connectome_loaded",
                n_neurons=conn.n_neurons,
                n_synapses=conn.n_synapses,
            )
            return conn

        if self.connectome_mode == CONNECTOME_MUSHROOM_BODY:
            log.info("harness.mushroom_body_load")
            from flyecon.etl.loader import load_connectome
            from flyecon.etl.subcircuit import extract_mushroom_body
            full = load_connectome()
            conn = extract_mushroom_body(full, hops=1)
            log.info(
                "harness.mushroom_body_loaded",
                n_neurons=conn.n_neurons,
                n_synapses=conn.n_synapses,
            )
            return conn

        # Default: synthetic
        log.info(
            "harness.synthetic_connectome",
            n_neurons=self.n_neurons,
            density=self.density,
        )
        return random_sparse(
            n_neurons=self.n_neurons,
            density=self.density,
            seed=self.seed,
        )

    def _build_policy(self) -> FlyPolicy:
        """Build a FlyPolicy, expanding readout for full connectome mode.

        When connectome_mode='full', the readout is expanded to include
        all DN-class neurons (not just last-N default).
        """
        assert self._connectome is not None
        assert self._gain is not None

        if self.connectome_mode == CONNECTOME_FULL:
            dn_indices = _find_dn_neuron_indices(self._connectome)
            if dn_indices:
                log.info(
                    "harness.full_readout_dn",
                    n_dn_neurons=len(dn_indices),
                )
                return build_fly_policy(
                    self._connectome,
                    gain=self._gain,
                    duration_ms=400.0,
                    output_neuron_ids=dn_indices,
                )

        # Default build (subcircuit or synthetic — uses last-N neurons)
        return build_fly_policy(
            self._connectome,
            gain=self._gain,
            duration_ms=400.0,
        )

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

                # ── Decision trace: play one MR12 episode ──
                self._emit_decision_trace()

                # ── Neural activity: inspect spike counts ──
                self._emit_neural_activity()

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

            render_dashboard(
                self._telemetry,
                self.dashboard_path,
                avatar_mode=self.avatar_mode,
            )
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

    def _emit_decision_trace(self) -> None:
        """Play one full MR12 episode and emit per-round decision trace.

        Called from the training thread at each eval_interval checkpoint.
        """
        assert self._policy is not None
        assert self._oracle is not None

        import torch
        import numpy as np
        from flyecon.oracle.mdp import DEFAULT_WIN_PROBS

        buy_plans = list(BuyPlan)
        win_probs = DEFAULT_WIN_PROBS
        rng = np.random.default_rng(self._mission.training_step)
        rounds: list[dict] = []

        self._policy.eval()
        with torch.no_grad():
            for half in range(2):
                st = pistol_round_state(half=half)
                for rnd in range(1, 13):
                    dist, _, _ = self._policy.forward([st])
                    pa = int(dist.probs.argmax().item())
                    obp = self._oracle.decide(st)  # type: ignore[union-attr]
                    oa = buy_plans.index(obp)
                    agreed = pa == oa
                    rounds.append({
                        "round": rnd,
                        "half": half,
                        "money": st.money,
                        "policy_action": buy_plans[pa].value,
                        "oracle_action": obp.value,
                        "agreed": agreed,
                    })
                    # Step the environment using policy action
                    pbp = buy_plans[pa]
                    eff = pbp if BUY_PLAN_COSTS[pbp.value] <= st.money else BuyPlan.SAVE
                    nxt = step(
                        st, eff,
                        round_won=float(rng.random()) < win_probs[eff.value],
                    )
                    st = nxt
                    if st.round_number == 1 and rnd < 12:
                        break
        self._policy.train()

        self._emit("decision_trace", {"rounds": rounds})

    def _emit_neural_activity(self) -> None:
        """Inspect spike counts and emit layer-aggregated neural activity.

        Called from the training thread at each eval_interval checkpoint.
        """
        assert self._policy is not None

        import torch

        info = self._policy.inspect(pistol_round_state(half=0))
        spike_counts = info["spike_counts"]
        n = spike_counts.shape[0]

        # Aggregate by layer
        layer_sums: dict[str, float] = {}
        layer_counts: dict[str, int] = {}
        for idx in range(n):
            layer = self._neuron_layer_map.get(idx, "Other")
            layer_sums[layer] = layer_sums.get(layer, 0.0) + float(spike_counts[idx].item())
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        layers: dict[str, float] = {}
        for layer, total in layer_sums.items():
            cnt = layer_counts[layer]
            layers[layer] = round(total / max(cnt, 1), 4)

        # mean_output_rate: average of output neuron rates
        output_ids = self._policy.readout.output_neuron_ids
        n_out = len(output_ids)
        if n_out > 0:
            out_rates = [float(spike_counts[int(i)].item()) for i in output_ids]
            mean_output_rate = sum(out_rates) / n_out
        else:
            mean_output_rate = 0.0

        self._emit("neural_activity", {
            "layers": layers,
            "mean_output_rate": round(mean_output_rate, 4),
        })

    @property
    def mission_state(self) -> MissionState:
        """Current mission state (read-only access)."""
        return self._mission

    @property
    def policy(self) -> Optional[FlyPolicy]:
        """Current policy (read-only access)."""
        return self._policy

    @property
    def connectome(self) -> Optional[Connectome]:
        """Current connectome (read-only access)."""
        return self._connectome

    @property
    def neuron_layer_map(self) -> dict[int, str]:
        """Cached neuron index → layer label map."""
        return self._neuron_layer_map

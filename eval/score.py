"""Factory eval harness for PROJECT FLY//ECON.

Six dimensions, each returning {"score": float, "details": str}.
Phase 1 implements economy_mdp; the rest are stubs returning 0.0.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from flyecon.state.constants import (
    BUY_PLAN_COSTS,
    LOSS_BONUS_LADDER,
    MAX_LOSS_STREAK,
    ROUNDS_PER_HALF,
    STARTING_MONEY,
    WIN_REWARD,
)
from flyecon.state.economy import BuyPlan, EconomyState, pistol_round_state, step


def _dim_result(score: float, details: str) -> dict[str, Any]:
    return {"score": score, "details": details}


def economy_mdp() -> dict[str, Any]:
    """Verify EconomyState transitions match MR12 rules exactly."""
    errors: list[str] = []
    checks_passed = 0
    total_checks = 0

    # Check 1: Pistol round starts at $800
    total_checks += 1
    ps = pistol_round_state()
    if ps.money == STARTING_MONEY:
        checks_passed += 1
    else:
        errors.append(f"Pistol start: expected {STARTING_MONEY}, got {ps.money}")

    # Check 2: Loss bonus ladder progression
    for streak in range(MAX_LOSS_STREAK + 1):
        total_checks += 1
        state = EconomyState(
            money=0,
            loss_streak=streak,
            round_number=1,
            half=0,
            opponent_loss_streak=0,
        )
        next_state = step(state, BuyPlan.SAVE, round_won=False)
        expected_money = LOSS_BONUS_LADDER[streak]
        if next_state.money == expected_money:
            checks_passed += 1
        else:
            errors.append(
                f"Loss bonus streak {streak}: expected {expected_money}, "
                f"got {next_state.money}"
            )

    # Check 3: Win gives base reward
    total_checks += 1
    state = EconomyState(money=0, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0)
    next_state = step(state, BuyPlan.SAVE, round_won=True)
    if next_state.money == WIN_REWARD:
        checks_passed += 1
    else:
        errors.append(f"Win reward: expected {WIN_REWARD}, got {next_state.money}")

    # Check 4: Win resets own loss streak
    total_checks += 1
    state = EconomyState(money=0, loss_streak=3, round_number=1, half=0, opponent_loss_streak=0)
    next_state = step(state, BuyPlan.SAVE, round_won=True)
    if next_state.loss_streak == 0:
        checks_passed += 1
    else:
        errors.append(f"Win should reset loss_streak to 0, got {next_state.loss_streak}")

    # Check 5: Win increments opponent loss streak
    total_checks += 1
    state = EconomyState(money=0, loss_streak=0, round_number=1, half=0, opponent_loss_streak=2)
    next_state = step(state, BuyPlan.SAVE, round_won=True)
    if next_state.opponent_loss_streak == 3:
        checks_passed += 1
    else:
        errors.append(
            f"Win should increment opponent streak, "
            f"expected 3, got {next_state.opponent_loss_streak}"
        )

    # Check 6: Loss resets opponent loss streak
    total_checks += 1
    state = EconomyState(money=0, loss_streak=0, round_number=1, half=0, opponent_loss_streak=3)
    next_state = step(state, BuyPlan.SAVE, round_won=False)
    if next_state.opponent_loss_streak == 0:
        checks_passed += 1
    else:
        errors.append(
            f"Loss should reset opponent streak to 0, got {next_state.opponent_loss_streak}"
        )

    # Check 7: Halftime reset (round 12 → next half)
    total_checks += 1
    state = EconomyState(
        money=5000, loss_streak=3, round_number=ROUNDS_PER_HALF, half=0, opponent_loss_streak=2
    )
    next_state = step(state, BuyPlan.SAVE, round_won=True)
    if (
        next_state.money == STARTING_MONEY
        and next_state.half == 1
        and next_state.round_number == 1
        and next_state.loss_streak == 0
    ):
        checks_passed += 1
    else:
        errors.append(
            f"Halftime reset failed: got money={next_state.money}, "
            f"half={next_state.half}, round={next_state.round_number}"
        )

    # Check 8: Buy plan deducts cost
    total_checks += 1
    state = EconomyState(money=5000, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0)
    next_state = step(state, BuyPlan.FULL_BUY, round_won=True)
    expected = min(5000 - BUY_PLAN_COSTS["FULL_BUY"] + WIN_REWARD, 16_000)
    if next_state.money == expected:
        checks_passed += 1
    else:
        errors.append(f"Full buy cost: expected {expected}, got {next_state.money}")

    score = checks_passed / total_checks if total_checks > 0 else 0.0
    details = f"{checks_passed}/{total_checks} checks passed"
    if errors:
        details += "; ERRORS: " + "; ".join(errors)
    return _dim_result(score, details)


def oracle_solver() -> dict[str, Any]:
    """Verify Oracle value iteration converges, policy is sane, Oracle beats random."""
    errors: list[str] = []
    checks_passed = 0
    total_checks = 0

    try:
        from flyecon.oracle.mdp import EconomyMDP, idx_from_money
        from flyecon.oracle.solver import solve, verify_against_random

        mdp = EconomyMDP()
        policy = solve(mdp, gamma=0.99, tol=1e-8)

        # Check 1: Convergence in < 100 iterations
        total_checks += 1
        if policy.iterations < 100:
            checks_passed += 1
        else:
            errors.append(f"Convergence took {policy.iterations} iterations (expected <100)")

        # Check 2: Policy never chooses FULL_BUY when money < cost
        total_checks += 1
        full_buy_idx = mdp.actions().index(BuyPlan.FULL_BUY)
        full_buy_cost = BUY_PLAN_COSTS["FULL_BUY"]
        max_poor_idx = idx_from_money(full_buy_cost - 100)
        violations = 0
        for s_idx in range(mdp.n_states):
            s = mdp.idx_to_state(s_idx)
            if s.money_idx <= max_poor_idx and policy.policy_table[s_idx] == full_buy_idx:
                violations += 1
        if violations == 0:
            checks_passed += 1
        else:
            errors.append(f"FULL_BUY chosen in {violations} unaffordable states")

        # Check 3: Pistol round action is not FULL_BUY
        total_checks += 1
        ps = pistol_round_state()
        pistol_action = policy.decide(ps)
        if pistol_action != BuyPlan.FULL_BUY:
            checks_passed += 1
        else:
            errors.append("Pistol round chose FULL_BUY (expected ECO or FORCE_BUY)")

        # Check 4: Oracle wins more rounds than random (≥1.0 extra wins,
        # equivalent to ≥$3250 in win-reward value, well above $500)
        total_checks += 1
        result = verify_against_random(policy, n_episodes=500, seed=42)
        if result.wins_advantage >= 1.0:
            checks_passed += 1
        else:
            errors.append(
                f"Oracle wins advantage: {result.wins_advantage:.2f} "
                f"(expected ≥1.0 rounds)"
            )

    except Exception as e:
        total_checks = 1
        errors.append(f"Oracle solver failed: {e}")

    score = checks_passed / total_checks if total_checks > 0 else 0.0
    details = f"{checks_passed}/{total_checks} checks passed"
    if errors:
        details += "; ERRORS: " + "; ".join(errors)
    return _dim_result(score, details)


def connectome_etl() -> dict[str, Any]:
    """Verify ETL sign convention, sparse format, int32 indices with synthetic data."""
    errors: list[str] = []
    checks_passed = 0
    total_checks = 0

    try:
        import pandas as pd
        import torch

        from flyecon.etl.loader import load_connectome_from_tables

        # Build a small synthetic connectome
        weights = pd.DataFrame(
            {
                "bodyId_pre": [1, 1, 2, 3, 4],
                "bodyId_post": [2, 3, 3, 4, 5],
                "weight": [10, 5, 8, 3, 7],
            }
        )
        nt = pd.DataFrame(
            {
                "bodyId": [1, 2, 3, 4, 5],
                "predictedNt": [
                    "acetylcholine",
                    "gaba",
                    "glutamate",
                    "dopamine",
                    "serotonin",
                ],
            }
        )
        ann = pd.DataFrame(
            {"bodyId": [1, 2, 3, 4, 5], "type": ["n1", "n2", "n3", "n4", "n5"]}
        )

        conn = load_connectome_from_tables(weights, nt, ann)

        # Check 1: Sign convention — ACh is positive
        total_checks += 1
        dense = conn.weight_matrix.to_dense()
        if dense[0, 1].item() > 0:
            checks_passed += 1
        else:
            errors.append(f"ACh edge should be positive, got {dense[0, 1].item()}")

        # Check 2: GABA is negative
        total_checks += 1
        if dense[1, 2].item() < 0:
            checks_passed += 1
        else:
            errors.append(f"GABA edge should be negative, got {dense[1, 2].item()}")

        # Check 3: Dopamine in separate matrix (not in weight matrix)
        total_checks += 1
        if dense[3, 4].item() == 0.0:
            checks_passed += 1
        else:
            errors.append(f"Dopamine edge should not be in weight_matrix, got {dense[3, 4].item()}")

        # Check 4: Dopamine is in dopamine_matrix
        total_checks += 1
        dopa_dense = conn.dopamine_matrix.to_dense()
        if dopa_dense[3, 4].item() > 0:
            checks_passed += 1
        else:
            errors.append("Dopamine edge missing from dopamine_matrix")

        # Check 5: Sparse CSR format
        total_checks += 1
        if conn.weight_matrix.layout == torch.sparse_csr:
            checks_passed += 1
        else:
            errors.append(f"Expected sparse_csr, got {conn.weight_matrix.layout}")

        # Check 6: int32 indices
        total_checks += 1
        if (
            conn.weight_matrix.crow_indices().dtype == torch.int32
            and conn.weight_matrix.col_indices().dtype == torch.int32
        ):
            checks_passed += 1
        else:
            errors.append(
                f"Expected int32 indices, got crow={conn.weight_matrix.crow_indices().dtype}, "
                f"col={conn.weight_matrix.col_indices().dtype}"
            )

    except Exception as e:
        total_checks = 1
        errors.append(f"Connectome ETL failed: {e}")

    score = checks_passed / total_checks if total_checks > 0 else 0.0
    details = f"{checks_passed}/{total_checks} checks passed"
    if errors:
        details += "; ERRORS: " + "; ".join(errors)
    return _dim_result(score, details)


def simulation_runs() -> dict[str, Any]:
    """Verify LIF simulation core: exact integration, refractory, calibration."""
    errors: list[str] = []
    checks_passed = 0
    total_checks = 0

    try:
        import math

        import torch

        from flyecon.etl.controls import random_sparse
        from flyecon.sim.calibration import calibrate_gain, preflight_gate
        from flyecon.sim.lif import C_M, LIFNetwork, _DECAY, _ONE_MINUS_DECAY
        from flyecon.state.constants import DT_MS, TAU_MS, V_REST_MV, V_THRESH_MV

        # Check 1: Zero input → no spikes, V stays at V_REST
        total_checks += 1
        conn10 = random_sparse(n_neurons=10, density=0.1, seed=42)
        net = LIFNetwork(conn10, gain=0.001)
        zero_inp = torch.zeros(10, dtype=torch.float32)
        counts = net.simulate(zero_inp, duration_ms=500.0)
        if float(counts.sum().item()) == 0.0:
            checks_passed += 1
        else:
            errors.append(f"Zero input produced spikes: {counts.sum().item()}")

        # Check 2: Exact integration matches forward Euler (dt=0.01ms) to <1%
        total_checks += 1
        I_test = 3.0  # subthreshold
        duration = 100.0
        # Exact
        net_e = LIFNetwork(conn10, gain=0.0)
        inp_e = torch.full((10,), I_test, dtype=torch.float32)
        for _ in range(int(duration / DT_MS)):
            net_e.step(inp_e)
        V_exact = net_e.V.clone()
        # Forward Euler dt=0.01
        dt_fe = 0.01
        V_fe = torch.full((10,), V_REST_MV, dtype=torch.float32)
        for _ in range(int(duration / dt_fe)):
            dV = (-(V_fe - V_REST_MV) / TAU_MS + I_test / C_M) * dt_fe
            V_fe = V_fe + dV
        max_err = float(torch.max(torch.abs(V_exact - V_fe) / (torch.abs(V_fe - V_REST_MV) + 1e-8)).item())
        if max_err < 0.01:
            checks_passed += 1
        else:
            errors.append(f"Exact vs Euler error: {max_err:.4f} (expected <0.01)")

        # Check 3: Refractory prevents double-spike within 2ms
        total_checks += 1
        conn1 = random_sparse(n_neurons=1, density=0.0, seed=0)
        net_r = LIFNetwork(conn1, gain=0.0)
        inp_r = torch.tensor([50.0], dtype=torch.float32)
        spike_times: list[float] = []
        for t in range(200):
            spikes = net_r.step(inp_r)
            if spikes[0]:
                spike_times.append(t * DT_MS)
        if len(spike_times) >= 2:
            min_isi = min(spike_times[i] - spike_times[i - 1] for i in range(1, len(spike_times)))
            if min_isi >= DT_MS + DT_MS:
                checks_passed += 1
            else:
                errors.append(f"Minimum ISI {min_isi}ms < {DT_MS + DT_MS}ms")
        else:
            errors.append(f"Too few spikes ({len(spike_times)}) to check refractory")

        # Check 4: Calibration finds healthy gain on mixed E/I network
        # All-excitatory networks can't be calibrated (gain only increases
        # rate).  A mixed E/I network with net inhibition lets gain control
        # the rate by modulating recurrent inhibitory feedback.
        total_checks += 1
        import numpy as np

        from flyecon.etl.loader import Connectome, build_csr_tensor

        _rng = np.random.default_rng(42)
        _n, _dens, _ei = 100, 0.15, 0.3
        _n_poss = _n * (_n - 1)
        _n_edges = int(_n_poss * _dens)
        _flat = _rng.choice(_n_poss, size=_n_edges, replace=False)
        _rows = _flat // (_n - 1)
        _cols = _flat % (_n - 1)
        _cols = np.where(_cols >= _rows, _cols + 1, _cols)
        _is_e = _rng.random(_n_edges) < _ei
        _vals = np.where(_is_e, 1.0, -5.0).astype(np.float32)
        _W = build_csr_tensor(
            _rows.astype(np.int64), _cols.astype(np.int64), _vals, _n,
        )
        _empty = build_csr_tensor(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            _n,
        )
        conn100 = Connectome(
            weight_matrix=_W, dopamine_matrix=_empty,
            body_ids=np.arange(_n, dtype=np.int64),
            neuron_types={}, n_neurons=_n, n_synapses=_n_edges,
            metadata={"control": "mixed_ei"},
        )
        cal = calibrate_gain(conn100, target_rate_hz=(1.0, 10.0), duration_ms=500.0)
        if cal.is_healthy:
            checks_passed += 1
        else:
            errors.append(f"Calibration failed: gain={cal.gain:.6f}, rate={cal.mean_rate_hz:.2f} Hz")

        # Check 5: Preflight rejects gain=0 with zero input
        total_checks += 1
        pf = preflight_gate(conn100, gain=0.0, duration_ms=500.0, input_amplitude=0.0)
        if not pf.all_pass:
            checks_passed += 1
        else:
            errors.append("Preflight should reject gain=0 with zero input")

    except Exception as e:
        total_checks = max(total_checks, 1)
        errors.append(f"Simulation runs failed: {e}")

    score = checks_passed / total_checks if total_checks > 0 else 0.0
    details = f"{checks_passed}/{total_checks} checks passed"
    if errors:
        details += "; ERRORS: " + "; ".join(errors)
    return _dim_result(score, details)


def fly_vs_oracle() -> dict[str, Any]:
    """Verify PPO policy produces valid actions and shows learning signal."""
    errors: list[str] = []
    checks_passed = 0
    total_checks = 0

    try:
        import torch

        from flyecon.etl.controls import random_sparse
        from flyecon.policy.ppo import PPOTrainer, build_fly_policy

        # Build a tiny test policy (no real connectome needed)
        conn = random_sparse(n_neurons=30, density=0.1, seed=42)
        policy = build_fly_policy(conn, gain=0.0, duration_ms=50.0)

        # Check 1: Policy produces valid Categorical distribution
        total_checks += 1
        states = [pistol_round_state()]
        dist, values, features = policy.forward(states)
        probs = dist.probs
        if not torch.isnan(probs).any() and torch.allclose(
            probs.sum(dim=-1), torch.ones(1), atol=1e-4,
        ):
            checks_passed += 1
        else:
            errors.append("Policy produces invalid distribution")

        # Check 2: PPO update runs without error, metrics finite
        total_checks += 1
        trainer = PPOTrainer(policy, n_steps=32, batch_size=16, n_epochs=2)
        rollouts = trainer.collect_rollouts()
        metrics = trainer.update(rollouts)
        if all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
            checks_passed += 1
        else:
            errors.append(f"PPO metrics not finite: {metrics}")

        # Check 3: Short training loop completes
        total_checks += 1
        tlog = trainer.train(n_iterations=3)
        if len(tlog.metrics) == 3:
            checks_passed += 1
        else:
            errors.append(f"Expected 3 metrics, got {len(tlog.metrics)}")

    except Exception as e:
        total_checks = max(total_checks, 1)
        errors.append(f"fly_vs_oracle failed: {e}")

    score = checks_passed / total_checks if total_checks > 0 else 0.0
    details = f"{checks_passed}/{total_checks} checks passed"
    if errors:
        details += "; ERRORS: " + "; ".join(errors)
    return _dim_result(score, details)


def dashboard_renders() -> dict[str, Any]:
    """Verify dashboard + avatar: render, FCI, self-contained HTML."""
    errors: list[str] = []
    checks_passed = 0
    total_checks = 0

    try:
        import tempfile
        import re
        from pathlib import Path

        from flyecon.dashboard.renderer import render_dashboard
        from flyecon.dashboard.telemetry import TelemetryEvent, TelemetryStore
        from flyecon.avatar.ascii import ASCIIAvatar
        from flyecon.dashboard.fci import compute_fci

        # Check 1: render_dashboard produces a non-empty HTML string
        total_checks += 1
        with tempfile.TemporaryDirectory() as tmpdir:
            telem_path = Path(tmpdir) / "telemetry.jsonl"
            store = TelemetryStore(telem_path)
            # Seed with a few events so the dashboard has data
            store.append(TelemetryEvent("fci", data={"fci": 0.65}))
            store.append(TelemetryEvent("mission_state", data={
                "uptime_seconds": 3600.0,
                "ladder_level": 0,
                "n_candidates": 1,
                "cycles_completed": 10,
            }))

            out_path = Path(tmpdir) / "dashboard.html"
            result_path = render_dashboard(store, out_path)
            html = result_path.read_text()

            if len(html) > 100 and "<html" in html.lower():
                checks_passed += 1
            else:
                errors.append(
                    f"render_dashboard produced {len(html)} chars, "
                    f"expected >100 with <html tag"
                )

        # Check 2: ASCIIAvatar.render() produces non-empty output for
        # various FCI values (high, mid, low)
        total_checks += 1
        avatar = ASCIIAvatar()
        fci_tests = [0.1, 0.5, 0.9]
        all_nonempty = True
        for fci_val in fci_tests:
            art = avatar.render(fci=fci_val)
            if not art or not art.strip():
                all_nonempty = False
                errors.append(f"ASCIIAvatar.render(fci={fci_val}) returned empty")
        if all_nonempty:
            checks_passed += 1

        # Check 3: ASCIIAvatar.render() handles event-specific poses
        total_checks += 1
        events_ok = True
        for event in ["round_win", "forced_eco", "oracle_duel"]:
            art = avatar.render(fci=0.5, event=event)
            if not art or not art.strip():
                events_ok = False
                errors.append(f"ASCIIAvatar.render(event={event!r}) returned empty")
        if events_ok:
            checks_passed += 1

        # Check 4: HTML is self-contained — no <script src="http...">,
        # <link href="http...">, or <img src="http..."> tags that would
        # require network access to render.  URLs *inside* inline JS
        # strings (e.g. plotly.js copyright notices, map attribution) are
        # fine — those are data, not resource loads.
        total_checks += 1
        with tempfile.TemporaryDirectory() as tmpdir:
            telem_path = Path(tmpdir) / "telemetry.jsonl"
            store = TelemetryStore(telem_path)
            store.append(TelemetryEvent("fci", data={"fci": 0.5}))
            out_path = Path(tmpdir) / "dashboard.html"
            result_path = render_dashboard(store, out_path)
            html = result_path.read_text()

            # Match only HTML tags that load external resources:
            # <script src="https://...">, <link ... href="https://...">,
            # <img src="https://...">, <iframe src="https://...">
            external_tags = re.findall(
                r'<(?:script|link|img|iframe)\b[^>]*(?:src|href)\s*=\s*["\']'
                r'https?://[^"\']*["\']',
                html,
                re.IGNORECASE,
            )
            if not external_tags:
                checks_passed += 1
            else:
                errors.append(
                    f"HTML loads {len(external_tags)} external resource(s): "
                    f"{external_tags[:3]}"
                )

        # Check 5: compute_fci returns valid values at extremes
        total_checks += 1
        fci_zero = compute_fci(0.0, 0.0, 0)
        fci_max = compute_fci(20.0, 16_000.0, 5)
        if 0.0 <= fci_zero <= 0.1 and 0.9 <= fci_max <= 1.0:
            checks_passed += 1
        else:
            errors.append(
                f"FCI extremes: zero-input={fci_zero:.3f} (expected ~0), "
                f"max-input={fci_max:.3f} (expected ~1)"
            )

    except Exception as e:
        total_checks = max(total_checks, 1)
        errors.append(f"dashboard_renders failed: {e}")

    score = checks_passed / total_checks if total_checks > 0 else 0.0
    details = f"{checks_passed}/{total_checks} checks passed"
    if errors:
        details += "; ERRORS: " + "; ".join(errors)
    return _dim_result(score, details)


def run_all() -> dict[str, Any]:
    """Run all eval dimensions and return aggregate results."""
    dimensions = {
        "economy_mdp": economy_mdp,
        "oracle_solver": oracle_solver,
        "connectome_etl": connectome_etl,
        "simulation_runs": simulation_runs,
        "fly_vs_oracle": fly_vs_oracle,
        "dashboard_renders": dashboard_renders,
    }

    results: dict[str, Any] = {}
    total_score = 0.0
    for name, fn in dimensions.items():
        result = fn()
        results[name] = result
        total_score += result["score"]

    results["aggregate_score"] = total_score / len(dimensions)
    return results


if __name__ == "__main__":
    results = run_all()
    print(json.dumps(results, indent=2))
    sys.exit(0 if results["aggregate_score"] > 0.0 else 1)

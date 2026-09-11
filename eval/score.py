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
    """Stub — returns 0.0 until Phase 3."""
    return _dim_result(0.0, "Not implemented until Phase 3")


def simulation_runs() -> dict[str, Any]:
    """Stub — returns 0.0 until Phase 4."""
    return _dim_result(0.0, "Not implemented until Phase 4")


def fly_vs_oracle() -> dict[str, Any]:
    """Stub — returns 0.0 until Phase 6."""
    return _dim_result(0.0, "Not implemented until Phase 6")


def dashboard_renders() -> dict[str, Any]:
    """Stub — returns 0.0 until Phase 8."""
    return _dim_result(0.0, "Not implemented until Phase 8")


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

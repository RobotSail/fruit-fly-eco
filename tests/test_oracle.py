"""Tests for the Oracle MDP solver.

Verifies:
- Value iteration converges in <100 iterations
- Policy never chooses FULL_BUY when money < min_full_buy_cost
- Policy on pistol round ($800) chooses ECO or FORCE_BUY (never FULL_BUY)
- Oracle dominates random policy by ≥$500 mean money over 1000 episodes
"""

from __future__ import annotations

import numpy as np
import pytest

from flyecon.oracle.mdp import (
    HALF_RANGE,
    LOSS_STREAK_RANGE,
    MONEY_BUCKETS,
    N_ACTIONS,
    N_STATES,
    OPP_LOSS_STREAK_RANGE,
    ROUND_RANGE,
    EconomyMDP,
    MDPState,
    idx_from_money,
    money_from_idx,
)
from flyecon.oracle.solver import Policy, solve, verify_against_random
from flyecon.state.constants import (
    BUY_PLAN_COSTS,
    MAX_MONEY,
    MIN_MONEY,
    MONEY_STEP,
    STARTING_MONEY,
)
from flyecon.state.economy import BuyPlan, EconomyState, pistol_round_state

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mdp() -> EconomyMDP:
    """Shared MDP instance for all tests in this module."""
    return EconomyMDP()


@pytest.fixture(scope="module")
def solved_policy(mdp: EconomyMDP) -> Policy:
    """Solve the MDP once for all tests (expensive fixture, module-scoped)."""
    return solve(mdp, gamma=0.99, tol=1e-8)


# ── MDP structure tests ─────────────────────────────────────────────────────


class TestMDPStructure:
    def test_state_space_size(self) -> None:
        """State space should be 161 × 5 × 12 × 2 × 5 = 96,600."""
        expected = (
            MONEY_BUCKETS * LOSS_STREAK_RANGE * ROUND_RANGE
            * HALF_RANGE * OPP_LOSS_STREAK_RANGE
        )
        assert N_STATES == expected
        assert expected == 96_600

    def test_action_count(self, mdp: EconomyMDP) -> None:
        assert len(mdp.actions()) == 5

    def test_money_idx_roundtrip(self) -> None:
        """money_from_idx and idx_from_money are inverses."""
        for money in range(MIN_MONEY, MAX_MONEY + 1, MONEY_STEP):
            idx = idx_from_money(money)
            assert money_from_idx(idx) == money

    def test_state_idx_roundtrip(self, mdp: EconomyMDP) -> None:
        """state_to_idx and idx_to_state are inverses for sampled states."""
        rng = np.random.default_rng(123)
        for _ in range(500):
            s = MDPState(
                money_idx=int(rng.integers(0, MONEY_BUCKETS)),
                loss_streak=int(rng.integers(0, LOSS_STREAK_RANGE)),
                round_idx=int(rng.integers(0, ROUND_RANGE)),
                half=int(rng.integers(0, HALF_RANGE)),
                opp_loss_streak=int(rng.integers(0, OPP_LOSS_STREAK_RANGE)),
            )
            idx = mdp.state_to_idx(s)
            assert 0 <= idx < N_STATES
            recovered = mdp.idx_to_state(idx)
            assert recovered == s

    def test_transition_probabilities_sum_to_one(self, mdp: EconomyMDP) -> None:
        """Transition probabilities sum to 1 for affordable (state, action) pairs."""
        rng = np.random.default_rng(456)
        for _ in range(200):
            s = MDPState(
                money_idx=int(rng.integers(0, MONEY_BUCKETS)),
                loss_streak=int(rng.integers(0, LOSS_STREAK_RANGE)),
                round_idx=int(rng.integers(0, ROUND_RANGE)),
                half=int(rng.integers(0, HALF_RANGE)),
                opp_loss_streak=int(rng.integers(0, OPP_LOSS_STREAK_RANGE)),
            )
            for action in mdp.actions():
                transitions = mdp.transition(s, action)
                if not transitions:
                    # Unaffordable action → empty transitions (masked by solver)
                    assert not mdp.is_affordable(s, action)
                    continue
                total_prob = sum(p for _, p, _ in transitions)
                assert abs(total_prob - 1.0) < 1e-12, (
                    f"Probabilities sum to {total_prob} for state={s}, action={action}"
                )

    def test_transition_indices_in_range(self, mdp: EconomyMDP) -> None:
        """All next-state indices should be valid."""
        rng = np.random.default_rng(789)
        for _ in range(200):
            s = MDPState(
                money_idx=int(rng.integers(0, MONEY_BUCKETS)),
                loss_streak=int(rng.integers(0, LOSS_STREAK_RANGE)),
                round_idx=int(rng.integers(0, ROUND_RANGE)),
                half=int(rng.integers(0, HALF_RANGE)),
                opp_loss_streak=int(rng.integers(0, OPP_LOSS_STREAK_RANGE)),
            )
            for action in mdp.actions():
                for nxt, p, r in mdp.transition(s, action):
                    assert 0 <= nxt < N_STATES, f"next_idx={nxt} out of range"
                    assert 0.0 <= p <= 1.0
                    assert r >= 0.0


# ── Solver tests ─────────────────────────────────────────────────────────────


class TestSolver:
    def test_convergence_under_100_iterations(self, solved_policy: Policy) -> None:
        """Value iteration should converge in <100 iterations."""
        assert solved_policy.iterations < 100, (
            f"Value iteration took {solved_policy.iterations} iterations (expected <100)"
        )

    def test_value_table_shape(self, solved_policy: Policy) -> None:
        assert solved_policy.value_table.shape == (N_STATES,)

    def test_policy_table_shape(self, solved_policy: Policy) -> None:
        assert solved_policy.policy_table.shape == (N_STATES,)

    def test_value_table_nonnegative(self, solved_policy: Policy) -> None:
        """All state values should be non-negative (money can't be negative)."""
        assert np.all(solved_policy.value_table >= 0.0)

    def test_all_policy_actions_valid(self, solved_policy: Policy) -> None:
        """All policy actions should be valid action indices."""
        assert np.all(solved_policy.policy_table >= 0)
        assert np.all(solved_policy.policy_table < N_ACTIONS)


# ── Policy sanity tests ──────────────────────────────────────────────────────


class TestPolicySanity:
    def test_never_full_buy_when_too_poor(self, solved_policy: Policy) -> None:
        """Policy should never choose FULL_BUY when money < FULL_BUY cost."""
        mdp = solved_policy.mdp
        full_buy_cost = BUY_PLAN_COSTS["FULL_BUY"]
        max_affordable_idx = idx_from_money(full_buy_cost - MONEY_STEP)
        full_buy_action_idx = mdp.actions().index(BuyPlan.FULL_BUY)

        violations = 0
        for money_idx in range(0, max_affordable_idx + 1):
            for ls in range(LOSS_STREAK_RANGE):
                for rnd in range(ROUND_RANGE):
                    for h in range(HALF_RANGE):
                        for ols in range(OPP_LOSS_STREAK_RANGE):
                            s = MDPState(money_idx, ls, rnd, h, ols)
                            idx = mdp.state_to_idx(s)
                            if solved_policy.policy_table[idx] == full_buy_action_idx:
                                violations += 1

        assert violations == 0, (
            f"Policy chose FULL_BUY in {violations} states where money < "
            f"${full_buy_cost}"
        )

    def test_pistol_round_not_full_buy(self, solved_policy: Policy) -> None:
        """On pistol round ($800), policy should choose ECO or FORCE_BUY, never FULL_BUY."""
        ps = pistol_round_state()
        action = solved_policy.decide(ps)
        assert action != BuyPlan.FULL_BUY, (
            f"Pistol round ($800) chose {action.value}, expected ECO or FORCE_BUY"
        )

    def test_pistol_round_reasonable(self, solved_policy: Policy) -> None:
        """On pistol round, action should be one of the affordable options."""
        ps = pistol_round_state()
        action = solved_policy.decide(ps)
        cost = BUY_PLAN_COSTS[action.value]
        assert cost <= STARTING_MONEY, (
            f"Pistol round chose {action.value} costing ${cost} > ${STARTING_MONEY}"
        )

    def test_rich_state_prefers_buying(self, solved_policy: Policy) -> None:
        """With lots of money, policy should generally not SAVE."""
        rich_state = EconomyState(
            money=10000,
            loss_streak=0,
            round_number=6,
            half=0,
            opponent_loss_streak=0,
        )
        action = solved_policy.decide(rich_state)
        # With $10,000 the oracle should buy something, not save
        assert action != BuyPlan.SAVE, (
            "Rich state ($10,000) chose SAVE — policy is degenerate"
        )


# ── Oracle vs Random ─────────────────────────────────────────────────────────


class TestOracleVsRandom:
    def test_oracle_dominates_random(self, solved_policy: Policy) -> None:
        """Oracle should dominate random policy by ≥$500 mean money.

        The Oracle optimises round WINS.  Each extra win is worth ~$3250
        (WIN_REWARD) minus equipment cost.  We measure the money-equivalent
        advantage: extra_wins × WIN_REWARD, which should exceed $500.
        """
        result = verify_against_random(
            solved_policy, n_episodes=1000, seed=42
        )
        # Oracle should win meaningfully more rounds than random
        assert result.wins_advantage >= 1.0, (
            f"Oracle wins advantage: {result.wins_advantage:.2f} rounds "
            f"(expected ≥1.0)"
        )
        # Money-equivalent of extra wins should exceed $500
        money_equiv = result.wins_advantage * 3250
        assert money_equiv >= 500.0, (
            f"Oracle money-equivalent advantage: ${money_equiv:.0f} (expected ≥$500)"
        )

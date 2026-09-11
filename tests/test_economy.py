"""Tests for flyecon/state/economy.py — MR12 economy transitions."""

from __future__ import annotations

import pytest

from flyecon.state.constants import (
    BUY_PLAN_COSTS,
    MAX_MONEY,
    ROUNDS_PER_HALF,
    STARTING_MONEY,
    WIN_REWARD,
)
from flyecon.state.economy import BuyPlan, EconomyState, pistol_round_state, step


class TestPistolRound:
    def test_pistol_start_money(self) -> None:
        """Pistol round starts at $800."""
        state = pistol_round_state()
        assert state.money == 800
        assert state.money == STARTING_MONEY

    def test_pistol_round_is_round_1(self) -> None:
        state = pistol_round_state()
        assert state.round_number == 1

    def test_pistol_half_0_by_default(self) -> None:
        state = pistol_round_state()
        assert state.half == 0

    def test_pistol_loss_streak_zero(self) -> None:
        state = pistol_round_state()
        assert state.loss_streak == 0
        assert state.opponent_loss_streak == 0


class TestLossBonusLadder:
    def test_loss_bonus_progression(self) -> None:
        """Verify loss-bonus ladder: 1400 → 1900 → 2400 → 2900 → 3400."""
        expected_bonuses = [1_400, 1_900, 2_400, 2_900, 3_400]
        for streak, expected in enumerate(expected_bonuses):
            state = EconomyState(
                money=0,
                loss_streak=streak,
                round_number=1,
                half=0,
                opponent_loss_streak=0,
            )
            next_state = step(state, BuyPlan.SAVE, round_won=False)
            assert next_state.money == expected, (
                f"Loss streak {streak}: expected ${expected}, got ${next_state.money}"
            )

    def test_loss_streak_increments(self) -> None:
        """Each loss increments loss_streak by 1, up to max."""
        state = EconomyState(
            money=5000, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0
        )
        for expected_streak in range(1, 5):
            state = step(state, BuyPlan.SAVE, round_won=False)
            assert state.loss_streak == expected_streak

    def test_loss_streak_caps_at_max(self) -> None:
        """Loss streak should not exceed MAX_LOSS_STREAK (4)."""
        state = EconomyState(
            money=5000, loss_streak=4, round_number=1, half=0, opponent_loss_streak=0
        )
        next_state = step(state, BuyPlan.SAVE, round_won=False)
        assert next_state.loss_streak == 4


class TestWinMechanics:
    def test_win_gives_base_reward(self) -> None:
        """Win awards $3250 base."""
        state = EconomyState(
            money=0, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0
        )
        next_state = step(state, BuyPlan.SAVE, round_won=True)
        assert next_state.money == WIN_REWARD
        assert next_state.money == 3_250

    def test_win_resets_own_loss_streak(self) -> None:
        state = EconomyState(
            money=0, loss_streak=4, round_number=1, half=0, opponent_loss_streak=0
        )
        next_state = step(state, BuyPlan.SAVE, round_won=True)
        assert next_state.loss_streak == 0

    def test_win_increments_opponent_loss_streak(self) -> None:
        state = EconomyState(
            money=0, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0
        )
        next_state = step(state, BuyPlan.SAVE, round_won=True)
        assert next_state.opponent_loss_streak == 1

    def test_win_resets_opponent_ladder(self) -> None:
        """When we win, the opponent's loss streak resets on their side.
        But from our perspective, we track opponent_loss_streak incrementing."""
        # We track opponent loss streak increasing when we win
        state = EconomyState(
            money=0, loss_streak=0, round_number=1, half=0, opponent_loss_streak=2
        )
        next_state = step(state, BuyPlan.SAVE, round_won=True)
        # Opponent lost again, so their streak increases
        assert next_state.opponent_loss_streak == 3

    def test_loss_resets_opponent_streak(self) -> None:
        """When we lose, the opponent won — their loss streak resets to 0."""
        state = EconomyState(
            money=5000, loss_streak=0, round_number=1, half=0, opponent_loss_streak=3
        )
        next_state = step(state, BuyPlan.SAVE, round_won=False)
        assert next_state.opponent_loss_streak == 0

    def test_kill_rewards_added(self) -> None:
        """Kill rewards are added on top of win reward."""
        state = EconomyState(
            money=0, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0
        )
        next_state = step(state, BuyPlan.SAVE, round_won=True, kill_reward=600)
        assert next_state.money == WIN_REWARD + 600


class TestHalftimeReset:
    def test_halftime_resets_to_pistol(self) -> None:
        """At round 12 of half 0, next state is pistol of half 1."""
        state = EconomyState(
            money=10_000,
            loss_streak=4,
            round_number=ROUNDS_PER_HALF,
            half=0,
            opponent_loss_streak=3,
        )
        next_state = step(state, BuyPlan.SAVE, round_won=True)
        assert next_state.money == STARTING_MONEY
        assert next_state.half == 1
        assert next_state.round_number == 1
        assert next_state.loss_streak == 0
        assert next_state.opponent_loss_streak == 0

    def test_halftime_reset_on_loss(self) -> None:
        """Halftime reset also happens on a loss in round 12."""
        state = EconomyState(
            money=5000,
            loss_streak=2,
            round_number=ROUNDS_PER_HALF,
            half=0,
            opponent_loss_streak=1,
        )
        next_state = step(state, BuyPlan.SAVE, round_won=False)
        assert next_state.half == 1
        assert next_state.money == STARTING_MONEY
        assert next_state.round_number == 1

    def test_second_half_last_round(self) -> None:
        """Round 12 of half 1 — match over, clamp to last round."""
        state = EconomyState(
            money=5000,
            loss_streak=0,
            round_number=ROUNDS_PER_HALF,
            half=1,
            opponent_loss_streak=0,
        )
        next_state = step(state, BuyPlan.SAVE, round_won=True)
        assert next_state.half == 1
        assert next_state.round_number == ROUNDS_PER_HALF


class TestBuyPlanCosts:
    def test_full_buy_deducts_cost(self) -> None:
        state = EconomyState(
            money=5000, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0
        )
        next_state = step(state, BuyPlan.FULL_BUY, round_won=True)
        expected = 5000 - BUY_PLAN_COSTS["FULL_BUY"] + WIN_REWARD
        assert next_state.money == expected

    def test_save_costs_nothing(self) -> None:
        state = EconomyState(
            money=5000, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0
        )
        next_state = step(state, BuyPlan.SAVE, round_won=True)
        assert next_state.money == 5000 + WIN_REWARD

    def test_cannot_go_below_zero_from_buy(self) -> None:
        """Even if buy cost exceeds money, money_after_buy is clamped to 0."""
        state = EconomyState(
            money=100, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0
        )
        next_state = step(state, BuyPlan.FULL_BUY, round_won=True)
        # money_after_buy = max(100 - 4750, 0) = 0
        assert next_state.money == WIN_REWARD

    def test_money_capped_at_max(self) -> None:
        """Money cannot exceed MAX_MONEY ($16,000)."""
        state = EconomyState(
            money=MAX_MONEY, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0
        )
        next_state = step(state, BuyPlan.SAVE, round_won=True)
        assert next_state.money == MAX_MONEY


class TestRoundAdvancement:
    def test_round_increments(self) -> None:
        state = EconomyState(
            money=5000, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0
        )
        next_state = step(state, BuyPlan.SAVE, round_won=True)
        assert next_state.round_number == 2

    def test_all_buy_plans_exist(self) -> None:
        """Verify all 5 buy plans are defined."""
        assert len(BuyPlan) == 5
        names = {bp.value for bp in BuyPlan}
        assert names == {"FULL_BUY", "FORCE_BUY", "HALF_BUY", "ECO", "SAVE"}


class TestEconomyStateValidation:
    def test_invalid_money_raises(self) -> None:
        with pytest.raises(ValueError, match="money"):
            EconomyState(money=-1, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0)

    def test_invalid_money_over_max_raises(self) -> None:
        with pytest.raises(ValueError, match="money"):
            EconomyState(
                money=MAX_MONEY + 1,
                loss_streak=0,
                round_number=1,
                half=0,
                opponent_loss_streak=0,
            )

    def test_invalid_loss_streak_raises(self) -> None:
        with pytest.raises(ValueError, match="loss_streak"):
            EconomyState(money=0, loss_streak=5, round_number=1, half=0, opponent_loss_streak=0)

    def test_invalid_round_raises(self) -> None:
        with pytest.raises(ValueError, match="round_number"):
            EconomyState(money=0, loss_streak=0, round_number=0, half=0, opponent_loss_streak=0)

    def test_invalid_half_raises(self) -> None:
        with pytest.raises(ValueError, match="half"):
            EconomyState(money=0, loss_streak=0, round_number=1, half=2, opponent_loss_streak=0)

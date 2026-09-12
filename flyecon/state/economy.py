"""MR12 economy state and transition function.

This module defines the ground-truth economy mechanics for Counter-Strike MR12.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flyecon.state.constants import (
    BUY_PLAN_COSTS,
    LOSS_BONUS_LADDER,
    MAX_LOSS_STREAK,
    MAX_MONEY,
    MIN_MONEY,
    ROUNDS_PER_HALF,
    STARTING_MONEY,
    WIN_REWARD,
)


class BuyPlan(Enum):
    """Available buy plan actions."""

    FULL_BUY = "FULL_BUY"
    FORCE_BUY = "FORCE_BUY"
    HALF_BUY = "HALF_BUY"
    ECO = "ECO"
    SAVE = "SAVE"


@dataclass(frozen=True, slots=True)
class EconomyState:
    """Immutable economy state for one team in an MR12 match.

    All money values are in integer USD.
    """

    money: int
    loss_streak: int
    round_number: int
    half: int
    opponent_loss_streak: int

    def __post_init__(self) -> None:
        """Validate invariants."""
        if not MIN_MONEY <= self.money <= MAX_MONEY:
            raise ValueError(f"money must be in [{MIN_MONEY}, {MAX_MONEY}], got {self.money}")
        if not 0 <= self.loss_streak <= MAX_LOSS_STREAK:
            raise ValueError(
                f"loss_streak must be in [0, {MAX_LOSS_STREAK}], got {self.loss_streak}"
            )
        if not 1 <= self.round_number <= ROUNDS_PER_HALF:
            raise ValueError(
                f"round_number must be in [1, {ROUNDS_PER_HALF}], got {self.round_number}"
            )
        if self.half not in (0, 1):
            raise ValueError(f"half must be 0 or 1, got {self.half}")
        if not 0 <= self.opponent_loss_streak <= MAX_LOSS_STREAK:
            raise ValueError(
                f"opponent_loss_streak must be in [0, {MAX_LOSS_STREAK}], "
                f"got {self.opponent_loss_streak}"
            )


def pistol_round_state(half: int = 0) -> EconomyState:
    """Create the initial state for a pistol round."""
    return EconomyState(
        money=STARTING_MONEY,
        loss_streak=0,
        round_number=1,
        half=half,
        opponent_loss_streak=0,
    )


def step(
    state: EconomyState,
    buy_plan: BuyPlan,
    round_won: bool,
    kill_reward: int = 0,
) -> EconomyState:
    """Advance the economy by one round using MR12 rules.

    Args:
        state: Current economy state.
        buy_plan: The buy plan chosen for this round.
        round_won: Whether the team won the round.
        kill_reward: Additional money from kills (default 0).

    Returns:
        New economy state after the round resolves.
    """
    cost = BUY_PLAN_COSTS[buy_plan.value]

    # Spend money on equipment (cannot go below 0)
    money_after_buy = max(state.money - cost, MIN_MONEY)

    if round_won:
        # Win: receive round reward + kill rewards
        new_money = min(money_after_buy + WIN_REWARD + kill_reward, MAX_MONEY)
        new_loss_streak = 0
        new_opponent_loss_streak = min(
            state.opponent_loss_streak + 1, MAX_LOSS_STREAK
        )
    else:
        # Loss: receive loss bonus based on loss streak + kill rewards
        loss_bonus = LOSS_BONUS_LADDER[min(state.loss_streak, MAX_LOSS_STREAK)]
        new_money = min(money_after_buy + loss_bonus + kill_reward, MAX_MONEY)
        new_loss_streak = min(state.loss_streak + 1, MAX_LOSS_STREAK)
        # Opponent won, so their loss streak resets
        new_opponent_loss_streak = 0

    # Advance round
    new_round = state.round_number + 1

    # Check for halftime reset
    if new_round > ROUNDS_PER_HALF:
        # Halftime: reset to pistol round of next half
        if state.half == 0:
            return EconomyState(
                money=STARTING_MONEY,
                loss_streak=0,
                round_number=1,
                half=1,
                opponent_loss_streak=0,
            )
        else:
            # Match over — return final state clamped to last round
            return EconomyState(
                money=new_money,
                loss_streak=new_loss_streak,
                round_number=ROUNDS_PER_HALF,
                half=1,
                opponent_loss_streak=new_opponent_loss_streak,
            )

    return EconomyState(
        money=new_money,
        loss_streak=new_loss_streak,
        round_number=new_round,
        half=state.half,
        opponent_loss_streak=new_opponent_loss_streak,
    )

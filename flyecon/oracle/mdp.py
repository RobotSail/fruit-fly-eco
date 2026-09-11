"""MR12 economy MDP for value-iteration solving.

Enumerates the full state space in discretized buckets and encodes
transition dynamics using the exact MR12 rules from flyecon.state.economy.

The Oracle's transition model IS the MR12 rules — not a separate
reimplementation. Every transition delegates to economy.step().

Reward: 1.0 for winning a round, 0.0 for losing. The Oracle maximises
expected discounted round wins. Money is a state variable that gates
which buy-plans are affordable — the Oracle learns to manage money
as a means to winning rounds, not as an end in itself.

Terminal states: round 12 of half 1 (match over). These are absorbing
with zero future reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyecon.state.constants import (
    BUY_PLAN_COSTS,
    MAX_LOSS_STREAK,
    MAX_MONEY,
    MIN_MONEY,
    MONEY_STEP,
    ROUNDS_PER_HALF,
)
from flyecon.state.economy import BuyPlan, EconomyState, step

# ── State space dimensions ───────────────────────────────────────────────────

MONEY_BUCKETS: int = (MAX_MONEY - MIN_MONEY) // MONEY_STEP + 1  # 161
LOSS_STREAK_RANGE: int = MAX_LOSS_STREAK + 1  # 5
ROUND_RANGE: int = ROUNDS_PER_HALF  # 12
HALF_RANGE: int = 2
OPP_LOSS_STREAK_RANGE: int = MAX_LOSS_STREAK + 1  # 5

N_STATES: int = (
    MONEY_BUCKETS
    * LOSS_STREAK_RANGE
    * ROUND_RANGE
    * HALF_RANGE
    * OPP_LOSS_STREAK_RANGE
)
"""Total states: 161 × 5 × 12 × 2 × 5 = 96,600."""

N_ACTIONS: int = len(BuyPlan)  # 5


# ── Win-probability model ────────────────────────────────────────────────────
# Parameterised: base win rate per buy-plan, tunable.
# Higher equipment spend → higher round-win probability.
# These are *not* hardcoded truths — they are documented modelling
# assumptions that can be calibrated against match data (e.g. Xenopoulos
# et al. 2021 OSE rankings).

DEFAULT_WIN_PROBS: dict[str, float] = {
    "FULL_BUY": 0.55,
    "FORCE_BUY": 0.40,
    "HALF_BUY": 0.47,
    "ECO": 0.25,
    "SAVE": 0.20,
}
"""P(win | buy_plan). Tunable modelling assumption."""


@dataclass(frozen=True, slots=True)
class MDPState:
    """Discretised MDP state (indices, not raw values)."""

    money_idx: int  # 0 .. MONEY_BUCKETS-1
    loss_streak: int  # 0 .. MAX_LOSS_STREAK
    round_idx: int  # 0 .. ROUND_RANGE-1  (round_number-1)
    half: int  # 0 or 1
    opp_loss_streak: int  # 0 .. MAX_LOSS_STREAK


def money_from_idx(idx: int) -> int:
    """Convert bucket index to dollar amount."""
    return MIN_MONEY + idx * MONEY_STEP


def idx_from_money(money: int) -> int:
    """Convert dollar amount to bucket index (floor to nearest step)."""
    clamped = max(MIN_MONEY, min(money, MAX_MONEY))
    return (clamped - MIN_MONEY) // MONEY_STEP


def mdp_state_to_economy(s: MDPState) -> EconomyState:
    """Convert an MDPState to an EconomyState for the step function."""
    return EconomyState(
        money=money_from_idx(s.money_idx),
        loss_streak=s.loss_streak,
        round_number=s.round_idx + 1,  # round_number is 1-indexed
        half=s.half,
        opponent_loss_streak=s.opp_loss_streak,
    )


def economy_to_mdp_state(e: EconomyState) -> MDPState:
    """Convert an EconomyState to an MDPState (discretised)."""
    return MDPState(
        money_idx=idx_from_money(e.money),
        loss_streak=e.loss_streak,
        round_idx=e.round_number - 1,
        half=e.half,
        opp_loss_streak=e.opponent_loss_streak,
    )


def is_terminal_state(s: MDPState) -> bool:
    """Check if a state is terminal (match over).

    Terminal = last round of second half (round_idx=11, half=1).
    After this round, the match is over — no more rounds to play.
    """
    return s.round_idx == ROUND_RANGE - 1 and s.half == 1


class EconomyMDP:
    """Full MR12 economy MDP with discretised state space.

    State space: money (161) × loss_streak (5) × round (12) × half (2)
                 × opp_loss_streak (5) ≈ 96,600 states.
    Action space: 5 BuyPlan variants.

    The transition function delegates to flyecon.state.economy.step() —
    the Oracle shares the exact same economy rules as the simulator.

    Reward: 1.0 for winning a round, 0.0 for losing. Money management
    is incentivised through the state dynamics (more money → can afford
    better buys → higher win probability).

    Terminal states: round 12 of half 1 (match over). The round is
    still played (win/lose), but future value is zero.
    """

    def __init__(
        self,
        win_probs: dict[str, float] | None = None,
    ) -> None:
        self.win_probs = win_probs or dict(DEFAULT_WIN_PROBS)
        self.n_states = N_STATES
        self.n_actions = N_ACTIONS
        self._actions = list(BuyPlan)

        # Strides for flat-index computation
        self._strides = np.array(
            [
                OPP_LOSS_STREAK_RANGE * HALF_RANGE * ROUND_RANGE * LOSS_STREAK_RANGE,
                OPP_LOSS_STREAK_RANGE * HALF_RANGE * ROUND_RANGE,
                OPP_LOSS_STREAK_RANGE * HALF_RANGE,
                OPP_LOSS_STREAK_RANGE,
                1,
            ],
            dtype=np.int64,
        )

        # Pre-compute affordability mask: can_afford[money_idx, action_idx]
        self.can_afford = np.zeros((MONEY_BUCKETS, N_ACTIONS), dtype=np.bool_)
        for m_idx in range(MONEY_BUCKETS):
            money = money_from_idx(m_idx)
            for a_idx, action in enumerate(self._actions):
                self.can_afford[m_idx, a_idx] = BUY_PLAN_COSTS[action.value] <= money

        # Pre-compute terminal state mask
        self.terminal_mask = np.zeros(N_STATES, dtype=np.bool_)
        for s_idx in range(N_STATES):
            s = self.idx_to_state(s_idx)
            if is_terminal_state(s):
                self.terminal_mask[s_idx] = True

    # ── Index conversion ─────────────────────────────────────────────────

    def state_to_idx(self, s: MDPState) -> int:
        """Flat index for an MDPState."""
        return int(
            s.money_idx * self._strides[0]
            + s.loss_streak * self._strides[1]
            + s.round_idx * self._strides[2]
            + s.half * self._strides[3]
            + s.opp_loss_streak * self._strides[4]
        )

    def idx_to_state(self, idx: int) -> MDPState:
        """Recover MDPState from flat index."""
        money_idx, rem = divmod(idx, int(self._strides[0]))
        loss_streak, rem = divmod(rem, int(self._strides[1]))
        round_idx, rem = divmod(rem, int(self._strides[2]))
        half, opp_loss_streak = divmod(rem, int(self._strides[3]))
        return MDPState(
            money_idx=money_idx,
            loss_streak=loss_streak,
            round_idx=round_idx,
            half=half,
            opp_loss_streak=opp_loss_streak,
        )

    # ── Action space ─────────────────────────────────────────────────────

    def actions(self) -> list[BuyPlan]:
        """Return all 5 BuyPlan variants."""
        return list(self._actions)

    def is_affordable(self, s: MDPState, action: BuyPlan) -> bool:
        """Check if an action is affordable at the given state."""
        a_idx = self._actions.index(action)
        return bool(self.can_afford[s.money_idx, a_idx])

    # ── Transition function ──────────────────────────────────────────────

    def transition(
        self, s: MDPState, action: BuyPlan
    ) -> list[tuple[int, float, float]]:
        """Compute transitions: List[(next_state_idx, probability, reward)].

        Delegates to economy.step() for exact MR12 mechanics.
        Reward: 1.0 for winning, 0.0 for losing.

        If the agent cannot afford the action, returns an empty list
        (the solver masks these out with -inf Q values).
        """
        a_idx = self._actions.index(action)
        if not self.can_afford[s.money_idx, a_idx]:
            return []  # unaffordable → masked by solver

        econ_state = mdp_state_to_economy(s)
        p_win = self.win_probs[action.value]
        p_lose = 1.0 - p_win

        transitions: list[tuple[int, float, float]] = []

        # Outcome: WIN → reward 1.0
        next_win = step(econ_state, action, round_won=True)
        mdp_win = economy_to_mdp_state(next_win)
        idx_win = self.state_to_idx(mdp_win)
        transitions.append((idx_win, p_win, 1.0))

        # Outcome: LOSE → reward 0.0
        next_lose = step(econ_state, action, round_won=False)
        mdp_lose = economy_to_mdp_state(next_lose)
        idx_lose = self.state_to_idx(mdp_lose)
        transitions.append((idx_lose, p_lose, 0.0))

        return transitions

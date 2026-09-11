"""Value-iteration solver for the MR12 economy MDP.

Solves V[s] = max_a Σ_{s'} P(s'|s,a)(R(s,a,s') + γ·V[s'])
using NumPy arrays. Zero connectome dependency.

Terminal states (round 12, half 1) use γ=0 so V reflects only
the immediate round outcome (match is over). This gives fast
convergence — information propagates backward from terminal
states in O(horizon) iterations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import structlog

from flyecon.oracle.mdp import (
    EconomyMDP,
    economy_to_mdp_state,
)
from flyecon.state.constants import BUY_PLAN_COSTS
from flyecon.state.economy import BuyPlan, EconomyState, pistol_round_state, step

log = structlog.get_logger()


@dataclass
class Policy:
    """Solved policy: maps states to actions and values.

    Stores the full value table and policy table as NumPy arrays
    for O(1) lookup.
    """

    value_table: np.ndarray  # shape [n_states], float64
    policy_table: np.ndarray  # shape [n_states], int (action index)
    mdp: EconomyMDP
    iterations: int
    solve_time_s: float

    def decide(self, state: EconomyState) -> BuyPlan:
        """Choose the optimal BuyPlan for a given economy state."""
        mdp_s = economy_to_mdp_state(state)
        idx = self.mdp.state_to_idx(mdp_s)
        action_idx = int(self.policy_table[idx])
        return self.mdp.actions()[action_idx]

    def value(self, state: EconomyState) -> float:
        """Return the estimated value of a given economy state."""
        mdp_s = economy_to_mdp_state(state)
        idx = self.mdp.state_to_idx(mdp_s)
        return float(self.value_table[idx])

    def decide_and_value(self, state: EconomyState) -> tuple[BuyPlan, float]:
        """Return (action, value) for a given economy state."""
        mdp_s = economy_to_mdp_state(state)
        idx = self.mdp.state_to_idx(mdp_s)
        action_idx = int(self.policy_table[idx])
        return self.mdp.actions()[action_idx], float(self.value_table[idx])

    def summary(self) -> str:
        """Return a human-readable summary of the solved policy."""
        actions = self.mdp.actions()
        lines = [
            f"Oracle Policy — solved in {self.iterations} iterations "
            f"({self.solve_time_s:.3f}s)",
            f"States: {self.mdp.n_states:,}  Actions: {self.mdp.n_actions}",
            f"Mean value: {self.value_table.mean():.4f}",
            f"Max value: {self.value_table.max():.4f}",
            "",
            "Action distribution:",
        ]
        for i, a in enumerate(actions):
            count = int(np.sum(self.policy_table == i))
            pct = 100.0 * count / self.mdp.n_states
            lines.append(f"  {a.value:12s}: {count:>6,} states ({pct:.1f}%)")

        # Pistol round recommendation
        ps = pistol_round_state()
        rec = self.decide(ps)
        lines.append(f"\nPistol round ($800) recommendation: {rec.value}")
        return "\n".join(lines)


def solve(
    mdp: EconomyMDP,
    gamma: float = 0.99,
    tol: float = 1e-8,
    max_iterations: int = 5000,
) -> Policy:
    """Solve the MDP via value iteration.

    Terminal states (last round of second half) use γ=0: they see
    only the immediate round outcome reward, no future value.
    This bounds the effective horizon and ensures fast convergence.

    Args:
        mdp: The economy MDP to solve.
        gamma: Discount factor for non-terminal states.
        tol: Convergence tolerance (max Bellman residual).
        max_iterations: Safety cap on iterations.

    Returns:
        Solved Policy with value table and policy table.
    """
    t0 = time.monotonic()

    n_states = mdp.n_states
    n_actions = mdp.n_actions

    V = np.zeros(n_states, dtype=np.float64)
    policy = np.zeros(n_states, dtype=np.int32)

    # Pre-compute all transitions for vectorised updates.
    # Each (state, action) has exactly 2 outcomes (win/lose).
    next_idx = np.zeros((n_states, n_actions, 2), dtype=np.int64)
    prob = np.zeros((n_states, n_actions, 2), dtype=np.float64)
    reward = np.zeros((n_states, n_actions, 2), dtype=np.float64)
    # Mask: True if the action is affordable at this state
    affordable = np.zeros((n_states, n_actions), dtype=np.bool_)

    log.info("oracle.precompute_transitions", n_states=n_states, n_actions=n_actions)

    actions = mdp.actions()
    for s_idx in range(n_states):
        state = mdp.idx_to_state(s_idx)
        for a_idx, action in enumerate(actions):
            transitions = mdp.transition(state, action)
            if transitions:
                affordable[s_idx, a_idx] = True
                for o_idx, (nxt, p, r) in enumerate(transitions):
                    next_idx[s_idx, a_idx, o_idx] = nxt
                    prob[s_idx, a_idx, o_idx] = p
                    reward[s_idx, a_idx, o_idx] = r

    t_precomp = time.monotonic() - t0
    log.info("oracle.transitions_precomputed", precompute_time_s=round(t_precomp, 3))

    # Per-state discount: γ for non-terminal, 0 for terminal states
    # Terminal states (round 12, half 1) see only immediate reward.
    gamma_arr = np.full(n_states, gamma, dtype=np.float64)
    gamma_arr[mdp.terminal_mask] = 0.0
    # Shape for broadcasting: [n_states, 1, 1]
    gamma_broadcast = gamma_arr[:, np.newaxis, np.newaxis]

    neg_inf_mask = ~affordable  # True where action is NOT affordable

    # Value iteration loop
    iteration = 0
    delta = float("inf")
    for iteration in range(1, max_iterations + 1):
        V_next = V[next_idx]  # shape [n_states, n_actions, 2]
        Q = np.sum(prob * (reward + gamma_broadcast * V_next), axis=2)

        # Mask unaffordable actions
        Q[neg_inf_mask] = -np.inf

        V_new = np.max(Q, axis=1)  # [n_states]
        policy_new = np.argmax(Q, axis=1).astype(np.int32)

        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        policy = policy_new

        if delta < tol:
            break

    solve_time = time.monotonic() - t0
    log.info(
        "oracle.solved",
        iterations=iteration,
        delta=delta,
        solve_time_s=round(solve_time, 3),
        mean_value=round(float(V.mean()), 4),
    )

    return Policy(
        value_table=V,
        policy_table=policy,
        mdp=mdp,
        iterations=iteration,
        solve_time_s=solve_time,
    )


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Result of simulating one full MR12 match."""

    total_money: float  # sum of end-of-round money across all rounds
    total_wins: int  # number of rounds won


def simulate_episode(
    policy_fn: callable,
    rng: np.random.Generator,
    mdp: EconomyMDP,
) -> EpisodeResult:
    """Simulate one full MR12 match using a policy.

    The policy_fn takes an EconomyState and returns a BuyPlan.
    Win/loss outcomes are sampled stochastically using the MDP's
    win-probability model.
    """
    total_money = 0.0
    total_wins = 0

    for half in range(2):
        state = pistol_round_state(half=half)
        for rnd in range(1, 13):  # 12 rounds per half
            action = policy_fn(state)
            # Determine effective action (can't buy what you can't afford)
            cost = BUY_PLAN_COSTS[action.value]
            if cost > state.money:
                effective = BuyPlan.SAVE
            else:
                effective = action

            p_win = mdp.win_probs[effective.value]
            won = rng.random() < p_win

            next_state = step(state, effective, round_won=won)
            total_money += next_state.money
            if won:
                total_wins += 1
            state = next_state

            # Check for halftime reset (round_number resets to 1)
            if state.round_number == 1 and rnd < 12:
                break  # halftime happened via step()

    return EpisodeResult(total_money=total_money, total_wins=total_wins)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of Oracle vs random comparison."""

    oracle_money_mean: float
    random_money_mean: float
    money_advantage: float
    oracle_wins_mean: float
    random_wins_mean: float
    wins_advantage: float
    n_episodes: int


def verify_against_random(
    policy: Policy,
    n_episodes: int = 1000,
    seed: int = 42,
) -> VerificationResult:
    """Compare Oracle policy vs random policy over n_episodes.

    The Oracle optimises round wins. It should win significantly more
    rounds than random. Each extra win is worth WIN_REWARD ($3250),
    so the Oracle's win advantage translates to substantial money-
    equivalent value even though it spends more on equipment.
    """
    rng = np.random.default_rng(seed)
    mdp = policy.mdp
    actions = mdp.actions()

    def oracle_policy(state: EconomyState) -> BuyPlan:
        return policy.decide(state)

    def random_policy(state: EconomyState) -> BuyPlan:
        return actions[int(rng.integers(0, len(actions)))]

    oracle_totals: list[float] = []
    random_totals: list[float] = []
    oracle_wins_list: list[int] = []
    random_wins_list: list[int] = []

    for _ in range(n_episodes):
        oracle_rng = np.random.default_rng(rng.integers(0, 2**32))
        random_rng = np.random.default_rng(rng.integers(0, 2**32))

        oracle_result = simulate_episode(oracle_policy, oracle_rng, mdp)
        random_result = simulate_episode(random_policy, random_rng, mdp)

        oracle_totals.append(oracle_result.total_money)
        random_totals.append(random_result.total_money)
        oracle_wins_list.append(oracle_result.total_wins)
        random_wins_list.append(random_result.total_wins)

    oracle_money_mean = float(np.mean(oracle_totals))
    random_money_mean = float(np.mean(random_totals))
    oracle_wins_mean = float(np.mean(oracle_wins_list))
    random_wins_mean = float(np.mean(random_wins_list))

    result = VerificationResult(
        oracle_money_mean=oracle_money_mean,
        random_money_mean=random_money_mean,
        money_advantage=oracle_money_mean - random_money_mean,
        oracle_wins_mean=oracle_wins_mean,
        random_wins_mean=random_wins_mean,
        wins_advantage=oracle_wins_mean - random_wins_mean,
        n_episodes=n_episodes,
    )

    log.info(
        "oracle.verify_against_random",
        n_episodes=n_episodes,
        oracle_money_mean=round(oracle_money_mean, 1),
        random_money_mean=round(random_money_mean, 1),
        money_advantage=round(result.money_advantage, 1),
        oracle_wins_mean=round(oracle_wins_mean, 2),
        random_wins_mean=round(random_wins_mean, 2),
        wins_advantage=round(result.wins_advantage, 2),
    )

    return result

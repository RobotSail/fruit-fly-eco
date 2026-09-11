"""Evaluation harness: compare a policy against an oracle on economy scenarios.

evaluate_policy() runs both functions on identical scenarios and produces
an EvalResult with value ratio, win rate, and per-scenario breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from flyecon.state.economy import BuyPlan, EconomyState


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Result for a single economy scenario."""

    state: EconomyState
    policy_action: BuyPlan
    oracle_action: BuyPlan
    policy_value: float
    oracle_value: float
    policy_matches_oracle: bool


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Aggregate evaluation result."""

    mean_policy_value: float
    mean_oracle_value: float
    value_ratio: float
    win_rate: float
    n_scenarios: int
    scenario_results: tuple[ScenarioResult, ...] = field(repr=False)


def evaluate_policy(
    policy_fn: Callable[[EconomyState], tuple[BuyPlan, float]],
    oracle_fn: Callable[[EconomyState], tuple[BuyPlan, float]],
    scenarios: Sequence[EconomyState],
) -> EvalResult:
    """Evaluate a policy against an oracle on a set of economy scenarios.

    Args:
        policy_fn: Takes an EconomyState, returns (BuyPlan, value_estimate).
        oracle_fn: Takes an EconomyState, returns (BuyPlan, value_estimate).
        scenarios: Sequence of EconomyState instances to evaluate on.

    Returns:
        EvalResult with aggregate and per-scenario metrics.
    """
    if not scenarios:
        return EvalResult(
            mean_policy_value=0.0,
            mean_oracle_value=0.0,
            value_ratio=0.0,
            win_rate=0.0,
            n_scenarios=0,
            scenario_results=(),
        )

    results: list[ScenarioResult] = []
    total_policy_value = 0.0
    total_oracle_value = 0.0
    wins = 0

    for state in scenarios:
        policy_action, policy_value = policy_fn(state)
        oracle_action, oracle_value = oracle_fn(state)

        matches = policy_action == oracle_action
        if matches:
            wins += 1

        results.append(
            ScenarioResult(
                state=state,
                policy_action=policy_action,
                oracle_action=oracle_action,
                policy_value=policy_value,
                oracle_value=oracle_value,
                policy_matches_oracle=matches,
            )
        )
        total_policy_value += policy_value
        total_oracle_value += oracle_value

    n = len(scenarios)
    mean_policy = total_policy_value / n
    mean_oracle = total_oracle_value / n
    ratio = mean_policy / mean_oracle if mean_oracle != 0.0 else 0.0

    return EvalResult(
        mean_policy_value=mean_policy,
        mean_oracle_value=mean_oracle,
        value_ratio=ratio,
        win_rate=wins / n,
        n_scenarios=n,
        scenario_results=tuple(results),
    )

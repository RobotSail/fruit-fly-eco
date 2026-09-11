"""Fly Confidence Index (FCI) — composite health metric for the avatar.

FCI ∈ [0, 1] bridges raw neural telemetry to operator-readable affect.
"""

from __future__ import annotations


def compute_fci(
    kc_mean_rate: float,
    recent_reward: float,
    consecutive_wins: int,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    kc_rate_max: float = 20.0,
    reward_max: float = 16_000.0,
    wins_max: int = 5,
) -> float:
    """Compute the Fly Confidence Index.

    Args:
        kc_mean_rate: Mean Kenyon-cell firing rate (Hz).
        recent_reward: Recent cumulative reward (money).
        consecutive_wins: Number of consecutive round wins.
        weights: Relative weights for (kc, reward, wins) components.
        kc_rate_max: Maximum expected KC rate for normalization.
        reward_max: Maximum expected reward for normalization.
        wins_max: Maximum expected consecutive wins for normalization.

    Returns:
        FCI value in [0, 1].
    """
    w_kc, w_reward, w_wins = weights

    # Normalize each component to [0, 1]
    kc_norm = max(0.0, min(kc_mean_rate / max(kc_rate_max, 1e-8), 1.0))
    reward_norm = max(0.0, min(recent_reward / max(reward_max, 1e-8), 1.0))
    wins_norm = max(0.0, min(consecutive_wins / max(wins_max, 1), 1.0))

    # Weighted sum, clamped to [0, 1]
    total_weight = w_kc + w_reward + w_wins
    if total_weight <= 0:
        return 0.0

    fci = (w_kc * kc_norm + w_reward * reward_norm + w_wins * wins_norm) / total_weight
    return max(0.0, min(fci, 1.0))


def recovery_subroutine(fci_history: list[float], threshold: float = 0.4, window: int = 3) -> bool:
    """Check if recovery actions should be triggered.

    Returns True if the last *window* FCI readings are all below *threshold*.
    This triggers recovery per README §5 Panel 1.
    """
    if len(fci_history) < window:
        return False
    return all(v < threshold for v in fci_history[-window:])

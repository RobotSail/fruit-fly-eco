"""Shared test fixtures for FLY//ECON."""

from __future__ import annotations

import pytest

from flyecon.state.economy import EconomyState, pistol_round_state


@pytest.fixture
def pistol_state() -> EconomyState:
    """Fresh pistol round state (half 0)."""
    return pistol_round_state(half=0)


@pytest.fixture
def mid_game_state() -> EconomyState:
    """Mid-game state with some money and loss streak."""
    return EconomyState(
        money=4_500,
        loss_streak=2,
        round_number=6,
        half=0,
        opponent_loss_streak=1,
    )

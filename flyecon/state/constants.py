"""All constants for PROJECT FLY//ECON.

Units convention:
  - All voltages in millivolts (mV)
  - All times in milliseconds (ms)
  - All money in USD integer dollars
"""

from __future__ import annotations

# ── LIF neuron parameters (all mV / ms) ──────────────────────────────────────
V_REST_MV: float = -52.0
"""Resting membrane potential (mV)."""

V_THRESH_MV: float = -45.0
"""Spike threshold potential (mV)."""

TAU_MS: float = 20.0
"""Membrane time constant (ms)."""

DT_MS: float = 5.0
"""Simulation time step (ms). Exact/exponential integration is unconditionally
stable at this step size."""

REFRACTORY_MS: float = 2.0
"""Absolute refractory period (ms)."""

SUBTHRESHOLD_RANGE_MV: float = V_THRESH_MV - V_REST_MV
"""Dynamic range between rest and threshold (mV). Should be 7.0 mV."""

# ── Neurotransmitter sign convention ──────────────────────────────────────────
# ACh is excitatory in Drosophila; GABA, Glutamate, Histamine are inhibitory.
# Dopamine is handled as a separate modulation tensor, not fast-synaptic.
NT_SIGN: dict[str, int] = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
}

DOPAMINE_NTS: list[str] = ["dopamine"]
"""Neurotransmitters treated as reward-modulation (separate tensor)."""

AMINERGIC_ZERO_CURRENT: list[str] = ["serotonin", "octopamine", "tyramine"]
"""Aminergic neuromodulators mapped to zero current."""

MIN_CONFIDENCE: float = 0.5
"""Minimum synapse confidence for inclusion in weight matrix."""

# ── MR12 economy constants ────────────────────────────────────────────────────
STARTING_MONEY: int = 800
"""Pistol round starting money ($)."""

MAX_MONEY: int = 16_000
"""Maximum money cap ($)."""

MIN_MONEY: int = 0
"""Minimum money floor ($)."""

MONEY_STEP: int = 100
"""Discretization step for Oracle MDP ($)."""

WIN_REWARD: int = 3_250
"""Base money awarded for winning a round ($)."""

LOSS_BONUS_LADDER: tuple[int, ...] = (1_400, 1_900, 2_400, 2_900, 3_400)
"""Loss bonus by consecutive loss streak (0-indexed). Streak 0 → $1400, etc."""

MAX_LOSS_STREAK: int = len(LOSS_BONUS_LADDER) - 1
"""Maximum loss streak index (4)."""

KILL_REWARDS: dict[str, int] = {
    "knife": 1_500,
    "pistol": 300,
    "smg": 600,
    "rifle": 300,
    "sniper": 100,
    "shotgun": 900,
    "grenade": 300,
    "zeus": 0,
}
"""Kill reward by weapon category ($)."""

ROUNDS_PER_HALF: int = 12
"""Number of rounds per half in MR12."""

NUM_HALVES: int = 2
"""Number of halves in a match."""

TOTAL_ROUNDS: int = ROUNDS_PER_HALF * NUM_HALVES
"""Maximum rounds in regulation (24)."""

# ── Buy plan cost estimates (representative loadout costs) ────────────────────
BUY_PLAN_COSTS: dict[str, int] = {
    "FULL_BUY": 4_750,
    "FORCE_BUY": 2_000,
    "HALF_BUY": 3_100,
    "ECO": 400,
    "SAVE": 0,
}
"""Approximate cost for each buy plan ($)."""

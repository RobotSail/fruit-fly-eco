"""Tests for flyecon/state/constants.py — units consistency and coverage."""

from __future__ import annotations

from flyecon.state.constants import (
    AMINERGIC_ZERO_CURRENT,
    DOPAMINE_NTS,
    DT_MS,
    LOSS_BONUS_LADDER,
    MAX_LOSS_STREAK,
    MIN_CONFIDENCE,
    NT_SIGN,
    REFRACTORY_MS,
    SUBTHRESHOLD_RANGE_MV,
    TAU_MS,
    V_REST_MV,
    V_THRESH_MV,
)


class TestLIFUnitsConsistency:
    """All LIF parameters must use consistent units (mV, ms)."""

    def test_voltages_are_in_mv_range(self) -> None:
        """Biologically plausible range: V_rest in [-60, -40], V_thresh in [-50, -35]."""
        assert -60.0 <= V_REST_MV <= -40.0, f"V_REST_MV={V_REST_MV} out of range"
        assert -50.0 <= V_THRESH_MV <= -35.0, f"V_THRESH_MV={V_THRESH_MV} out of range"

    def test_threshold_above_rest(self) -> None:
        """Threshold must be above rest for spiking to work."""
        assert V_THRESH_MV > V_REST_MV

    def test_subthreshold_range_is_7mv(self) -> None:
        """The dynamic range between rest and threshold is exactly 7.0 mV."""
        assert SUBTHRESHOLD_RANGE_MV == 7.0
        assert V_THRESH_MV - V_REST_MV == 7.0

    def test_time_constants_in_ms(self) -> None:
        """Time constants must be in milliseconds (plausible: 1–100 ms)."""
        assert 1.0 <= TAU_MS <= 100.0, f"TAU_MS={TAU_MS} not in ms range"
        assert 0.01 <= DT_MS <= 50.0, f"DT_MS={DT_MS} not in ms range"
        assert 0.5 <= REFRACTORY_MS <= 10.0, f"REFRACTORY_MS={REFRACTORY_MS} not in ms range"

    def test_dt_less_than_tau(self) -> None:
        """Time step must be smaller than the membrane time constant."""
        assert DT_MS < TAU_MS


class TestNTSignMap:
    """Neurotransmitter sign map must cover known invertebrate transmitters."""

    def test_ach_is_excitatory(self) -> None:
        assert NT_SIGN["acetylcholine"] == +1

    def test_gaba_is_inhibitory(self) -> None:
        assert NT_SIGN["gaba"] == -1

    def test_glutamate_is_inhibitory(self) -> None:
        """In Drosophila, glutamate is inhibitory (unlike vertebrates)."""
        assert NT_SIGN["glutamate"] == -1

    def test_histamine_is_inhibitory(self) -> None:
        assert NT_SIGN["histamine"] == -1

    def test_known_transmitters_covered(self) -> None:
        """All known fast-synaptic transmitters in Drosophila must be in the sign map."""
        required = {"acetylcholine", "gaba", "glutamate", "histamine"}
        assert required.issubset(NT_SIGN.keys())

    def test_dopamine_is_separate(self) -> None:
        """Dopamine must NOT be in the fast-synaptic sign map."""
        assert "dopamine" not in NT_SIGN
        assert "dopamine" in DOPAMINE_NTS

    def test_aminergic_neuromodulators_are_zero(self) -> None:
        """Serotonin, octopamine, tyramine → zero current."""
        for nt in AMINERGIC_ZERO_CURRENT:
            assert nt not in NT_SIGN, f"{nt} should not be in fast-synaptic sign map"

    def test_sign_values_are_plus_or_minus_one(self) -> None:
        """All signs must be exactly +1 or -1."""
        for nt, sign in NT_SIGN.items():
            assert sign in (+1, -1), f"NT_SIGN[{nt}] = {sign}, expected +1 or -1"


class TestConfidenceThreshold:
    def test_min_confidence_range(self) -> None:
        assert 0.0 < MIN_CONFIDENCE <= 1.0

    def test_min_confidence_value(self) -> None:
        assert MIN_CONFIDENCE == 0.5


class TestEconomyConstants:
    def test_loss_bonus_ladder_length(self) -> None:
        assert len(LOSS_BONUS_LADDER) == 5

    def test_loss_bonus_ladder_progression(self) -> None:
        """Loss bonus increases monotonically: 1400 → 1900 → 2400 → 2900 → 3400."""
        expected = (1_400, 1_900, 2_400, 2_900, 3_400)
        assert LOSS_BONUS_LADDER == expected

    def test_loss_bonus_ladder_increment(self) -> None:
        """Each step in the ladder adds $500."""
        for i in range(1, len(LOSS_BONUS_LADDER)):
            assert LOSS_BONUS_LADDER[i] - LOSS_BONUS_LADDER[i - 1] == 500

    def test_max_loss_streak_matches_ladder(self) -> None:
        assert MAX_LOSS_STREAK == len(LOSS_BONUS_LADDER) - 1

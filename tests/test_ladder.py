"""Tests for flyecon.resilience.ladder — degradation ladder with hysteresis."""

from __future__ import annotations

import pytest

from flyecon.resilience.ladder import (
    CPU_MIN_COUNT,
    DISK_CRITICAL_MB,
    DISK_LOW_MB,
    MAX_LEVEL,
    RAM_CRITICAL_MB,
    RAM_LOW_MB,
    DegradationLadder,
    ResourceReport,
    TrainingConfig,
    probe_resources,
)


def _report(
    ram_mb: float = 8000.0,
    cpus: int = 4,
    disk_mb: float = 10000.0,
    responsive: bool = True,
) -> ResourceReport:
    """Build a ResourceReport with convenient defaults."""
    return ResourceReport(
        available_ram_mb=ram_mb,
        cpu_count=cpus,
        available_disk_mb=disk_mb,
        training_responsive=responsive,
    )


class TestDecideLevel:
    """Level transitions based on resource reports."""

    def test_healthy_stays_at_zero(self) -> None:
        """Abundant resources keep the ladder at level 0."""
        ladder = DegradationLadder()
        level = ladder.decide_level(_report())
        assert level == 0

    def test_critical_ram_goes_to_5(self) -> None:
        """RAM below critical threshold → level 5."""
        ladder = DegradationLadder()
        level = ladder.decide_level(_report(ram_mb=RAM_CRITICAL_MB - 1))
        assert level == MAX_LEVEL

    def test_low_ram_goes_to_4(self) -> None:
        """RAM below low threshold but above critical → level 4."""
        ladder = DegradationLadder()
        level = ladder.decide_level(
            _report(ram_mb=RAM_LOW_MB - 1)
        )
        assert level == 4

    def test_low_disk_goes_to_3(self) -> None:
        """Disk below low threshold → level 3."""
        ladder = DegradationLadder()
        level = ladder.decide_level(
            _report(disk_mb=DISK_LOW_MB - 1)
        )
        assert level == 3

    def test_critical_disk_goes_to_5(self) -> None:
        """Disk below critical threshold → level 5."""
        ladder = DegradationLadder()
        level = ladder.decide_level(
            _report(disk_mb=DISK_CRITICAL_MB - 1)
        )
        assert level == MAX_LEVEL

    def test_unresponsive_goes_to_5(self) -> None:
        """Training not responsive → level 5."""
        ladder = DegradationLadder()
        level = ladder.decide_level(_report(responsive=False))
        assert level == MAX_LEVEL

    def test_step_up_is_immediate(self) -> None:
        """Stepping UP (degrading) is instant on resource shortage."""
        ladder = DegradationLadder()
        ladder.decide_level(_report())  # start at 0
        assert ladder.current_level == 0

        level = ladder.decide_level(_report(ram_mb=RAM_CRITICAL_MB - 1))
        assert level == MAX_LEVEL

    def test_step_down_requires_hysteresis(self) -> None:
        """Stepping DOWN requires consecutive healthy probes."""
        ladder = DegradationLadder(hysteresis_count=2)
        # Force to level 5
        ladder.decide_level(_report(ram_mb=RAM_CRITICAL_MB - 1))
        assert ladder.current_level == 5

        # First healthy probe — stays at 5 (step down by 1 requires 2)
        ladder.decide_level(_report())
        assert ladder.current_level == 5

        # Second healthy probe — steps down by 1 to 4
        ladder.decide_level(_report())
        assert ladder.current_level == 4

    def test_recovery_is_gradual(self) -> None:
        """Full recovery from level 5 → 0 requires multiple passes."""
        ladder = DegradationLadder(hysteresis_count=2)
        ladder.decide_level(_report(responsive=False))
        assert ladder.current_level == 5

        # Need 2 probes per level to step down
        for expected_level in [4, 3, 2, 1, 0]:
            ladder.decide_level(_report())
            ladder.decide_level(_report())
            assert ladder.current_level == expected_level


class TestApplyLevel:
    """Level → TrainingConfig mapping."""

    def test_level_0_full_operation(self) -> None:
        """Level 0 enables everything."""
        ladder = DegradationLadder()
        cfg = ladder.apply_level(0)
        assert cfg.training_enabled
        assert cfg.eval_enabled
        assert cfg.dashboard_enabled
        assert cfg.avatar_enabled
        assert cfg.n_candidates > 1

    def test_level_4_freezes_training(self) -> None:
        """Level 4 disables training but keeps eval + dashboard."""
        ladder = DegradationLadder()
        cfg = ladder.apply_level(4)
        assert not cfg.training_enabled
        assert cfg.eval_enabled
        assert cfg.dashboard_enabled
        assert cfg.avatar_enabled

    def test_level_5_patrolling(self) -> None:
        """Level 5 disables everything except dashboard + avatar."""
        ladder = DegradationLadder()
        cfg = ladder.apply_level(5)
        assert not cfg.training_enabled
        assert not cfg.eval_enabled
        assert cfg.dashboard_enabled
        assert cfg.avatar_enabled

    def test_all_levels_have_avatar(self) -> None:
        """Avatar renders at ALL degradation levels."""
        ladder = DegradationLadder()
        for level in range(MAX_LEVEL + 1):
            cfg = ladder.apply_level(level)
            assert cfg.avatar_enabled, f"Avatar disabled at level {level}"
            assert cfg.dashboard_enabled, f"Dashboard disabled at level {level}"


class TestProbeResources:
    """probe_resources() returns a valid ResourceReport."""

    def test_returns_report(self) -> None:
        """probe_resources returns non-negative values."""
        report = probe_resources()
        assert report.available_ram_mb > 0
        assert report.cpu_count >= 1
        assert report.available_disk_mb > 0
        assert report.training_responsive is True

"""Degradation ladder — 5 levels per README §3.9.

Level 0: Full operation (multiple candidates, full batch, all backends).
Level 1: One candidate instead of many.
Level 2: Smaller batch / shorter horizon.
Level 3: Cheapest backend only (CPU, smallest possible simulation).
Level 4: Freeze training; keep evaluating and dashboarding last live candidate.
Level 5: Poll-and-retry for resources; resume the instant they return.
         Patrolling. Avatar still renders.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()

MAX_LEVEL: int = 5
MIN_LEVEL: int = 0

# ── Resource thresholds ──────────────────────────────────────────────────────
# These define when to step up the degradation ladder.
RAM_CRITICAL_MB: int = 512
RAM_LOW_MB: int = 1024
DISK_CRITICAL_MB: int = 256
DISK_LOW_MB: int = 512
CPU_MIN_COUNT: int = 1


@dataclass(frozen=True, slots=True)
class ResourceReport:
    """Snapshot of available system resources."""

    available_ram_mb: float
    cpu_count: int
    available_disk_mb: float
    training_responsive: bool = True


@dataclass
class TrainingConfig:
    """Mutable training configuration adjusted by the ladder."""

    n_candidates: int = 3
    batch_size: int = 128
    horizon_steps: int = 128
    training_enabled: bool = True
    eval_enabled: bool = True
    dashboard_enabled: bool = True
    avatar_enabled: bool = True

    def describe(self) -> str:
        """Human-readable config summary."""
        parts = [
            f"candidates={self.n_candidates}",
            f"batch={self.batch_size}",
            f"horizon={self.horizon_steps}",
            f"train={'on' if self.training_enabled else 'off'}",
            f"eval={'on' if self.eval_enabled else 'off'}",
            f"dash={'on' if self.dashboard_enabled else 'off'}",
            f"avatar={'on' if self.avatar_enabled else 'off'}",
        ]
        return ", ".join(parts)


class DegradationLadder:
    """Manages resource-aware degradation with hysteresis.

    Stepping UP (degrading) is immediate on resource shortage.
    Stepping DOWN requires *hysteresis_count* consecutive healthy probes
    to avoid oscillation between levels.
    """

    def __init__(self, hysteresis_count: int = 2) -> None:
        self.current_level: int = 0
        self.hysteresis_count = hysteresis_count
        self._consecutive_healthy: int = 0

    def decide_level(
        self, report: ResourceReport, current_level: int | None = None
    ) -> int:
        """Decide the appropriate ladder level given a resource report.

        Returns the new level. Updates internal state.
        """
        if current_level is not None:
            self.current_level = current_level

        desired = self._desired_level(report)

        if desired > self.current_level:
            # Step up immediately on resource shortage
            self._consecutive_healthy = 0
            self.current_level = min(desired, MAX_LEVEL)
            log.info(
                "ladder_step_up",
                new_level=self.current_level,
                ram_mb=report.available_ram_mb,
                disk_mb=report.available_disk_mb,
            )
        elif desired < self.current_level:
            # Step down only after consecutive healthy probes (hysteresis)
            self._consecutive_healthy += 1
            if self._consecutive_healthy >= self.hysteresis_count:
                self.current_level = max(self.current_level - 1, MIN_LEVEL)
                self._consecutive_healthy = 0
                log.info(
                    "ladder_step_down",
                    new_level=self.current_level,
                    consecutive_healthy=self.hysteresis_count,
                )
        else:
            # Same level — reset healthy counter
            self._consecutive_healthy = 0

        return self.current_level

    def _desired_level(self, report: ResourceReport) -> int:
        """Compute the ideal level ignoring hysteresis."""
        if not report.training_responsive:
            return 5

        if (
            report.available_ram_mb < RAM_CRITICAL_MB
            or report.available_disk_mb < DISK_CRITICAL_MB
        ):
            return 5

        if report.available_ram_mb < RAM_LOW_MB:
            return 4

        if report.available_disk_mb < DISK_LOW_MB:
            return 3

        if report.cpu_count <= CPU_MIN_COUNT:
            return 2

        return 0

    def apply_level(
        self, level: int, config: TrainingConfig | None = None
    ) -> TrainingConfig:
        """Return a TrainingConfig adjusted for the given degradation level."""
        cfg = config or TrainingConfig()

        if level == 0:
            # Full operation
            return TrainingConfig(
                n_candidates=cfg.n_candidates,
                batch_size=cfg.batch_size,
                horizon_steps=cfg.horizon_steps,
                training_enabled=True,
                eval_enabled=True,
                dashboard_enabled=True,
                avatar_enabled=True,
            )

        if level == 1:
            return TrainingConfig(
                n_candidates=1,
                batch_size=cfg.batch_size,
                horizon_steps=cfg.horizon_steps,
                training_enabled=True,
                eval_enabled=True,
                dashboard_enabled=True,
                avatar_enabled=True,
            )

        if level == 2:
            return TrainingConfig(
                n_candidates=1,
                batch_size=max(cfg.batch_size // 2, 16),
                horizon_steps=max(cfg.horizon_steps // 2, 32),
                training_enabled=True,
                eval_enabled=True,
                dashboard_enabled=True,
                avatar_enabled=True,
            )

        if level == 3:
            return TrainingConfig(
                n_candidates=1,
                batch_size=16,
                horizon_steps=32,
                training_enabled=True,
                eval_enabled=True,
                dashboard_enabled=True,
                avatar_enabled=True,
            )

        if level == 4:
            return TrainingConfig(
                n_candidates=1,
                batch_size=16,
                horizon_steps=32,
                training_enabled=False,
                eval_enabled=True,
                dashboard_enabled=True,
                avatar_enabled=True,
            )

        # Level 5: patrolling — everything off except dashboard/avatar
        return TrainingConfig(
            n_candidates=0,
            batch_size=0,
            horizon_steps=0,
            training_enabled=False,
            eval_enabled=False,
            dashboard_enabled=True,
            avatar_enabled=True,
        )


def probe_resources() -> ResourceReport:
    """Probe current system resources.

    Uses ``psutil`` if available, falls back to ``os`` / ``shutil``.
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        available_ram_mb = mem.available / (1024 * 1024)
    except ImportError:
        # Rough fallback — assume 4 GB if psutil unavailable
        available_ram_mb = 4096.0

    cpu_count = os.cpu_count() or 1

    try:
        disk = shutil.disk_usage(".")
        available_disk_mb = disk.free / (1024 * 1024)
    except OSError:
        available_disk_mb = 10_000.0

    return ResourceReport(
        available_ram_mb=available_ram_mb,
        cpu_count=cpu_count,
        available_disk_mb=available_disk_mb,
        training_responsive=True,
    )

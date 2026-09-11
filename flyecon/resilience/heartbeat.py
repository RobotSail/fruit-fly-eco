"""Heartbeat writer + watcher for component health monitoring.

Each component writes a JSON heartbeat file on a fixed interval.
The watcher scans the directory and flags stale components as DEAD.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Generator

import structlog

log = structlog.get_logger()


class HealthStatus(str, Enum):
    """Component health status."""

    ALIVE = "ALIVE"
    DEAD = "DEAD"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """Health report for a single component."""

    component: str
    status: HealthStatus
    last_beat_ns: int  # monotonic nanoseconds of last beat
    wall_time: str  # ISO-8601 wall time of last beat
    detail: str = ""
    age_s: float = 0.0  # seconds since last beat


class HeartbeatWriter:
    """Writes periodic heartbeat JSON files for one component.

    Usage::

        writer = HeartbeatWriter("training", Path("heartbeats/"))
        with writer.alive():
            do_work()  # beats are written every interval_s automatically

    Or manually::

        writer.beat()  # one-shot write
    """

    def __init__(
        self,
        component: str,
        heartbeat_dir: Path,
        interval_s: float = 30.0,
    ) -> None:
        self.component = component
        self.heartbeat_dir = Path(heartbeat_dir)
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self, detail: str = "") -> None:
        """Write one heartbeat file."""
        self.heartbeat_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "component": self.component,
            "timestamp_ns": time.monotonic_ns(),
            "wall_time": datetime.now(timezone.utc).isoformat(),
            "status": "alive",
            "detail": detail,
        }
        path = self.heartbeat_dir / f"{self.component}.json"
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload))
        os.replace(str(tmp_path), str(path))

    @contextmanager
    def alive(self, detail: str = "") -> Generator[None, None, None]:
        """Context manager that writes heartbeats on a background thread."""
        self._stop_event.clear()
        self.beat(detail)

        def _loop() -> None:
            while not self._stop_event.wait(timeout=self.interval_s):
                try:
                    self.beat(detail)
                except Exception as exc:
                    log.warning(
                        "heartbeat_write_failed",
                        component=self.component,
                        error=str(exc),
                    )

        self._thread = threading.Thread(
            target=_loop, daemon=True, name=f"heartbeat-{self.component}"
        )
        self._thread.start()
        try:
            yield
        finally:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            self._thread = None


class HeartbeatWatcher:
    """Scans heartbeat directory and reports component health.

    A component is DEAD if its last beat is older than *timeout_s*.
    """

    def __init__(self, heartbeat_dir: Path, timeout_s: float = 90.0) -> None:
        self.heartbeat_dir = Path(heartbeat_dir)
        self.timeout_s = timeout_s

    def check_all(self) -> list[ComponentHealth]:
        """Read all heartbeat files and return health reports."""
        results: list[ComponentHealth] = []
        if not self.heartbeat_dir.exists():
            return results

        now_ns = time.monotonic_ns()

        for path in sorted(self.heartbeat_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                component = data.get("component", path.stem)
                ts_ns = data.get("timestamp_ns", 0)
                wall_time = data.get("wall_time", "")
                detail = data.get("detail", "")

                age_s = (now_ns - ts_ns) / 1e9
                status = HealthStatus.ALIVE if age_s <= self.timeout_s else HealthStatus.DEAD

                results.append(
                    ComponentHealth(
                        component=component,
                        status=status,
                        last_beat_ns=ts_ns,
                        wall_time=wall_time,
                        detail=detail,
                        age_s=age_s,
                    )
                )
            except Exception as exc:
                results.append(
                    ComponentHealth(
                        component=path.stem,
                        status=HealthStatus.UNKNOWN,
                        last_beat_ns=0,
                        wall_time="",
                        detail=f"parse error: {exc}",
                    )
                )
        return results

    def dead_components(self) -> list[str]:
        """Return names of components whose heartbeats are stale."""
        return [
            ch.component
            for ch in self.check_all()
            if ch.status == HealthStatus.DEAD
        ]

"""Append-only JSONL telemetry store for mission observability.

The TelemetryStore is a pure consumer of training/eval data.
It never blocks the training loop.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass
class TelemetryEvent:
    """A single timestamped telemetry event.

    Events include training metrics, eval results, component health,
    ladder level changes, FCI values, avatar state transitions.
    """

    event_type: str
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to a single JSON line."""
        return json.dumps(
            {
                "event_type": self.event_type,
                "timestamp": self.timestamp,
                "data": self.data,
            },
            default=str,
        )

    @classmethod
    def from_json(cls, line: str) -> TelemetryEvent:
        """Deserialize from a JSON line."""
        d = json.loads(line)
        return cls(
            event_type=d["event_type"],
            timestamp=d.get("timestamp", 0.0),
            data=d.get("data", {}),
        )


class TelemetryStore:
    """Append-only JSONL file for all mission telemetry.

    Usage::

        store = TelemetryStore(Path("telemetry.jsonl"))
        store.append(TelemetryEvent("training_step", data={"loss": 0.5}))
        latest = store.read_latest(n=100)
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TelemetryEvent) -> None:
        """Append one event as a JSON line. Fsyncs for durability."""
        line = event.to_json() + "\n"
        with open(self.path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def read_latest(self, n: int = 1000) -> list[TelemetryEvent]:
        """Read the last *n* events from the store.

        Reads the tail of the file efficiently for dashboard rendering.
        """
        if not self.path.exists():
            return []

        lines: list[str] = []
        try:
            with open(self.path) as f:
                all_lines = f.readlines()
                lines = all_lines[-n:]
        except Exception as exc:
            log.warning("telemetry_read_error", error=str(exc))
            return []

        events: list[TelemetryEvent] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(TelemetryEvent.from_json(line))
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("telemetry_parse_error", line=line[:80], error=str(exc))
                continue
        return events

    def count(self) -> int:
        """Count total events in the store."""
        if not self.path.exists():
            return 0
        with open(self.path) as f:
            return sum(1 for line in f if line.strip())

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

    def read_from_offset(self, byte_offset: int = 0) -> tuple[list[TelemetryEvent], int]:
        """Read new events starting from a byte offset.

        Opens the JSONL file, seeks to *byte_offset*, reads all complete
        lines from that point, and returns the parsed events together with
        the new byte offset (ready for the next call).

        Partial or corrupt lines are silently skipped — the returned offset
        always points past the last *successfully parsed* newline, so the
        next call will re-try any incomplete trailing data.

        Returns
        -------
        tuple[list[TelemetryEvent], int]
            (new_events, new_byte_offset).  Empty list + unchanged offset
            when the file does not exist or has no new data.
        """
        if not self.path.exists():
            return [], 0

        events: list[TelemetryEvent] = []
        new_offset = byte_offset

        try:
            with open(self.path, "rb") as f:
                f.seek(byte_offset)
                raw = f.read()
                new_offset = byte_offset + len(raw)
        except Exception as exc:
            log.warning("telemetry_read_offset_error", error=str(exc))
            return [], byte_offset

        # Walk through complete lines only
        text = raw.decode("utf-8", errors="replace")
        # If there's no trailing newline, the last chunk may be partial —
        # rewind offset to exclude it so the next call retries.
        if text and not text.endswith("\n"):
            last_nl = text.rfind("\n")
            if last_nl == -1:
                # Entire read is a partial line — return nothing, rewind
                return [], byte_offset
            # Exclude the partial tail
            new_offset = byte_offset + len(text[:last_nl + 1].encode("utf-8"))
            text = text[:last_nl + 1]

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(TelemetryEvent.from_json(line))
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning(
                    "telemetry_offset_parse_error", line=line[:80], error=str(exc),
                )
                continue

        return events, new_offset

    def count(self) -> int:
        """Count total events in the store."""
        if not self.path.exists():
            return 0
        with open(self.path) as f:
            return sum(1 for line in f if line.strip())

"""Tests for flyecon.resilience.heartbeat — writer + watcher."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from flyecon.resilience.heartbeat import (
    HealthStatus,
    HeartbeatWatcher,
    HeartbeatWriter,
)


@pytest.fixture()
def hb_dir(tmp_path: Path) -> Path:
    """Provide a temporary heartbeat directory."""
    return tmp_path / "heartbeats"


class TestHeartbeatWriter:
    """HeartbeatWriter creates valid JSON heartbeat files."""

    def test_beat_creates_json(self, hb_dir: Path) -> None:
        """A single beat() call creates a valid JSON file."""
        writer = HeartbeatWriter("training", hb_dir)
        writer.beat(detail="step 100")

        path = hb_dir / "training.json"
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["component"] == "training"
        assert data["status"] == "alive"
        assert data["detail"] == "step 100"
        assert "timestamp_ns" in data
        assert "wall_time" in data

    def test_beat_updates_file(self, hb_dir: Path) -> None:
        """Subsequent beats overwrite the same file."""
        writer = HeartbeatWriter("sim", hb_dir)
        writer.beat(detail="first")
        d1 = json.loads((hb_dir / "sim.json").read_text())

        time.sleep(0.01)
        writer.beat(detail="second")
        d2 = json.loads((hb_dir / "sim.json").read_text())

        assert d2["detail"] == "second"
        assert d2["timestamp_ns"] >= d1["timestamp_ns"]

    def test_alive_context_manager(self, hb_dir: Path) -> None:
        """alive() context manager writes at least one heartbeat."""
        writer = HeartbeatWriter("dashboard", hb_dir, interval_s=0.05)
        with writer.alive(detail="running"):
            time.sleep(0.12)

        path = hb_dir / "dashboard.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["component"] == "dashboard"


class TestHeartbeatWatcher:
    """HeartbeatWatcher detects stale heartbeats."""

    def test_detects_alive(self, hb_dir: Path) -> None:
        """Fresh heartbeat is reported as ALIVE."""
        writer = HeartbeatWriter("etl", hb_dir)
        writer.beat()

        watcher = HeartbeatWatcher(hb_dir, timeout_s=90)
        reports = watcher.check_all()
        assert len(reports) == 1
        assert reports[0].component == "etl"
        assert reports[0].status == HealthStatus.ALIVE

    def test_detects_stale(self, hb_dir: Path) -> None:
        """Old heartbeat with past timestamp is reported as DEAD."""
        hb_dir.mkdir(parents=True, exist_ok=True)
        # Write a heartbeat with a very old timestamp_ns
        old_ts = time.monotonic_ns() - int(200 * 1e9)  # 200 seconds ago
        path = hb_dir / "training.json"
        path.write_text(json.dumps({
            "component": "training",
            "timestamp_ns": old_ts,
            "wall_time": "2025-01-01T00:00:00+00:00",
            "status": "alive",
            "detail": "",
        }))

        watcher = HeartbeatWatcher(hb_dir, timeout_s=90)
        reports = watcher.check_all()
        assert len(reports) == 1
        assert reports[0].status == HealthStatus.DEAD

    def test_dead_components(self, hb_dir: Path) -> None:
        """dead_components() returns only stale component names."""
        hb_dir.mkdir(parents=True, exist_ok=True)

        # Fresh
        HeartbeatWriter("fresh", hb_dir).beat()

        # Stale
        old_ts = time.monotonic_ns() - int(200 * 1e9)
        (hb_dir / "stale.json").write_text(json.dumps({
            "component": "stale",
            "timestamp_ns": old_ts,
            "wall_time": "",
            "status": "alive",
            "detail": "",
        }))

        watcher = HeartbeatWatcher(hb_dir, timeout_s=90)
        dead = watcher.dead_components()
        assert "stale" in dead
        assert "fresh" not in dead

    def test_empty_dir(self, hb_dir: Path) -> None:
        """Watcher returns empty list for nonexistent directory."""
        watcher = HeartbeatWatcher(hb_dir, timeout_s=90)
        assert watcher.check_all() == []

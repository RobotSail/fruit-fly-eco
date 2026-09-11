"""Tests for flyecon.dashboard — telemetry, renderer, FCI."""

from __future__ import annotations

from pathlib import Path

import pytest

from flyecon.dashboard.fci import compute_fci, recovery_subroutine
from flyecon.dashboard.renderer import render_dashboard
from flyecon.dashboard.telemetry import TelemetryEvent, TelemetryStore


@pytest.fixture()
def store(tmp_path: Path) -> TelemetryStore:
    """Provide a fresh TelemetryStore in a temp directory."""
    return TelemetryStore(tmp_path / "telemetry.jsonl")


# ── TelemetryStore ───────────────────────────────────────────────────────────


class TestTelemetryStore:
    """Append-only JSONL store."""

    def test_append_and_read(self, store: TelemetryStore) -> None:
        """Events survive a write→read round trip."""
        store.append(TelemetryEvent("test", data={"value": 42}))
        events = store.read_latest(n=10)
        assert len(events) == 1
        assert events[0].event_type == "test"
        assert events[0].data["value"] == 42

    def test_multiple_events(self, store: TelemetryStore) -> None:
        """Multiple appends are all readable."""
        for i in range(5):
            store.append(TelemetryEvent("step", data={"i": i}))
        events = store.read_latest(n=10)
        assert len(events) == 5

    def test_read_latest_limit(self, store: TelemetryStore) -> None:
        """read_latest returns at most n events."""
        for i in range(10):
            store.append(TelemetryEvent("step", data={"i": i}))
        events = store.read_latest(n=3)
        assert len(events) == 3
        # Should be the last 3
        assert events[0].data["i"] == 7
        assert events[2].data["i"] == 9

    def test_empty_store(self, store: TelemetryStore) -> None:
        """Empty store returns empty list."""
        assert store.read_latest(n=10) == []

    def test_count(self, store: TelemetryStore) -> None:
        """count() returns total events."""
        assert store.count() == 0
        store.append(TelemetryEvent("a"))
        store.append(TelemetryEvent("b"))
        assert store.count() == 2


# ── FCI ──────────────────────────────────────────────────────────────────────


class TestFCI:
    """Fly Confidence Index computation."""

    def test_all_zero_gives_zero(self) -> None:
        """All-zero inputs → FCI near 0."""
        fci = compute_fci(kc_mean_rate=0.0, recent_reward=0.0, consecutive_wins=0)
        assert fci == pytest.approx(0.0, abs=0.01)

    def test_all_max_gives_one(self) -> None:
        """All-max inputs → FCI near 1."""
        fci = compute_fci(
            kc_mean_rate=20.0,
            recent_reward=16_000.0,
            consecutive_wins=5,
        )
        assert fci == pytest.approx(1.0, abs=0.01)

    def test_mid_values(self) -> None:
        """Mid-range inputs → FCI around 0.5."""
        fci = compute_fci(
            kc_mean_rate=10.0,
            recent_reward=8_000.0,
            consecutive_wins=2,
        )
        assert 0.3 < fci < 0.7

    def test_clamped_to_unit(self) -> None:
        """FCI is always in [0, 1] even with extreme inputs."""
        fci = compute_fci(kc_mean_rate=1000.0, recent_reward=1e9, consecutive_wins=100)
        assert 0.0 <= fci <= 1.0

        fci2 = compute_fci(kc_mean_rate=-10.0, recent_reward=-1000.0, consecutive_wins=-5)
        assert 0.0 <= fci2 <= 1.0


class TestRecoverySubroutine:
    """recovery_subroutine triggers on sustained low FCI."""

    def test_triggers_on_sustained_low(self) -> None:
        """3+ consecutive readings below threshold → True."""
        history = [0.3, 0.2, 0.1]
        assert recovery_subroutine(history, threshold=0.4, window=3) is True

    def test_no_trigger_when_mixed(self) -> None:
        """Mixed readings → False."""
        history = [0.3, 0.5, 0.2]
        assert recovery_subroutine(history, threshold=0.4, window=3) is False

    def test_no_trigger_insufficient_data(self) -> None:
        """Too few readings → False."""
        history = [0.1]
        assert recovery_subroutine(history, threshold=0.4, window=3) is False

    def test_empty_history(self) -> None:
        """Empty history → False."""
        assert recovery_subroutine([], threshold=0.4, window=3) is False


# ── Dashboard Renderer ───────────────────────────────────────────────────────


class TestDashboardRenderer:
    """render_dashboard produces valid self-contained HTML."""

    def test_renders_html(self, store: TelemetryStore, tmp_path: Path) -> None:
        """Dashboard produces a valid HTML file with no external resource loads."""
        # Add some telemetry events
        store.append(TelemetryEvent("fci", data={"fci": 0.65}))
        store.append(TelemetryEvent("mission_state", data={
            "uptime_seconds": 3600,
            "ladder_level": 0,
            "n_candidates": 1,
            "cycles_completed": 10,
        }))

        output = tmp_path / "dashboard.html"
        result = render_dashboard(store, output)

        assert result.exists()
        html = result.read_text()

        # Self-contained: no external <script src>, <link href>, or <img src>
        # (the inlined plotly.js may contain http:// in strings/comments, which
        # is fine — the dashboard loads no external resources)
        assert '<script src="http' not in html
        assert '<link href="http' not in html
        assert '<img src="http' not in html

    def test_all_panels_present(self, store: TelemetryStore, tmp_path: Path) -> None:
        """Dashboard contains all 10 panel div IDs."""
        store.append(TelemetryEvent("fci", data={"fci": 0.5}))
        output = tmp_path / "dashboard.html"
        render_dashboard(store, output)

        html = output.read_text()

        expected_panels = [
            "panel-avatar",
            "panel-fci",
            "panel-dopamine",
            "panel-distress",
            "panel-ladder",
            "panel-roi",
            "panel-sentiment",
            "panel-champions",
            "panel-grief",
            "panel-perpetuity",
            "panel-nanny",
        ]
        for panel_id in expected_panels:
            assert panel_id in html, f"Missing panel: {panel_id}"

    def test_contains_plotly(self, store: TelemetryStore, tmp_path: Path) -> None:
        """Dashboard contains inline Plotly (full or stub)."""
        store.append(TelemetryEvent("fci", data={"fci": 0.5}))
        output = tmp_path / "dashboard.html"
        render_dashboard(store, output)
        html = output.read_text()
        assert "Plotly" in html

    def test_contains_avatar(self, store: TelemetryStore, tmp_path: Path) -> None:
        """Dashboard contains the ASCII avatar."""
        store.append(TelemetryEvent("fci", data={"fci": 0.8}))
        output = tmp_path / "dashboard.html"
        render_dashboard(store, output)
        html = output.read_text()
        assert "<pre" in html

    def test_empty_telemetry(self, store: TelemetryStore, tmp_path: Path) -> None:
        """Dashboard renders even with no telemetry events."""
        output = tmp_path / "dashboard.html"
        result = render_dashboard(store, output)
        assert result.exists()
        html = result.read_text()
        assert "FLY//ECON" in html

    def test_uptime_banner(self, store: TelemetryStore, tmp_path: Path) -> None:
        """Dashboard has the uptime banner."""
        store.append(TelemetryEvent("mission_state", data={
            "uptime_seconds": 86400,
        }))
        output = tmp_path / "dashboard.html"
        render_dashboard(store, output)
        html = output.read_text()
        assert "1d" in html

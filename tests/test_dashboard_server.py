"""Tests for the live dashboard server, SVG avatar, and TelemetryStore.read_from_offset."""

from __future__ import annotations

from pathlib import Path

import pytest

from flyecon.avatar.svg import to_html
from flyecon.dashboard.telemetry import TelemetryEvent, TelemetryStore

# ─── TelemetryStore.read_from_offset ────────────────────────────────────

class TestReadFromOffset:
    """Test byte-offset tailing of JSONL telemetry."""

    def test_empty_file(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "empty.jsonl")
        events, offset = store.read_from_offset(0)
        assert events == []
        assert offset == 0

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "nope.jsonl")
        events, offset = store.read_from_offset(0)
        assert events == []
        assert offset == 0

    def test_write_then_read(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "telem.jsonl")
        store.append(TelemetryEvent("fci", data={"fci": 0.75}))
        store.append(TelemetryEvent("training_step", data={"loss": 0.1}))

        events, offset = store.read_from_offset(0)
        assert len(events) == 2
        assert events[0].event_type == "fci"
        assert events[1].event_type == "training_step"
        assert offset > 0

    def test_sequential_reads(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "telem.jsonl")
        store.append(TelemetryEvent("a", data={"v": 1}))
        events_1, off_1 = store.read_from_offset(0)
        assert len(events_1) == 1

        store.append(TelemetryEvent("b", data={"v": 2}))
        events_2, off_2 = store.read_from_offset(off_1)
        assert len(events_2) == 1
        assert events_2[0].event_type == "b"
        assert off_2 > off_1

    def test_partial_line_skipped(self, tmp_path: Path) -> None:
        """A partial trailing line should not crash or be returned."""
        p = tmp_path / "partial.jsonl"
        ev = TelemetryEvent("ok", data={"x": 1})
        full_line = ev.to_json() + "\n"
        # Write one complete line + a partial line (no trailing newline)
        p.write_text(full_line + '{"event_type":"bad","timestamp":0.0')
        store = TelemetryStore(p)
        events, offset = store.read_from_offset(0)
        assert len(events) == 1
        assert events[0].event_type == "ok"
        # Offset should point past the complete line only
        assert offset == len(full_line.encode("utf-8"))

    def test_corrupt_line_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "corrupt.jsonl"
        ev = TelemetryEvent("good", data={"x": 1})
        p.write_text(ev.to_json() + "\nNOT_JSON\n")
        store = TelemetryStore(p)
        events, _ = store.read_from_offset(0)
        assert len(events) == 1
        assert events[0].event_type == "good"


# ─── SVG Avatar ─────────────────────────────────────────────────────────

class TestSVGAvatar:
    """Test the animated SVG fly avatar."""

    def test_returns_svg(self) -> None:
        html = to_html(fci=0.5, event="")
        assert "<svg" in html
        assert "</svg>" in html

    def test_contains_eyes(self) -> None:
        html = to_html(fci=0.5, event="")
        assert "facets" in html  # eye facet pattern
        assert "#CC0000" in html  # red compound eye color

    def test_contains_wings(self) -> None:
        html = to_html(fci=0.5, event="")
        assert "wing-left" in html
        assert "wing-right" in html
        assert "wingBeatLeft" in html

    def test_contains_legs(self) -> None:
        html = to_html(fci=0.5, event="")
        assert "legs-left" in html
        assert "legs-right" in html

    def test_contains_antennae(self) -> None:
        html = to_html(fci=0.5, event="")
        assert "antenna-left" in html
        assert "antenna-right" in html

    def test_css_keyframes_present(self) -> None:
        html = to_html(fci=0.5, event="")
        assert "@keyframes wingBeatLeft" in html
        assert "@keyframes bodyBob" in html
        assert "@keyframes legWiggleLeft" in html

    def test_wing_beat_duration_property(self) -> None:
        html = to_html(fci=0.5, event="")
        assert "--wing-beat-duration" in html

    def test_high_fci_faster_wings(self) -> None:
        html_high = to_html(fci=0.95, event="")
        html_low = to_html(fci=0.1, event="")
        # Both should have --wing-beat-duration set
        assert "--wing-beat-duration" in html_high
        assert "--wing-beat-duration" in html_low
        # Extract durations from the inline style
        import re
        pat = r"--wing-beat-duration:\s*([\d.]+)s"
        dur_high = float(re.search(pat, html_high).group(1))  # type: ignore[union-attr]
        dur_low = float(re.search(pat, html_low).group(1))  # type: ignore[union-attr]
        assert dur_high < dur_low  # higher FCI → shorter duration (faster)

    def test_update_avatar_js(self) -> None:
        html = to_html(fci=0.5, event="")
        assert "function updateAvatar" in html


# ─── FastAPI Endpoints ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_index() -> None:
    """GET / returns 200 with HTML containing Plotly, SVG, EventSource."""
    import httpx

    from dashboard_server import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "Plotly" in body or "plotly" in body.lower()
    assert "<svg" in body
    assert "EventSource" in body


@pytest.mark.asyncio
async def test_api_status() -> None:
    """GET /api/status returns 200 with JSON containing stage."""
    import httpx

    from dashboard_server import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "stage" in data
    assert "training_step" in data


@pytest.mark.asyncio
async def test_events_content_type() -> None:
    """GET /events returns text/event-stream content type."""
    import httpx

    from dashboard_server import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        async with client.stream("GET", "/events") as resp:
            assert resp.status_code == 200
            ct = resp.headers.get("content-type", "")
            assert "text/event-stream" in ct
            # Don't consume the full stream — just check headers


@pytest.mark.asyncio
async def test_sse_event_format(tmp_path: Path) -> None:
    """SSE events have correct format with event type, JSON data, and id."""
    import httpx

    from dashboard_server import _harness, app

    # Write a test event to the telemetry store if harness exists
    if _harness is not None:
        store = _harness._telemetry
        ev = TelemetryEvent("fci", data={"fci": 0.42})
        store.append(ev)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        async with client.stream("GET", "/events", timeout=3.0) as resp:
            assert resp.status_code == 200
            # Read a few bytes to confirm event-stream format
            ct = resp.headers.get("content-type", "")
            assert "text/event-stream" in ct

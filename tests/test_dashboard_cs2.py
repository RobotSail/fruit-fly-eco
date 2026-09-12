"""Tests for the CS2 esports dashboard."""
import sys
from pathlib import Path

# Worktree import hack: add parent repo for flyecon, worktree root for dashboard_cs2
_WORKTREE_ROOT = Path(__file__).resolve().parent.parent
_PARENT_REPO = _WORKTREE_ROOT.parent.parent  # /home/osilkin/fruit-fly
if str(_PARENT_REPO) not in sys.path:
    sys.path.insert(0, str(_PARENT_REPO))
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

import pytest
from fastapi.testclient import TestClient


class TestDashboardCS2:
    """Test the CS2 dashboard FastAPI app."""

    def test_import(self):
        """Smoke test: module imports without error."""
        import dashboard_cs2
        assert hasattr(dashboard_cs2, "app")

    def test_root_returns_200(self):
        """Root endpoint returns 200 with substantial HTML content."""
        from dashboard_cs2 import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert len(response.text) > 100
        assert "FLY//ECON" in response.text or "fly" in response.text.lower()

    def test_root_contains_cs2_theme(self):
        """Root page has CS2-themed content."""
        from dashboard_cs2 import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        # Should contain match data or loading state
        assert "match" in response.text.lower() or "loading" in response.text.lower()

    def test_api_status(self):
        """Status endpoint returns valid JSON."""
        from dashboard_cs2 import app
        client = TestClient(app)
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "progress" in data

    def test_api_matches(self):
        """Matches endpoint returns list of matches."""
        from dashboard_cs2 import app
        client = TestClient(app)
        response = client.get("/api/matches")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 169
        assert "match_id" in data[0]
        assert "teams" in data[0]

    def test_api_match_not_found(self):
        """Non-existent match returns 404 or 503."""
        from dashboard_cs2 import app
        client = TestClient(app)
        response = client.get("/api/matches/nonexistent/rounds")
        assert response.status_code in (404, 503)

    def test_html_contains_svg_avatar(self):
        """Root page includes the animated SVG fly mascot."""
        from dashboard_cs2 import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "fly-svg" in response.text or "svg" in response.text.lower()

    def test_matches_list_available_immediately(self):
        """Match browser data is available even before reservoir finishes."""
        from dashboard_cs2 import MATCHES
        assert len(MATCHES) == 169
        assert all("match_id" in m for m in MATCHES)

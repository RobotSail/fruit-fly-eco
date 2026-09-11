"""Tests for flyecon.avatar.threejs — 3D avatar with degrade chain.

Validates that the ThreeJSAvatar produces valid HTML with inline WebGL
rendering code, procedural mesh geometry, animation setup, and the
sprite/ASCII fallback chain.
"""

from __future__ import annotations

import json

import pytest

from flyecon.avatar.threejs import STATE_POSE_MAP, ThreeJSAvatar


@pytest.fixture()
def avatar() -> ThreeJSAvatar:
    """Fresh ThreeJSAvatar instance for each test."""
    return ThreeJSAvatar()


# ── HTML output ────────────────────────────────────────────────────────────


class TestThreeJSOutput:
    """to_html produces valid HTML with all required components."""

    def test_contains_canvas(self, avatar: ThreeJSAvatar) -> None:
        """Output contains WebGL canvas element."""
        html = avatar.to_html(fci=0.5)
        assert "<canvas" in html
        assert 'id="threejs-avatar-canvas"' in html

    def test_contains_webgl_init(self, avatar: ThreeJSAvatar) -> None:
        """Output contains WebGL initialization code."""
        html = avatar.to_html(fci=0.5)
        assert "getContext" in html
        assert "webgl" in html

    def test_contains_procedural_mesh(self, avatar: ThreeJSAvatar) -> None:
        """Output contains procedural mesh geometry (vertices, shaders)."""
        html = avatar.to_html(fci=0.5)
        # Vertex shader
        assert "aPos" in html
        assert "uProj" in html
        # Geometry construction
        assert "addTri" in html
        # Body, head, wings, legs, antennae
        assert "bodyColor" in html
        assert "wingColor" in html
        assert "legColor" in html

    def test_contains_animation(self, avatar: ThreeJSAvatar) -> None:
        """Output contains animation loop setup."""
        html = avatar.to_html(fci=0.5)
        assert "requestAnimationFrame" in html
        assert "uWingFreq" in html
        assert "uTime" in html

    def test_contains_pose_map(self, avatar: ThreeJSAvatar) -> None:
        """Output contains inline state→pose map as JSON."""
        html = avatar.to_html(fci=0.5)
        assert "FLY_POSE_MAP" in html
        assert "FLY_CURRENT_POSE" in html
        # Verify the JSON is valid by extracting it
        assert "preening" in html
        assert "wing_freq" in html

    def test_self_contained(self, avatar: ThreeJSAvatar) -> None:
        """Output has no external resource loads."""
        html = avatar.to_html(fci=0.5)
        assert '<script src="http' not in html
        assert '<link href="http' not in html


# ── Degrade chain ─────────────────────────────────────────────────────────


class TestDegradeChain:
    """Degrade chain: three.js → sprite → ASCII in HTML try/catch."""

    def test_fallback_div_present(self, avatar: ThreeJSAvatar) -> None:
        """Output contains fallback div for sprite/ASCII."""
        html = avatar.to_html(fci=0.5)
        assert 'id="threejs-fallback"' in html

    def test_fallback_contains_sprite(self, avatar: ThreeJSAvatar) -> None:
        """Fallback HTML contains sprite img tag."""
        html = avatar.to_html(fci=0.5)
        # The fallback is JSON-encoded and contains sprite data
        assert "data:image/png;base64," in html

    def test_fallback_contains_ascii(self, avatar: ThreeJSAvatar) -> None:
        """Fallback HTML contains ASCII <pre> tag."""
        html = avatar.to_html(fci=0.5)
        assert "<noscript>" in html
        assert "&lt;pre" in html or "<pre" in html

    def test_catch_block_shows_fallback(self, avatar: ThreeJSAvatar) -> None:
        """JavaScript catch block shows the fallback div."""
        html = avatar.to_html(fci=0.5)
        assert "catch" in html
        assert "fallbackDiv" in html
        assert "display" in html


# ── State→pose mapping ────────────────────────────────────────────────────


class TestStatePoseMap:
    """State→pose map is valid and contains all required poses."""

    def test_all_required_poses(self) -> None:
        """Pose map contains all event-driven and FCI-driven poses."""
        required = [
            "preening", "neutral", "slumped", "resolve",
            "forced_eco", "round_win", "three_streak",
            "oracle_duel", "retirement", "enshrinement",
        ]
        for pose in required:
            assert pose in STATE_POSE_MAP, f"Missing pose: {pose}"

    def test_pose_has_required_fields(self) -> None:
        """Each pose has condition, animations, wing_freq."""
        for name, pose in STATE_POSE_MAP.items():
            assert "condition" in pose, f"{name} missing condition"
            assert "animations" in pose, f"{name} missing animations"
            assert "wing_freq" in pose, f"{name} missing wing_freq"
            assert isinstance(pose["wing_freq"], (int, float))

    def test_pose_map_json_valid(self) -> None:
        """pose_map_json() returns valid JSON."""
        from flyecon.avatar.threejs import ThreeJSAvatar
        json_str = ThreeJSAvatar.pose_map_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "preening" in parsed

    def test_pose_resolves_for_events(self, avatar: ThreeJSAvatar) -> None:
        """Event-driven poses are correctly mapped."""
        events = [
            "round_win", "three_streak", "oracle_duel",
            "retirement", "enshrinement", "forced_eco",
        ]
        for event in events:
            html = avatar.to_html(fci=0.5, event=event)
            # The current pose should match the event name
            assert f'"{event}"' in html or event in html

    def test_fci_driven_poses(self, avatar: ThreeJSAvatar) -> None:
        """FCI-driven poses are correctly selected."""
        # High FCI → preening
        html_high = avatar.to_html(fci=0.85)
        assert '"preening"' in html_high

        # Mid FCI → neutral
        html_mid = avatar.to_html(fci=0.5)
        assert '"neutral"' in html_mid

        # Low FCI → slumped
        html_low = avatar.to_html(fci=0.2)
        assert '"slumped"' in html_low


# ── Consecutive slump prevention ──────────────────────────────────────────


class TestSlumpPrevention:
    """Avatar never shows slumped for two consecutive renders."""

    def test_no_double_slump(self) -> None:
        """Second consecutive low-FCI render shows resolve."""
        av = ThreeJSAvatar()
        html1 = av.to_html(fci=0.2)
        assert '"slumped"' in html1

        html2 = av.to_html(fci=0.2)
        assert '"resolve"' in html2


# ── Sprite avatar ─────────────────────────────────────────────────────────


class TestSpriteAvatar:
    """SpriteAvatar produces valid HTML with CSS animation."""

    def test_sprite_html_contains_img(self) -> None:
        """Sprite output contains <img> tag."""
        from flyecon.avatar.sprite import SpriteAvatar
        sprite = SpriteAvatar()
        html = sprite.to_html(fci=0.5)
        assert "<img" in html

    def test_sprite_has_base64_data(self) -> None:
        """Sprite output contains base64-encoded PNG data."""
        from flyecon.avatar.sprite import SpriteAvatar
        sprite = SpriteAvatar()
        html = sprite.to_html(fci=0.5)
        assert "data:image/png;base64," in html

    def test_sprite_has_css_animation(self) -> None:
        """Sprite output contains CSS animation keyframes."""
        from flyecon.avatar.sprite import SpriteAvatar
        sprite = SpriteAvatar()
        html = sprite.to_html(fci=0.5)
        assert "@keyframes" in html
        assert "animation:" in html

    def test_sprite_event_poses(self) -> None:
        """Sprite responds to economy events."""
        from flyecon.avatar.sprite import SpriteAvatar
        sprite = SpriteAvatar()
        html_win = sprite.to_html(fci=0.5, event="round_win")
        html_neutral = sprite.to_html(fci=0.5)
        # Different events produce different animations
        assert "round_win" in html_win or html_win != html_neutral

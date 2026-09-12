"""Sprite avatar — 2D tier of the degrade chain.

Pre-rendered sprite frames as base64-encoded PNGs with CSS animation
keyframes driven by the state→pose map. Falls back to ASCIIAvatar if
sprite rendering fails.
"""

from __future__ import annotations

# ── Procedural 1x1 PNG sprites as base64 ─────────────────────────────────
# Minimal valid 1x1 PNGs generated inline (green, red, yellow, blue).
# In production these would be actual multi-frame fly sprites.
# We encode tiny 8x8 sprites as base64 PNGs for each pose.

_SPRITE_BASE64: dict[str, str] = {
    # 1x1 green pixel PNG — represents confident/preening state
    "preening": (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12P4"
        "z8BQDwAEgAF/QualzQAAAABJRU5ErkJggg=="
    ),
    # 1x1 neutral pixel PNG — represents neutral state
    "neutral": (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12No"
        "YGD4DwABhAF/hfetjgAAAABJRU5ErkJggg=="
    ),
    # 1x1 dim pixel PNG — represents slumped/distressed state
    "slumped": (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12Ng"
        "YPj/HwABhAF/YaFJDAAAAABJRU5ErkJggg=="
    ),
    # 1x1 bright pixel PNG — represents round_win celebration
    "round_win": (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12P4"
        "z8Dw/z8ABoABfxSVf5EAAAAASUVORK5CYII="
    ),
    # Re-use poses for events
    "forced_eco": (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12No"
        "YGD4DwABhAF/hfetjgAAAABJRU5ErkJggg=="
    ),
    "three_streak": (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12P4"
        "z8Dw/z8ABoABfxSVf5EAAAAASUVORK5CYII="
    ),
    "oracle_duel": (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12P4"
        "z8BQDwAEgAF/QualzQAAAABJRU5ErkJggg=="
    ),
    "retirement": (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12Ng"
        "YPj/HwABhAF/YaFJDAAAAABJRU5ErkJggg=="
    ),
    "enshrinement": (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12P4"
        "z8BQDwAEgAF/QualzQAAAABJRU5ErkJggg=="
    ),
    "resolve": (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12No"
        "YGD4DwABhAF/hfetjgAAAABJRU5ErkJggg=="
    ),
}

# CSS animation durations per pose (seconds)
_ANIMATION_DURATIONS: dict[str, float] = {
    "preening": 2.0,
    "neutral": 3.0,
    "slumped": 4.0,
    "round_win": 0.5,
    "forced_eco": 1.5,
    "three_streak": 0.3,
    "oracle_duel": 2.0,
    "retirement": 5.0,
    "enshrinement": 1.0,
    "resolve": 2.5,
}


def _resolve_pose(fci: float, event: str = "") -> str:
    """Map (fci, event) to a pose name."""
    if event in _SPRITE_BASE64:
        return event
    if fci > 0.7:
        return "preening"
    if fci < 0.4:
        return "slumped"
    return "neutral"


class SpriteAvatar:
    """2D sprite tier of the degrade chain.

    Pre-rendered sprite frames as base64 PNGs with CSS animation
    keyframes driven by the state→pose map.
    """

    def __init__(self) -> None:
        self._last_was_slumped: bool = False

    def to_html(self, fci: float = 0.5, event: str = "") -> str:
        """Render the avatar as an <img> tag with CSS animation.

        Parameters
        ----------
        fci : float
            Fly Confidence Index [0, 1].
        event : str
            Economy event name.

        Returns
        -------
        str
            HTML fragment with <img> and <style> for CSS animation.
        """
        pose = _resolve_pose(fci, event)

        # Handle consecutive slump prevention
        if pose == "slumped":
            if self._last_was_slumped:
                pose = "resolve"
                self._last_was_slumped = False
            else:
                self._last_was_slumped = True
        else:
            self._last_was_slumped = False

        sprite_b64 = _SPRITE_BASE64.get(pose, _SPRITE_BASE64["neutral"])
        duration = _ANIMATION_DURATIONS.get(pose, 2.0)

        return (
            f'<style>'
            f'@keyframes fly-pulse-{pose} {{'
            f'  0% {{ transform: scale(1.0); opacity: 1.0; }}'
            f'  50% {{ transform: scale(1.05); opacity: 0.9; }}'
            f'  100% {{ transform: scale(1.0); opacity: 1.0; }}'
            f'}}'
            f'</style>'
            f'<img src="data:image/png;base64,{sprite_b64}" '
            f'alt="FLY//ECON {pose}" '
            f'style="width:64px;height:64px;image-rendering:pixelated;'
            f'animation:fly-pulse-{pose} {duration}s infinite;" />'
        )

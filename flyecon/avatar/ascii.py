"""ASCII avatar — bottom tier of the degrade chain.

Works at ALL 5 degradation levels. The fly is always visible.
"""

from __future__ import annotations

# ── Pose definitions ─────────────────────────────────────────────────────────
# Each pose is a multi-line ASCII art string representing the fly's state.
# We use regular strings with explicit backslash escaping because raw strings
# cannot end with an odd number of backslashes (Python parser limitation).

_POSE_PREENING = "\n".join([
    "    \\   /",
    "  ---◉---",
    "   / | \\",
    "  / /|\\ \\",
    "    / \\",
])

_POSE_STRIDING = "\n".join([
    "       /",
    "  ( ◉ )>",
    "   /|\\",
    "  / | \\",
    "   / \\",
])

_POSE_NEUTRAL = "\n".join([
    "  ( o )>",
    "   /|\\",
    "  / | \\",
    "   / \\",
])

_POSE_SLUMPED = "\n".join([
    "  ( . )>",
    "   /|",
    "  / |",
    "   / \\",
])

_POSE_RESOLVE = "\n".join([
    "  ( o )>!",
    "   /|\\",
    "  / | \\",
    "   / \\",
])

_POSE_FORCED_ECO = "\n".join([
    "  ( o )> ...",
    "   /|\\  $$$",
    "  / | \\",
    "   / \\",
])

_POSE_ROUND_WIN = "\n".join([
    "    *",
    "  \\ ◉ /",
    "   \\|/",
    "    |",
    "   / \\",
])

_POSE_THREE_STREAK = "\n".join([
    "  * * *",
    "  \\ ◉ /",
    "   \\|/",
    "    |",
    "   / \\",
])

_POSE_ORACLE_DUEL = "  <( ◉ ) |TABLE| ( ◉ )>"

_POSE_RETIREMENT = "  ( o )>  →  [GRIEF COUNSEL]"

_POSE_ENSHRINEMENT = "\n".join([
    "       🏆",
    "  ( ◉ )>",
    "   /|\\",
    "  / | \\",
    "   / \\",
])


class ASCIIAvatar:
    """Renders the fly as multi-line ASCII art.

    State→pose map is deterministic: given (fci, event, ladder_level),
    the output is always the same string. The avatar never shows
    "discouraged" (slumped) for two consecutive renders — after one
    low-FCI frame, it shows resolve.
    """

    def __init__(self) -> None:
        self._last_was_slumped: bool = False

    def render(
        self,
        fci: float,
        event: str = "",
        ladder_level: int = 0,
    ) -> str:
        """Render the fly's current pose as ASCII art.

        Args:
            fci: Fly Confidence Index [0, 1].
            event: Economy event name (e.g. "round_win", "forced_eco").
            ladder_level: Current degradation ladder level (0–5).

        Returns:
            Multi-line ASCII art string.
        """
        # Event-specific poses take priority
        if event == "oracle_duel":
            self._last_was_slumped = False
            return _POSE_ORACLE_DUEL

        if event == "retirement":
            self._last_was_slumped = False
            return _POSE_RETIREMENT

        if event == "enshrinement":
            self._last_was_slumped = False
            return _POSE_ENSHRINEMENT

        if event == "three_streak":
            self._last_was_slumped = False
            return _POSE_THREE_STREAK

        if event == "round_win":
            self._last_was_slumped = False
            return _POSE_ROUND_WIN

        if event == "forced_eco":
            self._last_was_slumped = False
            return _POSE_FORCED_ECO

        # FCI-driven poses
        if fci > 0.7:
            self._last_was_slumped = False
            return _POSE_PREENING

        if fci < 0.4:
            if self._last_was_slumped:
                # Never discouraged for two consecutive renders
                self._last_was_slumped = False
                return _POSE_RESOLVE
            else:
                self._last_was_slumped = True
                return _POSE_SLUMPED

        # Neutral: 0.4 ≤ fci ≤ 0.7
        self._last_was_slumped = False
        return _POSE_NEUTRAL

    def to_html(
        self,
        fci: float = 0.5,
        event: str = "",
        ladder_level: int = 0,
    ) -> str:
        """Render the avatar and wrap it in an HTML <pre> tag.

        Returns a self-contained HTML fragment with monospace font.
        """
        art = self.render(fci, event, ladder_level)
        escaped = (
            art.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return (
            '<pre style="font-family: monospace; font-size: 16px; '
            'line-height: 1.2; white-space: pre;">'
            f"{escaped}</pre>"
        )

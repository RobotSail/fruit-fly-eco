"""Tests for flyecon.avatar.ascii — state→pose map and HTML output."""

from __future__ import annotations

import pytest

from flyecon.avatar.ascii import ASCIIAvatar


@pytest.fixture()
def avatar() -> ASCIIAvatar:
    """Fresh avatar instance for each test."""
    return ASCIIAvatar()


class TestPoseMap:
    """Each state→pose mapping produces the expected ASCII art pattern."""

    def test_preening_high_fci(self, avatar: ASCIIAvatar) -> None:
        """FCI > 0.7 → preening pose with spread wings."""
        art = avatar.render(fci=0.85)
        assert "◉" in art

    def test_neutral_mid_fci(self, avatar: ASCIIAvatar) -> None:
        """0.4 < FCI < 0.7 → neutral pose."""
        art = avatar.render(fci=0.5)
        assert "o" in art

    def test_slumped_low_fci(self, avatar: ASCIIAvatar) -> None:
        """FCI < 0.4 → slumped pose (first time)."""
        art = avatar.render(fci=0.2)
        assert "." in art

    def test_round_win(self, avatar: ASCIIAvatar) -> None:
        """round_win event → celebratory pose."""
        art = avatar.render(fci=0.5, event="round_win")
        assert "◉" in art
        assert "*" in art

    def test_three_streak(self, avatar: ASCIIAvatar) -> None:
        """three_streak event → triple celebration."""
        art = avatar.render(fci=0.5, event="three_streak")
        assert art.count("*") >= 3

    def test_oracle_duel(self, avatar: ASCIIAvatar) -> None:
        """oracle_duel event → two flies facing across table."""
        art = avatar.render(fci=0.5, event="oracle_duel")
        assert "TABLE" in art

    def test_retirement(self, avatar: ASCIIAvatar) -> None:
        """retirement event → walking toward grief counsel."""
        art = avatar.render(fci=0.5, event="retirement")
        assert "GRIEF COUNSEL" in art

    def test_enshrinement(self, avatar: ASCIIAvatar) -> None:
        """enshrinement event → trophy pose."""
        art = avatar.render(fci=0.5, event="enshrinement")
        assert "🏆" in art

    def test_forced_eco(self, avatar: ASCIIAvatar) -> None:
        """forced_eco event → pacing with rubbing-foreleg."""
        art = avatar.render(fci=0.5, event="forced_eco")
        assert "..." in art

    def test_event_overrides_fci(self, avatar: ASCIIAvatar) -> None:
        """Event-specific pose takes priority over FCI-driven pose."""
        art_win = avatar.render(fci=0.1, event="round_win")
        art_low = avatar.render(fci=0.1)
        assert art_win != art_low


class TestConsecutiveSlump:
    """Avatar never shows 'discouraged' for two consecutive low-FCI renders."""

    def test_no_double_slump(self) -> None:
        """Second consecutive low-FCI render shows resolve, not slump."""
        avatar = ASCIIAvatar()
        art1 = avatar.render(fci=0.2)
        assert "." in art1  # slumped

        art2 = avatar.render(fci=0.2)
        assert "!" in art2  # resolve
        assert "." not in art2

    def test_slump_after_intervening_event(self) -> None:
        """Slump is allowed again after an intervening non-slump render."""
        avatar = ASCIIAvatar()
        avatar.render(fci=0.2)  # slumped
        avatar.render(fci=0.8)  # preening (resets slump flag)
        art3 = avatar.render(fci=0.2)
        assert "." in art3  # slumped again is OK


class TestToHtml:
    """to_html() wraps ASCII art in <pre> tags."""

    def test_pre_tag(self, avatar: ASCIIAvatar) -> None:
        """Output contains <pre> and </pre> tags."""
        html = avatar.to_html(fci=0.5)
        assert "<pre" in html
        assert "</pre>" in html

    def test_monospace_font(self, avatar: ASCIIAvatar) -> None:
        """Output specifies monospace font."""
        html = avatar.to_html(fci=0.5)
        assert "monospace" in html

    def test_html_escaping(self, avatar: ASCIIAvatar) -> None:
        """HTML special characters are escaped."""
        html = avatar.to_html(fci=0.5)
        # The neutral pose has )> which should be escaped to )&gt;
        assert "&gt;" in html

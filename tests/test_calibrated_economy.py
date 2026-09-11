"""Integration tests for calibrated economy probabilities."""
import pytest
from flyecon.oracle.calibration import CALIBRATED_WIN_PROBS, load_calibrated_win_probs


def test_calibrated_probs_have_all_keys():
    """All 5 buy-plan categories must be present."""
    required = {"FULL_BUY", "FORCE_BUY", "HALF_BUY", "ECO", "SAVE"}
    assert required == set(CALIBRATED_WIN_PROBS.keys())


def test_calibrated_probs_are_valid_probabilities():
    """All values must be in (0, 1)."""
    for k, v in CALIBRATED_WIN_PROBS.items():
        assert 0 < v < 1, f"{k} = {v} not a valid probability"


def test_calibrated_probs_ordering():
    """FULL_BUY > HALF_BUY > FORCE_BUY > ECO > SAVE (economic logic)."""
    p = CALIBRATED_WIN_PROBS
    assert p["FULL_BUY"] > p["HALF_BUY"], "Full buy should beat half buy"
    assert p["HALF_BUY"] > p["FORCE_BUY"], "Half buy should beat force buy"
    assert p["FORCE_BUY"] > p["ECO"], "Force buy should beat eco"
    assert p["ECO"] > p["SAVE"], "Eco should beat save"


def test_load_falls_back_to_static():
    """load_calibrated_win_probs returns static table when no cache exists."""
    from pathlib import Path
    # Use a path that definitely doesn't exist
    probs = load_calibrated_win_probs(cache_path=Path("/tmp/nonexistent_cache.json"))
    assert probs == CALIBRATED_WIN_PROBS


def test_mdp_uses_calibrated_probs():
    """EconomyMDP should default to calibrated (not old hardcoded) probs."""
    from flyecon.oracle.mdp import EconomyMDP
    mdp = EconomyMDP()
    # Should use calibrated values, not old hardcoded ones
    assert mdp.win_probs["FULL_BUY"] == CALIBRATED_WIN_PROBS["FULL_BUY"]
    assert mdp.win_probs["ECO"] == CALIBRATED_WIN_PROBS["ECO"]

"""Tests for flyecon.resilience.checkpoint — atomic write, load, prune."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from flyecon.resilience.checkpoint import (
    CheckpointBundle,
    MissionStage,
    MissionState,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture()
def ckpt_dir(tmp_path: Path) -> Path:
    """Provide a temporary checkpoint directory."""
    return tmp_path / "checkpoints"


def _make_bundle(step: int = 0, stage: MissionStage = MissionStage.TRAINING) -> CheckpointBundle:
    """Create a minimal checkpoint bundle."""
    return CheckpointBundle(
        model_state={"weight": [1.0, 2.0, 3.0]},
        optimizer_state={"lr": 3e-4},
        rng_state={"seed": 42},
        mission_state=MissionState(
            stage=stage,
            candidate_id=1,
            training_step=step,
            ladder_level=0,
            last_eval_score=0.5,
            cycles_completed=step,
            uptime_seconds=float(step * 10),
            task_pointer=f"training:candidate_1:step_{step}",
        ),
        metadata={"candidate_id": 1, "training_step": step},
    )


class TestSaveAndLoad:
    """save_checkpoint → load_checkpoint round-trip."""

    def test_round_trip(self, ckpt_dir: Path) -> None:
        """Saved bundle can be loaded back."""
        bundle = _make_bundle(step=100)
        save_checkpoint(bundle, ckpt_dir)

        loaded = load_checkpoint(ckpt_dir)
        assert loaded is not None
        assert loaded.mission_state.training_step == 100
        assert loaded.mission_state.stage == MissionStage.TRAINING
        assert loaded.metadata["candidate_id"] == 1

    def test_load_newest(self, ckpt_dir: Path) -> None:
        """load_checkpoint returns the checkpoint with the highest step."""
        save_checkpoint(_make_bundle(step=10), ckpt_dir, max_kept=5)
        save_checkpoint(_make_bundle(step=20), ckpt_dir, max_kept=5)
        save_checkpoint(_make_bundle(step=30), ckpt_dir, max_kept=5)

        loaded = load_checkpoint(ckpt_dir)
        assert loaded is not None
        assert loaded.mission_state.training_step == 30

    def test_max_kept_pruning(self, ckpt_dir: Path) -> None:
        """Old checkpoints beyond max_kept are pruned."""
        for s in range(5):
            save_checkpoint(_make_bundle(step=s * 10), ckpt_dir, max_kept=3)

        remaining = list(ckpt_dir.glob("ckpt_*"))
        # Filter out any tmp dirs
        remaining = [p for p in remaining if not p.name.startswith("ckpt_tmp_")]
        assert len(remaining) == 3

    def test_cold_start_returns_none(self, ckpt_dir: Path) -> None:
        """load_checkpoint returns None when no checkpoints exist."""
        assert load_checkpoint(ckpt_dir) is None

    def test_corrupt_fallback(self, ckpt_dir: Path) -> None:
        """If newest checkpoint is corrupt, fall back to the next one."""
        save_checkpoint(_make_bundle(step=10), ckpt_dir, max_kept=5)
        save_checkpoint(_make_bundle(step=20), ckpt_dir, max_kept=5)

        # Corrupt the newest by removing mission_state.json
        newest = ckpt_dir / "ckpt_00000020"
        (newest / "mission_state.json").unlink()

        loaded = load_checkpoint(ckpt_dir)
        assert loaded is not None
        assert loaded.mission_state.training_step == 10


class TestMissionState:
    """MissionState serialization round-trip."""

    def test_to_dict_from_dict(self) -> None:
        """MissionState survives JSON serialization."""
        ms = MissionState(
            stage=MissionStage.EVALUATING,
            candidate_id=7,
            training_step=128000,
            ladder_level=2,
            last_eval_score=0.73,
            cycles_completed=42,
            uptime_seconds=86400.0,
            task_pointer="training:candidate_7:step_128000",
        )
        d = ms.to_dict()
        # Verify it's JSON-safe
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        ms2 = MissionState.from_dict(d2)

        assert ms2.stage == MissionStage.EVALUATING
        assert ms2.candidate_id == 7
        assert ms2.training_step == 128000
        assert ms2.task_pointer == "training:candidate_7:step_128000"

    def test_all_stages(self) -> None:
        """Every MissionStage value can be serialized."""
        for stage in MissionStage:
            ms = MissionState(stage=stage)
            d = ms.to_dict()
            ms2 = MissionState.from_dict(d)
            assert ms2.stage == stage

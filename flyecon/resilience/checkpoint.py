"""Atomic checkpoint bundles for crash-safe persistence.

Write to temp dir → fsync → os.replace (atomic rename) → prune old.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import structlog

log = structlog.get_logger()


class MissionStage(str, Enum):
    """Pipeline stage the mission is currently executing."""

    ETL = "ETL"
    CALIBRATION = "CALIBRATION"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    PATROLLING = "PATROLLING"


@dataclass
class MissionState:
    """Resumable mission state — enables goto-resume, not re-plan.

    Serialized as JSON alongside model weights in each checkpoint bundle.
    """

    stage: MissionStage = MissionStage.ETL
    candidate_id: int = 0
    training_step: int = 0
    ladder_level: int = 0
    last_eval_score: float = 0.0
    cycles_completed: int = 0
    uptime_seconds: float = 0.0
    task_pointer: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        d = asdict(self)
        d["stage"] = self.stage.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MissionState:
        """Deserialize from a dictionary."""
        d = dict(d)  # shallow copy
        d["stage"] = MissionStage(d["stage"])
        return cls(**d)


@dataclass
class CheckpointBundle:
    """Everything needed to resume from a crash.

    - model_state: encoder + readout + value head state_dicts
    - optimizer_state: optimizer state_dict
    - rng_state: dict with torch / numpy / python random states
    - mission_state: MissionState object
    - metadata: candidate_id, training_step, timestamp, git_commit
    """

    model_state: dict[str, Any] = field(default_factory=dict)
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    rng_state: dict[str, Any] = field(default_factory=dict)
    mission_state: MissionState = field(default_factory=MissionState)
    metadata: dict[str, Any] = field(default_factory=dict)


def _fsync_directory(path: Path) -> None:
    """Fsync a directory to flush metadata to disk."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    """Fsync a single file to flush its contents to disk."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_checkpoint(
    bundle: CheckpointBundle,
    checkpoint_dir: Path,
    max_kept: int = 3,
) -> Path:
    """Atomically write a checkpoint bundle to disk.

    1. Serialize to a temp directory inside *checkpoint_dir*.
    2. ``os.fsync`` every written file.
    3. ``os.replace`` temp → ``ckpt_{step:08d}`` (atomic on POSIX).
    4. Prune oldest checkpoints beyond *max_kept*.

    Returns the path to the newly created checkpoint directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = bundle.mission_state.training_step
    ts = time.time()
    bundle.metadata.setdefault("timestamp", ts)
    bundle.metadata.setdefault("training_step", step)

    # Write into a temp directory so a crash mid-write never corrupts
    tmp_dir = Path(tempfile.mkdtemp(prefix="ckpt_tmp_", dir=str(checkpoint_dir)))
    try:
        # Model + optimizer state → torch format
        import torch

        torch.save(bundle.model_state, tmp_dir / "model_state.pt")
        torch.save(bundle.optimizer_state, tmp_dir / "optimizer_state.pt")

        # RNG state → torch format (may contain non-JSON-serializable objects)
        torch.save(bundle.rng_state, tmp_dir / "rng_state.pt")

        # Mission state → JSON
        ms_path = tmp_dir / "mission_state.json"
        ms_path.write_text(json.dumps(bundle.mission_state.to_dict(), indent=2))

        # Metadata → JSON
        meta_path = tmp_dir / "metadata.json"
        # Make sure metadata is JSON-serializable
        safe_meta = {k: v for k, v in bundle.metadata.items() if isinstance(v, (str, int, float, bool, type(None)))}
        meta_path.write_text(json.dumps(safe_meta, indent=2))

        # fsync all files
        for f in tmp_dir.iterdir():
            _fsync_file(f)
        _fsync_directory(tmp_dir)

        # Atomic rename
        final_path = checkpoint_dir / f"ckpt_{step:08d}"
        if final_path.exists():
            shutil.rmtree(final_path)
        os.replace(str(tmp_dir), str(final_path))
        _fsync_directory(checkpoint_dir)

        log.info(
            "checkpoint_saved",
            path=str(final_path),
            step=step,
        )

    except BaseException:
        # Clean up temp dir on any failure
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    # Prune old checkpoints
    _prune_checkpoints(checkpoint_dir, max_kept)

    return final_path


def _prune_checkpoints(checkpoint_dir: Path, max_kept: int) -> None:
    """Remove oldest checkpoints beyond *max_kept*."""
    ckpts = sorted(_list_checkpoint_dirs(checkpoint_dir), key=_ckpt_step)
    while len(ckpts) > max_kept:
        old = ckpts.pop(0)
        log.info("checkpoint_pruned", path=str(old))
        shutil.rmtree(old)


def _list_checkpoint_dirs(checkpoint_dir: Path) -> list[Path]:
    """List valid checkpoint directories sorted by step number."""
    results: list[Path] = []
    if not checkpoint_dir.exists():
        return results
    for entry in checkpoint_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("ckpt_") and not entry.name.startswith("ckpt_tmp_"):
            results.append(entry)
    return results


def _ckpt_step(ckpt_dir: Path) -> int:
    """Extract step number from checkpoint directory name."""
    try:
        return int(ckpt_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _is_valid_checkpoint(ckpt_dir: Path) -> bool:
    """Check whether a checkpoint directory contains the required files."""
    required = ["mission_state.json", "metadata.json"]
    return all((ckpt_dir / f).exists() for f in required)


def load_checkpoint(checkpoint_dir: Path) -> Optional[CheckpointBundle]:
    """Load the newest valid checkpoint, falling back to older ones.

    Returns ``None`` if no valid checkpoint exists (cold start).
    """
    checkpoint_dir = Path(checkpoint_dir)
    candidates = sorted(
        _list_checkpoint_dirs(checkpoint_dir),
        key=_ckpt_step,
        reverse=True,
    )

    for ckpt in candidates:
        try:
            bundle = _load_single(ckpt)
            log.info(
                "checkpoint_loaded",
                path=str(ckpt),
                step=bundle.mission_state.training_step,
            )
            return bundle
        except Exception as exc:
            log.warning(
                "checkpoint_corrupt",
                path=str(ckpt),
                error=str(exc),
            )
            continue

    log.info("no_valid_checkpoint", dir=str(checkpoint_dir))
    return None


def _load_single(ckpt_dir: Path) -> CheckpointBundle:
    """Load a single checkpoint directory into a CheckpointBundle."""
    if not _is_valid_checkpoint(ckpt_dir):
        raise ValueError(f"Missing required files in {ckpt_dir}")

    import torch

    model_state: dict[str, Any] = {}
    optimizer_state: dict[str, Any] = {}
    rng_state: dict[str, Any] = {}

    model_path = ckpt_dir / "model_state.pt"
    if model_path.exists():
        model_state = torch.load(model_path, map_location="cpu", weights_only=False)

    opt_path = ckpt_dir / "optimizer_state.pt"
    if opt_path.exists():
        optimizer_state = torch.load(opt_path, map_location="cpu", weights_only=False)

    rng_path = ckpt_dir / "rng_state.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)

    ms_data = json.loads((ckpt_dir / "mission_state.json").read_text())
    mission_state = MissionState.from_dict(ms_data)

    meta_data = json.loads((ckpt_dir / "metadata.json").read_text())

    return CheckpointBundle(
        model_state=model_state,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
        mission_state=mission_state,
        metadata=meta_data,
    )

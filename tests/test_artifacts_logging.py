"""Unit tests for artifact and image logging in RunTracker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mangosta.tracking.tracker import RunTracker, quick_track


def test_log_artifact_file(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    tracker = RunTracker(experiment="test_artifacts", run_name="run_01", root_dir=exp_dir)

    extra_file = tmp_path / "notes.txt"
    extra_file.write_text("Validation notes and qualitative feedback")

    dest = tracker.log_artifact(extra_file)
    assert dest.exists()
    assert dest.name == "notes.txt"
    assert dest.parent == tracker.artifacts_dir

    tracker.finish()

    # Verify run.json metadata
    with open(tracker.run_dir / "run.json") as f:
        meta = json.load(f)
    assert "artifacts" in meta
    assert len(meta["artifacts"]) == 1
    assert meta["artifacts"][0]["name"] == "notes.txt"
    assert meta["artifacts"][0]["type"] == "file"


def test_log_image_tensor_chw(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    with quick_track(experiment="vision", run_name="run_img", root_dir=exp_dir) as run:
        # Dummy CHW RGB tensor in range [0, 1]
        sample_tensor = torch.rand(3, 64, 64)
        img_dest = run.log_image("val_sample_epoch1", sample_tensor, step=1)
        assert img_dest.exists()

    with open(run.run_dir / "run.json") as f:
        meta = json.load(f)
    assert len(meta["artifacts"]) == 1
    assert meta["artifacts"][0]["name"] == "val_sample_epoch1.png"
    assert meta["artifacts"][0]["type"] == "image"
    assert meta["artifacts"][0]["step"] == 1


def test_log_image_from_existing_file(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    sample_file = tmp_path / "pred.png"
    sample_file.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    with quick_track(experiment="vision", run_name="run_copy", root_dir=exp_dir) as run:
        dest = run.log_image("pred", sample_file, step=2)
        assert dest.exists()
        assert dest.read_bytes() == sample_file.read_bytes()


def test_log_artifact_nonexistent_raises(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    tracker = RunTracker(experiment="err", run_name="run_err", root_dir=exp_dir)

    with pytest.raises(FileNotFoundError, match="Artifact file not found"):
        tracker.log_artifact(tmp_path / "nonexistent.csv")

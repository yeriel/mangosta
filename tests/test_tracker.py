"""Unit tests for mangosta.tracker: atomic tracking, dataset lineage, and quick_track."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mangosta.tracking.tracker import RunTracker, compute_sha256, quick_track


def test_compute_sha256_file_and_directory(tmp_path: Path) -> None:
    # Single file
    f1 = tmp_path / "data.csv"
    f1.write_text("col1,col2\n1,2\n3,4")
    digest1 = compute_sha256(f1)
    assert isinstance(digest1, str)
    assert len(digest1) == 64

    # Directory with multiple files
    sub = tmp_path / "dataset"
    sub.mkdir()
    (sub / "train.csv").write_text("train data")
    (sub / "val.csv").write_text("val data")

    dir_digest = compute_sha256(sub)
    assert isinstance(dir_digest, str)
    assert len(dir_digest) == 64

    # Determinism
    assert compute_sha256(sub) == dir_digest

    # Modifying a file changes hash
    (sub / "val.csv").write_text("modified val data")
    assert compute_sha256(sub) != dir_digest


def test_run_tracker_lifecycle(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    tracker = RunTracker(experiment="test_exp", run_name="run_01", root_dir=exp_dir)

    assert tracker.run_dir.exists()
    assert (tracker.run_dir / "run.json").exists()

    tracker.log_param("lr", 0.001)
    tracker.log_params({"batch_size": 32, "arch": "resnet"})

    # Log metrics
    tracker.log_metric("loss", 0.5, step=0)
    tracker.log_metric("loss", 0.3, step=1)
    tracker.log_metric("accuracy", 0.85, step=1)

    # Verify atomic file contents
    metrics_file = tracker.run_dir / "metrics.jsonl"
    assert metrics_file.exists()
    lines = metrics_file.read_text().strip().split("\n")
    assert len(lines) == 3

    metrics = tracker.get_metrics()
    assert len(metrics["loss"]) == 2
    assert metrics["loss"][1]["value"] == 0.3

    tracker.finish()
    assert tracker.status == "completed"

    with open(tracker.run_dir / "run.json") as f:
        meta = json.load(f)
    assert meta["status"] == "completed"
    assert meta["params"]["lr"] == 0.001
    assert meta["params"]["batch_size"] == 32


def test_quick_track_context_manager(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    dataset_file = tmp_path / "dataset.txt"
    dataset_file.write_text("sample dataset content")

    with quick_track(run_name="quick_run", root_dir=exp_dir) as run:
        run.log_param("optimizer", "adam")
        run.log_metric("train_loss", 0.42, step=0)
        h = run.set_dataset(dataset_file)
        assert len(h) == 64

    # Verify completed
    with open(run.run_dir / "run.json") as f:
        meta = json.load(f)
    assert meta["status"] == "completed"
    assert meta["dataset_hash"] == h


def test_quick_track_marks_failed_on_exception(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"

    with pytest.raises(RuntimeError, match="Training crashed"):
        with quick_track(run_name="crash_run", root_dir=exp_dir) as run:
            run.log_metric("loss", 1.0, step=0)
            raise RuntimeError("Training crashed")

    with open(run.run_dir / "run.json") as f:
        meta = json.load(f)
    assert meta["status"] == "failed"

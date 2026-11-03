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

    tracker.log_params({"lr": 0.001, "batch_size": 32, "arch": "resnet"})

    # Log metrics
    tracker.log_metrics({"loss": 0.5}, step=0)
    tracker.log_metrics({"loss": 0.3, "accuracy": 0.85}, step=1)

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
        run.log_params({"optimizer": "adam"})
        run.log_metrics({"train_loss": 0.42}, step=0)
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
            run.log_metrics({"loss": 1.0}, step=0)
            raise RuntimeError("Training crashed")

    with open(run.run_dir / "run.json") as f:
        meta = json.load(f)
    assert meta["status"] == "failed"


def test_run_tracker_log_metrics_batch(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    tracker = RunTracker(experiment="batch_metrics_exp", run_name="run_batch", root_dir=exp_dir)

    # 1. Log multiple metrics in one step
    tracker.log_metrics({"train_loss": 0.35, "train_acc": 0.88, "lr": 0.001}, step=1)

    # 2. Log another batch without explicit step (auto step)
    tracker.log_metrics({"val_loss": 0.40, "val_acc": 0.85})

    # 3. Empty dictionary should be a safe no-op
    tracker.log_metrics({})

    metrics = tracker.get_metrics()
    assert "train_loss" in metrics
    assert "train_acc" in metrics
    assert "lr" in metrics
    assert "val_loss" in metrics
    assert "val_acc" in metrics

    assert metrics["train_loss"][0]["value"] == 0.35
    assert metrics["train_loss"][0]["step"] == 1
    assert metrics["train_acc"][0]["step"] == 1
    assert metrics["lr"][0]["step"] == 1

    # Verify metrics.jsonl has exactly 5 lines (3 from step 1, 2 from auto step)
    lines = (tracker.run_dir / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 5

    tracker.finish()


def test_run_tracker_sync_interval_and_flush_disk(tmp_path: Path) -> None:
    from unittest.mock import patch

    exp_dir = tmp_path / "experiments"
    tracker = RunTracker(
        experiment="sync_exp",
        run_name="run_sync",
        root_dir=exp_dir,
        sync_interval=10,
    )

    with patch("os.fsync") as mock_fsync:
        # Step 0 -> 0 % 10 == 0 -> should fsync
        tracker.log_metrics({"loss": 0.9}, step=0)
        assert mock_fsync.call_count == 1

        # Steps 1 to 9 -> should NOT fsync
        for s in range(1, 10):
            tracker.log_metrics({"loss": 0.9 - s * 0.05}, step=s)
        assert mock_fsync.call_count == 1

        # Step 10 -> 10 % 10 == 0 -> should fsync
        tracker.log_metrics({"loss": 0.4}, step=10)
        assert mock_fsync.call_count == 2

        # Step 11 -> no fsync
        tracker.log_metrics({"loss": 0.39}, step=11)
        assert mock_fsync.call_count == 2

        # Explicit manual flush_disk -> should fsync
        tracker.flush_disk()
        assert mock_fsync.call_count == 3

    # On finish(), flush_disk is automatically invoked
    with patch("os.fsync") as mock_fsync:
        tracker.finish()
        # mock_fsync called at least once by flush_disk, plus once by _save_metadata
        assert mock_fsync.call_count >= 1


def test_run_tracker_sync_interval_validation_and_zero(tmp_path: Path) -> None:
    from unittest.mock import patch

    exp_dir = tmp_path / "experiments"

    # Negative sync_interval raises ValueError
    with pytest.raises(ValueError, match="sync_interval must be non-negative"):
        RunTracker(root_dir=exp_dir, sync_interval=-1)

    # sync_interval=0 disables periodic fsync entirely (pure userspace flush)
    tracker = RunTracker(root_dir=exp_dir, sync_interval=0)
    with patch("os.fsync") as mock_fsync:
        for s in range(25):
            tracker.log_metrics({"loss": 0.5}, step=s)
        assert mock_fsync.call_count == 0

    tracker.finish()



"""Unit tests for the enhanced mangosta view server and REST endpoints."""

from __future__ import annotations

from pathlib import Path

from mangosta.tracking.tracker import quick_track
from mangosta.tracking.viewer import _load_run_details, _scan_experiments


def test_scan_experiments_and_load_details(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"

    with quick_track(experiment="telemetry", run_name="run_alpha", root_dir=exp_dir) as run:
        run.log_params({"lr": 0.01, "batch_size": 32})
        run.log_metrics({"loss": 0.5}, step=1)
        run.log_metrics({"loss": 0.3}, step=2)

        # Log artifact file and dummy image
        extra_file = tmp_path / "summary.txt"
        extra_file.write_text("ok")
        run.log_artifact(extra_file)

    scanned = _scan_experiments(exp_dir)
    assert "telemetry" in scanned
    assert len(scanned["telemetry"]) == 1
    assert scanned["telemetry"][0]["run_name"] == "run_alpha"

    details = _load_run_details(exp_dir, "telemetry", "run_alpha")
    assert details is not None
    assert details["experiment"] == "telemetry"
    assert details["params"]["lr"] == 0.01
    assert len(details["metrics"]["loss"]) == 2
    assert len(details["artifacts"]) == 1
    assert details["artifacts"][0]["name"] == "summary.txt"

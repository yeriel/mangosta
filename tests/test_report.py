"""Unit tests for mangosta.report: static HTML report generation."""

from __future__ import annotations

from pathlib import Path

import torch

from mangosta.bundle import save_bundle
from mangosta.report import generate_report
from mangosta.tracker import RunTracker
from tests.conftest import DummyMLP


def test_generate_report_from_run_dir(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    tracker = RunTracker(experiment="vision_exp", run_name="run_01", root_dir=exp_dir)
    tracker.log_param("lr", 0.001)
    tracker.log_param("epochs", 10)
    tracker.log_metric("train_loss", 0.8, step=0)
    tracker.log_metric("train_loss", 0.4, step=1)
    tracker.log_metric("train_loss", 0.2, step=2)
    tracker.finish()

    output_html = tmp_path / "report_run.html"
    report_path = generate_report(tracker.run_dir, output_html)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "mangosta Report" in content
    assert "vision_exp" in content
    assert "train_loss" in content
    assert "0.001" in content
    assert "<svg" in content


def test_generate_report_from_bundle(tmp_path: Path) -> None:
    model = DummyMLP(in_features=4, out_features=2)
    bundle_path = tmp_path / "model.bundle"

    save_bundle(
        output_path=bundle_path,
        model=model,
        family="resnet_custom",
        version="2.1.0",
        init_args={"in_features": 4, "out_features": 2},
        sample_input=torch.randn(1, 4),
        metrics={"test_acc": 0.94},
    )

    output_html = tmp_path / "report_bundle.html"
    report_path = generate_report(bundle_path, output_html)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "resnet_custom" in content
    assert "2.1.0" in content
    assert "I/O Schema Contract" in content

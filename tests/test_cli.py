"""Unit tests for mangosta.cli: terminal inspector and report commands."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mangosta.bundle import save_bundle
from mangosta.cli import inspect_bundle, run_report
from tests.conftest import DummyMLP


def test_cli_inspect_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = DummyMLP(in_features=4, out_features=2)
    bundle_path = tmp_path / "model.bundle"

    save_bundle(
        output_path=bundle_path,
        model=model,
        family="transformer_lite",
        version="0.5.0",
        init_args={"in_features": 4, "out_features": 2},
        sample_input=torch.randn(1, 4),
        metrics={"f1": 0.88},
    )

    inspect_bundle(str(bundle_path))
    captured = capsys.readouterr().out
    assert "MANGOSTA BUNDLE INSPECTOR" in captured
    assert "transformer_lite" in captured
    assert "0.5.0" in captured
    assert "I/O Schema Contract" in captured
    assert "f1: 0.88" in captured


def test_cli_run_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = DummyMLP(in_features=4, out_features=2)
    bundle_path = tmp_path / "model.bundle"
    out_html = tmp_path / "out.html"

    save_bundle(
        output_path=bundle_path,
        model=model,
        family="mlp",
        version="1.0.0",
    )

    run_report(str(bundle_path), str(out_html))
    assert out_html.exists()
    captured = capsys.readouterr().out
    assert "Report successfully generated" in captured

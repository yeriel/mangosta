"""Tests for mangosta CLI subcommands: inspect, report, view, and inspect --json."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from mangosta.artifacts import save_bundle
from mangosta.cli.main import inspect_bundle, main
from mangosta.export import export_onnx


class MiniModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture
def safetensors_bundle(tmp_path: Path) -> Path:
    model = MiniModel()
    bundle_path = tmp_path / "model.bundle"
    save_bundle(
        output_path=bundle_path,
        model=model,
        family="mini_mlp",
        version="1.2.0",
        sample_input=torch.randn(1, 4),
        config={"learning_rate": 0.01},
        metrics={"accuracy": 0.95},
    )
    return bundle_path


@pytest.fixture
def onnx_bundle(tmp_path: Path) -> Path:
    model = MiniModel().eval()
    bundle_path = tmp_path / "model.onnx.bundle"
    export_onnx(
        model_or_bundle=model,
        output_path=bundle_path,
        sample_input=torch.randn(1, 4),
        as_bundle=True,
    )
    return bundle_path


def test_inspect_text_safetensors(safetensors_bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test standard human-readable terminal inspect on SafeTensors bundle."""
    inspect_bundle(str(safetensors_bundle), as_json=False)
    captured = capsys.readouterr().out

    assert "MANGOSTA BUNDLE INSPECTOR" in captured
    assert "Family:         mini_mlp" in captured
    assert "Version:        1.2.0" in captured
    assert "Weights Format: safetensors" in captured
    assert "model.safetensors" in captured
    assert "accuracy: 0.95" in captured
    assert "Model Tensors / Layers" in captured
    assert "fc.weight" in captured


def test_inspect_text_onnx(onnx_bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test standard human-readable terminal inspect on ONNX bundle."""
    inspect_bundle(str(onnx_bundle), as_json=False)
    captured = capsys.readouterr().out

    assert "MANGOSTA BUNDLE INSPECTOR" in captured
    assert "Weights Format: onnx" in captured
    assert "model.onnx" in captured


def test_inspect_json_safetensors(safetensors_bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test machine-readable inspect --json on SafeTensors bundle."""
    inspect_bundle(str(safetensors_bundle), as_json=True)
    captured = capsys.readouterr().out

    data = json.loads(captured)
    assert data["family"] == "mini_mlp"
    assert data["version"] == "1.2.0"
    assert data["format"] == "safetensors"
    assert data["weights_sha256"] is not None
    assert len(data["weights_sha256"]) == 64
    assert data["metrics"]["accuracy"] == 0.95

    # Check tensors list extracted via zero-torch header parsing
    assert isinstance(data["tensors"], list)
    tensor_names = [t["name"] for t in data["tensors"]]
    assert "fc.weight" in tensor_names
    assert "fc.bias" in tensor_names


def test_inspect_json_onnx(onnx_bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test machine-readable inspect --json on ONNX bundle."""
    inspect_bundle(str(onnx_bundle), as_json=True)
    captured = capsys.readouterr().out

    data = json.loads(captured)
    assert data["format"] == "onnx"
    assert data["weights_sha256"] is not None
    assert "model.onnx" in data["files"]


def test_inspect_missing_file_raises_exit(capsys: pytest.CaptureFixture[str]) -> None:
    """Test inspect on nonexistent path exits with code 1."""
    with pytest.raises(SystemExit) as exc:
        inspect_bundle("nonexistent_bundle.bundle", as_json=False)
    assert exc.value.code == 1

    with pytest.raises(SystemExit) as exc_json:
        inspect_bundle("nonexistent_bundle.bundle", as_json=True)
    assert exc_json.value.code == 1


def test_cli_main_inspect_json(safetensors_bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI dispatch via main() for mangosta inspect <bundle> --json."""
    with patch("sys.argv", ["mangosta", "inspect", str(safetensors_bundle), "--json"]):
        main()

    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["family"] == "mini_mlp"
    assert data["format"] == "safetensors"


def test_cli_main_report(safetensors_bundle: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI dispatch via main() for mangosta report."""
    out_html = tmp_path / "cli_report.html"
    with patch("sys.argv", ["mangosta", "report", str(safetensors_bundle), "-o", str(out_html)]):
        main()

    assert out_html.exists()
    assert "<!DOCTYPE html>" in out_html.read_text(encoding="utf-8")


def test_cli_main_view_dry(tmp_path: Path) -> None:
    """Test CLI view command validation."""
    with patch("sys.argv", ["mangosta", "view", str(tmp_path / "nonexistent")]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

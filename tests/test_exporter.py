"""Unit tests for mangosta.exporter: standalone ONNX export, dict mapping, and precision."""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch
from torch import nn

from mangosta.bundle import save_bundle
from mangosta.exporter import export_onnx
from tests.conftest import DummyMLP


class MultiOutputDictNet(nn.Module):
    """Model matching the user's ParticleSizingNet pattern returning dictionary outputs."""

    def __init__(self, in_features: int = 4) -> None:
        super().__init__()
        self.fc_mask = nn.Linear(in_features, 2)
        self.fc_radius = nn.Linear(in_features, 1)
        self.fc_gradient = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "semantic_mask": self.fc_mask(x),
            "radius_map": self.fc_radius(x),
            "gradient_map": self.fc_gradient(x),
        }


def test_export_onnx_with_dict_output_mapping(tmp_path: Path) -> None:
    model = MultiOutputDictNet(in_features=4)
    onnx_path = tmp_path / "model.onnx"

    out_file = export_onnx(
        model_or_bundle=model,
        output_path=onnx_path,
        sample_input_shape=(2, 4),
        output_mapping=["semantic_mask", "radius_map", "gradient_map"],
        input_names=["input0"],
        output_names=["output0", "output1", "output2"],
        dynamic_axes={
            "input0": {0: "batch"},
            "output0": {0: "batch"},
            "output1": {0: "batch"},
            "output2": {0: "batch"},
        },
        verify=True,
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 0

    # Load with onnx and check outputs
    model_proto = onnx.load(str(out_file))
    onnx.checker.check_model(model_proto)

    graph_outputs = [o.name for o in model_proto.graph.output]
    assert graph_outputs == ["output0", "output1", "output2"]


def test_export_onnx_from_bundle(tmp_path: Path) -> None:
    from mangosta.registry import ModelRegistry

    ModelRegistry.register_model("dummy_mlp", "1.0.0", DummyMLP)

    # Save a bundle first
    model = DummyMLP(in_features=4, out_features=2)
    bundle_path = tmp_path / "simple.bundle"
    save_bundle(
        output_path=bundle_path,
        model=model,
        family="dummy_mlp",
        version="1.0.0",
        init_args={"in_features": 4, "out_features": 2},
    )

    onnx_path = tmp_path / "from_bundle.onnx"
    out_file = export_onnx(
        model_or_bundle=bundle_path,
        output_path=onnx_path,
        sample_input_shape=(1, 4),
        verify=True,
    )

    assert out_file.exists()
    model_proto = onnx.load(str(out_file))
    onnx.checker.check_model(model_proto)


def test_export_onnx_with_weights_file(tmp_path: Path) -> None:
    model = DummyMLP(in_features=4, out_features=2)
    weights_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict()}, weights_path)

    fresh_model = DummyMLP(in_features=4, out_features=2)
    onnx_path = tmp_path / "with_weights.onnx"

    out_file = export_onnx(
        model_or_bundle=fresh_model,
        output_path=onnx_path,
        weights=weights_path,
        sample_input=torch.randn(1, 4),
        verify=True,
    )

    assert out_file.exists()


def test_export_onnx_missing_dict_key_raises(tmp_path: Path) -> None:
    model = MultiOutputDictNet(in_features=4)
    onnx_path = tmp_path / "fail.onnx"

    with pytest.raises(KeyError, match="Missing expected keys"):
        export_onnx(
            model_or_bundle=model,
            output_path=onnx_path,
            sample_input_shape=(1, 4),
            output_mapping=["semantic_mask", "non_existent_key"],
        )

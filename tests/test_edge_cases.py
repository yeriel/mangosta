"""Additional unit tests to harden edge cases and ensure >85% test coverage across Mangosta."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import mangosta
from mangosta.artifacts import load_bundle, resume_training, save_bundle
from mangosta.core.registry import ModelRegistry, register_component
from mangosta.core.schema import ModelSchema, TensorSpec, infer_schema, infer_schema_from_sample
from mangosta.exceptions import (
    BundleCorruptedError,
    ComponentNotFoundError,
    ModelNotFoundError,
    RegistryCollisionError,
    ResumeStateError,
    SchemaValidationError,
)
from mangosta.export import export_onnx
from mangosta.serving import ONNXRunner
from mangosta.tracking import RunTracker


class LinearToy(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_schema_edge_cases_and_checks() -> None:
    spec = TensorSpec(name="t", shape=[-1, 4], dtype="float32")
    assert spec.shape == [None, 4]
    
    # Check dictionary conversion
    d = spec.to_dict()
    assert d["shape"] == [None, 4]
    restored = TensorSpec.from_dict(d)
    assert restored.shape == [None, 4]

    # ModelSchema checking output helpers
    schema = infer_schema_from_sample(
        sample_input=np.zeros((2, 4), dtype=np.float32),
        sample_output=np.zeros((2, 2), dtype=np.float32),
    )
    assert schema.is_valid_output(np.zeros((3, 2), dtype=np.float32)) is True
    assert schema.is_valid_output(np.zeros((3, 3), dtype=np.float32)) is False
    
    ok, err = schema.check_output(np.zeros((3, 3), dtype=np.float32))
    assert ok is False
    assert "dimension 1 mismatch" in str(err)

    # Sequence structure validation
    seq_schema = infer_schema_from_sample(
        sample_input=[np.zeros((1, 4), dtype=np.float32), np.zeros((1, 2), dtype=np.float32)],
        sample_output=np.zeros((1, 1), dtype=np.float32),
    )
    assert seq_schema.input_structure == "sequence"
    assert seq_schema.is_valid_input([np.zeros((2, 4), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]) is True
    # Sequence length mismatch
    with pytest.raises(SchemaValidationError, match="sequence length mismatch"):
        seq_schema.validate_input([np.zeros((2, 4), dtype=np.float32)])

    # Mapping structure validation
    map_schema = infer_schema_from_sample(
        sample_input={"x": np.zeros((1, 4), dtype=np.float32)},
        sample_output={"y": np.zeros((1, 2), dtype=np.float32)},
    )
    assert map_schema.input_structure == "mapping"
    assert map_schema.output_structure == "mapping"
    assert map_schema.is_valid_input({"x": np.zeros((3, 4), dtype=np.float32)}) is True
    # Missing key
    with pytest.raises(SchemaValidationError, match="missing required key"):
        map_schema.validate_input({"other": np.zeros((3, 4), dtype=np.float32)})


def test_registry_edge_cases() -> None:
    ModelRegistry.clear()
    
    @register_component("loss", "cross_entropy")
    def my_ce(a: Any, b: Any) -> float:
        return 0.0

    assert ModelRegistry.get_component("loss", "cross_entropy") is my_ce

    # Duplicate component collision
    with pytest.raises(RegistryCollisionError):
        register_component("loss", "cross_entropy")(lambda a, b: 1.0)

    # Component not found
    with pytest.raises(ComponentNotFoundError):
        ModelRegistry.get_component("loss", "nonexistent")

    # Non-subclass registration fails
    with pytest.raises(TypeError):
        ModelRegistry.register_model("dummy", "1.0.0", object)  # type: ignore

    # Model not found error hints
    with pytest.raises(ModelNotFoundError):
        ModelRegistry.get_model_cls("unregistered_family")


def test_tracker_image_and_artifacts_edge_cases(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    with RunTracker(experiment="edge_exp", run_name="run_01", root_dir=exp_dir) as tracker:
        # Grayscale NumPy image (2D)
        gray = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        p1 = tracker.log_image("gray_sample", gray, step=0)
        assert p1.exists()

        # RGBA NumPy image (3D, 4 channels)
        rgba = np.random.randint(0, 255, (32, 32, 4), dtype=np.uint8)
        p2 = tracker.log_image("rgba_sample", rgba, step=1)
        assert p2.exists()

        # Existing image file copying
        local_img = tmp_path / "test_copy.png"
        local_img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        p3 = tracker.log_image("copied", local_img, step=2)
        assert p3.exists()

        # Missing artifact raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            tracker.log_artifact(tmp_path / "missing_file.txt")

        # get_metrics
        tracker.log_metrics({"loss": 0.42}, step=0)
        metrics = tracker.get_metrics()
        assert "loss" in metrics
        assert len(metrics["loss"]) == 1


def test_loader_and_bundle_edge_cases(tmp_path: Path) -> None:
    # Save bundle with config as dict and string
    model = LinearToy()
    b1 = tmp_path / "cfg_dict.bundle"
    save_bundle(b1, model, config={"lr": 0.01})
    loaded1 = load_bundle(b1, model=LinearToy())
    assert loaded1.config == {"lr": 0.01}
    assert "LoadedBundle" in repr(loaded1)

    # Config as raw string
    b2 = tmp_path / "cfg_str.bundle"
    save_bundle(b2, model, config="raw_string_config")
    loaded2 = load_bundle(b2, model=LinearToy())
    assert loaded2.config == "raw_string_config"

    # Nonexistent bundle
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "ghost.bundle")

    # Nonexistent resume_training
    with pytest.raises(FileNotFoundError):
        resume_training(tmp_path / "ghost.bundle", model=LinearToy())

    # Bundle missing train_state
    with pytest.raises(ResumeStateError):
        resume_training(b1, model=LinearToy())


def test_onnx_runner_and_export_edge_cases(tmp_path: Path) -> None:
    model = LinearToy().eval()
    onnx_bundle = tmp_path / "toy.onnx.bundle"
    export_onnx(
        model,
        onnx_bundle,
        sample_input=torch.randn(1, 4),
        as_bundle=True,
        config={"threshold": 0.5},
    )

    runner = ONNXRunner(onnx_bundle)
    assert runner.config == {"threshold": 0.5}
    assert "LoadedONNXBundle" in repr(runner)

    # Predict with PyTorch tensor directly on LoadedONNXBundle (automatic torch -> numpy conversion)
    t_in = torch.randn(2, 4)
    out_from_torch = runner.predict(t_in)
    assert isinstance(out_from_torch, np.ndarray)
    assert out_from_torch.shape == (2, 2)

    # Standalone ONNX file loading
    raw_onnx = tmp_path / "raw.onnx"
    export_onnx(model, raw_onnx, sample_input=torch.randn(1, 4), as_bundle=False)
    runner_raw = ONNXRunner(raw_onnx)
    assert runner_raw.predict(np.zeros((1, 4), dtype=np.float32)).shape == (1, 2)


def test_public_api_getattr_safety() -> None:
    # Existing symbols resolve cleanly
    assert mangosta.__version__ == "0.3.0"
    assert hasattr(mangosta, "save_bundle")
    assert hasattr(mangosta, "load_bundle")
    assert hasattr(mangosta, "RunTracker")
    assert hasattr(mangosta, "ONNXRunner")

    # Nonexistent attribute raises AttributeError
    with pytest.raises(AttributeError):
        _ = mangosta.nonexistent_symbol  # type: ignore

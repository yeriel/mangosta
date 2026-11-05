"""Comprehensive unit and integration tests for zero-PyTorch ONNX serving and contract enforcement."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from mangosta.artifacts import load_bundle, read_bundle_manifest
from mangosta.core.schema import ModelSchema, TensorSpec, infer_schema_from_sample
from mangosta.exceptions import BundleCorruptedError, SchemaValidationError
from mangosta.export import export_onnx
from mangosta.serving import LoadedONNXBundle, ONNXRunner


class ServingMLP(nn.Module):
    """Canonical MLP for testing ONNX export and serving."""

    def __init__(self, in_features: int = 8, hidden_features: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class MultiInputNet(nn.Module):
    """Model taking multiple tensor inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.linear_a = nn.Linear(4, 8)
        self.linear_b = nn.Linear(6, 8)
        self.out = nn.Linear(8, 2)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.linear_a(a) + self.linear_b(b)))


def test_export_onnx_as_bundle(tmp_path: Path) -> None:
    """Test exporting a PyTorch model into an autonomous ONNX bundle (.onnx.bundle)."""
    model = ServingMLP(8, 16, 2)
    model.eval()
    sample_input = torch.randn(2, 8, dtype=torch.float32)

    bundle_path = tmp_path / "model.onnx.bundle"
    exported_path = export_onnx(
        model_or_bundle=model,
        output_path=bundle_path,
        sample_input=sample_input,
        as_bundle=True,
        config={"preprocessing": {"normalize": True}},
        metadata={"author": "mangosta", "target": "edge"},
    )

    assert exported_path == bundle_path
    assert bundle_path.exists()

    # Check zero leftover temporary files
    leftover_tmps = list(tmp_path.glob(".tmp_*"))
    assert len(leftover_tmps) == 0, f"Leftover temporary files detected: {leftover_tmps}"

    # Verify internal structure of ONNX bundle ZIP
    with zipfile.ZipFile(bundle_path, "r") as zf:
        namelist = zf.namelist()
        assert "model.onnx" in namelist
        assert "manifest.json" in namelist
        assert "config.yaml" in namelist
        assert "model.safetensors" not in namelist

    manifest = read_bundle_manifest(bundle_path)
    assert manifest["weights_format"] == "onnx"
    assert manifest["family"] == "servingmlp"
    assert manifest["version"] == "0.1.0"
    assert "model_sha256" in manifest
    assert len(manifest["model_sha256"]) == 64
    assert manifest["metadata"]["author"] == "mangosta"
    assert manifest["schema"] is not None
    assert manifest["schema"]["input_structure"] == "tensor"


def test_load_onnx_bundle_and_predict(tmp_path: Path) -> None:
    """Test loading an ONNX bundle with load_bundle and performing inference with bit-to-bit precision."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = ServingMLP(8, 16, 2)
    model.eval()
    sample_input = torch.randn(1, 8, dtype=torch.float32)

    bundle_path = tmp_path / "model_prod.onnx.bundle"
    export_onnx(
        model_or_bundle=model,
        output_path=bundle_path,
        sample_input=sample_input,
        as_bundle=True,
    )

    # Load bundle via universal loader
    loaded = load_bundle(bundle_path)
    assert isinstance(loaded, LoadedONNXBundle)
    assert loaded.family == "servingmlp"
    assert loaded.schema is not None

    # Test inference with pure NumPy array
    np_test = np.random.randn(5, 8).astype(np.float32)
    onnx_pred = loaded.predict(np_test)

    # Compare against PyTorch forward pass
    with torch.inference_mode():
        torch_pred = model(torch.from_numpy(np_test)).numpy()

    np.testing.assert_allclose(onnx_pred, torch_pred, rtol=1e-5, atol=1e-5)


def test_onnx_serving_schema_contract_validation(tmp_path: Path) -> None:
    """Test that schema violations on NumPy arrays are rejected before reaching C++ ONNX runtime."""
    model = ServingMLP(8, 16, 2)
    model.eval()
    sample_input = torch.randn(1, 8, dtype=torch.float32)

    bundle_path = tmp_path / "model_contract.onnx.bundle"
    export_onnx(
        model_or_bundle=model,
        output_path=bundle_path,
        sample_input=sample_input,
        as_bundle=True,
    )

    loaded = load_bundle(bundle_path)

    # 1. Shape mismatch (expected 8 features, passed 10)
    bad_shape_input = np.random.randn(2, 10).astype(np.float32)
    with pytest.raises(SchemaValidationError) as exc_shape:
        loaded.predict(bad_shape_input)
    assert "dimension 1 mismatch" in str(exc_shape.value)

    # 2. Dtype mismatch (expected float32, passed float64)
    bad_dtype_input = np.random.randn(2, 8).astype(np.float64)
    with pytest.raises(SchemaValidationError) as exc_dtype:
        loaded.predict(bad_dtype_input)
    assert "expected dtype float32, got float64" in str(exc_dtype.value)

    # 3. Non-array type violation
    with pytest.raises(SchemaValidationError) as exc_type:
        loaded.predict("not_a_tensor_or_array")
    assert "must be a torch.Tensor or numpy.ndarray" in str(exc_type.value)


def test_onnx_bundle_tamper_detection(tmp_path: Path) -> None:
    """Test cryptographic integrity verification: 1 byte altered in model.onnx halts loading."""
    model = ServingMLP(8, 16, 2)
    bundle_path = tmp_path / "valid.onnx.bundle"
    export_onnx(
        model_or_bundle=model,
        output_path=bundle_path,
        sample_input=torch.randn(1, 8),
        as_bundle=True,
    )

    tampered_bundle = tmp_path / "tampered.onnx.bundle"
    with zipfile.ZipFile(bundle_path, "r") as src, zipfile.ZipFile(tampered_bundle, "w") as dst:
        for item in src.infolist():
            content = src.read(item.filename)
            if item.filename == "model.onnx":
                # Alter 1 byte in binary graph
                content = content[:-1] + b"\x00"
            dst.writestr(item, content)

    with pytest.raises(BundleCorruptedError) as exc_info:
        load_bundle(tampered_bundle)

    assert "cryptographic integrity check failed" in str(exc_info.value)
    assert "corrupted or tampered" in str(exc_info.value)


def test_onnx_runner_standalone(tmp_path: Path) -> None:
    """Test direct usage of ONNXRunner for standalone .onnx files and bundles."""
    model = ServingMLP(8, 16, 2)
    raw_onnx_path = tmp_path / "standalone.onnx"
    export_onnx(
        model_or_bundle=model,
        output_path=raw_onnx_path,
        sample_input=torch.randn(1, 8),
        as_bundle=False,
    )
    assert raw_onnx_path.exists()

    # Create explicit schema
    schema = infer_schema_from_sample(
        sample_input=np.zeros((1, 8), dtype=np.float32),
        sample_output=np.zeros((1, 2), dtype=np.float32),
    )

    runner = ONNXRunner(raw_onnx_path, schema=schema)
    assert len(runner.input_names) == 1
    assert len(runner.output_names) == 1

    x = np.random.randn(3, 8).astype(np.float32)
    out = runner.predict(x)
    assert out.shape == (3, 2)

    # Schema validation with standalone runner
    with pytest.raises(SchemaValidationError):
        runner.predict(np.random.randn(3, 9).astype(np.float32))


def test_infer_schema_from_sample_pure_numpy() -> None:
    """Test inferring a ModelSchema using exclusively NumPy arrays without PyTorch."""
    sample_in = np.zeros((2, 12), dtype=np.float32)
    sample_out = np.zeros((2, 4), dtype=np.float32)

    schema = infer_schema_from_sample(sample_in, sample_out, dynamic_batch_dim=0)
    assert schema.input_structure == "tensor"
    assert schema.output_structure == "tensor"
    assert schema.inputs[0].shape == [None, 12]
    assert schema.inputs[0].dtype == "float32"
    assert schema.outputs[0].shape == [None, 4]
    assert schema.outputs[0].dtype == "float32"

    # Validation succeeds on valid NumPy array
    schema.validate_input(np.random.randn(10, 12).astype(np.float32))

    # Validation fails on invalid dimension
    with pytest.raises(SchemaValidationError):
        schema.validate_input(np.random.randn(10, 15).astype(np.float32))


def test_multi_input_onnx_bundle_serving(tmp_path: Path) -> None:
    """Test exporting and serving a multi-input model with contract validation."""
    model = MultiInputNet()
    model.eval()
    sample_a = torch.randn(1, 4, dtype=torch.float32)
    sample_b = torch.randn(1, 6, dtype=torch.float32)

    bundle_path = tmp_path / "multi.onnx.bundle"
    export_onnx(
        model_or_bundle=model,
        output_path=bundle_path,
        sample_input=(sample_a, sample_b),
        input_names=["input_a", "input_b"],
        output_names=["output"],
        as_bundle=True,
    )

    loaded = load_bundle(bundle_path)
    assert len(loaded.input_names) == 2

    # Predict with tuple of NumPy arrays
    np_a = np.random.randn(3, 4).astype(np.float32)
    np_b = np.random.randn(3, 6).astype(np.float32)

    out = loaded.predict((np_a, np_b))
    assert out.shape == (3, 2)

    # Schema rejects dict because export sample was a sequence tuple
    with pytest.raises(SchemaValidationError):
        loaded.predict({"input_a": np_a, "input_b": np_b})

    # With validate=False, ONNXRunner still resolves dict feeds properly
    out_dict_unvalidated = loaded.predict({"input_a": np_a, "input_b": np_b}, validate=False)
    np.testing.assert_allclose(out, out_dict_unvalidated)

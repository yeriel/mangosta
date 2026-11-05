"""Shared PyTorch test fixtures and dummy architectures for mangosta tests."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn

from mangosta.core import ModelRegistry


class DummyMLP(nn.Module):
    """Simple linear model for testing."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyMultiInput(nn.Module):
    """Model taking multiple tensor arguments."""

    def __init__(self, feat_dim: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim * 2, 2)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([a, b], dim=-1))


class DummyDictInput(nn.Module):
    """Model taking keyword / mapping tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.fc(x)
        if mask is not None:
            out = out * mask
        return out


class NonContiguousNet(nn.Module):
    """Model with parameters that are non-contiguous in memory to test SafeTensors normalization."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Create non-contiguous tensor via transpose
        base = torch.randn(out_features, in_features)
        transposed = base.t()
        assert not transposed.is_contiguous(), "Expected non-contiguous tensor"
        self.weight = nn.Parameter(transposed)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


class MultiOutputDictNet(nn.Module):
    """Model returning multiple tensor heads in a dictionary."""

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


def assert_bundle_integrity(bundle_path: Path) -> dict:
    """Audit bundle ZIP archive layout and cryptographic SafeTensors hash."""
    import hashlib
    import json
    import zipfile

    assert bundle_path.exists(), f"Bundle file does not exist: {bundle_path}"
    with zipfile.ZipFile(bundle_path, "r") as zf:
        namelist = zf.namelist()
        assert "manifest.json" in namelist, "manifest.json missing from bundle"
        assert "model.safetensors" in namelist, "model.safetensors missing from bundle"
        assert "weights.pt" not in namelist, "Forbidden legacy weights.pt found in bundle"

        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest.get("weights_format") == "safetensors"
        assert "weights_sha256" in manifest
        expected_sha = manifest["weights_sha256"]
        assert len(expected_sha) == 64

        actual_sha = hashlib.sha256(zf.read("model.safetensors")).hexdigest()
        assert actual_sha == expected_sha, f"SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"

        return manifest



@pytest.fixture(autouse=True)
def clean_registry() -> Generator[None, None, None]:
    """Ensure registry state is clean before and after each test."""
    ModelRegistry.clear()
    yield
    ModelRegistry.clear()


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    """Provide clean temporary directory."""
    return tmp_path

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

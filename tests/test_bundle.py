"""Unit tests for mangosta.bundle: bundle creation, ZIP layout, and checkpointing."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import torch
import torch.optim as optim

from mangosta.bundle import read_bundle_manifest, save_bundle
from mangosta.exceptions import BundleCorruptedError
from tests.conftest import DummyMLP


def test_save_bundle_inference_mode(tmp_path: Path) -> None:
    model = DummyMLP(in_features=4, out_features=2)
    bundle_path = tmp_path / "model.bundle"

    saved = save_bundle(
        output_path=bundle_path,
        model=model,
        family="mlp",
        version="1.0.0",
        init_args={"in_features": 4, "out_features": 2},
        sample_input=torch.randn(1, 4),
        is_checkpoint=False,
        config={"lr": 0.01, "batch_size": 32},
    )

    assert saved.exists()
    with zipfile.ZipFile(saved, "r") as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert "weights.pt" in names
        assert "config.yaml" in names
        assert "train_state.pt" not in names

    manifest = read_bundle_manifest(saved)
    assert manifest["family"] == "mlp"
    assert manifest["version"] == "1.0.0"
    assert manifest["is_checkpoint"] is False
    assert manifest["schema"] is not None
    assert manifest["schema"]["inputs"][0]["shape"] == [None, 4]


def test_save_bundle_checkpoint_mode(tmp_path: Path) -> None:
    model = DummyMLP(in_features=4, out_features=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

    # Perform a dummy optimizer step so state is populated
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    bundle_path = tmp_path / "checkpoint.bundle"
    saved = save_bundle(
        output_path=bundle_path,
        model=model,
        family="mlp",
        version="1.0.0",
        init_args={"in_features": 4, "out_features": 2},
        is_checkpoint=True,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        step=150,
        metrics={"loss": 0.25},
    )

    assert saved.exists()
    with zipfile.ZipFile(saved, "r") as zf:
        names = zf.namelist()
        assert "train_state.pt" in names

    manifest = read_bundle_manifest(saved)
    assert manifest["is_checkpoint"] is True
    assert manifest["epoch"] == 3
    assert manifest["step"] == 150
    assert manifest["metrics"]["loss"] == 0.25


def test_corrupted_bundle_handling(tmp_path: Path) -> None:
    bad_zip = tmp_path / "corrupted.bundle"
    bad_zip.write_bytes(b"not a valid zip file")

    with pytest.raises(BundleCorruptedError, match="corrupted ZIP archive"):
        read_bundle_manifest(bad_zip)

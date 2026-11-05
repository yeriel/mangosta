"""Unit tests for mangosta.bundle: bundle creation, ZIP layout, and checkpointing."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import torch
import torch.optim as optim

from mangosta.artifacts.bundle import read_bundle_manifest, save_bundle
from mangosta.exceptions import BundleCorruptedError
from tests.conftest import DummyMLP, NonContiguousNet, assert_bundle_integrity


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
        config={"lr": 0.01, "batch_size": 32},
    )

    assert saved.exists()
    with zipfile.ZipFile(saved, "r") as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert "model.safetensors" in names
        assert "weights.pt" not in names
        assert "config.yaml" in names
        assert "train_state.pt" not in names

    manifest = read_bundle_manifest(saved)
    assert manifest["family"] == "mlp"
    assert manifest["version"] == "1.0.0"
    assert manifest["is_checkpoint"] is False
    assert manifest["weights_format"] == "safetensors"
    assert "weights_sha256" in manifest
    assert len(manifest["weights_sha256"]) == 64
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


def test_save_bundle_smart_defaults_inference_mode(tmp_path: Path) -> None:
    model = DummyMLP(in_features=4, out_features=2)
    bundle_path = tmp_path / "smart_default_infer.bundle"

    # Only output_path and model passed
    saved = save_bundle(bundle_path, model)

    assert saved.exists()
    manifest = read_bundle_manifest(saved)
    assert manifest["family"] == "dummymlp"
    assert manifest["version"] == "0.1.0"
    assert manifest["is_checkpoint"] is False

    with zipfile.ZipFile(saved, "r") as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert "model.safetensors" in names
        assert "weights.pt" not in names
        assert "train_state.pt" not in names


def test_save_bundle_smart_defaults_checkpoint_mode(tmp_path: Path) -> None:
    model = DummyMLP(in_features=4, out_features=2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    bundle_path = tmp_path / "smart_default_ckpt.bundle"

    # Optimizer passed without explicit is_checkpoint or family
    saved = save_bundle(bundle_path, model, optimizer=optimizer, epoch=4)

    assert saved.exists()
    manifest = read_bundle_manifest(saved)
    assert manifest["family"] == "dummymlp"
    assert manifest["version"] == "0.1.0"
    assert manifest["is_checkpoint"] is True
    assert manifest["epoch"] == 4
    assert manifest["weights_format"] == "safetensors"
    assert len(manifest["weights_sha256"]) == 64

    with zipfile.ZipFile(saved, "r") as zf:
        names = zf.namelist()
        assert "model.safetensors" in names
        assert "train_state.pt" in names


def test_save_bundle_cleanup_tmp_on_failure(tmp_path: Path) -> None:
    model = DummyMLP(in_features=4, out_features=2)
    bundle_path = tmp_path / "fail.bundle"

    from unittest.mock import patch

    # Simulate an error during zip writing
    with patch("zipfile.ZipFile", side_effect=RuntimeError("Disk failure simulation")):
        with pytest.raises(RuntimeError, match="Disk failure simulation"):
            save_bundle(bundle_path, model)

    # Verify that NO temporary .tmp_* files remain in the target directory
    tmp_files = list(tmp_path.glob("*.tmp_*"))
    assert len(tmp_files) == 0, f"Found orphaned temp files: {tmp_files}"
    assert not bundle_path.exists()


def test_save_bundle_non_contiguous_weights_handling(tmp_path: Path) -> None:
    """SafeTensors requires contiguous buffers; save_bundle must normalize non-contiguous parameters."""
    model = NonContiguousNet(in_features=4, out_features=2)
    bundle_path = tmp_path / "non_contiguous.bundle"

    saved = save_bundle(bundle_path, model)
    assert saved.exists()

    manifest = assert_bundle_integrity(saved)
    assert manifest["family"] == "noncontiguousnet"
    assert manifest["weights_format"] == "safetensors"


def test_save_bundle_cleanup_on_replace_failure(tmp_path: Path) -> None:
    """If os.replace fails (e.g. disk full), the finally block must clean up the .tmp_* file."""
    model = DummyMLP(in_features=4, out_features=2)
    bundle_path = tmp_path / "replace_fail.bundle"

    from unittest.mock import patch

    with patch("os.replace", side_effect=OSError("Disk full simulation during rename")):
        with pytest.raises(OSError, match="Disk full simulation"):
            save_bundle(bundle_path, model)

    # Ensure no leftover temp files
    tmp_files = list(tmp_path.glob("*.tmp_*"))
    assert len(tmp_files) == 0, f"Found leftover temp files: {tmp_files}"
    assert not bundle_path.exists()




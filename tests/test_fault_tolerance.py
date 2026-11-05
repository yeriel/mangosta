"""Fault tolerance, corruption resistance, and atomic safety test suite."""

from __future__ import annotations

import os
import random
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from mangosta.artifacts import load_bundle, resume_training, save_bundle
from mangosta.exceptions import BundleCorruptedError
from tests.conftest import DummyMLP


def test_truncated_bundle_rejected(tmp_path: Path) -> None:
    """Half-written bundle (simulating sudden SIGKILL or power outage) must raise BundleCorruptedError."""
    model = DummyMLP(in_features=4, out_features=2)
    valid_bundle = tmp_path / "valid.bundle"
    save_bundle(valid_bundle, model)

    raw_bytes = valid_bundle.read_bytes()
    truncated_bundle = tmp_path / "truncated.bundle"
    # Cut in half
    truncated_bundle.write_bytes(raw_bytes[: len(raw_bytes) // 2])

    with pytest.raises(BundleCorruptedError, match="corrupted ZIP archive"):
        load_bundle(truncated_bundle, model=DummyMLP(4, 2))

    with pytest.raises(BundleCorruptedError, match="corrupted ZIP archive"):
        resume_training(truncated_bundle, model=DummyMLP(4, 2))


def test_zero_byte_bundle_rejected(tmp_path: Path) -> None:
    """Empty 0-byte file must be caught immediately as corrupted archive."""
    empty_bundle = tmp_path / "empty.bundle"
    empty_bundle.write_bytes(b"")

    with pytest.raises(BundleCorruptedError):
        load_bundle(empty_bundle, model=DummyMLP(4, 2))


def test_random_byte_fuzzing_rejected(tmp_path: Path) -> None:
    """Random noise injected into bundle must be cleanly intercepted by cryptographic or ZIP checks."""
    model = DummyMLP(in_features=4, out_features=2)
    bundle_path = tmp_path / "fuzz_target.bundle"
    save_bundle(bundle_path, model)

    raw_bytes = bytearray(bundle_path.read_bytes())
    # Corrupt 20 random byte positions
    rng = random.Random(42)
    for _ in range(20):
        pos = rng.randint(0, len(raw_bytes) - 1)
        raw_bytes[pos] = rng.randint(0, 255)

    fuzzed_bundle = tmp_path / "fuzzed.bundle"
    fuzzed_bundle.write_bytes(bytes(raw_bytes))

    with pytest.raises(BundleCorruptedError):
        load_bundle(fuzzed_bundle, model=DummyMLP(4, 2))


def test_atomic_save_preserves_existing_bundle_on_failure(tmp_path: Path) -> None:
    """If save_bundle fails during write, the pre-existing bundle must remain 100% intact."""
    model1 = DummyMLP(in_features=4, out_features=2)
    dest_bundle = tmp_path / "production_model.bundle"

    # 1. Create original valid bundle
    save_bundle(dest_bundle, model1, version="1.0.0")
    original_bytes = dest_bundle.read_bytes()
    assert len(original_bytes) > 0

    # 2. Attempt to overwrite with a failing save_bundle (simulated disk/serialization crash)
    model2 = DummyMLP(in_features=4, out_features=2)
    with patch("zipfile.ZipFile.writestr", side_effect=OSError("Disk write error mid-stream")):
        with pytest.raises(OSError, match="Disk write error"):
            save_bundle(dest_bundle, model2, version="2.0.0")

    # 3. Verify original bundle is still completely intact
    assert dest_bundle.exists()
    assert dest_bundle.read_bytes() == original_bytes

    # Verify model1 can still be loaded without issue
    loaded = load_bundle(dest_bundle, model=DummyMLP(4, 2))
    assert loaded.version == "1.0.0"

    # Verify zero leftover .tmp_* files
    tmp_files = list(tmp_path.glob("*.tmp_*"))
    assert len(tmp_files) == 0, f"Leftover temp files found: {tmp_files}"

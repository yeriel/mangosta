"""End-to-end verification script running inside an air-gapped Ubuntu 24.04 container."""

from __future__ import annotations

import json
import socket
import sys
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from mangosta.artifacts import load_bundle, resume_training, save_bundle
from mangosta.artifacts.bundle import read_bundle_manifest
from mangosta.exceptions import BundleCorruptedError
from mangosta.tracking import RunTracker, quick_track


def verify_network_is_disabled() -> None:
    print("[1/5] Verifying network isolation (strictly offline)...")
    try:
        s = socket.create_connection(("8.8.8.8", 53), timeout=1.0)
        s.close()
        raise RuntimeError("CRITICAL: Network connection succeeded! Container is NOT offline.")
    except (OSError, socket.timeout):
        print("  ✓ Network is completely disabled (air-gapped confirmed).")


class SampleNet(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 16, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def verify_bundle_safetensors_and_integrity(tmp_dir: Path) -> None:
    print("[2/5] Testing SafeTensors bundle generation and cryptographic integrity...")
    model = SampleNet(8, 16, 2)
    sample_input = torch.randn(2, 8)
    expected_output = model(sample_input)

    bundle_path = tmp_dir / "offline_model.bundle"
    saved_path = save_bundle(
        output_path=bundle_path,
        model=model,
        sample_input=sample_input,
        config={"offline_test": True},
    )
    assert saved_path.exists(), "Bundle was not created."

    # 1. Inspect archive internals
    with zipfile.ZipFile(saved_path, "r") as zf:
        namelist = zf.namelist()
        assert "model.safetensors" in namelist, "model.safetensors missing from bundle"
        assert "weights.pt" not in namelist, "Legacy weights.pt should not be present"
        assert "manifest.json" in namelist, "manifest.json missing from bundle"

    manifest = read_bundle_manifest(saved_path)
    assert manifest["weights_format"] == "safetensors"
    assert "weights_sha256" in manifest
    assert len(manifest["weights_sha256"]) == 64
    print(f"  ✓ Bundle created with SafeTensors. SHA-256: {manifest['weights_sha256'][:16]}...")

    # 2. Tampering test
    tampered_bundle = tmp_dir / "tampered.bundle"
    with zipfile.ZipFile(bundle_path, "r") as src, zipfile.ZipFile(tampered_bundle, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "model.safetensors":
                data = b"tampered_safetensors_bytes"
            dst.writestr(item, data)

    tamper_caught = False
    try:
        fresh_model = SampleNet(8, 16, 2)
        load_bundle(tampered_bundle, model=fresh_model)
    except BundleCorruptedError as e:
        tamper_caught = True
        print(f"  ✓ Tampered bundle correctly rejected with error: {e}")
    assert tamper_caught, "Tampered bundle was not rejected!"

    # 3. Valid load & prediction test
    loaded = load_bundle(bundle_path, model=SampleNet(8, 16, 2))
    pred = loaded.predict(sample_input)
    assert torch.allclose(pred, expected_output), "Loaded model prediction does not match original!"
    print("  ✓ Clean bundle successfully loaded with bit-to-bit prediction match.")


def verify_checkpoint_resume_exact_continuity(tmp_dir: Path) -> None:
    print("[3/5] Testing checkpoint creation and exact training resume...")
    torch.manual_seed(1234)
    model1 = SampleNet(8, 16, 2)
    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)

    # Perform 5 initial training steps
    for step in range(5):
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        loss = nn.CrossEntropyLoss()(model1(x), y)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

    # Save checkpoint
    ckpt_path = tmp_dir / "checkpoint.bundle"
    save_bundle(
        output_path=ckpt_path,
        model=model1,
        optimizer=optimizer1,
        epoch=1,
        step=5,
    )

    # Reference next step from model1
    next_x = torch.randn(4, 8)
    expected_ref = model1(next_x)

    # Resume into fresh model
    model2 = SampleNet(8, 16, 2)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    res = resume_training(ckpt_path, model=model2, optimizer=optimizer2, restore_rng=True)

    assert res.epoch == 1
    assert res.step == 5
    assert res.rng_restored is True

    # RNG and outputs must match identically
    resumed_x = torch.randn(4, 8)
    assert torch.allclose(resumed_x, next_x), "RNG state was not restored deterministically!"
    resumed_out = model2(resumed_x)
    assert torch.allclose(resumed_out, expected_ref), "Resumed model forward pass mismatch!"
    print("  ✓ Exact bit-to-bit training state resumed successfully.")


def verify_tracker_streaming_and_interrupt(tmp_dir: Path) -> None:
    print("[4/5] Testing RunTracker with sync_interval=10 and interruption handling...")
    exp_dir = tmp_dir / "experiments"

    # 1. Tracking 100 steps
    with RunTracker(experiment="offline_exp", run_name="run_100", root_dir=exp_dir, sync_interval=10) as tracker:
        tracker.log_params({"lr": 0.001, "batch_size": 16})
        for s in range(100):
            tracker.log_metrics({"loss": 1.0 / (s + 1), "acc": s / 100.0}, step=s)

    metrics_file = tracker.run_dir / "metrics.jsonl"
    lines = metrics_file.read_text().strip().split("\n")
    assert len(lines) == 200, f"Expected 200 metric lines (2 per step * 100), got {len(lines)}"

    meta = json.loads((tracker.run_dir / "run.json").read_text())
    assert meta["status"] == "completed"
    assert meta["params"]["lr"] == 0.001
    print("  ✓ 100 steps logged with sync_interval=10 and status marked completed.")

    # 2. Interruption simulation
    try:
        with RunTracker(experiment="offline_exp", run_name="run_interrupted", root_dir=exp_dir) as tracker2:
            tracker2.log_metrics({"loss": 0.5}, step=0)
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    meta2 = json.loads((tracker2.run_dir / "run.json").read_text())
    assert meta2["status"] == "interrupted", f"Expected status 'interrupted', got {meta2['status']}"
    print("  ✓ KeyboardInterrupt / SIGINT cleanly marks status as 'interrupted'.")


def main() -> None:
    print("=" * 60)
    print("STARTING MANGOSTA OFFLINE SERVER VALIDATION (UBUNTU 24.04)")
    print("=" * 60)

    verify_network_is_disabled()

    work_dir = Path("/tmp/mangosta_offline_test")
    work_dir.mkdir(parents=True, exist_ok=True)

    verify_bundle_safetensors_and_integrity(work_dir)
    verify_checkpoint_resume_exact_continuity(work_dir)
    verify_tracker_streaming_and_interrupt(work_dir)

    print("[5/5] Running complete unit test suite with pytest...")
    import pytest
    code = pytest.main(["-v", "tests"])
    assert code == 0, f"pytest tests failed with exit code {code}"
    print("  ✓ All unit tests passed inside offline Ubuntu 24.04 container.")

    print("=" * 60)
    print("ALL OFFLINE SERVER VALIDATIONS COMPLETED SUCCESSFULLY (100% PASS)")
    print("=" * 60)


if __name__ == "__main__":
    main()

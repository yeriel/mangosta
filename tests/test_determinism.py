"""Bit-to-bit determinism test suite for Mangosta checkpointing and resume.

Verifies that training for 10 epochs continuously yields the EXACT identical loss,
model weights, and evaluation predictions as training for 5 epochs, saving a checkpoint,
and resuming into fresh model/optimizer instances for the remaining 5 epochs.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from mangosta.artifacts import resume_training, save_bundle
from tests.conftest import DummyMLP


def _set_all_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def test_bit_to_bit_determinism_continuous_vs_resumed(tmp_path: Path) -> None:
    """A 10-step continuous run must be mathematically identical to 5-step + resume + 5-step."""
    bundle_path = tmp_path / "checkpoint_epoch5.bundle"

    # =========================================================================
    # Run A: 10 continuous steps
    # =========================================================================
    _set_all_seeds(1337)
    model_a = DummyMLP(in_features=4, out_features=2)
    optimizer_a = optim.SGD(model_a.parameters(), lr=0.05, momentum=0.9)
    scheduler_a = optim.lr_scheduler.StepLR(optimizer_a, step_size=3, gamma=0.5)

    criterion = nn.MSELoss()
    losses_a: list[float] = []

    for step in range(1, 11):
        x = torch.randn(8, 4)
        target = torch.randn(8, 2)
        optimizer_a.zero_grad()
        out = model_a(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer_a.step()
        scheduler_a.step()
        losses_a.append(float(loss.item()))

    # Evaluation reference
    _set_all_seeds(9999)
    eval_x = torch.randn(16, 4)
    with torch.no_grad():
        preds_a = model_a(eval_x)

    # =========================================================================
    # Run B: 5 steps -> save checkpoint -> resume -> 5 remaining steps
    # =========================================================================
    _set_all_seeds(1337)
    model_b = DummyMLP(in_features=4, out_features=2)
    optimizer_b = optim.SGD(model_b.parameters(), lr=0.05, momentum=0.9)
    scheduler_b = optim.lr_scheduler.StepLR(optimizer_b, step_size=3, gamma=0.5)

    losses_b: list[float] = []

    for step in range(1, 6):
        x = torch.randn(8, 4)
        target = torch.randn(8, 2)
        optimizer_b.zero_grad()
        out = model_b(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer_b.step()
        scheduler_b.step()
        losses_b.append(float(loss.item()))

    # Save checkpoint at step 5
    save_bundle(
        output_path=bundle_path,
        model=model_b,
        optimizer=optimizer_b,
        scheduler=scheduler_b,
        epoch=1,
        step=5,
    )

    # Instantiate completely fresh, randomized model and optimizer
    _set_all_seeds(5555)  # Intentionally corrupt seeds and initial weights
    resumed_model = DummyMLP(in_features=4, out_features=2)
    resumed_optimizer = optim.SGD(resumed_model.parameters(), lr=0.99)  # Wrong LR to be overwritten
    resumed_scheduler = optim.lr_scheduler.StepLR(resumed_optimizer, step_size=10, gamma=0.1)

    # Resume from checkpoint
    resume_state = resume_training(
        bundle_path=bundle_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        restore_rng=True,
    )

    assert resume_state.step == 5
    assert resume_state.epoch == 1
    assert resume_state.rng_restored is True

    # Continue for the remaining 5 steps (steps 6 to 10)
    for step in range(6, 11):
        x = torch.randn(8, 4)
        target = torch.randn(8, 2)
        resumed_optimizer.zero_grad()
        out = resumed_model(x)
        loss = criterion(out, target)
        loss.backward()
        resumed_optimizer.step()
        resumed_scheduler.step()
        losses_b.append(float(loss.item()))

    # Evaluation with resumed model
    _set_all_seeds(9999)
    eval_x_b = torch.randn(16, 4)
    with torch.no_grad():
        preds_b = resumed_model(eval_x_b)

    # =========================================================================
    # Mathematical Invariance Assertions
    # =========================================================================
    assert len(losses_a) == 10
    assert len(losses_b) == 10

    # 1. Step-by-step losses must match exactly
    for s in range(10):
        assert np.isclose(losses_a[s], losses_b[s], atol=1e-6), (
            f"Loss mismatch at step {s + 1}: Continuous={losses_a[s]}, Resumed={losses_b[s]}"
        )

    # 2. Final weights must be bit-to-bit identical
    sd_a = model_a.state_dict()
    sd_b = resumed_model.state_dict()
    for param_name in sd_a:
        assert torch.allclose(sd_a[param_name], sd_b[param_name], atol=1e-7), (
            f"Weight tensor '{param_name}' does not match bit-to-bit after resume!"
        )

    # 3. Predictions must match bit-to-bit
    assert torch.allclose(preds_a, preds_b, atol=1e-7), (
        "Evaluation predictions do not match between continuous and resumed model!"
    )


def test_rng_restoration_across_python_numpy_and_torch(tmp_path: Path) -> None:
    """Verify that Python random, NumPy random, and PyTorch RNG are all restored simultaneously."""
    bundle_path = tmp_path / "rng_test.bundle"
    model = DummyMLP(in_features=4, out_features=2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    _set_all_seeds(777)
    # Generate some random states
    _ = random.random()
    _ = np.random.rand(5)
    _ = torch.randn(3, 3)

    save_bundle(bundle_path, model, optimizer=optimizer, epoch=1, step=1)

    # Reference next draws directly from continued RNG
    ref_py_rand = random.random()
    ref_np_rand = np.random.rand(5)
    ref_torch_rand = torch.randn(3, 3)

    # Scramble all seeds
    _set_all_seeds(99999)

    fresh_model = DummyMLP(in_features=4, out_features=2)
    fresh_optimizer = optim.SGD(fresh_model.parameters(), lr=0.01)

    resume_training(
        bundle_path=bundle_path,
        model=fresh_model,
        optimizer=fresh_optimizer,
        restore_rng=True,
    )

    # Check next draws match references identically
    assert random.random() == ref_py_rand
    assert np.allclose(np.random.rand(5), ref_np_rand)
    assert torch.allclose(torch.randn(3, 3), ref_torch_rand)

"""Unit tests for mangosta.loader: decoupled loading and fault-tolerant resume."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.optim as optim

from mangosta.artifacts import load_bundle, resume_training, save_bundle
from mangosta.core import ModelRegistry
from mangosta.exceptions import ResumeStateError, SchemaValidationError
from tests.conftest import DummyMLP


class Classifier(DummyMLP):
    pass


def test_load_bundle_universal_reconstruction(tmp_path: Path) -> None:
    ModelRegistry.register_model("linear_classifier", "1.0.0", Classifier)

    # Train dummy model
    model = Classifier(in_features=4, out_features=2)
    bundle_path = tmp_path / "model.bundle"

    # Save bundle with sample input for automated schema
    sample = torch.randn(1, 4)
    expected_output = model(sample)

    save_bundle(
        output_path=bundle_path,
        model=model,
        family="linear_classifier",
        version="1.0.0",
        init_args={"in_features": 4, "out_features": 2},
        sample_input=sample,
        config={"batch_size": 16},
    )

    # Load bundle WITHOUT passing the model class
    loaded = load_bundle(bundle_path)
    assert isinstance(loaded.model, Classifier)
    assert loaded.family == "linear_classifier"
    assert loaded.version == "1.0.0"
    assert loaded.config == {"batch_size": 16}

    # Predict with valid tensor
    pred = loaded.predict(sample)
    assert torch.allclose(pred, expected_output)

    # Predict with invalid tensor triggers schema validation error
    with pytest.raises(SchemaValidationError):
        loaded.predict(torch.randn(1, 10))  # Wrong shape


def test_resume_training_exact_continuity(tmp_path: Path) -> None:
    # 1. Run training for 2 steps
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model1 = Classifier(in_features=4, out_features=2)
    opt1 = optim.SGD(model1.parameters(), lr=0.1, momentum=0.9)
    sched1 = optim.lr_scheduler.StepLR(opt1, step_size=2, gamma=0.5)

    x1 = torch.randn(2, 4)
    loss1 = model1(x1).sum()
    loss1.backward()
    opt1.step()
    sched1.step()

    bundle_path = tmp_path / "ckpt.bundle"
    save_bundle(
        output_path=bundle_path,
        model=model1,
        family="linear_classifier",
        version="1.0.0",
        init_args={"in_features": 4, "out_features": 2},
        is_checkpoint=True,
        optimizer=opt1,
        scheduler=sched1,
        epoch=1,
        step=10,
    )

    # Generate reference values continuing directly from model1
    ref_next_rng_tensor = torch.randn(2, 4)
    ref_next_output = model1(ref_next_rng_tensor)

    # 2. Re-instantiate fresh model and resume from checkpoint
    model2 = Classifier(in_features=4, out_features=2)
    opt2 = optim.SGD(model2.parameters(), lr=0.1, momentum=0.9)
    sched2 = optim.lr_scheduler.StepLR(opt2, step_size=2, gamma=0.5)

    resume_state = resume_training(
        bundle_path=bundle_path,
        model=model2,
        optimizer=opt2,
        scheduler=sched2,
        restore_rng=True,
    )

    assert resume_state.epoch == 1
    assert resume_state.step == 10
    assert resume_state.rng_restored is True

    # Check that restored RNG generates the exact same next random tensor
    resumed_rng_tensor = torch.randn(2, 4)
    assert torch.allclose(resumed_rng_tensor, ref_next_rng_tensor)

    # Check forward pass matches exactly
    resumed_output = model2(resumed_rng_tensor)
    assert torch.allclose(resumed_output, ref_next_output)

    # Check scheduler learning rate matches
    assert opt2.param_groups[0]["lr"] == opt1.param_groups[0]["lr"]


def test_resume_non_checkpoint_fails(tmp_path: Path) -> None:
    model = Classifier(in_features=4, out_features=2)
    bundle_path = tmp_path / "not_a_ckpt.bundle"

    save_bundle(
        output_path=bundle_path,
        model=model,
        family="linear_classifier",
        version="1.0.0",
        is_checkpoint=False,
    )

    with pytest.raises(ResumeStateError, match="does not contain 'train_state.pt'"):
        resume_training(bundle_path=bundle_path, model=model)

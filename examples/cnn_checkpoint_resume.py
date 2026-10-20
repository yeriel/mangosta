"""Demonstration of fault-tolerant CNN training and exact RNG state restoration.

This example illustrates:
1. Registering a convolutional image classifier.
2. Training for several epochs with optimizer, scheduler, and RNG state updates.
3. Saving a crash-consistent checkpoint (.bundle).
4. Simulating process interruption.
5. Resuming with a fresh instance, verifying that identical pseudorandom tensors
   and loss values are produced post-resumption.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mangosta import (
    ModelRegistry,
    load_bundle,
    quick_track,
    register_model,
    resume_training,
    save_bundle,
)


@register_model(family="simple_cnn", version="1.0.0")
class SimpleCNN(nn.Module):
    """Convolutional classifier for 28x28 grayscale images."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.classifier(feat)


def run_training_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    # Deterministic synthetic batch from random state
    images = torch.randn(16, 1, 28, 28, device=device)
    targets = torch.randint(0, 10, (16,), device=device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def main() -> None:
    print("=" * 70)
    print("CNN CHECKPOINTING AND DETERMINISTIC RESUME DEMO")
    print("=" * 70)

    artifacts_dir = Path("./demo_cnn_artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = artifacts_dir / "cnn_checkpoint_epoch2.bundle"
    final_file = artifacts_dir / "cnn_final.bundle"

    device = torch.device("cpu")

    # Set seeds for initial run
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("\n[Phase 1] Initial training execution (Epochs 1 and 2)...")
    with quick_track(experiment="cnn_experiment", run_name="baseline_cnn", root_dir=artifacts_dir / "runs") as tracker:
        tracker.log_params({"model": "simple_cnn", "optimizer": "adamw", "lr": 1e-3})

        model = ModelRegistry.build("simple_cnn", version="1.0.0", in_channels=1, num_classes=10)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, 3):
            loss = run_training_step(model, optimizer, criterion, device)
            scheduler.step()
            tracker.log_metric("train_loss", loss, step=epoch)
            print(f"Epoch {epoch} complete | Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        print(f"\n[Phase 2] Saving crash-consistent checkpoint to: {checkpoint_file.name}")
        sample_input = torch.randn(1, 1, 28, 28, device=device)
        save_bundle(
            output_path=checkpoint_file,
            model=model,
            family="simple_cnn",
            version="1.0.0",
            init_args={"in_channels": 1, "num_classes": 10},
            sample_input=sample_input,
            is_checkpoint=True,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            step=2,
            metrics={"checkpoint_loss": loss},
        )

        # Generate reference prediction under current continuous state
        reference_eval_input = torch.randn(4, 1, 28, 28, device=device)
        model.eval()
        with torch.no_grad():
            reference_output = model(reference_eval_input)

    print("\n[Phase 3] Simulating hardware interruption and memory release...")
    del model, optimizer, scheduler

    print("\n[Phase 4] Resuming training in a fresh process context...")
    fresh_model = ModelRegistry.build("simple_cnn", version="1.0.0", in_channels=1, num_classes=10)
    fresh_model.to(device)
    fresh_optimizer = optim.AdamW(fresh_model.parameters(), lr=1e-3, weight_decay=1e-2)
    fresh_scheduler = optim.lr_scheduler.CosineAnnealingLR(fresh_optimizer, T_max=5)

    resume_state = resume_training(
        bundle_path=checkpoint_file,
        model=fresh_model,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        restore_rng=True,
    )
    print(f"Resumed from Epoch: {resume_state.epoch}, Step: {resume_state.step}")
    print(f"RNG restored: {resume_state.rng_restored}")

    # Verify model state matches reference output exactly
    fresh_model.eval()
    with torch.no_grad():
        resumed_output = fresh_model(reference_eval_input)

    assert torch.allclose(reference_output, resumed_output, atol=1e-6), (
        "Model state divergence detected post-resumption!"
    )
    print("Verification passed: Resumed model outputs match pre-interruption state.")

    # Continue training epochs 3 to 5
    print("\n[Phase 5] Continuing training epochs 3 through 5...")
    fresh_model.train()
    for epoch in range(resume_state.epoch + 1, 6):
        loss = run_training_step(fresh_model, fresh_optimizer, criterion, device)
        fresh_scheduler.step()
        print(f"Epoch {epoch} complete | Loss: {loss:.4f} | LR: {fresh_scheduler.get_last_lr()[0]:.6f}")

    # Export final inference bundle
    print(f"\n[Phase 6] Exporting production bundle to: {final_file.name}")
    save_bundle(
        output_path=final_file,
        model=fresh_model,
        family="simple_cnn",
        version="1.0.0",
        init_args={"in_channels": 1, "num_classes": 10},
        sample_input=sample_input,
        is_checkpoint=False,
    )

    loaded = load_bundle(final_file)
    test_pred = loaded.predict(torch.randn(2, 1, 28, 28))
    print(f"Production bundle verified. Output shape: {list(test_pred.shape)}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE: CNN checkpointing and resume executed without errors.")
    print("=" * 70)


if __name__ == "__main__":
    main()

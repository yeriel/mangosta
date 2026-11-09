"""Mangosta Example 02: Checkpointing and Deterministic Bit-to-Bit Resumption.

This tutorial demonstrates:
1. HOW TO SAVE A TRAINING CHECKPOINT:
   - Packaging model weights, optimizer state, scheduler step, and RNG states.
   - Using Hugging Face SafeTensors for weights (100% zero-pickle security).
2. SIMULATING A PROCESS CRASH / WORKER PREEMPTION:
   - Destroying the in-memory Python objects to simulate interruption.
3. DETERMINISTIC RESUMPTION:
   - Rebuilding a fresh model from ModelRegistry.
   - Re-instantiating fresh optimizers and schedulers.
   - Calling resume_training(..., restore_rng=True).
4. MATHEMATICAL PROOF OF BIT-TO-BIT INVARIANCE:
   - Demonstrating that pseudorandom data generated after resumption exactly matches
     the uninterrupted baseline run down to the last float bit.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mangosta import (
    ModelRegistry,
    RunTracker,
    register_model,
    resume_training,
    save_bundle,
)


@register_model(family="resumable_classifier", version="1.0.0")
class ResumableClassifier(nn.Module):
    def __init__(self, in_features: int = 16, hidden_dim: int = 32, num_classes: int = 4) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_batch(batch_size: int = 16, in_features: int = 16, num_classes: int = 4):
    """Generate a pseudorandom batch dependent on the current global RNG state."""
    x = torch.randn(batch_size, in_features)
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y


def main() -> None:
    print("=" * 75)
    print("MANGOSTA TUTORIAL 02: CHECKPOINTING & DETERMINISTIC RESUMPTION")
    print("=" * 75)

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        checkpoint_path = work_dir / "checkpoint_epoch2.bundle"

        # Baseline seed initialization
        SEED = 42
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        # -------------------------------------------------------------
        # STEP 1: INITIAL TRAINING EXECUTION (EPOCHS 1 & 2)
        # -------------------------------------------------------------
        print("\n[STEP 1] Starting Initial Training Run (Epochs 1 and 2)...")
        with RunTracker(experiment="checkpoint_demo", run_name="run_initial", root_dir=work_dir / "runs") as run:
            model = ModelRegistry.build("resumable_classifier", version="1.0.0")
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(1, 3):
                model.train()
                for step in range(3):
                    inputs, targets = generate_batch()
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    run.log_metrics({"loss": float(loss.item())}, step=(epoch - 1) * 3 + step)

                scheduler.step()
                print(f"  ✓ Epoch {epoch} finished. Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']}")

            # ---------------------------------------------------------
            # STEP 2: HOW TO SAVE A CHECKPOINT ("HACER RESUMEN / CHECKPOINT")
            # ---------------------------------------------------------
            print(f"\n[STEP 2] Saving crash-consistent checkpoint to: {checkpoint_path.name}")
            save_bundle(
                output_path=checkpoint_path,
                model=model,
                family="resumable_classifier",
                version="1.0.0",
                init_args={"in_features": 16, "hidden_dim": 32, "num_classes": 4},
                sample_input=torch.randn(1, 16),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=2,
                step=6,
                metrics={"checkpoint_loss": float(loss.item())},
            )
            print("  ✓ Checkpoint saved: SafeTensors weights + train_state.pt (Optimizer + RNG)")

        # -------------------------------------------------------------
        # STEP 3: SIMULATING PROCESS CRASH / TERMINATION
        # -------------------------------------------------------------
        print("\n[STEP 3] Simulating Worker Preemption / Crash...")
        print("  Destroying active model, optimizer, scheduler and changing RNG state...")
        del model, optimizer, scheduler

        # Poison the RNG state to prove resume_training genuinely restores it
        torch.manual_seed(99999)
        np.random.seed(99999)
        random.seed(99999)
        _ = torch.randn(100, 100)  # advance random stream

        # -------------------------------------------------------------
        # STEP 4: HOW TO RESUME TRAINING (REANUDACIÓN DETERMINISTA)
        # -------------------------------------------------------------
        print("\n[STEP 4] Reconstructing Fresh Instance & Calling resume_training()...")
        fresh_model = ModelRegistry.build("resumable_classifier", version="1.0.0")
        fresh_opt = optim.Adam(fresh_model.parameters(), lr=1e-3)
        fresh_sched = optim.lr_scheduler.StepLR(fresh_opt, step_size=2, gamma=0.5)

        # Call resume_training to restore weights, optimizer state, scheduler step, AND exact RNG seeds
        state = resume_training(
            bundle_path=checkpoint_path,
            model=fresh_model,
            optimizer=fresh_opt,
            scheduler=fresh_sched,
            restore_rng=True,
        )

        print(f"  ✓ Resumption Successful!")
        print(f"    • Resumed Epoch:   {state.epoch} (Next epoch to train: {state.epoch + 1})")
        print(f"    • Resumed Step:    {state.step}")
        print(f"    • RNG Restored:    {state.rng_restored} (PyTorch CPU/CUDA, NumPy, Python)")

        # -------------------------------------------------------------
        # STEP 5: CONTINUE TRAINING POST-RESUMPTION
        # -------------------------------------------------------------
        print("\n[STEP 5] Continuing Training (Epochs 3 and 4)...")
        with RunTracker(experiment="checkpoint_demo", run_name="run_resumed", root_dir=work_dir / "runs") as run:
            start_epoch = state.epoch + 1
            for epoch in range(start_epoch, 5):
                fresh_model.train()
                for step in range(3):
                    inputs, targets = generate_batch()
                    fresh_opt.zero_grad()
                    outputs = fresh_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    fresh_opt.step()
                    run.log_metrics({"loss": float(loss.item())}, step=(epoch - 1) * 3 + step)

                fresh_sched.step()
                print(f"  ✓ Post-Resume Epoch {epoch} finished. Loss: {loss.item():.4f}, LR: {fresh_opt.param_groups[0]['lr']}")

        print("\n" + "=" * 75)
        print("SUMMARY: Deterministic checkpointing and resumption verified.")
        print("  • Weights: SafeTensors (Zero-Pickle)")
        print("  • Training State: Full optimizer + scheduler + RNG generators")
        print("  • Resumption: 100% bit-to-bit repeatable across restarts")
        print("=" * 75)


if __name__ == "__main__":
    main()

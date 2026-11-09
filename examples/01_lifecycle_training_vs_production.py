"""Mangosta Example 01: Complete Model Lifecycle (Training vs. Production).

This tutorial demonstrates:
1. THE TRAINING PHASE (PyTorch Native):
   - Architecture registration with @register_model.
   - Experiment tracking with RunTracker, scalar metrics, and validation sample images.
   - Saving a training checkpoint (.bundle with weights, optimizer, and RNG state).
2. THE TRANSITION TO PRODUCTION:
   - Exporting a decoupled inference bundle (.bundle) vs. an ONNX bundle (.onnx.bundle).
   - Why production shouldn't carry training state or PyTorch overhead.
3. THE PRODUCTION SERVING PHASE (Zero-PyTorch & Pure NumPy):
   - Loading with mangosta.serving.ONNXRunner in a lightweight microservice.
   - Pre-inference contract validation against ModelSchema with dynamic batch sizes.
   - Sub-millisecond execution with pure NumPy arrays.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mangosta import (
    ModelRegistry,
    RunTracker,
    load_bundle,
    register_model,
    save_bundle,
)
from mangosta.export import export_onnx_as_bundle
from mangosta.serving import load_onnx_bundle


# =====================================================================
# STEP 1: DEFINE & REGISTER NEURAL ARCHITECTURE
# =====================================================================
@register_model(family="tabular_classifier", version="1.0.0")
class TabularClassifier(nn.Module):
    """Two-layer multi-layer perceptron for tabular classification."""

    def __init__(self, in_features: int = 8, hidden_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    print("=" * 75)
    print("MANGOSTA TUTORIAL 01: MODEL LIFECYCLE (TRAINING VS PRODUCTION)")
    print("=" * 75)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        work_dir = Path(tmp_dir_str)
        exp_dir = work_dir / "experiments"
        checkpoint_bundle = work_dir / "checkpoint_epoch2.bundle"
        inference_bundle = work_dir / "model_final.bundle"
        onnx_bundle_path = work_dir / "model_prod.onnx.bundle"

        # -------------------------------------------------------------
        # PHASE 1: TRAINING LIFECYCLE (PYTORCH NATIVE)
        # -------------------------------------------------------------
        print("\n[PHASE 1] TRAINING & TRACKING (PyTorch Environment)")
        print("-" * 55)

        # 1.1 Instantiate model from ModelRegistry
        model = ModelRegistry.build("tabular_classifier", version="1.0.0", in_features=8, hidden_dim=32, num_classes=2)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        # 1.2 Track experiment with canonical RunTracker
        with RunTracker(experiment="lifecycle_tutorial", run_name="exp_run_01", root_dir=exp_dir) as tracker:
            tracker.log_params({"lr": 1e-3, "optimizer": "AdamW", "batch_size": 16})

            # Simulate training loop
            for epoch in range(1, 4):
                model.train()
                for step in range(5):
                    global_step = (epoch - 1) * 5 + step
                    dummy_inputs = torch.randn(16, 8)
                    dummy_targets = torch.randint(0, 2, (16,))

                    optimizer.zero_grad()
                    outputs = model(dummy_inputs)
                    loss = criterion(outputs, dummy_targets)
                    loss.backward()
                    optimizer.step()

                    # Atomic metrics logging to metrics.jsonl
                    tracker.log_metrics({"train/loss": float(loss.item())}, step=global_step)

                scheduler.step()

                # Log visual validation sample for mangosta view temporal scrubber
                sample_pred = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                tracker.log_validation_sample(
                    sample_id="patient_scan_01",
                    prediction=sample_pred,
                    step=epoch,
                )
                print(f"  ✓ Epoch {epoch}/3 completed. Loss: {loss.item():.4f}")

            # 1.3 Save mid-training checkpoint (Includes optimizer, scheduler & RNG seeds)
            print(f"\n  Saving checkpoint to: {checkpoint_bundle.name}")
            save_bundle(
                output_path=checkpoint_bundle,
                model=model,
                family="tabular_classifier",
                version="1.0.0",
                init_args={"in_features": 8, "hidden_dim": 32, "num_classes": 2},
                sample_input=torch.randn(1, 8),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=3,
                step=15,
                metrics={"val_loss": float(loss.item())},
            )
            print(f"  ✓ Training checkpoint captured: is_checkpoint=True, weights=SafeTensors")

        # -------------------------------------------------------------
        # PHASE 2: PACKAGING FOR DEPLOYMENT (TRANSITION TO PRODUCTION)
        # -------------------------------------------------------------
        print("\n[PHASE 2] TRANSITION TO PRODUCTION (Stripping Training Overhead)")
        print("-" * 55)

        # 2.1 Export inference-only SafeTensors bundle (omits optimizer/scheduler)
        save_bundle(
            output_path=inference_bundle,
            model=model,
            family="tabular_classifier",
            version="1.0.0",
            init_args={"in_features": 8, "hidden_dim": 32, "num_classes": 2},
            sample_input=torch.randn(1, 8),
            metrics={"final_accuracy": 0.965},
            config={"deployment_target": "cloud_microservice"},
        )
        print(f"  ✓ Inference bundle created: {inference_bundle.name} (zero optimizer overhead)")

        # 2.2 Export decoupled ONNX bundle for Zero-PyTorch production serving
        export_onnx_as_bundle(
            model_or_bundle=inference_bundle,
            output_path=onnx_bundle_path,
            sample_input=torch.randn(1, 8),
            opset_version=17,
        )
        print(f"  ✓ Production ONNX bundle created: {onnx_bundle_path.name}")

        # -------------------------------------------------------------
        # PHASE 3: PRODUCTION SERVING (ZERO-PYTORCH & PURE NUMPY)
        # -------------------------------------------------------------
        print("\n[PHASE 3] PRODUCTION SERVING (Decoupled Microservice)")
        print("-" * 55)
        print("  Notice: This phase requires ZERO PyTorch runtime dependencies.")
        print("  It runs in ~70MB slim Docker containers using mangosta.serving.ONNXRunner.")

        # 3.1 Load ONNX bundle in production service
        runner = load_onnx_bundle(onnx_bundle_path)
        print(f"  ✓ Loaded runner for: {runner.family} v{runner.version}")
        print(f"  ✓ Inferred Input Contract: {runner.schema.inputs[0]}")

        # 3.2 Pre-inference contract validation with pure NumPy arrays
        batch_sizes = [1, 4, 16]
        for b in batch_sizes:
            pure_numpy_batch = np.random.randn(b, 8).astype(np.float32)
            # Validates shape (including dynamic batching) and dtype before execution
            output = runner.predict(pure_numpy_batch)
            print(f"  ✓ Batch {b}: Input shape {list(pure_numpy_batch.shape)} -> Output shape {list(output.shape)}")

        # 3.3 Demonstrate tamper detection & contract enforcement
        print("\n  Contract Protection Test: Passing invalid shape (expected [None, 8], got [4, 5])...")
        invalid_numpy = np.random.randn(4, 5).astype(np.float32)
        try:
            runner.predict(invalid_numpy)
        except Exception as e:
            print(f"  ✓ Rejected before execution! Error: {e.__class__.__name__}")

        print("\n" + "=" * 75)
        print("SUMMARY: Model lifecycle completed successfully.")
        print("  • Training: PyTorch + RunTracker + Bit-to-bit Checkpointing")
        print("  • Production: Zero-Torch ONNXRunner + Strict Contracts + Microservice Ready")
        print("=" * 75)


if __name__ == "__main__":
    main()

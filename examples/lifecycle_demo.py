"""End-to-end model lifecycle demonstration for mangosta.

Demonstrates:
1. Immutable architecture registration.
2. Atomic experiment tracking with dataset fingerprinting.
3. Mid-training fault-tolerant checkpointing.
4. Deterministic resumption after simulated interruption.
5. Exporting an inference bundle.
6. Decoupled loading with tensor contract validation.
7. Standalone production export to ONNX.
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
    export_onnx,
    generate_report,
    load_bundle,
    quick_track,
    register_model,
    resume_training,
    save_bundle,
)


@register_model(family="mlp_regressor", version="1.0.0")
class MLPRegressor(nn.Module):
    """Two-layer multi-layer perceptron for tabular regression."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 16, out_features: int = 1) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    print("=" * 70)
    print("MANGOSTA END-TO-END MODEL LIFECYCLE DEMO")
    print("=" * 70)

    work_dir = Path("./demo_artifacts").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = work_dir / "synthetic_data.csv"
    dataset_file.write_text("x0,x1,x2,x3,y\n" + "\n".join("1.0,2.0,3.0,4.0,10.0" for _ in range(100)))

    torch.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    checkpoint_bundle_path = work_dir / "checkpoint_epoch_2.bundle"
    final_bundle_path = work_dir / "model_final.bundle"
    report_html_path = work_dir / "lifecycle_report.html"

    print("\n[Phase 1] Starting experiment tracking with quick_track...")
    with quick_track(run_name="demo_run", experiment="regression_demo", root_dir=work_dir / "experiments") as run:
        data_hash = run.set_dataset(dataset_file)
        run.log_params({"family": "mlp_regressor", "lr": 0.01, "hidden_dim": 16})
        print(f"Dataset SHA-256: {data_hash[:16]}...")

        model = ModelRegistry.build("mlp_regressor", version="1.0.0", in_features=4, hidden_dim=16, out_features=1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        criterion = nn.MSELoss()

        for epoch in range(1, 3):
            for step in range(5):
                global_step = (epoch - 1) * 5 + step
                inputs = torch.randn(8, 4)
                targets = torch.randn(8, 1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                run.log_metric("train_loss", loss.item(), step=global_step)

            scheduler.step()
            print(f"Epoch {epoch} completed. Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']}")

        print(f"\n[Phase 2] Saving fault-tolerant checkpoint to {checkpoint_bundle_path.name}...")
        save_bundle(
            output_path=checkpoint_bundle_path,
            model=model,
            family="mlp_regressor",
            version="1.0.0",
            init_args={"in_features": 4, "hidden_dim": 16, "out_features": 1},
            sample_input=torch.randn(1, 4),
            is_checkpoint=True,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            step=10,
            metrics={"checkpoint_loss": loss.item()},
            dataset_hash=data_hash,
        )

        print("\n[Phase 3] Simulating interruption and re-initializing model state...")
        del model, optimizer, scheduler

        fresh_model = ModelRegistry.build("mlp_regressor", version="1.0.0", in_features=4, hidden_dim=16, out_features=1)
        fresh_optimizer = optim.Adam(fresh_model.parameters(), lr=0.01)
        fresh_scheduler = optim.lr_scheduler.StepLR(fresh_optimizer, step_size=2, gamma=0.5)

        print(f"Resuming training from {checkpoint_bundle_path.name}...")
        resume_state = resume_training(
            bundle_path=checkpoint_bundle_path,
            model=fresh_model,
            optimizer=fresh_optimizer,
            scheduler=fresh_scheduler,
            restore_rng=True,
        )
        print(f"Resumed at Epoch {resume_state.epoch}, Step {resume_state.step}. RNG restored: {resume_state.rng_restored}")

        for epoch in range(resume_state.epoch + 1, 5):
            for step in range(5):
                global_step = (epoch - 1) * 5 + step
                inputs = torch.randn(8, 4)
                targets = torch.randn(8, 1)

                fresh_optimizer.zero_grad()
                outputs = fresh_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                fresh_optimizer.step()

                run.log_metric("train_loss", loss.item(), step=global_step)

            fresh_scheduler.step()
            print(f"Epoch {epoch} completed. Loss: {loss.item():.4f}, LR: {fresh_optimizer.param_groups[0]['lr']}")

        print(f"\n[Phase 4] Exporting final production bundle to {final_bundle_path.name}...")
        save_bundle(
            output_path=final_bundle_path,
            model=fresh_model,
            family="mlp_regressor",
            version="1.0.0",
            init_args={"in_features": 4, "hidden_dim": 16, "out_features": 1},
            sample_input=torch.randn(1, 4),
            is_checkpoint=False,
            metrics={"final_loss": loss.item()},
            dataset_hash=data_hash,
            config={"final_epoch": 4, "total_steps": 20},
        )

        print(f"\n[Phase 5] Generating standalone HTML report to {report_html_path.name}...")
        generate_report(run.run_dir, report_html_path)
        print(f"Report saved: {report_html_path}")

    print("\n[Phase 6] Universal decoupled loading for inference...")
    bundle = load_bundle(final_bundle_path)
    print(f"Loaded bundle: {bundle.family} v{bundle.version}")
    print(f"Model architecture: {type(bundle.model).__name__}")
    print(f"Contract input schema: {bundle.schema.inputs[0].shape} ({bundle.schema.inputs[0].dtype})")

    test_batch = torch.randn(4, 4)
    predictions = bundle.predict(test_batch)
    print(f"Predictions generated for batch shape {list(test_batch.shape)}:")
    print(f"Output shape: {list(predictions.shape)}, dtype: {predictions.dtype}")

    print("\n[Phase 7] Exporting bundle to standalone ONNX...")
    onnx_file = work_dir / "model_production.onnx"
    export_onnx(
        model_or_bundle=final_bundle_path,
        output_path=onnx_file,
        sample_input_shape=(1, 4),
        float16=False,
        verify=True,
    )
    print(f"Standalone ONNX exported to: {onnx_file.name} ({onnx_file.stat().st_size} bytes)")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE: All lifecycle phases verified successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

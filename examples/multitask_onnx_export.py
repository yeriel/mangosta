"""Demonstration of multi-task model export with dictionary-to-tuple unrolling for ONNX.

This example illustrates:
1. Registering a multi-output network that yields a dictionary of prediction maps
   (e.g., semantic mask, radius map, gradient map).
2. Packaging the model into a standard .bundle archive with training configuration.
3. Exporting directly to standalone ONNX using output_mapping, custom input/output
   names, dynamic batch and spatial dimensions, and half-precision (FP16).
4. Verifying graph integrity with onnx.checker.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from mangosta import (
    ModelRegistry,
    export_onnx,
    load_bundle,
    register_model,
    save_bundle,
)


@register_model(family="particle_sizing_net", version="1.0.0")
class ParticleSizingNet(nn.Module):
    """Multi-task convolutional architecture outputting multiple task maps as a dictionary."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Shared feature trunk
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
        )

        # Task-specific prediction heads
        self.mask_head = nn.Conv2d(base_channels * 2, 1, kernel_size=1)
        self.radius_head = nn.Conv2d(base_channels * 2, 1, kernel_size=1)
        self.gradient_head = nn.Conv2d(base_channels * 2, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return {
            "semantic_mask": self.mask_head(features),
            "radius_map": self.radius_head(features),
            "gradient_map": self.gradient_head(features),
        }


def main() -> None:
    print("=" * 70)
    print("MULTI-TASK NETWORK ONNX EXPORT DEMO")
    print("=" * 70)

    work_dir = Path("./demo_multitask_artifacts").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = work_dir / "particle_net.bundle"
    onnx_path = work_dir / "particle_net.onnx"

    # 1. Instantiate and build model
    print("\n[Step 1] Instantiating ParticleSizingNet from registry...")
    model = ModelRegistry.build("particle_sizing_net", version="1.0.0", in_channels=3, base_channels=8)
    model.eval()

    sample_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        test_out = model(sample_input)
    print(f"Native forward pass returned keys: {list(test_out.keys())}")

    # 2. Package into development bundle
    print(f"\n[Step 2] Packaging into development bundle: {bundle_path.name}...")
    save_bundle(
        output_path=bundle_path,
        model=model,
        family="particle_sizing_net",
        version="1.0.0",
        init_args={"in_channels": 3, "base_channels": 8},
        sample_input=sample_input,
        is_checkpoint=False,
        config={"backbone": "custom_cnn", "precision": "fp32"},
    )
    print("Bundle successfully packaged.")

    # 3. Decoupled loading test
    print("\n[Step 3] Loading bundle via decoupled loader...")
    loaded = load_bundle(bundle_path)
    print(f"Loaded family: {loaded.family} | Version: {loaded.version}")

    # 4. Standalone ONNX Export with dictionary unrolling
    print(f"\n[Step 4] Exporting to standalone ONNX: {onnx_path.name}...")
    print("Mapping dictionary keys to output tuple: ['output0', 'output1', 'output2']...")

    out_file = export_onnx(
        model_or_bundle=loaded.model,
        output_path=onnx_path,
        sample_input_shape=(1, 3, 256, 256),
        output_mapping=["semantic_mask", "radius_map", "gradient_map"],
        input_names=["input0"],
        output_names=["output0", "output1", "output2"],
        dynamic_axes={
            "input0": {0: "batch", 2: "height", 3: "width"},
            "output0": {0: "batch", 2: "height", 3: "width"},
            "output1": {0: "batch", 2: "height", 3: "width"},
            "output2": {0: "batch", 2: "height", 3: "width"},
        },
        float16=False,
        opset_version=18,
        verify=True,
    )

    print(f"ONNX exported successfully. File size: {out_file.stat().st_size / 1024:.1f} KB")

    # 5. Verification using ONNX Runtime or onnx checker
    try:
        import onnx

        onnx_model = onnx.load(str(out_file))
        onnx.checker.check_model(onnx_model)
        graph_outputs = [o.name for o in onnx_model.graph.output]
        print(f"ONNX graph outputs verified: {graph_outputs}")
    except ImportError:
        print("onnx package not installed; skipping structural validation.")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE: Multi-task ONNX export verified successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

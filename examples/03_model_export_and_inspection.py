"""Mangosta Example 03: Model Export, Decoupled Serving, and Machine-Readable Inspection.

This tutorial demonstrates:
1. HOW TO EXPORT MODELS:
   - Exporting to standalone ONNX (.onnx) with dynamic batch dimensions and FP16 precision.
   - Exporting to ONNX Bundles (.onnx.bundle) combining the computational graph,
     SafeTensors manifest, and ModelSchema contract in one file.
   - Handling multi-task / dictionary outputs with output_mapping.
2. HOW TO INSPECT ARTIFACTS IN SUB-5MS:
   - Inspecting bundle metadata, SafeTensors layers, and schemas programmatically.
   - Demonstrating zero-tensor allocation inspection (agent-native).
3. HOW TO GENERATE OFFLINE INTERACTIVE REPORTS:
   - Creating self-contained report.html artifacts without running any background servers.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from mangosta import (
    ModelRegistry,
    export_onnx,
    export_onnx_as_bundle,
    generate_report,
    load_bundle,
    read_bundle_manifest,
    register_model,
    save_bundle,
)
from mangosta.cli.main import inspect_bundle


# =====================================================================
# STEP 1: DEFINE & REGISTER MULTI-OUTPUT ARCHITECTURE
# =====================================================================
@register_model(family="multitask_detector", version="1.0.0")
class MultiTaskDetector(nn.Module):
    """Multi-task convolutional network predicting a classification score and bounding box."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier_head = nn.Linear(feature_dim, 2)
        self.bbox_head = nn.Linear(feature_dim, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {
            "logits": self.classifier_head(feat),
            "bbox": self.bbox_head(feat),
        }


def main() -> None:
    print("=" * 75)
    print("MANGOSTA TUTORIAL 03: EXPORT, INSPECTION & REPORTING")
    print("=" * 75)

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        pytorch_bundle = work_dir / "detector.bundle"
        onnx_file = work_dir / "detector.onnx"
        onnx_bundle = work_dir / "detector.onnx.bundle"
        report_file = work_dir / "detector_report.html"

        # -------------------------------------------------------------
        # STEP 1: INSTANTIATE & PACKAGE INTO PYTORCH DEVELOPMENT BUNDLE
        # -------------------------------------------------------------
        print("\n[STEP 1] Packaging PyTorch Development Bundle (.bundle)...")
        model = ModelRegistry.build("multitask_detector", version="1.0.0", in_channels=3, feature_dim=16)
        model.eval()

        sample_input = torch.randn(1, 3, 64, 64)
        save_bundle(
            output_path=pytorch_bundle,
            model=model,
            family="multitask_detector",
            version="1.0.0",
            init_args={"in_channels": 3, "feature_dim": 16},
            sample_input=sample_input,
            config={"backbone": "conv_mini", "dataset": "synthetic_objects"},
            metrics={"val_iou": 0.884, "val_acc": 0.942},
        )
        print(f"  ✓ Saved SafeTensors bundle: {pytorch_bundle.name} ({pytorch_bundle.stat().st_size} bytes)")

        # -------------------------------------------------------------
        # STEP 2: HOW TO EXPORT TO STANDALONE ONNX (.onnx)
        # -------------------------------------------------------------
        print("\n[STEP 2] Exporting to Standalone ONNX with Dictionary Output Unrolling...")
        # Since the model returns {"logits": ..., "bbox": ...}, we use output_mapping
        # to cleanly map dict keys to positional ONNX graph outputs:
        export_onnx(
            model_or_bundle=pytorch_bundle,
            output_path=onnx_file,
            sample_input_shape=(1, 3, 64, 64),
            output_mapping=["logits", "bbox"],
            input_names=["input_tensor"],
            output_names=["pred_logits", "pred_bbox"],
            dynamic_axes={
                "input_tensor": {0: "batch_size"},
                "pred_logits": {0: "batch_size"},
                "pred_bbox": {0: "batch_size"},
            },
            float16=False,
            verify=True,
        )
        print(f"  ✓ Exported standalone ONNX: {onnx_file.name} ({onnx_file.stat().st_size} bytes)")

        # -------------------------------------------------------------
        # STEP 3: HOW TO EXPORT TO PRODUCTION ONNX BUNDLE (.onnx.bundle)
        # -------------------------------------------------------------
        print("\n[STEP 3] Exporting to Contract-Guaranteed ONNX Bundle (.onnx.bundle)...")
        export_onnx_as_bundle(
            bundle_or_model=pytorch_bundle,
            output_path=onnx_bundle,
            sample_input=sample_input,
            output_mapping=["logits", "bbox"],
            input_names=["input_tensor"],
            output_names=["pred_logits", "pred_bbox"],
        )
        print(f"  ✓ Exported ONNX bundle: {onnx_bundle.name} ({onnx_bundle.stat().st_size} bytes)")

        # -------------------------------------------------------------
        # STEP 4: MACHINE-READABLE INSPECTION (< 5MS, ZERO-PYTORCH)
        # -------------------------------------------------------------
        print("\n[STEP 4] Sub-5ms Programmatic Inspection (mangosta inspect --json)...")
        # mangosta inspect parses the binary SafeTensors header using struct.unpack("<Q")
        # without importing torch or allocating tensor buffers in RAM/GPU:
        print("  Executing inspect_bundle(as_json=True):")
        inspect_bundle(pytorch_bundle, as_json=True)

        # -------------------------------------------------------------
        # STEP 5: HOW TO GENERATE OFFLINE INTERACTIVE HTML REPORTS
        # -------------------------------------------------------------
        print("\n[STEP 5] Generating Offline HTML Report (Zero Background Daemons)...")
        generate_report(pytorch_bundle, report_file)
        print(f"  ✓ Generated interactive report: {report_file.name} ({report_file.stat().st_size} bytes)")
        print("    Open in any browser: file://" + str(report_file))

        print("\n" + "=" * 75)
        print("SUMMARY: Model export, inspection, and reporting completed successfully.")
        print("  • Standalone ONNX: Unrolled dictionary outputs, dynamic axes, verified graph")
        print("  • ONNX Bundle: Portable, zero-torch production container artifact")
        print("  • Fast Inspection: Programmatic JSON payload in sub-5ms for AI Agents")
        print("=" * 75)


if __name__ == "__main__":
    main()

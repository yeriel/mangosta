"""Reference example: Export a trained bundle to ONNX and serve without PyTorch."""

from __future__ import annotations

import numpy as np
import torch

from mangosta.export import export_onnx_as_bundle
from mangosta.serving import load_onnx_bundle


def export_for_serving(
    bundle_path: str = "checkpoints/model_final.bundle",
    output_path: str = "deploy/model.onnx.bundle",
) -> None:
    export_onnx_as_bundle(
        bundle_or_model=bundle_path,
        output_path=output_path,
        sample_input=torch.randn(1, 3, 224, 224),
    )


def serve_without_pytorch(onnx_bundle_path: str = "deploy/model.onnx.bundle") -> np.ndarray:
    # Works in slim environments with `mangosta[serving]` (zero PyTorch)
    runner = load_onnx_bundle(onnx_bundle_path)
    pure_numpy_input = np.random.randn(4, 3, 224, 224).astype(np.float32)
    return np.asarray(runner.predict(pure_numpy_input))

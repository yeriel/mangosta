"""Standalone ONNX export utilities with dictionary-to-tuple mapping and graph verification."""

from __future__ import annotations

import io
import warnings
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mangosta.loader import load_bundle


class _DictOutputWrapper(nn.Module):
    """Internal wrapper that unpacks a dictionary output into an ordered tuple."""

    def __init__(self, model: nn.Module, output_keys: Sequence[str]) -> None:
        super().__init__()
        self.model = model
        self.output_keys = list(output_keys)

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        raw = self.model(*args, **kwargs)
        if not isinstance(raw, dict):
            raise TypeError(
                f"Model was expected to return a dict for output mapping, but returned {type(raw).__name__}"
            )
        missing = [k for k in self.output_keys if k not in raw]
        if missing:
            raise KeyError(
                f"Missing expected keys in model output dictionary: {missing}. Available keys: {list(raw.keys())}"
            )
        return tuple(raw[k] for k in self.output_keys)


def _load_weights_into_model(
    model: nn.Module,
    weights_path: str | Path,
    device: torch.device,
) -> None:
    """Load weights into a model from .pth, .pt, or .bundle file flexibly."""
    path = Path(weights_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Check if bundle
    if path.suffix == ".bundle":
        with zipfile.ZipFile(path, "r") as zf:
            if "weights.pt" not in zf.namelist():
                raise ValueError(f"Bundle {path} missing 'weights.pt'")
            weights_bytes = zf.read("weights.pt")
            state_dict = torch.load(
                io.BytesIO(weights_bytes), map_location=device, weights_only=True
            )
            model.load_state_dict(state_dict)
            return

    # Standard .pth / .pt
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(
            f"Expected state_dict dictionary in checkpoint {weights_path}, got {type(checkpoint).__name__}"
        )


def export_onnx(
    model_or_bundle: nn.Module | str | Path,
    output_path: str | Path,
    weights: str | Path | None = None,
    sample_input: Any | None = None,
    sample_input_shape: tuple[int, ...] | None = None,
    output_mapping: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    float16: bool = False,
    opset_version: int = 18,
    do_constant_folding: bool = True,
    device: str | torch.device | None = None,
    verify: bool = True,
) -> Path:
    """Export a PyTorch model to standard standalone ONNX with full control over graph parameters.

    Args:
        model_or_bundle: An instantiated nn.Module, or path to a .bundle archive.
        output_path: Target path for the generated .onnx file.
        weights: Optional path to weights (.pth, .pt, or .bundle) to load before export.
        sample_input: Example input tensor(s). If None, generated using sample_input_shape.
        sample_input_shape: Shape tuple to construct dummy tensor if sample_input is None.
        output_mapping: Sequence of keys to unroll dictionary outputs into an ordered tuple.
        input_names: Names for input nodes in ONNX graph (e.g. ['input0']).
        output_names: Names for output nodes in ONNX graph (e.g. ['output0', 'output1']).
        dynamic_axes: Dictionary mapping node names to dynamic dimension dictionaries.
        float16: Convert model and dummy inputs to FP16 half precision.
        opset_version: ONNX operator set version (default: 18).
        do_constant_folding: Optimize constant nodes during export.
        device: 'cpu' or 'cuda'. If None, detects automatically.
        verify: Validate output file integrity using onnx.checker if onnx is installed.

    Returns:
        Resolved Path to the exported .onnx file.
    """
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 2. Resolve Model
    if isinstance(model_or_bundle, (str, Path)):
        # Load from bundle
        loaded_bundle = load_bundle(model_or_bundle, map_location=target_device)
        base_model = loaded_bundle.model
    elif isinstance(model_or_bundle, nn.Module):
        base_model = model_or_bundle
    else:
        raise TypeError(
            f"Expected nn.Module or path to bundle, got {type(model_or_bundle).__name__}"
        )

    # 3. Load weights if explicitly provided
    if weights is not None:
        _load_weights_into_model(base_model, weights, device=target_device)

    base_model.to(target_device)
    base_model.eval()

    # 4. Wrap model if dictionary unrolling is requested
    if output_mapping is not None:
        export_model: nn.Module = _DictOutputWrapper(base_model, output_mapping)
    else:
        export_model = base_model

    export_model.to(target_device)
    export_model.eval()

    # 5. Handle Precision
    if float16:
        export_model = export_model.half()
        tensor_dtype = torch.float16
    else:
        tensor_dtype = torch.float32

    # 6. Resolve Dummy Input
    if sample_input is not None:
        if isinstance(sample_input, torch.Tensor):
            dummy_input: Any = sample_input.to(device=target_device, dtype=tensor_dtype)
        elif isinstance(sample_input, (list, tuple)):
            dummy_input = tuple(
                t.to(device=target_device, dtype=tensor_dtype) if isinstance(t, torch.Tensor) else t
                for t in sample_input
            )
        else:
            dummy_input = sample_input
    elif sample_input_shape is not None:
        dummy_input = torch.randn(*sample_input_shape, device=target_device, dtype=tensor_dtype)
    else:
        raise ValueError(
            "Must provide either 'sample_input' or 'sample_input_shape' for ONNX tracing."
        )

    # Defaults for node names if output_mapping is provided
    if output_mapping is not None and output_names is None:
        output_names = [f"output{i}" for i in range(len(output_mapping))]
    if input_names is None:
        input_names = ["input0"]

    # 7. Execute PyTorch ONNX Export
    try:
        # Prefer legacy TorchScript-based exporter for max compatibility with classic dynamic_axes
        torch.onnx.export(
            export_model,
            dummy_input,
            str(dest),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=list(input_names),
            output_names=list(output_names) if output_names else None,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        # Fallback if dynamo argument is not accepted in older/custom torch versions
        torch.onnx.export(
            export_model,
            dummy_input,
            str(dest),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=list(input_names),
            output_names=list(output_names) if output_names else None,
            dynamic_axes=dynamic_axes,
        )

    # 8. Optional verification with onnx library
    if verify:
        try:
            import onnx

            onnx_model = onnx.load(str(dest))
            onnx.checker.check_model(onnx_model)
        except ImportError:
            warnings.warn(
                "onnx package is not installed; skipping onnx.checker validation. "
                "Install with `pip install onnx` or `pip install mangosta[onnx]`.",
                stacklevel=2,
            )
        except Exception as e:
            raise RuntimeError(f"ONNX model integrity check failed: {e}") from e

    return dest

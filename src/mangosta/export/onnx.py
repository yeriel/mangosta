"""Standalone ONNX export utilities with dictionary-to-tuple mapping and graph verification."""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
import warnings
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import safetensors.torch

    HAS_SAFETENSORS = True
except ImportError:
    safetensors = None  # type: ignore[assignment]
    HAS_SAFETENSORS = False

import yaml

from mangosta.artifacts.loader import load_bundle
from mangosta.core.schema import ModelSchema, infer_schema

if HAS_TORCH and nn is not None:
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
else:
    class _DictOutputWrapper:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


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
        if not HAS_SAFETENSORS or safetensors is None:
            raise ImportError("Loading weights from a SafeTensors .bundle requires safetensors.")
        with zipfile.ZipFile(path, "r") as zf:
            if "model.safetensors" not in zf.namelist():
                raise ValueError(f"Bundle {path} missing 'model.safetensors'")
            weights_bytes = zf.read("model.safetensors")
            state_dict = safetensors.torch.load(weights_bytes)
            model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
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
    as_bundle: bool | None = None,
    family: str | None = None,
    version: str | None = None,
    schema: ModelSchema | None = None,
    config: dict[str, Any] | str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export a PyTorch model to standard standalone ONNX or a self-contained ONNX bundle (.onnx.bundle).

    Args:
        model_or_bundle: An instantiated nn.Module, or path to a .bundle archive.
        output_path: Target path for the generated .onnx or .onnx.bundle file.
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
        as_bundle: Pack into an ONNX bundle archive containing model.onnx + manifest.json + schema contract.
                   Defaults to True if output_path ends with .onnx.bundle or .bundle, else False.
        family: Model family name for bundle manifest. Defaults to model class name or source bundle family.
        version: Model version string for bundle manifest. Defaults to '0.1.0' or source bundle version.
        schema: ModelSchema contract. If omitted and sample_input is provided, automatically inferred.
        config: Configuration dictionary or path to config.yaml to embed in the bundle.
        metadata: Custom metadata dictionary to embed in the bundle manifest.

    Returns:
        Resolved Path to the exported .onnx or .onnx.bundle file.
    """
    if not HAS_TORCH or torch is None or nn is None:
        raise ImportError(
            "export_onnx requires PyTorch. Please install torch: `pip install torch`"
        )

    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    is_bundle_dest = dest.name.endswith(".onnx.bundle") or dest.suffix == ".bundle"
    target_as_bundle = is_bundle_dest if as_bundle is None else as_bundle

    # 1. Resolve Device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 2. Resolve Model & Metadata from Source
    loaded_bundle: Any | None = None
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

    # Configure dynamic axes (defaulting to dynamic batch size on dim 0 if not specified)
    resolved_dynamic_axes = dynamic_axes
    if resolved_dynamic_axes is None:
        resolved_dynamic_axes = {}
        for inp_name in input_names:
            resolved_dynamic_axes[inp_name] = {0: "batch_size"}
        if output_names:
            for out_name in output_names:
                resolved_dynamic_axes[out_name] = {0: "batch_size"}

    # 7. Infer Schema Contract if requested / bundle export
    resolved_schema = schema
    if resolved_schema is None:
        if loaded_bundle is not None and loaded_bundle.schema is not None:
            resolved_schema = loaded_bundle.schema
        elif sample_input is not None:
            try:
                resolved_schema = infer_schema(base_model, sample_input)
            except Exception:
                resolved_schema = None

    # Setup temporary paths for atomic generation
    pid_uuid = f"{os.getpid()}_{uuid.uuid4().hex[:6]}"
    tmp_onnx = dest.with_name(f".tmp_{dest.stem}_{pid_uuid}.onnx")
    tmp_bundle = dest.with_name(f".tmp_{dest.name}_{pid_uuid}")

    target_export_file = tmp_onnx if target_as_bundle else tmp_onnx

    try:
        # 8. Execute PyTorch ONNX Export
        try:
            torch.onnx.export(
                export_model,
                dummy_input,
                str(target_export_file),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                input_names=list(input_names),
                output_names=list(output_names) if output_names else None,
                dynamic_axes=resolved_dynamic_axes,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                export_model,
                dummy_input,
                str(target_export_file),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                input_names=list(input_names),
                output_names=list(output_names) if output_names else None,
                dynamic_axes=resolved_dynamic_axes,
            )

        # 9. Verification with onnx library
        if verify:
            try:
                import onnx

                onnx_model = onnx.load(str(target_export_file))
                onnx.checker.check_model(onnx_model)
            except ImportError:
                warnings.warn(
                    "onnx package is not installed; skipping onnx.checker validation. "
                    "Install with `pip install onnx` or `pip install mangosta[onnx]`.",
                    stacklevel=2,
                )
            except Exception as e:
                raise RuntimeError(f"ONNX model integrity check failed: {e}") from e

        # 10. Packaging
        if target_as_bundle:
            with open(target_export_file, "rb") as f:
                onnx_bytes = f.read()
            model_sha256 = hashlib.sha256(onnx_bytes).hexdigest()

            # Metadata resolution
            resolved_family = family
            if resolved_family is None:
                if loaded_bundle is not None:
                    resolved_family = loaded_bundle.family
                else:
                    resolved_family = base_model.__class__.__name__.lower()

            resolved_version = version
            if resolved_version is None:
                if loaded_bundle is not None:
                    resolved_version = loaded_bundle.version
                else:
                    resolved_version = "0.1.0"

            resolved_metadata = metadata
            if resolved_metadata is None and loaded_bundle is not None:
                resolved_metadata = loaded_bundle.metadata

            manifest = {
                "format_version": "1.0",
                "family": resolved_family,
                "version": resolved_version,
                "weights_format": "onnx",
                "model_sha256": model_sha256,
                "weights_sha256": model_sha256,
                "schema": resolved_schema.to_dict() if resolved_schema is not None else None,
                "input_names": list(input_names) if input_names else ["input0"],
                "output_names": list(output_names) if output_names else None,
                "metadata": resolved_metadata or {},
            }

            # Config resolution
            config_str: str | None = None
            resolved_config = config
            if resolved_config is None and loaded_bundle is not None:
                resolved_config = loaded_bundle.config

            if resolved_config is not None:
                if isinstance(resolved_config, (str, Path)) and Path(resolved_config).exists():
                    with open(resolved_config, "r", encoding="utf-8") as f:
                        config_str = f.read()
                elif isinstance(resolved_config, dict):
                    config_str = yaml.safe_dump(resolved_config, sort_keys=False)
                elif isinstance(resolved_config, str):
                    config_str = resolved_config

            with zipfile.ZipFile(tmp_bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
                zf.writestr("model.onnx", onnx_bytes)
                if config_str is not None:
                    zf.writestr("config.yaml", config_str.encode("utf-8"))

            os.replace(tmp_bundle, dest)
        else:
            os.replace(target_export_file, dest)

    finally:
        # Guarantee zero orphaned temporary files
        for p in (tmp_onnx, tmp_bundle):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    return dest


def export_onnx_as_bundle(*args: Any, **kwargs: Any) -> Path:
    """Convenience alias for export_onnx(..., as_bundle=True).

    Exports a PyTorch model into a self-contained ONNX bundle (.onnx.bundle)
    containing model.onnx, cryptographic SHA-256 manifest, and ModelSchema contract.
    """
    if "bundle_or_model" in kwargs and "model_or_bundle" not in kwargs:
        kwargs["model_or_bundle"] = kwargs.pop("bundle_or_model")
    kwargs["as_bundle"] = True
    return export_onnx(*args, **kwargs)

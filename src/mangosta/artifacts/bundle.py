"""Self-contained model bundle packaging with fault-tolerant checkpointing."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import random
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import safetensors.torch
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    safetensors = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False

import yaml

from mangosta.core.schema import ModelSchema, infer_schema
from mangosta.exceptions import BundleCorruptedError


def _capture_rng_states() -> dict[str, Any]:
    """Capture comprehensive RNG states across Torch (CPU and CUDA), Python random, and NumPy."""
    if not HAS_TORCH or torch is None:
        return {}

    rng: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python_random": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            rng["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass

    try:
        import numpy as np

        rng["numpy_random"] = np.random.get_state()
    except ImportError:
        pass

    return rng


def _restore_rng_states(rng_states: dict[str, Any]) -> None:
    """Restore RNG states for exact training reproducibility."""
    if not isinstance(rng_states, dict):
        return

    if HAS_TORCH and torch is not None:
        if "torch_cpu" in rng_states and rng_states["torch_cpu"] is not None:
            torch.set_rng_state(rng_states["torch_cpu"])

        if (
            "torch_cuda" in rng_states
            and rng_states["torch_cuda"] is not None
            and torch.cuda.is_available()
        ):
            try:
                torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
            except Exception:
                pass

    if "python_random" in rng_states and rng_states["python_random"] is not None:
        random.setstate(rng_states["python_random"])

    if "numpy_random" in rng_states and rng_states["numpy_random"] is not None:
        try:
            import numpy as np

            np.random.set_state(rng_states["numpy_random"])
        except ImportError:
            pass


def save_bundle(
    output_path: str | Path,
    model: Any,
    family: str | None = None,
    version: str = "0.1.0",
    init_args: dict[str, Any] | None = None,
    sample_input: Any | None = None,
    schema: ModelSchema | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    metrics: dict[str, Any] | None = None,
    dataset_hash: str | None = None,
    config: dict[str, Any] | str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Pack a PyTorch model and its complete runtime context into a self-contained .bundle archive.

    Guarantees atomic creation via temporary file swap.

    Smart defaults:
    - ``family``: Defaults to ``model.__class__.__name__.lower()`` if omitted.
    - ``version``: Defaults to ``"0.1.0"`` if omitted.
    - Training state: If ``optimizer``, ``scheduler``, or ``scaler`` is provided,
      the bundle is automatically marked and packed with training state for resumption.
    """
    if not HAS_TORCH or torch is None or safetensors is None:
        raise ImportError(
            "save_bundle requires PyTorch and SafeTensors. Please install torch and safetensors: `pip install torch safetensors`"
        )
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(f"{dest.name}.tmp_{os.getpid()}_{uuid.uuid4().hex[:6]}")

    # Smart defaults
    resolved_family = family
    resolved_version = version

    if resolved_family is None:
        # 1. Check if model class or instance was registered with @register_model
        model_cls = model if isinstance(model, type) else model.__class__
        registered_family = getattr(model, "_mangosta_family", None) or getattr(model_cls, "_mangosta_family", None)
        if registered_family:
            resolved_family = registered_family
            if version == "0.1.0":
                resolved_version = getattr(model, "_mangosta_version", None) or getattr(model_cls, "_mangosta_version", None) or version
        else:
            # 2. Check if model class is in ModelRegistry
            try:
                from mangosta.core.registry import ModelRegistry

                for (fam, ver), registered_cls in ModelRegistry._models.items():
                    if registered_cls is model_cls:
                        resolved_family = fam
                        if version == "0.1.0":
                            resolved_version = ver
                        break
            except Exception:
                pass

        # 3. Fallback to snake_case / lower of class name
        if resolved_family is None:
            resolved_family = model.__class__.__name__.lower()

    # Smart auto-detection of init_args if not provided
    resolved_init_args = init_args
    if resolved_init_args is None:
        resolved_init_args = {}
        try:
            import inspect

            sig = inspect.signature(model.__class__.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if hasattr(model, param_name):
                    val = getattr(model, param_name)
                    if isinstance(val, (int, float, str, bool, list, dict, tuple)):
                        resolved_init_args[param_name] = val
                elif param.default is not inspect.Parameter.empty:
                    val = param.default
                    if isinstance(val, (int, float, str, bool, list, dict, tuple)):
                        resolved_init_args[param_name] = val
        except Exception:
            resolved_init_args = {}

    is_checkpoint = optimizer is not None or scheduler is not None or scaler is not None

    # Infer schema if sample_input is given and schema not explicitly provided
    if schema is None and sample_input is not None:
        schema = infer_schema(model, sample_input)

    # Serialize model weights with SafeTensors (zero pickle) and compute SHA-256 fingerprint
    state_dict = model.state_dict()
    contiguous_weights = {
        k: v.contiguous() if isinstance(v, torch.Tensor) else v
        for k, v in state_dict.items()
    }
    weights_bytes = safetensors.torch.save(contiguous_weights)
    weights_sha256 = hashlib.sha256(weights_bytes).hexdigest()

    # Build manifest
    manifest: dict[str, Any] = {
        "format_version": "1.0",
        "family": resolved_family,
        "version": resolved_version,
        "init_args": resolved_init_args,
        "is_checkpoint": is_checkpoint,
        "weights_format": "safetensors",
        "weights_sha256": weights_sha256,
        "epoch": epoch,
        "step": step,
        "dataset_hash": dataset_hash,
        "schema": schema.to_dict() if schema is not None else None,
        "metrics": metrics or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "env": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        },
        "metadata": metadata or {},
    }

    # Resolve config dictionary / string
    config_str: str | None = None
    if config is not None:
        if isinstance(config, (str, Path)) and Path(config).exists():
            with open(config, "r", encoding="utf-8") as f:
                config_str = f.read()
        elif isinstance(config, dict):
            config_str = yaml.safe_dump(config, sort_keys=False)
        elif isinstance(config, str):
            config_str = config

    try:
        with zipfile.ZipFile(tmp_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. Manifest
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            zf.writestr("manifest.json", manifest_bytes)

            # 2. Config if provided
            if config_str is not None:
                zf.writestr("config.yaml", config_str.encode("utf-8"))

            # 3. Model weights in SafeTensors format
            zf.writestr("model.safetensors", weights_bytes)

            # 4. Training state (only if is_checkpoint is True)
            if is_checkpoint:
                train_state: dict[str, Any] = {
                    "epoch": epoch,
                    "step": step,
                    "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else None),
                    "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
                    "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
                    "rng_states": _capture_rng_states(),
                }
                train_buffer = io.BytesIO()
                torch.save(train_state, train_buffer)
                zf.writestr("train_state.pt", train_buffer.getvalue())

        # Atomic rename to prevent corrupt bundles on unexpected termination
        os.replace(tmp_dest, dest)
    finally:
        # Guarantee zero orphaned .tmp files if writing or replacing failed
        if tmp_dest.exists():
            try:
                tmp_dest.unlink()
            except OSError:
                pass

    return dest


def read_bundle_manifest(bundle_path: str | Path) -> dict[str, Any]:
    """Read and return only the manifest.json from a .bundle file without loading weights."""
    path = Path(bundle_path)
    if not path.is_file():
        raise FileNotFoundError(f"Bundle file not found: {bundle_path}")

    try:
        with zipfile.ZipFile(path, "r") as zf:
            if "manifest.json" not in zf.namelist():
                raise BundleCorruptedError(f"Missing 'manifest.json' in bundle: {bundle_path}")
            data = zf.read("manifest.json")
            manifest = json.loads(data.decode("utf-8"))
            return dict(manifest) if isinstance(manifest, dict) else {}
    except zipfile.BadZipFile as e:
        raise BundleCorruptedError(f"Invalid or corrupted ZIP archive: {bundle_path}") from e

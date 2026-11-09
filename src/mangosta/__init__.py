"""mangosta: Lightweight MLOps and Model Lifecycle Management for PyTorch and Production Serving."""

from __future__ import annotations

from typing import Any

# Detect PyTorch presence
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

# Scaffolding / CLI
from mangosta.cli.init_project import init_project

# Core contracts and schema (works with or without PyTorch)
from mangosta.core.schema import (
    ModelSchema,
    TensorSpec,
    infer_schema_from_sample,
)

# Exceptions (zero PyTorch dependency)
from mangosta.exceptions import (
    BundleCorruptedError,
    BundleError,
    ComponentNotFoundError,
    MangostaError,
    ModelNotFoundError,
    RegistryCollisionError,
    RegistryError,
    ResumeStateError,
    SchemaError,
    SchemaValidationError,
)

# Serving abstractions (zero PyTorch dependency)
from mangosta.serving.runner import (
    LoadedONNXBundle,
    ONNXRunner,
    load_onnx_bundle,
)

# Universal bundle loader & manifest reader
from mangosta.artifacts.loader import load_bundle
from mangosta.artifacts.bundle import read_bundle_manifest

# Training & PyTorch-specific components (loaded conditionally)
if HAS_TORCH:
    from mangosta.artifacts import (
        LoadedBundle,
        ResumeState,
        resume_training,
        save_bundle,
    )
    from mangosta.core import (
        ModelRegistry,
        infer_schema,
        register_component,
        register_model,
    )
    from mangosta.export import export_onnx, export_onnx_as_bundle
    from mangosta.tracking import (
        RunTracker,
        compute_sha256,
        generate_report,
        quick_track,
        start_viewer,
    )

__version__ = "0.3.0"

_BASE_ALL = [
    "__version__",
    # CLI / Scaffolding
    "init_project",
    # Core contracts
    "ModelSchema",
    "TensorSpec",
    "infer_schema_from_sample",
    # Serving
    "load_bundle",
    "read_bundle_manifest",
    "LoadedONNXBundle",
    "ONNXRunner",
    "load_onnx_bundle",
    # Exceptions
    "MangostaError",
    "RegistryError",
    "RegistryCollisionError",
    "ModelNotFoundError",
    "ComponentNotFoundError",
    "SchemaError",
    "SchemaValidationError",
    "BundleError",
    "BundleCorruptedError",
    "ResumeStateError",
]

_TORCH_ALL = [
    # Core Registry & Schema
    "register_model",
    "register_component",
    "ModelRegistry",
    "infer_schema",
    # Tracking
    "RunTracker",
    "quick_track",
    "compute_sha256",
    "generate_report",
    "start_viewer",
    # PyTorch Artifacts
    "save_bundle",
    "resume_training",
    "LoadedBundle",
    "ResumeState",
    # Export
    "export_onnx",
    "export_onnx_as_bundle",
]

if HAS_TORCH:
    __all__ = _BASE_ALL + _TORCH_ALL
else:
    __all__ = _BASE_ALL


def __getattr__(name: str) -> Any:
    if not HAS_TORCH and name in _TORCH_ALL:
        raise ImportError(
            f"'{name}' requires PyTorch to be installed.\n"
            "To use training, tracking, and export features, please install PyTorch:\n"
            "  `pip install torch` or `pip install 'mangosta[train]'`\n"
            "For production serving without PyTorch, use `mangosta.load_bundle()`, "
            "`mangosta.ONNXRunner`, or `mangosta.ModelSchema`."
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

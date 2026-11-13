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

# Universal bundle loader & manifest reader
from mangosta.artifacts.bundle import inspect_bundle_metadata, read_bundle_manifest
from mangosta.artifacts.loader import load_bundle

# Scaffolding / CLI
from mangosta.cli.init_project import init_project

# Core contracts, config, and schema (works with or without PyTorch)
from mangosta.core.config import (
    ExperimentConfig,
    MangostaConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
    seed_everything,
)
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
    ConfigError,
    MangostaError,
    ModelNotFoundError,
    RegistryCollisionError,
    RegistryError,
    ResumeStateError,
    SchemaError,
    SchemaValidationError,
    TrainerNotFoundError,
)

# Serving abstractions (zero PyTorch dependency)
from mangosta.serving.runner import (
    LoadedONNXBundle,
    ONNXRunner,
    load_onnx_bundle,
)

# Training & PyTorch-specific components (loaded conditionally; listed in _TORCH_ALL)
if HAS_TORCH:
    from mangosta.artifacts import (  # noqa: F401
        LoadedBundle,
        ResumeState,
        resume_training,
        save_bundle,
    )
    from mangosta.core import (  # noqa: F401
        ModelRegistry,
        TrainerRegistry,
        infer_schema,
        register_component,
        register_model,
        register_trainer,
    )
    from mangosta.engine import BaseTrainer  # noqa: F401
    from mangosta.export import export_onnx, export_onnx_as_bundle  # noqa: F401
    from mangosta.tracking import (  # noqa: F401
        RunTracker,
        compute_sha256,
        quick_track,
        start_viewer,
    )

__version__ = "0.4.0"

__all__ = [
    "__version__",
    # CLI / Scaffolding
    "init_project",
    # Core config & contracts
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "MangostaConfig",
    "load_config",
    "seed_everything",
    "ModelSchema",
    "TensorSpec",
    "infer_schema_from_sample",
    # Serving
    "load_bundle",
    "read_bundle_manifest",
    "inspect_bundle_metadata",
    "LoadedONNXBundle",
    "ONNXRunner",
    "load_onnx_bundle",
    # Exceptions
    "MangostaError",
    "ConfigError",
    "RegistryError",
    "RegistryCollisionError",
    "ModelNotFoundError",
    "TrainerNotFoundError",
    "ComponentNotFoundError",
    "SchemaError",
    "SchemaValidationError",
    "BundleError",
    "BundleCorruptedError",
    "ResumeStateError",
]

_TORCH_ALL = [
    # Core Registry, Engine & Schema
    "register_model",
    "register_trainer",
    "register_component",
    "ModelRegistry",
    "TrainerRegistry",
    "BaseTrainer",
    "infer_schema",
    # Tracking
    "RunTracker",
    "quick_track",
    "compute_sha256",
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
    __all__ += _TORCH_ALL


def __getattr__(name: str) -> Any:
    if not HAS_TORCH and name in _TORCH_ALL:
        raise ImportError(
            f"'{name}' requires PyTorch to be installed.\n"
            "To use training, tracking, and export features, please install:\n"
            "  `uv add 'mangosta[full]'` (or `pip install 'mangosta[full]'`)\n"
            "For production serving without PyTorch, use `mangosta.load_bundle()`, "
            "`mangosta.ONNXRunner`, or `mangosta.ModelSchema`."
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

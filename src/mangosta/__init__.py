"""mangosta: Lightweight MLOps and Model Lifecycle Management for PyTorch."""

from __future__ import annotations

from mangosta.artifacts import (
    LoadedBundle,
    ResumeState,
    load_bundle,
    read_bundle_manifest,
    resume_training,
    save_bundle,
)
from mangosta.cli.init_project import init_project
from mangosta.core import (
    ModelRegistry,
    ModelSchema,
    TensorSpec,
    infer_schema,
    register_component,
    register_model,
)
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
from mangosta.export import export_onnx
from mangosta.tracking import (
    RunTracker,
    compute_sha256,
    generate_report,
    quick_track,
    start_viewer,
)

__version__ = "0.2.0a1"

__all__ = [
    "__version__",
    # CLI / Scaffolding
    "init_project",
    # Core
    "register_model",
    "register_component",
    "ModelRegistry",
    "infer_schema",
    "ModelSchema",
    "TensorSpec",
    # Tracking
    "RunTracker",
    "quick_track",
    "compute_sha256",
    "generate_report",
    "start_viewer",
    # Artifacts
    "save_bundle",
    "load_bundle",
    "resume_training",
    "read_bundle_manifest",
    "LoadedBundle",
    "ResumeState",
    # Export
    "export_onnx",
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

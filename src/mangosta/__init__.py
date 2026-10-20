"""mangosta: Lightweight MLOps and Model Lifecycle Management for PyTorch."""

from __future__ import annotations

from mangosta.bundle import read_bundle_manifest, save_bundle
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
from mangosta.loader import LoadedBundle, ResumeState, load_bundle, resume_training
from mangosta.registry import ModelRegistry, register_component, register_model
from mangosta.report import generate_report
from mangosta.schema import ModelSchema, TensorSpec, infer_schema
from mangosta.tracker import RunTracker, compute_sha256, quick_track

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Registry
    "register_model",
    "register_component",
    "ModelRegistry",
    # Schema
    "infer_schema",
    "ModelSchema",
    "TensorSpec",
    # Tracker
    "RunTracker",
    "quick_track",
    "compute_sha256",
    # Bundle & Loader
    "save_bundle",
    "load_bundle",
    "resume_training",
    "read_bundle_manifest",
    "LoadedBundle",
    "ResumeState",
    # Report
    "generate_report",
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

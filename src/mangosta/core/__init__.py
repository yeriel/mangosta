"""Core abstractions: architecture registry and tensor contracts."""

from __future__ import annotations

from mangosta.core.registry import (
    ModelRegistry,
    register_component,
    register_model,
)
from mangosta.core.schema import (
    ModelSchema,
    TensorSpec,
    infer_schema,
)

__all__ = [
    "ModelRegistry",
    "register_model",
    "register_component",
    "TensorSpec",
    "ModelSchema",
    "infer_schema",
]

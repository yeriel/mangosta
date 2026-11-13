"""Core abstractions: architecture registry and tensor contracts."""

from __future__ import annotations

from mangosta.core.config import (
    ExperimentConfig,
    MangostaConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
    seed_everything,
)
from mangosta.core.registry import (
    ModelRegistry,
    TrainerRegistry,
    register_component,
    register_model,
    register_trainer,
)
from mangosta.core.schema import (
    ModelSchema,
    TensorSpec,
    infer_schema,
    infer_schema_from_sample,
)

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "MangostaConfig",
    "load_config",
    "seed_everything",
    "ModelRegistry",
    "TrainerRegistry",
    "register_model",
    "register_trainer",
    "register_component",
    "TensorSpec",
    "ModelSchema",
    "infer_schema",
    "infer_schema_from_sample",
]

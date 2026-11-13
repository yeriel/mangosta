"""Strict typed configuration dataclasses and deterministic seeding for Mangosta."""

from __future__ import annotations

import difflib
import os
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mangosta.exceptions import ConfigError

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


def seed_everything(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if HAS_TORCH and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def _validate_unknown_keys(
    raw_section: dict[str, Any],
    valid_keys: set[str],
    section_name: str,
    source: str,
) -> None:
    """Raise ConfigError with a 'Did you mean' suggestion if an unknown key is present."""
    for key in raw_section:
        if key not in valid_keys:
            matches = difflib.get_close_matches(key, sorted(valid_keys), n=1, cutoff=0.5)
            hint = f" Did you mean '{matches[0]}'?" if matches else ""
            raise ConfigError(
                f"Unknown parameter '{key}' in section '{section_name}' ({source}).{hint}\n"
                f"Valid parameters for '{section_name}' are: {sorted(valid_keys)}"
            )


@dataclass
class ExperimentConfig:
    """Experiment-level metadata and execution settings."""

    name: str = "baseline"
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError(f"experiment.name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ConfigError(f"experiment.seed must be an integer, got {self.seed!r}")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ConfigError(
                f"experiment.device must be one of 'auto', 'cpu', 'cuda', 'mps', got {self.device!r}"
            )


@dataclass
class ModelConfig:
    """Model registry coordinates and constructor arguments."""

    family: str = ""
    version: str = "1.0.0"
    init_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.family, str):
            raise ConfigError(f"model.family must be a string, got {self.family!r}")
        self.version = str(self.version)
        if not isinstance(self.init_args, dict):
            raise ConfigError(f"model.init_args must be a dictionary, got {type(self.init_args).__name__}")


@dataclass
class TrainingConfig:
    """Training hyperparameters and MLOps automation controls."""

    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    amp: bool = False
    grad_clip_norm: float | None = None
    patience: int | None = None
    monitor: str = "val_loss"
    monitor_mode: str = "min"

    def __post_init__(self) -> None:
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs < 1:
            raise ConfigError(f"training.epochs must be a positive integer, got {self.epochs!r}")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ConfigError(f"training.batch_size must be a positive integer, got {self.batch_size!r}")
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        if not isinstance(self.amp, bool):
            raise ConfigError(f"training.amp must be a boolean, got {self.amp!r}")
        if self.grad_clip_norm is not None:
            self.grad_clip_norm = float(self.grad_clip_norm)
            if self.grad_clip_norm <= 0:
                raise ConfigError(
                    f"training.grad_clip_norm must be > 0 or None, got {self.grad_clip_norm!r}"
                )
        if self.patience is not None:
            if not isinstance(self.patience, int) or isinstance(self.patience, bool) or self.patience < 1:
                raise ConfigError(
                    f"training.patience must be a positive integer or None, got {self.patience!r}"
                )
        if self.monitor_mode not in ("min", "max"):
            raise ConfigError(
                f"training.monitor_mode must be 'min' or 'max', got {self.monitor_mode!r}"
            )


@dataclass
class MangostaConfig:
    """Root typed configuration for Mangosta pipelines."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: dict[str, Any] = field(default_factory=dict)
    custom: dict[str, Any] = field(default_factory=dict)
    _explicit_fields: dict[str, set[str]] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "dict") -> MangostaConfig:
        """Build and validate a MangostaConfig from a raw dictionary."""
        if not isinstance(raw, dict):
            raise ConfigError(f"Configuration root must be a dictionary, got {type(raw).__name__}")

        valid_root = {"experiment", "model", "training", "data", "custom"}
        _validate_unknown_keys(raw, valid_root, "root", source)

        explicit: dict[str, set[str]] = {}

        # 1. Experiment section
        raw_exp = raw.get("experiment") or {}
        if not isinstance(raw_exp, dict):
            raise ConfigError(f"Section 'experiment' must be a mapping, got {type(raw_exp).__name__}")
        exp_keys = {f.name for f in fields(ExperimentConfig)}
        _validate_unknown_keys(raw_exp, exp_keys, "experiment", source)
        explicit["experiment"] = set(raw_exp.keys())
        exp_cfg = ExperimentConfig(**raw_exp)

        # 2. Model section (supports both explicit init_args and flat model keys)
        raw_model = dict(raw.get("model") or {})
        if not isinstance(raw_model, dict):
            raise ConfigError(f"Section 'model' must be a mapping, got {type(raw_model).__name__}")
        family = raw_model.pop("family", "")
        version = str(raw_model.pop("version", "1.0.0"))
        init_args = dict(raw_model.pop("init_args", {}) or {})
        if raw_model:
            # Backward-compatible flat kwargs inside model: merge into init_args
            init_args.update(raw_model)
        explicit["model"] = {"family", "version", "init_args"} if (raw.get("model")) else set()
        model_cfg = ModelConfig(family=family, version=version, init_args=init_args)

        # 3. Training section
        raw_train = raw.get("training") or {}
        if not isinstance(raw_train, dict):
            raise ConfigError(f"Section 'training' must be a mapping, got {type(raw_train).__name__}")
        train_keys = {f.name for f in fields(TrainingConfig)}
        _validate_unknown_keys(raw_train, train_keys, "training", source)
        explicit["training"] = set(raw_train.keys())
        train_cfg = TrainingConfig(**raw_train)

        # 4. Free-form user sections
        raw_data = raw.get("data") or {}
        raw_custom = raw.get("custom") or {}
        if not isinstance(raw_data, dict):
            raise ConfigError(f"Section 'data' must be a mapping, got {type(raw_data).__name__}")
        if not isinstance(raw_custom, dict):
            raise ConfigError(f"Section 'custom' must be a mapping, got {type(raw_custom).__name__}")

        return cls(
            experiment=exp_cfg,
            model=model_cfg,
            training=train_cfg,
            data=dict(raw_data),
            custom=dict(raw_custom),
            _explicit_fields=explicit,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain dictionary (excluding internal tracking)."""
        return {
            "experiment": asdict(self.experiment),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": dict(self.data),
            "custom": dict(self.custom),
        }

    def format_summary(self, resolved_device: str | None = None) -> str:
        """Return a clean summary table showing which values came from YAML vs defaults."""
        dev_str = resolved_device or self.experiment.device
        lines = [
            "╭──────────────────── Mangosta Training Config ────────────────────╮",
            f"│ Experiment : {self.experiment.name} (seed={self.experiment.seed}, device={dev_str})",
            f"│ Model      : {self.model.family or '<unregistered>'} v{self.model.version}",
            f"│   init_args: {self.model.init_args}",
            "│ Training   :",
        ]
        train_explicit = self._explicit_fields.get("training", set())
        for f in fields(TrainingConfig):
            val = getattr(self.training, f.name)
            origin = "(from YAML)" if f.name in train_explicit else "(default)"
            lines.append(f"│   {f.name:<15} = {str(val):<10} {origin}")
        lines.append("╰──────────────────────────────────────────────────────────────────╯")
        return "\n".join(lines)


def load_config(path_or_dict: str | Path | dict[str, Any] | MangostaConfig) -> MangostaConfig:
    """Load and strictly validate a MangostaConfig from a YAML file, dict, or existing instance."""
    if isinstance(path_or_dict, MangostaConfig):
        return path_or_dict
    if isinstance(path_or_dict, dict):
        return MangostaConfig.from_dict(path_or_dict, source="dict")

    path = Path(path_or_dict)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: '{path}'")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML file '{path}': {e}") from e

    return MangostaConfig.from_dict(raw, source=str(path))

"""Universal decoupled model loading and fault-tolerant training resumption."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import yaml

from mangosta.artifacts.bundle import _restore_rng_states
from mangosta.core.registry import ModelRegistry
from mangosta.core.schema import ModelSchema
from mangosta.exceptions import BundleCorruptedError, ResumeStateError


@dataclass
class ResumeState:
    """State restored from a training checkpoint bundle."""

    epoch: int
    step: int
    manifest: dict[str, Any]
    rng_restored: bool


class LoadedBundle:
    """Encapsulates a loaded model ready for inference with automated contract validation."""

    def __init__(
        self,
        model: nn.Module,
        manifest: dict[str, Any],
        schema: ModelSchema | None = None,
        config: dict[str, Any] | str | None = None,
        bundle_path: Path | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.manifest = manifest
        self.schema = schema
        self.config = config
        self.bundle_path = bundle_path

    @property
    def family(self) -> str:
        return self.manifest.get("family", "unknown")

    @property
    def version(self) -> str:
        return self.manifest.get("version", "unknown")

    @property
    def metadata(self) -> dict[str, Any]:
        return self.manifest.get("metadata", {})

    def predict(self, input_data: Any, validate: bool = True) -> Any:
        """Run forward inference, optionally verifying the input against the schema contract."""
        if validate and self.schema is not None:
            self.schema.validate_input(input_data)

        with torch.inference_mode():
            if isinstance(input_data, torch.Tensor):
                output = self.model(input_data)
            elif isinstance(input_data, (list, tuple)):
                output = self.model(*input_data)
            elif isinstance(input_data, Mapping):
                output = self.model(**input_data)
            else:
                output = self.model(input_data)

        if validate and self.schema is not None:
            self.schema.validate_output(output)

        return output


def load_bundle(
    bundle_path: str | Path,
    map_location: str | torch.device = "cpu",
    model: nn.Module | None = None,
) -> LoadedBundle:
    """Universal loader that reconstructs the architecture from the registry and restores weights.

    If `model` is not provided, the architecture is dynamically reconstructed
    using the model family, version, and init_args stored in manifest.json.
    """
    path = Path(bundle_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Bundle file not found: {bundle_path}")

    try:
        with zipfile.ZipFile(path, "r") as zf:
            namelist = zf.namelist()
            if "manifest.json" not in namelist:
                raise BundleCorruptedError(f"Missing 'manifest.json' in {bundle_path}")
            if "weights.pt" not in namelist:
                raise BundleCorruptedError(f"Missing 'weights.pt' in {bundle_path}")

            manifest_data = zf.read("manifest.json")
            manifest = json.loads(manifest_data.decode("utf-8"))

            config: dict[str, Any] | str | None = None
            if "config.yaml" in namelist:
                cfg_raw = zf.read("config.yaml").decode("utf-8")
                try:
                    config = yaml.safe_load(cfg_raw)
                except Exception:
                    config = cfg_raw

            # Reconstruct model architecture
            if model is None:
                family = manifest.get("family")
                version = manifest.get("version")
                init_args = manifest.get("init_args", {})
                if not family:
                    raise BundleCorruptedError("Manifest is missing 'family' field.")
                model = ModelRegistry.build(family, version=version, **init_args)

            # Load model state_dict
            weights_bytes = zf.read("weights.pt")
            weights_buffer = io.BytesIO(weights_bytes)
            state_dict = torch.load(weights_buffer, map_location=map_location, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(map_location)
            model.eval()

            # Reconstruct schema if present
            schema: ModelSchema | None = None
            if manifest.get("schema") is not None:
                schema = ModelSchema.from_dict(manifest["schema"])

            return LoadedBundle(
                model=model,
                manifest=manifest,
                schema=schema,
                config=config,
                bundle_path=path,
            )

    except zipfile.BadZipFile as e:
        raise BundleCorruptedError(f"Invalid or corrupted ZIP archive: {bundle_path}") from e


def resume_training(
    bundle_path: str | Path,
    model: nn.Module,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> ResumeState:
    """Restore the complete training state from a fault-tolerant checkpoint bundle.

    Loads model weights, optimizer state, scheduler state, GradScaler state,
    and restores exact RNG generator states.
    """
    path = Path(bundle_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Bundle file not found: {bundle_path}")

    try:
        with zipfile.ZipFile(path, "r") as zf:
            namelist = zf.namelist()
            if "train_state.pt" not in namelist:
                raise ResumeStateError(
                    f"Bundle '{bundle_path}' does not contain 'train_state.pt'. "
                    "Make sure it was saved with is_checkpoint=True."
                )

            manifest_data = zf.read("manifest.json")
            manifest = json.loads(manifest_data.decode("utf-8"))

            # 1. Restore model weights
            weights_bytes = zf.read("weights.pt")
            state_dict = torch.load(
                io.BytesIO(weights_bytes),
                map_location=map_location,
                weights_only=True,
            )
            model.load_state_dict(state_dict)

            # 2. Restore training state
            train_bytes = zf.read("train_state.pt")
            train_state = torch.load(
                io.BytesIO(train_bytes),
                map_location=map_location,
                weights_only=False,  # Needs full python objects for RNG/optimizer states
            )

            # Optimizer
            if optimizer is not None and train_state.get("optimizer_state_dict"):
                optimizer.load_state_dict(train_state["optimizer_state_dict"])

            # Scheduler
            if scheduler is not None and train_state.get("scheduler_state_dict"):
                scheduler.load_state_dict(train_state["scheduler_state_dict"])

            # Scaler
            if scaler is not None and train_state.get("scaler_state_dict"):
                scaler.load_state_dict(train_state["scaler_state_dict"])

            # RNG states
            rng_restored = False
            if restore_rng and train_state.get("rng_states"):
                _restore_rng_states(train_state["rng_states"])
                rng_restored = True

            return ResumeState(
                epoch=train_state.get("epoch", 0),
                step=train_state.get("step", 0),
                manifest=manifest,
                rng_restored=rng_restored,
            )

    except zipfile.BadZipFile as e:
        raise BundleCorruptedError(f"Corrupted bundle archive: {bundle_path}") from e

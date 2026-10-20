"""Input/Output schema inference and tensor signature validation for PyTorch models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn

from mangosta.exceptions import SchemaValidationError


def _normalize_dtype(dtype: torch.dtype | str) -> str:
    """Normalize a torch dtype or string representation into a canonical string."""
    if isinstance(dtype, torch.dtype):
        return str(dtype).replace("torch.", "")
    s = str(dtype).strip().lower()
    if s.startswith("torch."):
        s = s[6:]
    return s


def _get_torch_dtype(dtype_str: str) -> torch.dtype | None:
    """Resolve a string to a torch.dtype if possible."""
    norm = _normalize_dtype(dtype_str)
    return getattr(torch, norm, None)


@dataclass
class TensorSpec:
    """Specification of an expected tensor shape and data type."""

    shape: list[int | None]
    dtype: str
    name: str = "tensor"

    def __post_init__(self) -> None:
        self.dtype = _normalize_dtype(self.dtype)
        # Convert any negative dimension (e.g. -1) to None for consistency
        self.shape = [None if (dim is None or dim == -1) else int(dim) for dim in self.shape]

    def matches(
        self, tensor: torch.Tensor, allow_batch_dynamic: bool = True
    ) -> tuple[bool, str | None]:
        """Check if a tensor matches this specification.

        Returns (is_match, error_message).
        """
        if not isinstance(tensor, torch.Tensor):
            return False, f"Expected torch.Tensor for '{self.name}', got {type(tensor).__name__}"

        actual_dtype = _normalize_dtype(tensor.dtype)
        if actual_dtype != self.dtype:
            return (
                False,
                f"Tensor '{self.name}' expected dtype {self.dtype}, got {actual_dtype}",
            )

        actual_shape = list(tensor.shape)
        if len(actual_shape) != len(self.shape):
            return (
                False,
                f"Tensor '{self.name}' expected {len(self.shape)} dimensions {self.shape}, got {len(actual_shape)} dimensions {actual_shape}",
            )

        for i, (expected_dim, actual_dim) in enumerate(zip(self.shape, actual_shape)):
            if expected_dim is None:
                # Dynamic dimension, any size >= 0 is allowed
                continue
            if expected_dim != actual_dim:
                return (
                    False,
                    f"Tensor '{self.name}' dimension {i} mismatch: expected {expected_dim}, got {actual_dim} (expected {self.shape}, got {actual_shape})",
                )

        return True, None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorSpec:
        return cls(
            name=data.get("name", "tensor"),
            shape=list(data.get("shape", [])),
            dtype=data.get("dtype", "float32"),
        )


@dataclass
class ModelSchema:
    """I/O Contract specifying shapes and dtypes for inputs and outputs."""

    inputs: list[TensorSpec] | dict[str, TensorSpec]
    outputs: list[TensorSpec] | dict[str, TensorSpec]
    input_structure: str = "tensor"  # "tensor", "sequence", or "mapping"
    output_structure: str = "tensor"  # "tensor", "sequence", or "mapping"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_input(self, input_data: Any) -> None:
        """Validate input data against the schema. Raises SchemaValidationError on mismatch."""
        self._validate_structure_and_specs(
            data=input_data,
            specs=self.inputs,
            expected_structure=self.input_structure,
            direction="Input",
        )

    def validate_output(self, output_data: Any) -> None:
        """Validate output data against the schema. Raises SchemaValidationError on mismatch."""
        self._validate_structure_and_specs(
            data=output_data,
            specs=self.outputs,
            expected_structure=self.output_structure,
            direction="Output",
        )

    def _validate_structure_and_specs(
        self,
        data: Any,
        specs: list[TensorSpec] | dict[str, TensorSpec],
        expected_structure: str,
        direction: str,
    ) -> None:
        if expected_structure == "tensor":
            if not isinstance(data, torch.Tensor):
                raise SchemaValidationError(
                    f"{direction} must be a torch.Tensor, got {type(data).__name__}"
                )
            spec = specs[0] if isinstance(specs, list) else list(specs.values())[0]
            ok, err = spec.matches(data)
            if not ok:
                raise SchemaValidationError(f"{direction} schema violation: {err}")

        elif expected_structure == "sequence":
            if not isinstance(data, (list, tuple)):
                raise SchemaValidationError(
                    f"{direction} must be a sequence (list or tuple), got {type(data).__name__}"
                )
            spec_list = specs if isinstance(specs, list) else list(specs.values())
            if len(data) != len(spec_list):
                raise SchemaValidationError(
                    f"{direction} sequence length mismatch: expected {len(spec_list)} items, got {len(data)}"
                )
            for idx, (item, spec) in enumerate(zip(data, spec_list)):
                ok, err = spec.matches(item)
                if not ok:
                    raise SchemaValidationError(f"{direction} item [{idx}] schema violation: {err}")

        elif expected_structure == "mapping":
            if not isinstance(data, Mapping):
                raise SchemaValidationError(
                    f"{direction} must be a mapping/dict, got {type(data).__name__}"
                )
            spec_map = specs if isinstance(specs, dict) else {s.name: s for s in specs}
            for k, spec in spec_map.items():
                if k not in data:
                    raise SchemaValidationError(f"{direction} mapping missing required key: '{k}'")
                ok, err = spec.matches(data[k])
                if not ok:
                    raise SchemaValidationError(f"{direction} key '{k}' schema violation: {err}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize schema to a JSON-compatible dictionary."""
        if isinstance(self.inputs, list):
            inputs_serialized: Any = [s.to_dict() for s in self.inputs]
        else:
            inputs_serialized = {k: s.to_dict() for k, s in self.inputs.items()}

        if isinstance(self.outputs, list):
            outputs_serialized: Any = [s.to_dict() for s in self.outputs]
        else:
            outputs_serialized = {k: s.to_dict() for k, s in self.outputs.items()}

        return {
            "inputs": inputs_serialized,
            "outputs": outputs_serialized,
            "input_structure": self.input_structure,
            "output_structure": self.output_structure,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSchema:
        """Construct ModelSchema from serialized dictionary."""
        raw_inputs = data.get("inputs", [])
        raw_outputs = data.get("outputs", [])
        input_structure = data.get("input_structure", "tensor")
        output_structure = data.get("output_structure", "tensor")

        if isinstance(raw_inputs, list):
            inputs: list[TensorSpec] | dict[str, TensorSpec] = [
                TensorSpec.from_dict(item) for item in raw_inputs
            ]
        else:
            inputs = {k: TensorSpec.from_dict(v) for k, v in raw_inputs.items()}

        if isinstance(raw_outputs, list):
            outputs: list[TensorSpec] | dict[str, TensorSpec] = [
                TensorSpec.from_dict(item) for item in raw_outputs
            ]
        else:
            outputs = {k: TensorSpec.from_dict(v) for k, v in raw_outputs.items()}

        return cls(
            inputs=inputs,
            outputs=outputs,
            input_structure=input_structure,
            output_structure=output_structure,
            metadata=data.get("metadata", {}),
        )


def _spec_from_tensor(
    tensor: torch.Tensor, name: str = "tensor", dynamic_batch_dim: int | None = 0
) -> TensorSpec:
    shape: list[int | None] = list(tensor.shape)
    if dynamic_batch_dim is not None and len(shape) > dynamic_batch_dim:
        shape[dynamic_batch_dim] = None
    return TensorSpec(name=name, shape=shape, dtype=_normalize_dtype(tensor.dtype))


def infer_schema(
    model: nn.Module,
    sample_input: Any,
    dynamic_batch_dim: int | None = 0,
) -> ModelSchema:
    """Non-invasively inspect a model's input and output tensor signatures.

    Executes a single forward pass with `torch.inference_mode()` while preserving
    the model's training state.

    Args:
        model: PyTorch Module to inspect.
        sample_input: A torch.Tensor, sequence of Tensors, or Mapping of name -> Tensor.
        dynamic_batch_dim: Dimension index to treat as dynamic batch size (default 0).
                           Pass None to disable dynamic batching.

    Returns:
        A ModelSchema object representing the model's signature contract.
    """
    was_training = model.training
    model.eval()

    try:
        # Determine input structure and specs
        if isinstance(sample_input, torch.Tensor):
            input_structure = "tensor"
            inputs: list[TensorSpec] | dict[str, TensorSpec] = [
                _spec_from_tensor(sample_input, name="input", dynamic_batch_dim=dynamic_batch_dim)
            ]
            with torch.inference_mode():
                outputs_raw = model(sample_input)

        elif isinstance(sample_input, (list, tuple)):
            input_structure = "sequence"
            inputs = [
                _spec_from_tensor(t, name=f"input_{i}", dynamic_batch_dim=dynamic_batch_dim)
                for i, t in enumerate(sample_input)
            ]
            with torch.inference_mode():
                outputs_raw = model(*sample_input)

        elif isinstance(sample_input, Mapping):
            input_structure = "mapping"
            inputs = {
                k: _spec_from_tensor(v, name=k, dynamic_batch_dim=dynamic_batch_dim)
                for k, v in sample_input.items()
            }
            with torch.inference_mode():
                outputs_raw = model(**sample_input)
        else:
            raise TypeError(
                f"sample_input must be a Tensor, Sequence of Tensors, or Mapping of Tensors, got {type(sample_input).__name__}"
            )

        # Determine output structure and specs
        if isinstance(outputs_raw, torch.Tensor):
            output_structure = "tensor"
            outputs: list[TensorSpec] | dict[str, TensorSpec] = [
                _spec_from_tensor(outputs_raw, name="output", dynamic_batch_dim=dynamic_batch_dim)
            ]
        elif isinstance(outputs_raw, (list, tuple)):
            output_structure = "sequence"
            outputs = [
                _spec_from_tensor(t, name=f"output_{i}", dynamic_batch_dim=dynamic_batch_dim)
                for i, t in enumerate(outputs_raw)
            ]
        elif isinstance(outputs_raw, Mapping):
            output_structure = "mapping"
            outputs = {
                k: _spec_from_tensor(v, name=k, dynamic_batch_dim=dynamic_batch_dim)
                for k, v in outputs_raw.items()
            }
        else:
            raise TypeError(f"Model returned unsupported output type: {type(outputs_raw).__name__}")

        return ModelSchema(
            inputs=inputs,
            outputs=outputs,
            input_structure=input_structure,
            output_structure=output_structure,
        )

    finally:
        model.train(was_training)

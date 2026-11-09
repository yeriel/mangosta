# API Reference: Schema & Contracts (`mangosta.core.schema`)

The schema module defines formal data contracts for tensor inputs and outputs. It guarantees that models in training, serving, and export pipelines receive expected tensor shapes, data types, and dictionary/tuple structures.

---

## `TensorSpec`

```python
@dataclass
class TensorSpec:
    shape: list[int | None]
    dtype: str
    name: str = "tensor"
    optional: bool = False
```

Specification of an individual tensor's shape, dtype, and name.

### Fields
- **`shape`**: Dimensions of the tensor. Use `None` or `-1` for dynamic dimensions (e.g. `[None, 3, 224, 224]` for dynamic batch size).
- **`dtype`**: Standard dtype string (e.g. `"float32"`, `"int64"`, `"float16"`, `"uint8"`, `"bool"`).
- **`name`**: Descriptive identifier for the tensor node.
- **`optional`**: Whether this tensor can be omitted in a multi-input signature.

### Methods
- **`validate(tensor: Any, context: str = "") -> None`**: Validates whether the given tensor (PyTorch `Tensor` or NumPy `ndarray`) satisfies shape and dtype constraints. Raises `SchemaValidationError` with actionable error diagnostics on mismatch.
- **`to_dict() -> dict[str, Any]`**: Serialize spec to a JSON-compatible dictionary.
- **`from_dict(data: dict[str, Any]) -> TensorSpec`**: Reconstruct spec from dictionary.

---

## `ModelSchema`

```python
@dataclass
class ModelSchema:
    inputs: list[TensorSpec] | dict[str, TensorSpec]
    outputs: list[TensorSpec] | dict[str, TensorSpec]
    input_structure: str = "tensor"    # "tensor", "sequence", or "mapping"
    output_structure: str = "tensor"   # "tensor", "sequence", or "mapping"
    metadata: dict[str, Any] = field(default_factory=dict)
```

Complete input/output data contract for a model.

### Structure Types
- **`"tensor"`**: The model accepts or returns a single tensor/array.
- **`"sequence"`**: The model accepts or returns a sequence (`tuple` or `list`) of tensors.
- **`"mapping"`**: The model accepts or returns a key-value dictionary of tensors (`dict[str, Tensor]`).

### Methods
- **`validate_input(input_data: Any) -> None`**: Validate runtime inputs before forward execution.
- **`validate_output(output_data: Any) -> None`**: Validate model outputs after execution.
- **`check_input(input_data: Any) -> bool`**: Returns `True` if valid, `False` otherwise (non-raising check).
- **`check_output(output_data: Any) -> bool`**: Returns `True` if valid, `False` otherwise (non-raising check).
- **`to_dict() -> dict[str, Any]`**: JSON-compatible serialization.
- **`from_dict(data: dict[str, Any]) -> ModelSchema`**: Reconstruct from serialized dictionary.

---

## Schema Inference Functions

### `infer_schema(model: nn.Module, sample_input: Any) -> ModelSchema`
Inspect the model dynamically by executing a single dry forward pass with `sample_input` under `torch.no_grad()`. Infers both input and output shapes, dtypes, and structural containers (`tensor`, `sequence`, or `mapping`).

Dynamic batch dimensions (dim 0) are automatically detected and replaced with `None`.

### `infer_schema_from_sample(sample_input: Any, sample_output: Any) -> ModelSchema`
Construct a `ModelSchema` directly from sample inputs and outputs without running a forward pass. Useful when working with offline traces or mock tensors.

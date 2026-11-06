# API Reference: Export (`mangosta.export`)

The export module handles compiling and packaging PyTorch models or SafeTensors bundles into standardized ONNX binaries and `.onnx.bundle` deployment packages.

---

## `export_onnx`

```python
def export_onnx(
    model_or_bundle: nn.Module | str | Path,
    output_path: str | Path,
    weights: str | Path | None = None,
    sample_input: Any | None = None,
    sample_input_shape: tuple[int, ...] | None = None,
    output_mapping: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    float16: bool = False,
    opset_version: int = 18,
    do_constant_folding: bool = True,
    schema: ModelSchema | None = None,
    config: dict[str, Any] | str | Path | None = None,
    as_bundle: bool | None = None,
    family: str | None = None,
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    verify: bool = True,
) -> Path:
```

Exports a PyTorch model or `.bundle` archive to ONNX.

### Parameters
- **`model_or_bundle`**: An initialized `nn.Module` instance, or a path to an existing `.bundle` file.
- **`output_path`**: Destination file path (`.onnx` or `.onnx.bundle`).
- **`weights`**: Optional path to a `.pth`, `.pt`, or `.safetensors` file to populate the model before export.
- **`sample_input`**: Real tensor or sequence of tensors used for graph tracing.
- **`sample_input_shape`**: Shape tuple used to generate dummy random inputs when `sample_input` is omitted (e.g. `(1, 3, 224, 224)`).
- **`output_mapping`**: Ordered list of dictionary keys. When your PyTorch model outputs a `dict[str, Tensor]`, ONNX exporter unrolls the dictionary into a tuple matching this order.
- **`input_names`**: Names assigned to ONNX input graph nodes (defaults to `["input0"]`).
- **`output_names`**: Names assigned to ONNX output graph nodes.
- **`dynamic_axes`**: Custom dynamic dimensions mapping. Defaults to dynamic batch dimension on dim 0 for all inputs and outputs.
- **`float16`**: When `True`, casts the model and sample inputs to `torch.float16` for reduced memory and accelerated GPU/NPU inference.
- **`opset_version`**: ONNX opset version (default: 18).
- **`as_bundle`**:
  - `True`: Package into a `.onnx.bundle` containing `manifest.json`, `model.onnx`, and optional `config.yaml`.
  - `False`: Export standalone `.onnx` file.
  - `None` (default): Infers based on whether `output_path` ends with `.bundle`.
- **`verify`**: When `True` (default), uses `onnx.checker` to validate graph integrity immediately after generation.

### Example
```python
from mangosta import export_onnx

export_onnx(
    model_or_bundle="checkpoints/epoch_10.bundle",
    output_path="exports/resnet.onnx.bundle",
    sample_input_shape=(1, 3, 224, 224),
    float16=True,
    as_bundle=True,
)
```

# API Reference: Export & Slim Serving (`mangosta.export` / `mangosta.serving`)

Compile `.bundle` or PyTorch modules into verified `.onnx` / `.onnx.bundle` packages (`mangosta[full]`), then serve them with pure NumPy + ONNX Runtime without PyTorch (`mangosta[serving]`).

---

## 1. Export (`mangosta.export`)

### `export_onnx`

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
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    float16: bool = False,
    opset_version: int = 18,
    do_constant_folding: bool = True,
    device: str | torch.device | None = None,
    verify: bool = True,
    as_bundle: bool | None = None,
    family: str | None = None,
    version: str | None = None,
    schema: ModelSchema | None = None,
    config: dict[str, Any] | str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path
```

Key behaviors:

- **Schema Shape Auto-Inference**: If `model_or_bundle` is a `.bundle` containing a `ModelSchema` and both `sample_input` and `sample_input_shape` are omitted, dummy tensors are generated automatically from `schema.inputs` (substituting `1` for dynamic dimensions).
- **Dictionary Unrolling (`output_mapping`)**: Wraps models returning `dict[str, Tensor]` into ordered positional ONNX outputs.
- **Bundle Detection**: When `as_bundle=None` (default), packs a `.onnx.bundle` if `output_path` ends with `.onnx.bundle` or `.bundle`, else writes a standalone `.onnx` file.

### `export_onnx_as_bundle`

```python
def export_onnx_as_bundle(*args: Any, **kwargs: Any) -> Path
```

Convenience wrapper forcing `as_bundle=True` (also accepts `bundle_or_model=` as an alias for `model_or_bundle=`).

---

## 2. Zero-PyTorch Serving (`mangosta.serving`)

### `load_onnx_bundle`

```python
def load_onnx_bundle(
    bundle_path: str | Path,
    providers: list[str] | None = None,
) -> LoadedONNXBundle
```

Verifies `model.onnx` SHA-256 against `manifest.json`, loads the graph directly from memory into `onnxruntime.InferenceSession` (default `["CPUExecutionProvider"]`), and reconstructs `ModelSchema`.

### `LoadedONNXBundle`

```python
class LoadedONNXBundle:
    session: ort.InferenceSession
    manifest: dict[str, Any]
    schema: ModelSchema | None
    config: dict[str, Any] | str | None
    bundle_path: Path | None
    input_names: list[str]
    output_names: list[str]
```

| Member | Behavior |
| :--- | :--- |
| `predict(input_data: Any, validate: bool = True) -> Any` | Validates input against `ModelSchema`, runs ONNX Runtime inference, and reconstructs `tensor`, `sequence`, or `mapping` outputs. |
| `__call__(input_data, validate=True)` | Alias for `predict()`. |
| `get_runner()` | Returns `self`. |

### `ONNXRunner`

```python
class ONNXRunner(LoadedONNXBundle):
    def __init__(
        self,
        model_path_or_bundle: str | Path,
        schema: ModelSchema | None = None,
        config: dict[str, Any] | str | None = None,
        providers: list[str] | None = None,
    ) -> None
```

Accepts **either** a standalone `.onnx` file path or a `.onnx.bundle` archive. Also provides `ONNXRunner.from_bundle(bundle_path, providers=None)`.

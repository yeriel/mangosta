# API Reference: Slim Serving (`mangosta.serving`)

The serving module enables zero-PyTorch production model serving using ONNX Runtime. Microservices and container images can discard PyTorch (`~2.5GB`), CUDA toolkits, and compilation toolchains, deploying ultralight containers (`~50MB`) while retaining full contract validation.

---

## `load_onnx_bundle`

```python
def load_onnx_bundle(
    bundle_path: str | Path,
    providers: list[str] | None = None,
) -> LoadedONNXBundle:
```

Load an ONNX bundle archive (`.onnx.bundle`) with cryptographic validation and contract reconstruction.

### Features
- **Zero-PyTorch**: Requires only `onnxruntime` and standard library modules (`zipfile`, `hashlib`, `json`).
- **Integrity Verification**: Verifies SHA-256 fingerprint of `model.onnx` before loading.
- **In-Memory Loading**: Directly mounts ONNX model bytes into the ONNX Runtime C++ engine without extracting temporary files to disk.

---

## `LoadedONNXBundle`

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

Encapsulates a loaded ONNX model bundle ready for forward inference.

### Methods
- **`predict(input_data: Any, validate: bool = True) -> Any`**:
  Execute inference against the underlying ONNX Runtime session.
  - Automatically converts inputs (NumPy array, torch.Tensor, list, or dict) to the expected input feed format.
  - Validates shapes and dtypes against `ModelSchema` when `validate=True`.
  - Reconstructs original output structures (`tensor`, `sequence`, or `mapping`) according to the schema contract.
- **`__call__(input_data: Any, validate: bool = True) -> Any`**:
  Alias for `predict()`.
- **`get_runner() -> LoadedONNXBundle`**:
  Self-reference convenience method.

---

## `ONNXRunner`

```python
class ONNXRunner(LoadedONNXBundle):
    def __init__(
        self,
        model_path_or_bundle: str | Path,
        schema: ModelSchema | None = None,
        config: dict[str, Any] | str | None = None,
        providers: list[str] | None = None,
    ) -> None:
```

Universal runner supporting either standalone `.onnx` files or `.onnx.bundle` archives.

### Class Methods
- **`from_bundle(bundle_path: str | Path, providers: list[str] | None = None) -> LoadedONNXBundle`**:
  Load from `.onnx.bundle` archive.
- **`from_model_path(model_path: str | Path, providers: list[str] | None = None) -> ONNXRunner`**:
  Load directly from a standalone `.onnx` file.

# API Reference: Artifacts & Bundles (`mangosta.artifacts`)

The artifacts module handles atomic packaging, cryptographic verification, and resumption for PyTorch models and training checkpoints using standard ZIP archives and SafeTensors format (zero-pickle execution).

---

## `save_bundle`

```python
def save_bundle(
    output_path: str | Path,
    model: nn.Module,
    family: str | None = None,
    version: str = "0.1.0",
    init_args: dict[str, Any] | None = None,
    is_checkpoint: bool = False,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int | None = None,
    step: int | None = None,
    sample_input: Any | None = None,
    schema: ModelSchema | None = None,
    dataset_hash: str | None = None,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
```

Atomically pack a trained PyTorch model and its complete lifecycle metadata into a self-contained `.bundle` file.

### Smart Defaults & Guarantees
- **Atomic Creation**: Writes to a temporary `.tmp_<pid>_<uuid>` file and performs an atomic filesystem rename (`os.replace`) upon completion.
- **Zero-Pickle Weights**: Model weights are serialized exclusively using Hugging Face SafeTensors (`model.safetensors`).
- **Cryptographic Fingerprint**: Computes SHA-256 hash of the weights and writes it into `manifest.json`.
- **Automatic Schema Inference**: If `sample_input` is provided and `schema` is omitted, `infer_schema(model, sample_input)` is invoked automatically.
- **Smart Family Fallback**: Defaults to `model.__class__.__name__.lower()` if `family` is omitted.
- **Checkpoint Detection**: Automatically sets `is_checkpoint = True` if `optimizer`, `scheduler`, or `scaler` is supplied.

### Resumption States
When `is_checkpoint=True`, the bundle saves:
- Optimizer state dict (`optimizer.state_dict()`)
- Learning rate scheduler state dict (`scheduler.state_dict()`)
- PyTorch AMP GradScaler state dict (`scaler.state_dict()`)
- RNG states: PyTorch CPU, PyTorch CUDA (all devices), Python `random`, and NumPy `np.random`

---

## `load_bundle`

```python
def load_bundle(
    bundle_path: str | Path,
    map_location: Any = "cpu",
    model: nn.Module | None = None,
) -> LoadedBundle | LoadedONNXBundle:
```

Universal loader for `.bundle` archives.

### Behaviors
1. **Cryptographic Integrity Verification**: Verifies SHA-256 of `model.safetensors` against `manifest.json`. If tampered or corrupted, raises `BundleCorruptedError`.
2. **Architecture Reconstruction**: If `model` is omitted, resolves `family` and `version` from `ModelRegistry` and instantiates the class using `manifest["init_args"]`.
3. **ONNX Detection**: If the archive is an ONNX bundle (`.onnx.bundle` or `weights_format == 'onnx'`), routes automatically to `load_onnx_bundle` without requiring PyTorch.

---

## `resume_training`

```python
def resume_training(
    bundle_path: str | Path,
    model: nn.Module,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    restore_rng: bool = True,
    map_location: Any = "cpu",
) -> ResumeState:
```

Restore complete training state from a checkpoint bundle for exact bit-to-bit resumption after preemption or SIGINT.

### Returns
`ResumeState` dataclass containing:
- `epoch`: `int | None` (the epoch at checkpoint time)
- `step`: `int | None` (the step at checkpoint time)
- `metrics`: `dict[str, float]` (metrics recorded at checkpoint time)

---

## `LoadedBundle`

```python
class LoadedBundle:
    model: nn.Module
    manifest: dict[str, Any]
    schema: ModelSchema | None
    config: dict[str, Any] | str | None
    bundle_path: Path | None
```

Wrapper around a loaded PyTorch model providing schema contract enforcement.

### Methods
- **`predict(input_data: Any, validate: bool = True) -> Any`**: Forward pass in `eval()` mode. Validates tensor shapes and dtypes against `self.schema` before execution.
- **`__call__(input_data: Any, validate: bool = True) -> Any`**: Alias for `predict()`.

---

## `read_bundle_manifest`

```python
def read_bundle_manifest(bundle_path: str | Path) -> dict[str, Any]:
```

Fast manifest reader. Extracts and parses `manifest.json` from a `.bundle` archive without decompressing model weights or allocating GPU/CPU tensor buffers.

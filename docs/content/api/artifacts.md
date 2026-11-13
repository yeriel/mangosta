# API Reference: Artifacts & Bundles (`mangosta.artifacts`)

Handles atomic `.bundle` ZIP packaging with zero-pickle SafeTensors weights, cryptographic SHA-256 verification, zero-torch metadata inspection, and bit-to-bit checkpoint recovery.

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
) -> Path
```

Atomically writes `.bundle` via temporary file + `os.replace`:

- Serializes weights into `model.safetensors` and hashes them (`SHA-256`) into `manifest.json`.
- Infers `ModelSchema` automatically when `sample_input` is provided and `schema` is omitted.
- Sets `is_checkpoint=True` automatically when `optimizer`, `scheduler`, or `scaler` is provided, saving `train_state.pt` with optimizer, scheduler, scaler, and 4 RNG states (PyTorch CPU, all CUDA devices, NumPy, Python).

---

## `load_bundle`

```python
def load_bundle(
    bundle_path: str | Path,
    map_location: Any = "cpu",
    model: nn.Module | None = None,
) -> LoadedBundle | LoadedONNXBundle
```

1. Verifies SHA-256 checksum (`BundleCorruptedError` on tampering/corruption).
2. Reconstructs the model from `ModelRegistry` using `family`, `version`, and `init_args` unless `model=` is supplied.
3. Automatically routes `.onnx.bundle` archives to `load_onnx_bundle` (zero PyTorch).

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
) -> ResumeState
```

Restores weights, optimizer/scheduler/scaler states, and RNG states from a checkpoint bundle. Raises `ResumeStateError` if `train_state.pt` is missing. Returns `ResumeState(epoch, step, metrics, rng_restored)`.

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

Methods: `predict(input_data: Any, validate: bool = True) -> Any`, `__call__(...)`.

---

## Zero-Torch Readers

### `read_bundle_manifest`

```python
def read_bundle_manifest(bundle_path: str | Path) -> dict[str, Any]
```

Extracts and parses `manifest.json` without importing PyTorch or reading weights.

### `inspect_bundle_metadata`

```python
def inspect_bundle_metadata(bundle_path: str | Path) -> dict[str, Any]
```

Parses `manifest.json`, optional `config.yaml`, ZIP file list, and the 8-byte binary SafeTensors header **without importing PyTorch**, returning `total_params`, `weights_size_bytes`, and per-tensor metadata (`name`, `shape`, `dtype`, `param_count`, `size_bytes`).

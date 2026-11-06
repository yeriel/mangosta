# API Reference: Tracking (`mangosta.tracking`)

The tracking module provides zero-daemon, thread-safe, and crash-consistent experiment logging for PyTorch training pipelines. All metadata, step metrics, and validation artifacts are stored as plain files (`run.json`, `metrics.jsonl`, `artifacts/`) directly in your workspace.

---

## `RunTracker`

```python
class RunTracker:
    def __init__(
        self,
        experiment: str = "default",
        run_name: str | None = None,
        root_dir: str | Path = "./runs",
        auto_git: bool = True,
    ) -> None:
```

Context manager and logger for experiment runs.

### Parameters
- **`experiment`** (*str*, default `"default"`): High-level experiment name or grouping (e.g. `"vision_baseline"`).
- **`run_name`** (*str | None*, optional): Unique name for this run. If omitted, generates a timestamped identifier (e.g. `"run_20261106_143000"`).
- **`root_dir`** (*str | Path*, default `"./runs"`): Directory where experiments and runs are stored.
- **`auto_git`** (*bool*, default `True`): When `True`, captures git commit hash, branch name, and dirty status without modifying git history.

### Context Management
`RunTracker` implements the context manager protocol (`__enter__` and `__exit__`). Upon clean exit, the run status in `run.json` is set to `"completed"`. If an unhandled exception occurs, the status is set to `"failed"` and the traceback is recorded before re-raising the exception.

```python
from mangosta import RunTracker

with RunTracker(experiment="cifar10", run_name="resnet18_lr1e-3") as run:
    run.log_params({"lr": 1e-3, "batch_size": 64})
    for epoch in range(5):
        run.log_metric("train/loss", 0.42, step=epoch)
```

---

### Methods

#### `log_param(key: str, value: Any) -> None`
Log a single parameter or hyperparameter to `run.json`.
- **`key`**: Name of the hyperparameter.
- **`value`**: JSON-serializable value (strings, numbers, booleans, lists, dicts).

#### `log_params(params: Mapping[str, Any]) -> None`
Log multiple hyperparameters atomically to `run.json`.
- **`params`**: Dictionary of key-value pairs.

#### `log_metric(name: str, value: float | int, step: int | None = None) -> None`
Append a single metric observation with timestamp and optional step to `metrics.jsonl`.
- **`name`**: Metric identifier (e.g. `"train/loss"`, `"val/accuracy"`).
- **`value`**: Numeric scalar value.
- **`step`**: Integer step or epoch index.

#### `log_metrics(metrics: Mapping[str, float | int], step: int | None = None) -> None`
Append multiple metric observations sharing the same step atomically.

#### `log_artifact(local_path: str | Path, artifact_name: str | None = None) -> Path`
Copy a file (e.g. CSV, confusion matrix JSON, plot) into the run's `artifacts/` folder and track its SHA-256 fingerprint in `run.json`.
- **`local_path`**: Path to the source file on disk.
- **`artifact_name`**: Optional destination name inside `artifacts/`. Defaults to the filename.

#### `log_image(name: str, image: Any, step: int | None = None) -> Path`
Save an evaluation image into `artifacts/` and register it in `run.json`.
- **`name`**: Name prefix for the image.
- **`image`**: Supported inputs:
  - `torch.Tensor` (shape `[C, H, W]`, `[1, H, W]`, `[3, H, W]`, or `[H, W]`, float `[0, 1]` or uint8 `[0, 255]`)
  - `numpy.ndarray` (shape `[H, W]`, `[H, W, 1]`, `[H, W, 3]`, or `[H, W, 4]`)
  - Existing image file path (`.png`, `.jpg`, `.jpeg`, `.ppm`)
- **`step`**: Optional training step index appended to the filename.

#### `set_dataset(path: str | Path, compute_hash: bool = True) -> str`
Record the dataset path and calculate its cryptographic SHA-256 fingerprint (streaming chunks to avoid memory overhead).
- Returns the hexadecimal SHA-256 digest string.

---

## `quick_track`

```python
def quick_track(
    experiment: str = "default",
    run_name: str | None = None,
    root_dir: str | Path = "./runs",
    params: dict[str, Any] | None = None,
) -> RunTracker:
```

Factory helper returning a ready-to-use `RunTracker` instance with optional pre-populated hyperparameters.

---

## Utility Functions

### `compute_sha256(file_path: str | Path, chunk_size: int = 65536) -> str`
Stream file contents and return its SHA-256 checksum string. Memory footprint is strictly bounded to `chunk_size` bytes.

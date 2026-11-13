# API Reference: Tracking & Viewer (`mangosta.tracking`)

Zero-daemon, thread-safe local experiment tracking (`run.json`, `metrics.jsonl`, `artifacts/`) and the ephemeral `mangosta view` web server.

---

## `RunTracker`

```python
class RunTracker:
    def __init__(
        self,
        experiment: str = "default",
        run_name: str | None = None,
        root_dir: str | Path = "experiments",
        sync_interval: int = 10,
    ) -> None
```

Writes run logs under `<root_dir>/<experiment>/<run_name>/` and installs signal handlers (`SIGINT`, `SIGTERM`) on the main thread so interrupted runs are marked `"interrupted"` cleanly.

### Context Manager Protocol

```python
from mangosta import RunTracker

with RunTracker(experiment="cifar10", run_name="resnet18_lr1e-3") as run:
    run.log_params({"lr": 1e-3, "batch_size": 64})
    run.log_metrics({"train_loss": 0.42}, step=1)
```

On normal exit, status becomes `"completed"`. On an unhandled exception, status is recorded as `"failed"` before re-raising.

### Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| `log_params` | `(params: dict[str, Any]) -> None` | Update and atomically save parameters into `run.json`. |
| `set_dataset` | `(dataset_path: str \| Path) -> str` | Compute SHA-256 of a dataset file or directory tree and store in `run.json`. |
| `log_metrics` | `(metrics: dict[str, float \| int], step: int \| None = None, timestamp: float \| None = None) -> None` | Append step metrics atomically to `metrics.jsonl`. Syncs disk every `sync_interval` writes. |
| `log_artifact` | `(source_path: str \| Path, name: str \| None = None) -> Path` | Copy any file into `artifacts/` and register it in `run.json`. |
| `log_image` | `(name: str, image: Any, step: int \| None = None, tag: str \| None = None, sample_id: str \| None = None) -> Path` | Save PNG (stdlib zlib, or PPM fallback) from `torch.Tensor`, `np.ndarray`, or image path into `artifacts/`. |
| `log_validation_sample` | `(sample_id: str, prediction: Any, input_image: Any \| None = None, ground_truth: Any \| None = None, step: int \| None = None) -> dict[str, Path]` | Save `Input` / `Ground Truth` / `Prediction` comparisons per step for the **Validation** timelapse viewer. |
| `get_metrics` | `() -> dict[str, list[dict[str, Any]]]` | Read logged series back as `{metric_name: [...]}`. |
| `flush_disk` | `() -> None` | Flush buffered metric writes explicitly. |
| `finish` | `(status: str = "completed") -> None` | Mark run status and write final timestamps (called automatically on `__exit__`). |

---

## `quick_track`

```python
@contextmanager
def quick_track(
    run_name: str | None = None,
    experiment: str = "default",
    root_dir: str | Path = "experiments",
    sync_interval: int = 10,
) -> Iterator[RunTracker]
```

Lightweight context manager helper:

```python
from mangosta import quick_track

with quick_track(run_name="baseline_exp") as run:
    run.log_params({"lr": 0.001})
    run.log_metrics({"loss": 0.42}, step=0)
```

---

## `start_viewer`

```python
def start_viewer(
    target_path: str | Path = "experiments",
    port: int = 8501,
    open_browser: bool = True,
) -> None
```

Starts the ephemeral stdlib `http.server` dashboard (also launched via `uv run mangosta view`).

---

## `compute_sha256`

```python
def compute_sha256(path: str | Path, chunk_size: int = 65536) -> str
```

Streams a single file or recursively walks a directory sorted by relative path to produce a deterministic SHA-256 fingerprint without loading large datasets into RAM.

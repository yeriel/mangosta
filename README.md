# mangosta

**Lightweight MLOps and Model Lifecycle Management for PyTorch.**

> *"Simple by default, strict on demand"* - Functional in three lines of code for rapid experimentation, yet robust enough for production pipelines and autonomous AI agents through formal contracts, deterministic fault tolerance, and immutable architecture versioning.

---

## Overview

Modern MLOps frameworks frequently impose heavy operational complexity: daemon background services, external databases, cloud platform dependencies, and brittle networking setups.

`mangosta` is engineered to provide local-first, serverless model lifecycle primitives directly within Python and standard file systems. It eliminates infrastructure overhead while delivering strict guarantees regarding reproducibility, tensor contracts, and crash-consistent persistence.

### Architectural Principles

- **Zero Service Dependencies**: No background daemons, web servers, or remote databases. All operations use direct, atomic file system calls (JSONL, standard ZIP archives, and optional SQLite).
- **Strict Immutability**: Architectures registered in `ModelRegistry` cannot be silently overwritten, preventing silent behavioral divergence between training runs.
- **Contract-Driven Inference**: Non-invasive tensor signature inference captures shapes (including dynamic batch dimensions) and data types, enforcing runtime verification before execution.
- **Deterministic Fault Tolerance**: Checkpoints capture complete training state including optimizer parameters, learning rate schedulers, gradient scalers, and pseudo-random number generator (RNG) states across Python, PyTorch (CPU and CUDA), and NumPy.
- **Decoupled Production Export**: Development artifacts (`.bundle`) remain optimized for training continuity and auditability, while standalone production formats (such as ONNX) are cleanly exported with full user control over dynamic axes, precision, and output restructuring.

---

## Comparison with Existing Tools

| Capability | mangosta | MLflow | Weights & Biases | ClearML |
| :--- | :--- | :--- | :--- | :--- |
| **Infrastructure Overhead** | None (Local files only) | Local/Remote server | Cloud daemon / SaaS | Server / Agent daemon |
| **External Services Required** | Zero | Web server + DB | API backend | Server stack + DB |
| **Persistence Mechanism** | Atomic JSONL & ZIP | SQLite / File store | Cloud storage | MongoDB / MinIO |
| **Fault-Tolerant RNG Restoration** | Native (CPU, CUDA, NumPy, Py) | Manual | Manual | Manual |
| **Tensor Contract Validation** | Native (Pre-inference) | Signature schema | Logged schema only | Logged schema only |
| **Packaging Architecture** | Modular (KISS & SOLID) | Monolithic package | Monolithic client | Complex SDK |
| **Offline Operation** | 100% Offline | Partial | Requires offline mode | Requires local server |

---

## Installation

Install using `uv` (recommended):

```bash
uv add mangosta
```

Or using standard `pip`:

```bash
pip install mangosta
```

### Optional Dependencies

For ONNX graph verification using `onnx.checker`:

```bash
uv add "mangosta[onnx]"
# or
pip install "mangosta[onnx]"
```

---

## Quickstart

### 1. Atomic Local Tracking

Track hyperparameters and step-level metrics with atomic disk flushing:

```python
from mangosta import quick_track

with quick_track(experiment="vision", run_name="baseline_resnet") as run:
    run.log_params({"learning_rate": 1e-3, "batch_size": 32, "weight_decay": 1e-4})
    run.set_dataset("data/processed_images.parquet")
    
    for epoch in range(10):
        # Training iteration ...
        run.log_metric("train_loss", 0.342, step=epoch)
        run.log_metric("val_accuracy", 0.912, step=epoch)
```

### 2. Architecture Registration

Register neural network architectures under a family name and semantic version:

```python
import torch
import torch.nn as nn
from mangosta import register_model

@register_model(family="classifier", version="1.0.0")
class ConvClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).flatten(1))
```

### 3. Checkpointing and Fault-Tolerant Resumption

Save complete training state including optimizer, scheduler, and all random generator states:

```python
from mangosta import save_bundle, resume_training, ModelRegistry

# Save crash-consistent checkpoint
save_bundle(
    output_path="checkpoints/epoch_5.bundle",
    model=model,
    family="classifier",
    version="1.0.0",
    init_args={"in_channels": 3, "num_classes": 10},
    sample_input=torch.randn(1, 3, 32, 32),
    is_checkpoint=True,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=5,
    step=1250,
)

# Resume deterministically following an interruption
fresh_model = ModelRegistry.build("classifier", version="1.0.0", in_channels=3, num_classes=10)
fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=5)

state = resume_training(
    bundle_path="checkpoints/epoch_5.bundle",
    model=fresh_model,
    optimizer=fresh_optimizer,
    scheduler=fresh_scheduler,
    restore_rng=True,
)

start_epoch = state.epoch + 1
start_step = state.step
```

### 4. Decoupled Production Inference

Load models dynamically without importing the underlying model class in your serving scripts:

```python
import torch
from mangosta import load_bundle

# Reconstructs architecture via registry and mounts weights
bundle = load_bundle("artifacts/model_final.bundle")

# Inference with automated input signature validation
input_tensor = torch.randn(4, 3, 32, 32)
predictions = bundle.predict(input_tensor)
```

### 5. Standalone ONNX Export

Export directly to standard ONNX with automatic dictionary-to-tuple output unrolling and float16 precision:

```python
from mangosta import export_onnx

export_onnx(
    model_or_bundle="artifacts/model_final.bundle",
    output_path="exports/model.onnx",
    sample_input_shape=(1, 3, 512, 512),
    output_mapping=["semantic_mask", "radius_map", "gradient_map"],
    input_names=["input0"],
    output_names=["output0", "output1", "output2"],
    dynamic_axes={
        "input0": {0: "batch", 2: "height", 3: "width"},
        "output0": {0: "batch", 2: "height", 3: "width"},
        "output1": {0: "batch", 2: "height", 3: "width"},
        "output2": {0: "batch", 2: "height", 3: "width"},
    },
    float16=True,
    opset_version=18,
    verify=True,
)
```

---

## Command Line Interface (CLI)

`mangosta` provides a zero-daemon CLI utility:

```bash
# Inspect internal bundle contents, tensor signatures, and metadata
mangosta inspect checkpoints/epoch_5.bundle

# Generate a self-contained, offline interactive HTML report
mangosta report experiments/vision/baseline_resnet -o report.html
mangosta report checkpoints/epoch_5.bundle -o report.html

# Launch an ephemeral browser viewer (terminates cleanly with Ctrl+C)
mangosta view ./experiments --port 8501

# Export a bundle directly to ONNX from the terminal
mangosta export-onnx checkpoints/epoch_5.bundle -o model.onnx --fp16 --input-shape 1 3 32 32 -m models.classifier
```

---

## Package Architecture

The library follows Single Responsibility (SRP) and Open-Closed (OCP) design principles:

```text
src/mangosta/
|-- __init__.py           # Unified high-level public API
|-- exceptions.py         # Domain-specific exception hierarchy
|-- core/                 # Architecture registry and tensor contracts
|   |-- registry.py       # ModelRegistry, @register_model, dynamic SemVer resolution
|   `-- schema.py         # TensorSpec, ModelSchema, infer_schema
|-- artifacts/            # Persistence, atomic packaging, and loading
|   |-- bundle.py         # Atomic ZIP creation, RNG serialization, checkpointing
|   `-- loader.py         # Decoupled reconstruction, LoadedBundle, resume_training
|-- tracking/             # Experiment tracking and lineage
|   |-- tracker.py        # RunTracker, quick_track, SHA-256 dataset fingerprinting
|   `-- report.py         # Offline interactive HTML report generator
|-- export/               # Standalone production exports
|   `-- onnx.py           # ONNX exporter, dictionary unrolling, precision handling
`-- cli/                  # Command-line interface
    `-- main.py           # CLI entrypoints (inspect, report, view, export-onnx)
```

---

## Development

Run tests and verification suites locally:

```bash
# Sync virtual environment
uv sync

# Run automated tests with coverage
uv run pytest -v --cov=mangosta

# Run linter and formatting checks
uv run ruff check src tests examples
uv run ruff format --check src tests examples

# Build package distributions (sdist and wheel)
uv build
```

---

## License

MIT License.

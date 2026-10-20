---
name: mangosta
description: >-
  Manages PyTorch model lifecycle without external daemons or web services:
  immutable architecture registration, contract-based tensor signature validation,
  atomic local experiment tracking, fault-tolerant checkpointing with RNG state restoration,
  and decoupled ONNX export. Activate this skill whenever the user is building, tracking,
  packaging, resuming, or exporting PyTorch models.
---

# mangosta: Agent Engineering Guide

`mangosta` is a zero-daemon, local-first library for managing the PyTorch model lifecycle.
It replaces complex infrastructure (MLflow, ClearML, W&B) with direct, atomic file system operations,
formal contracts, and deterministic training state persistence.

## Architecture and Responsibilities

When generating code or debugging with `mangosta`, adhere to its modular package boundaries:

- `mangosta.core`: Model architecture registration (`ModelRegistry`, `@register_model`) and tensor I/O contracts (`ModelSchema`, `TensorSpec`, `infer_schema`).
- `mangosta.artifacts`: Atomic ZIP `.bundle` creation (`save_bundle`), decoupled inference mounting (`load_bundle`), and fault-tolerant state recovery (`resume_training`).
- `mangosta.tracking`: Atomic step-level metrics logging (`RunTracker`, `quick_track`) and dataset fingerprinting (`compute_sha256`).
- `mangosta.export`: Standalone production export utilities (`export_onnx`) with dictionary-to-tuple output unrolling and float16 precision.
- `mangosta.cli`: Zero-daemon terminal commands (`inspect`, `report`, `view`, `export-onnx`).

All primary entrypoints are exposed at the top-level package namespace (`from mangosta import ...`).

---

## Agent Operational Rules

1. **Never Bypass the Registry for Production Bundles**:
   Always register neural network architectures using `@register_model(family="<name>", version="<semver>")`.
   This guarantees that `load_bundle()` can reconstruct the network without hardcoded class imports in serving scripts.

2. **Enforce Collision Protection**:
   `ModelRegistry.register_model()` rejects duplicate `(family, version)` pairs with `RegistryCollisionError`.
   Never overwrite versions silently; bump the semantic version (e.g., `1.0.0` -> `1.1.0`) unless explicitly permitted via `allow_override=True`.

3. **Separate Checkpoints from Final Inference Bundles**:
   - Training checkpoints: set `is_checkpoint=True`, pass `optimizer`, `scheduler`, and `scaler`. This serializes `train_state.pt` including random generator states across PyTorch, NumPy, and Python.
   - Inference bundles: set `is_checkpoint=False`. This eliminates optimizer weights and keeps artifacts lightweight.

4. **Verify Dynamic Batch Dimensions in Contracts**:
   Always pass `sample_input` to `save_bundle()` or use `infer_schema(model, sample_input)`.
   Dimension 0 is automatically converted to dynamic batching (`None`).

5. **Decouple Production Formats**:
   The `.bundle` archive is designed for internal development, training resumption, and contract verification.
   For deployment to Triton, TensorRT, or ONNX Runtime, use `export_onnx()` to create a standalone `.onnx` file outside the bundle.

---

## Canonical Patterns

### 1. Model Registration and Contract Definition

```python
import torch
import torch.nn as nn
from mangosta import register_model

@register_model(family="image_classifier", version="1.0.0")
class ImageClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).flatten(1))
```

### 2. Atomic Experiment Tracking

```python
from mangosta import quick_track

with quick_track(experiment="vision_models", run_name="exp_run_01") as run:
    # Compute SHA-256 fingerprint for data lineage audit
    data_hash = run.set_dataset("data/training_split.parquet")
    run.log_params({"lr": 1e-3, "batch_size": 32, "weight_decay": 1e-4})

    for step, loss_value in enumerate(train_losses):
        run.log_metric("train_loss", loss_value, step=step)
```

### 3. Checkpointing and Deterministic Resumption

```python
from mangosta import save_bundle, resume_training, ModelRegistry

# Saving mid-training checkpoint
save_bundle(
    output_path="checkpoints/checkpoint_epoch_10.bundle",
    model=model,
    family="image_classifier",
    version="1.0.0",
    init_args={"in_channels": 3, "num_classes": 10},
    sample_input=torch.randn(1, 3, 224, 224),
    is_checkpoint=True,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=10,
    step=5000,
)

# Resuming deterministically after interruption
fresh_model = ModelRegistry.build("image_classifier", version="1.0.0", in_channels=3, num_classes=10)
fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
fresh_sched = torch.optim.lr_scheduler.StepLR(fresh_opt, step_size=5)

state = resume_training(
    bundle_path="checkpoints/checkpoint_epoch_10.bundle",
    model=fresh_model,
    optimizer=fresh_opt,
    scheduler=fresh_sched,
    restore_rng=True,
)

start_epoch = state.epoch + 1
start_step = state.step
```

### 4. Decoupled Inference with Contract Validation

```python
import torch
from mangosta import load_bundle

bundle = load_bundle("artifacts/model_final.bundle")
# Validates shape and dtype against schema before executing forward pass
predictions = bundle.predict(torch.randn(8, 3, 224, 224))
```

### 5. Standalone ONNX Export with Dictionary Output Unrolling

When a model outputs a dictionary of tensors (e.g. `{"mask": ..., "coords": ...}`), pass `output_mapping`:

```python
from mangosta import export_onnx

export_onnx(
    model_or_bundle="artifacts/model_final.bundle",
    output_path="model.onnx",
    sample_input_shape=(1, 3, 512, 512),
    output_mapping=["mask", "coords"],
    input_names=["input0"],
    output_names=["output_mask", "output_coords"],
    dynamic_axes={
        "input0": {0: "batch", 2: "height", 3: "width"},
        "output_mask": {0: "batch", 2: "height", 3: "width"},
        "output_coords": {0: "batch"},
    },
    float16=True,
    verify=True,
)
```

---

## Exception Handling Guide

| Exception | Root Cause | Solution |
| :--- | :--- | :--- |
| `RegistryCollisionError` | Attempted to register already existing `(family, version)`. | Bump semantic version or set `allow_override=True`. |
| `ModelNotFoundError` | Family or version not registered in `ModelRegistry`. | Ensure class module is imported before `load_bundle()` or `build()`. |
| `SchemaValidationError` | Input tensor shape or dtype violates inferred contract. | Verify incoming tensor dimensionality against `bundle.schema`. |
| `ResumeStateError` | Bundle loaded with `resume_training` lacks `train_state.pt`. | Ensure bundle was saved with `is_checkpoint=True`. |
| `BundleCorruptedError` | Archive damaged or missing `manifest.json`/`weights.pt`. | Check storage integrity or re-export artifact. |

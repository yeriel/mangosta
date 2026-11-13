# Mangosta

**Lightweight MLOps and Model Lifecycle Management for PyTorch.**

Train with PyTorch. Track without daemons. Serve in `~70MB` containers without PyTorch.

---

## Key Features

- **Zero Daemons, Zero Servers**: All runs, metrics, and models live as atomic local files (`JSONL`, `run.json`, `.bundle` ZIPs) instead of background databases.
- **Zero-Pickle Weights**: Model weights serialize exclusively with Hugging Face **SafeTensors** and a verified SHA-256 checksum.
- **White-Box Training (`BaseTrainer`)**: Handles device resolution, AMP, gradient clipping, LR schedulers, `EarlyStopping`, tracking, and checkpointing without hiding your PyTorch code.
- **Deterministic Resume**: Checkpoints snapshot optimizer, scheduler, `GradScaler`, and 4 RNG states (PyTorch CPU, CUDA, NumPy, Python) for exact bit-to-bit recovery.
- **Shape & Dtype Contracts (`ModelSchema`)**: Invalid inputs fail immediately with `SchemaValidationError` before reaching your model.
- **Slim Production Serving**: Export `.bundle` archives to `.onnx.bundle` and run inference with pure NumPy (`mangosta[serving]`, no PyTorch installed).

---

## Installation

Mangosta separates heavy training tools from slim production serving:

=== "Training & Full Dev (`[full]`)"

    ```bash
    uv add "mangosta[full]"
    ```

    Includes PyTorch, SafeTensors, ONNX, and ONNX Runtime.

=== "Zero-PyTorch Serving (`[serving]`)"

    ```bash
    uv add "mangosta[serving]"
    ```

    Includes NumPy, ONNX Runtime, PyYAML, and `packaging` — **no PyTorch (`~70MB` container)**.

=== "Minimal Core (no extras)"

    ```bash
    uv add mangosta
    ```

    Schema contracts, strict config validation, and `<5ms` `.bundle` inspection (`mangosta inspect --json`).

---

## A 60-Second Example

### 1. Train & Package in PyTorch (`mangosta[full]`)

Define your model, subclass `BaseTrainer`, and call `.fit()`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mangosta import BaseTrainer, load_config, register_model, register_trainer

@register_model(family="mlp", version="1.0.0")  # (1)!
class MLP(nn.Module):
    def __init__(self, in_features: int = 8, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@register_trainer(family="mlp", version="1.0.0")
class MLPTrainer(BaseTrainer):  # (2)!
    def __init__(self, model: nn.Module, config=None, **kwargs) -> None:
        super().__init__(model, config, criterion=nn.CrossEntropyLoss(), **kwargs)

cfg = load_config({
    "experiment": {"name": "quickstart", "seed": 42},
    "model": {"family": "mlp", "version": "1.0.0"},
    "training": {"epochs": 3, "learning_rate": 1e-3},
})
x, y = torch.randn(64, 8), torch.randint(0, 3, (64,))
loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

trainer = MLPTrainer(MLP(), config=cfg)
result = trainer.fit(train_loader=loader, val_loader=loader)  # (3)!
```

1. Registers `MLP` with SemVer so checkpoints can reconstruct the exact architecture later without `pickle`.
2. Passing `criterion=` to `BaseTrainer` gives you `training_step` and `validation_step` out of the box—or override them directly for custom losses/metrics.
3. Automatically logs `metrics.jsonl`, captures git/dataset provenance, and writes `model_checkpoint.bundle`, `model_best.bundle`, and `model_final.bundle`.

### 2. Inspect, View & Export from the CLI

```bash
# Inspect any .bundle in <5ms without importing PyTorch
uv run mangosta inspect checkpoints/model_final.bundle --json

# Open the interactive dashboard (metrics, model comparison, validation timelapses & bundle diff)
uv run mangosta view

# Export to a zero-PyTorch ONNX bundle (shape auto-inferred from ModelSchema)
uv run mangosta export-onnx checkpoints/model_final.bundle -o deploy/model.onnx.bundle -m my_models
```

### 3. Serve in Production Without PyTorch (`mangosta[serving]`)

```python
import numpy as np
from mangosta import load_onnx_bundle

runner = load_onnx_bundle("deploy/model.onnx.bundle")
preds = runner.predict(np.random.randn(4, 8).astype(np.float32))
```

!!! tip "Automatic Contract Validation"
    If a client sends shape `(4, 5)` instead of `(4, 8)`, `runner.predict(...)` rejects it immediately with `SchemaValidationError` before invoking ONNX Runtime.

---

## Where to Go Next

Start with the step-by-step **Tutorial - User Guide** to build your first project from scratch:

1. **[First Project (`mangosta init`)](tutorial/01_first_project.md)** — Scaffold a Clean Architecture repo with `AGENTS.md` and AI skill support.
2. **[Config & Registries](tutorial/02_config_and_registries.md)** — Strict YAML configs with typo hints and SemVer model/trainer catalogs.
3. **[Training with `BaseTrainer`](tutorial/03_training_with_basetrainer.md)** — White-box hooks, AMP, schedulers, and `EarlyStopping`.
4. **[Checkpoints & RNG Resume](tutorial/04_checkpoints_and_resume.md)** — Crash-safe atomic saves and bit-to-bit deterministic resumption.
5. **[Tracking & `mangosta view`](tutorial/05_visual_tracking_and_view.md)** — Compare runs/models, scrub validation timelapses, and inspect SafeTensors layers.
6. **[ONNX Export & Serving](tutorial/06_onnx_export_and_serving.md)** — Deploy slim zero-PyTorch microservices with shape contracts.

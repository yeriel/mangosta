# Guide 01: Lifecycle — Training vs. Production

> **Companion script:** [`examples/01_lifecycle_training_vs_production.py`](https://github.com/yeriel/mangosta/blob/main/examples/01_lifecycle_training_vs_production.py)  
> **Run it locally:** `uv run python examples/01_lifecycle_training_vs_production.py`

This guide walks through the core workflow of Mangosta: **how to train in PyTorch, strip training overhead, and serve in production with pure NumPy (`~70MB` container, zero PyTorch).**

---

## Step 1: Register Your Architecture (`@register_model`)

Decorate your `nn.Module` with `@register_model(family, version)`. This allows Mangosta to reconstruct the model later from `manifest.json` without using `pickle`:

```python
import torch
import torch.nn as nn
from mangosta import ModelRegistry, register_model

@register_model(family="tabular_classifier", version="1.0.0")
class TabularClassifier(nn.Module):
    def __init__(self, in_features: int = 8, hidden_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Instantiate from the registry
model = ModelRegistry.build(
    "tabular_classifier",
    version="1.0.0",
    in_features=8,
    hidden_dim=32,
    num_classes=2,
)
```

---

## Step 2: Train, Track & Save a Checkpoint (`.bundle`)

Use `RunTracker` as a context manager to log hyperparameters, step metrics (`metrics.jsonl`), and validation samples. Pass `optimizer` and `scheduler` to `save_bundle` to create a resumable checkpoint:

```python
import numpy as np
import torch.optim as optim
from mangosta import RunTracker, save_bundle

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
criterion = nn.CrossEntropyLoss()

with RunTracker(experiment="lifecycle_tutorial", run_name="exp_run_01") as tracker:
    tracker.log_params({"lr": 1e-3, "optimizer": "AdamW", "batch_size": 16})

    for epoch in range(1, 4):
        model.train()
        inputs = torch.randn(16, 8)
        targets = torch.randint(0, 2, (16,))

        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 1. Log scalar metric
        tracker.log_metrics({"train/loss": float(loss.item())}, step=epoch)

        # 2. Log visual validation sample for mangosta view
        sample_pred = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        tracker.log_validation_sample(sample_id="scan_01", prediction=sample_pred, step=epoch)

    # Save mid-training checkpoint (includes SafeTensors weights + optimizer + 4 RNG states)
    save_bundle(
        output_path="checkpoint_epoch3.bundle",
        model=model,
        family="tabular_classifier",
        version="1.0.0",
        init_args={"in_features": 8, "hidden_dim": 32, "num_classes": 2},
        sample_input=torch.randn(1, 8),
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        step=15,
    )
```

---

## Step 3: Transition to Production (Strip Training State)

Production microservices should never carry optimizer states or require a `2.5GB` PyTorch installation. Export a clean inference bundle or an ONNX bundle (`.onnx.bundle`):

```python
from mangosta.export import export_onnx_as_bundle

# Option A: Lightweight PyTorch Inference Bundle (omits optimizer/scheduler)
save_bundle(
    output_path="model_final.bundle",
    model=model,
    family="tabular_classifier",
    version="1.0.0",
    init_args={"in_features": 8, "hidden_dim": 32, "num_classes": 2},
    sample_input=torch.randn(1, 8),
    metrics={"final_accuracy": 0.965},
)

# Option B: Production ONNX Bundle (for Zero-PyTorch serving)
export_onnx_as_bundle(
    model_or_bundle="model_final.bundle",
    output_path="model_prod.onnx.bundle",
    sample_input=torch.randn(1, 8),
    opset_version=17,
)
```

---

## Step 4: Serve in Production with Pure NumPy (`mangosta[serving]`)

In your production container (`uv add "mangosta[serving]"`), load `model_prod.onnx.bundle` with `load_onnx_bundle`. It verifies the SHA-256 checksum and enforces the `ModelSchema` contract (`[None, 8]`, `float32`) before running ONNX Runtime:

```python
import numpy as np
from mangosta.serving import load_onnx_bundle

runner = load_onnx_bundle("model_prod.onnx.bundle")

# Dynamic batching works automatically (batch sizes 1, 4, 16...)
for batch_size in [1, 4, 16]:
    x_numpy = np.random.randn(batch_size, 8).astype(np.float32)
    preds = runner.predict(x_numpy)
    print(f"Batch {batch_size} -> Output shape: {preds.shape}")

# Invalid shapes or dtypes are rejected BEFORE hitting the C++ engine:
bad_input = np.random.randn(4, 5).astype(np.float32)
# runner.predict(bad_input)  # Raises SchemaValidationError!
```

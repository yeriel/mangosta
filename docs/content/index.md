# Mangosta

> **Lightweight MLOps and Model Lifecycle Management for PyTorch.**  
> Train with PyTorch. Track without daemons. Serve in ~70MB containers without PyTorch.

---

## Quick Install (with `uv`)

Mangosta is designed to be used inside a **`uv`** virtual environment. It separates heavy training dependencies from lightweight production serving:

```bash
# 1. Create and activate a uv environment
uv venv
source .venv/bin/activate

# 2. Install full toolkit (Training + ONNX + Serving)
uv add "mangosta[all]"
```

*(Need only training or only a ~70MB zero-PyTorch production container? See [Installation & Flags](installation.md)).*

---

## What Can You Do with Mangosta?

In most ML projects, moving from a research script to production is painful: `.pt` files rely on insecure `pickle`, interrupted runs lose their random state, experiment trackers require heavy servers, and production Docker containers weigh `2.5GB+` because they carry PyTorch.

Mangosta solves the **entire lifecycle** with 4 simple primitives:

```mermaid
flowchart LR
    A["1. Train & Track\n@register_model + RunTracker"] --> B["2. Checkpoint & Resume\nsave_bundle + resume_training"]
    B --> C["3. Export & Inspect\nexport_onnx_as_bundle + inspect --json"]
    C --> D["4. Zero-Torch Serving\nload_onnx_bundle (Pure NumPy)"]

    classDef default fill:#1e293b,stroke:#3b82f6,stroke-width:1px,color:#f8fafc;
    classDef highlight fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    class B,D highlight;
```

---

## 60-Second End-to-End Example

### Phase 1: Train & Save in PyTorch (`mangosta[train]`)

```python
import torch
import torch.nn as nn
from mangosta import ModelRegistry, RunTracker, register_model, save_bundle
from mangosta.export import export_onnx_as_bundle

# 1. Register architecture with semantic versioning
@register_model(family="tabular_mlp", version="1.0.0")
class TabularMLP(nn.Module):
    def __init__(self, in_features: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 2. Build model & track metrics locally (zero daemons)
model = ModelRegistry.build("tabular_mlp", version="1.0.0", in_features=8, num_classes=2)
sample_x = torch.randn(1, 8)

with RunTracker(experiment="quickstart", run_name="run_01") as tracker:
    tracker.log_params({"lr": 1e-3, "arch": "tabular_mlp"})
    tracker.log_metrics({"train/loss": 0.31}, step=1)

# 3. Save SafeTensors bundle & export ONNX production bundle
save_bundle("model.bundle", model=model, sample_input=sample_x)
export_onnx_as_bundle("model.bundle", "model.onnx.bundle", sample_input=sample_x)
```

### Phase 2: Serve in Production Without PyTorch (`mangosta[serving]`)

In your production container, you **do not need PyTorch installed**. `load_onnx_bundle` validates input shapes and dtypes automatically before running inference with pure NumPy:

```python
import numpy as np
from mangosta.serving import load_onnx_bundle

runner = load_onnx_bundle("model.onnx.bundle")

# Validates contract ([None, 8], float32) before running ONNX Runtime
batch = np.random.randn(4, 8).astype(np.float32)
predictions = runner.predict(batch)
```

### Phase 3: Inspect & Visualize from CLI

```bash
# Inspect bundle metadata, contract schema, and layers in < 5ms (zero PyTorch import)
uv run mangosta inspect model.bundle --json

# Launch zero-daemon local dashboard (curves + validation image timelapses)
uv run mangosta view
```

---

## Follow the Learning Path

Everything in Mangosta is taught through 4 progressive steps matching the runnable scripts in [`examples/`](examples.md):

| Step | Guide | What You Learn |
| :-: | :--- | :--- |
| **01** | **[Lifecycle: Training vs. Production](guides/01_fast_prototyping.md)** | How to separate PyTorch training (`.bundle`) from Zero-PyTorch serving (`.onnx.bundle`). |
| **02** | **[Checkpoints & Deterministic Resume](guides/02_fault_tolerance_and_resumption.md)** | How to save crash-consistent checkpoints and resume bit-to-bit (including CPU/CUDA/NumPy RNG). |
| **03** | **[Export, `<5ms` Inspection & Reports](guides/03_production_serving_and_contracts.md)** | How to export multi-output models to ONNX, inspect headers in `<5ms`, and generate offline HTML reports. |
| **04** | **[Visual Tracking & `mangosta view`](guides/05_visualization_and_timelapses.md)** | How to log scalar curves and validation sample timelapses (Input vs. GT vs. Prediction). |

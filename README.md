# mangosta

**Lightweight MLOps and Model Lifecycle Management for PyTorch.**

> *"Simple by default, strict on demand"* — Instant setup for solo experimentation and research, engineered with the architectural rigor required by production pipelines and autonomous AI coding agents: formal tensor contracts, deterministic fault tolerance, zero-daemon persistence, and machine-readable CLI interfaces.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-informational.svg)](https://mypy-lang.org/)
[![Documentation](https://img.shields.io/badge/docs-yeriel.github.io%2Fmangosta-blue)](https://yeriel.github.io/mangosta/)

---

## Why Mangosta?

Modern MLOps stacks frequently impose overwhelming operational complexity: background daemons, external database engines, cloud platform lock-in, and fragile networking.

`mangosta` provides **local-first, serverless model lifecycle primitives** directly within Python and standard file systems. It eliminates infrastructure overhead while delivering strict guarantees regarding reproducibility, tensor contracts, and crash-consistent persistence.

```mermaid
flowchart LR
    A["Scaffold Project<br/><code>uv run mangosta init</code>"] --> B["Train & Track<br/><code>RunTracker (JSONL)</code>"]
    B --> C["Atomic Bundle<br/><code>.bundle (SafeTensors)</code>"]
    C --> D["Inspection & Audit<br/><code>uv run mangosta inspect --json</code>"]
    C --> E["Production Serving<br/><code>.onnx.bundle (Zero-Torch)</code>"]
```

### Architectural Principles

- **Zero Service Dependencies**: No background daemons, Redis, or databases. Everything is stored in human-readable, crash-consistent flat files (JSONL, standard ZIP archives, SafeTensors).
- **Zero-Pickle Security**: Weights are serialized exclusively using Hugging Face SafeTensors (`model.safetensors`). Arbitrary code execution during deserialization is physically impossible.
- **Contract-Driven Inference**: Non-invasive tensor signatures validate shapes (including dynamic batch dimensions) and data types before runtime execution.
- **Deterministic Bit-to-Bit Resumption**: Captures complete training snapshots (optimizer, scheduler, GradScaler, and RNG states across PyTorch CPU/CUDA, Python, and NumPy).
- **Slim Production Serving**: Deploy high-performance ONNX microservices (`~70MB` Docker containers) without installing PyTorch (`~2.5GB`).
- **AI Agent-Native**: Machine-readable `uv run mangosta inspect <bundle> --json` parses SafeTensors binary headers in `< 5ms` with zero tensor allocations, tailored for LLMs and CI/CD pipelines.

---

## Technical Comparison

| Capability | **mangosta** | **Vanilla PyTorch** | **MLflow** | **Weights & Biases** |
| :--- | :--- | :--- | :--- | :--- |
| **Operational Overhead** | **Zero** (Local files only) | None | Local/Remote server | Cloud daemon / SaaS |
| **External Dependencies** | **None** (Self-contained) | None | Web server + DB | API backend |
| **Persistence Mechanism** | **Atomic JSONL & ZIP** | Ad-hoc `.pt` / `.pth` | SQLite / File store | Cloud storage |
| **Pickle Security** | **100% SafeTensors (Zero Pickle)** | Vulnerable (`pickle`) | Ad-hoc (usually pickle) | Ad-hoc (usually pickle) |
| **Deterministic RNG Recovery** | **Native** (CPU, CUDA, NumPy, Py) | Manual code required | None (manual) | None (manual) |
| **Tensor Contract Validation** | **Native** (Pre-inference) | Manual assertions | Model signatures | None |
| **Machine-Readable CLI** | **`--json` in < 5ms** | None | Limited CLI | CLI client |
| **Slim Microservice Image** | **~70MB (Zero-PyTorch)** | ~2.5GB - 6GB | ~2.5GB - 6GB | ~2.5GB - 6GB |
| **100% Offline Capability** | **Yes** | Yes | Partial | Offline queue mode |

---

## Installation (with `uv`)

Mangosta standardizes on **`uv`** as the package and environment manager.

### Recommended for Development: `mangosta[full]`

For local development, research, training, and ONNX export, install the complete toolkit:

```bash
# In your project:
uv add "mangosta[full]"
```

### Installation Targets

| Target | Command | What It Installs | When to Use |
| :--- | :--- | :--- | :--- |
| **Full Toolkit** *(Recommended)* | `uv add "mangosta[full]"` | PyTorch, SafeTensors, ONNX, ONNX Runtime | **Development, training, and ONNX export** |
| **Production Serving** | `uv add "mangosta[serving]"` | ONNX Runtime, NumPy, PyYAML | **Zero-PyTorch containers (`~70MB`)** |
| **Minimal Base** | `uv add mangosta` | NumPy, PyYAML, Packaging | CLI scaffolding (`init`) & JSON inspection only |

---

## 3-Minute Quickstart

### 1. Scaffold a Clean Architecture Project

Run the scaffolding CLI command using `uv run`:

```bash
uv run mangosta init my_project
cd my_project
```

Scaffolds a modular project with `configs/`, `src/`, `entrypoints/`, `AGENTS.md`, and `.agents/skills/mangosta/`.

### 2. Register Architecture & Track Training

```python
import torch
import torch.nn as nn
from mangosta import register_model, RunTracker, save_bundle

# 1. Register model architecture with semantic versioning
@register_model(family="sensor_mlp", version="1.0.0")
class SensorMLP(nn.Module):
    def __init__(self, in_features: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

model = SensorMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 2. Track experiment with atomic local flushing
with RunTracker(experiment="sensor_exp", run_name="baseline") as run:
    run.log_params({"lr": 1e-3, "batch_size": 32})

    for epoch in range(5):
        loss = 0.42 / (epoch + 1)
        run.log_metrics({"train/loss": loss}, step=epoch)

        # Log validation sample for visual inspection in mangosta view (Input vs GT vs Pred)
        run.log_validation_sample(
            sample_id="sample_01",
            prediction=torch.rand(3, 32, 32),
            input_image=torch.rand(3, 32, 32),
            step=epoch,
        )

    # 3. Save atomic SafeTensors bundle with contract and checkpoint state
    save_bundle(
        output_path="checkpoints/epoch_5.bundle",
        model=model,
        sample_input=torch.randn(1, 8),
        optimizer=optimizer,
        epoch=5,
    )
```

### 3. Inspect in Terminal or Browser

All Mangosta CLI commands run via `uv run mangosta`:

```bash
# Human-readable terminal summary
uv run mangosta inspect checkpoints/epoch_5.bundle

# Machine-readable JSON summary for AI agents or CI/CD (< 5ms, zero PyTorch allocation)
uv run mangosta inspect checkpoints/epoch_5.bundle --json

# Launch zero-daemon local visual dashboard (Ctrl+C stops cleanly)
uv run mangosta view
```

### 4. Zero-PyTorch Production Serving

Export the bundle to an ONNX bundle:

```bash
uv run mangosta export-onnx checkpoints/epoch_5.bundle -o exports/model.onnx.bundle --input-shape 1 8 -m src.models
```

Serve in production without installing PyTorch:

```python
from mangosta.serving import load_onnx_bundle
import numpy as np

# In production container (requires only `uv add "mangosta[serving]"`, ~70MB container):
runner = load_onnx_bundle("exports/model.onnx.bundle")

# Automated tensor shape & dtype validation against ModelSchema contract
features = np.random.randn(4, 8).astype(np.float32)
predictions = runner.predict(features)
```

---

## Command Line Interface (CLI)

Mangosta commands are invoked inside your project using `uv run mangosta <command>`:

| Command | Description |
| :--- | :--- |
| `uv run mangosta init [name]` | Scaffold a production-grade Clean Architecture project (`uv sync` runs automatically). |
| `uv run mangosta inspect <bundle> [--diff <other>] [--config] [--json]` | Inspect or compare `.bundle` archives, tensor layers, and contracts without importing PyTorch. |
| `uv run mangosta view [target] [--port 8501]` | Start the ephemeral local visual dashboard (Real Data Timelapse, A/B/GT grid, Multi-Run & Bundle Compare, EMA smoothing, Live Watch). |
| `uv run mangosta export-onnx <bundle> -o <out>` | Export bundle directly to standalone ONNX or `.onnx.bundle` (auto-infers `--input-shape` from `ModelSchema`). |

---

## Documentation

Full documentation, architecture guides, and API reference are available at:
👉 **[yeriel.github.io/mangosta](https://yeriel.github.io/mangosta/)**

To run the documentation locally:
```bash
uv --directory docs run zensical serve
```

---

## AI Agent Integration (Antigravity & Cursor Skill)

`mangosta` includes a native Agent Skill specification at [`src/mangosta/skills/mangosta/SKILL.md`](src/mangosta/skills/mangosta/SKILL.md). When copied to `.agents/skills/mangosta/SKILL.md` in your projects, AI assistants (Cursor, Claude Code, Antigravity) automatically follow defensive patterns:
- Always registering architectures with `@register_model`.
- Using `save_bundle` and `resume_training` instead of unsafe `pickle`/`torch.save`.
- Querying bundle metadata via `uv run mangosta inspect <bundle> --json` to prevent token waste.

---

## Local Development & Testing

Build and verify the package locally:

```bash
# Build wheel and sdist distributions locally
uv build

# Run strict PEP 561 static type checks
uv run mypy --config-file pyproject.toml src/mangosta

# Run linter
uv run ruff check src/mangosta
```

To test in a separate project locally:
```bash
# In your test project directory:
uv add --editable "/path/to/mangosta[full]"
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

"""Project initializer scaffolding for mangosta."""

from __future__ import annotations

from pathlib import Path


def init_project(project_name: str = ".", template: str = "default") -> Path:
    """Scaffold a lean, production-grade PyTorch project powered by Mangosta."""
    dest = Path(project_name).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Directory Tree (Zero empty artifact directories created upfront)
    dirs = [
        dest / "configs",
        dest / "entrypoints",
        dest / "src",
        dest / ".agents" / "skills" / "mangosta",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Package __init__.py files
    (dest / "src" / "__init__.py").touch(exist_ok=True)
    (dest / "entrypoints" / "__init__.py").touch(exist_ok=True)

    # 3. Project configuration & models
    _write_configs(dest)
    _write_source(dest)

    # 4. Clean entrypoints (train, infer, benchmark)
    _write_entrypoints(dest)

    # 5. Metadata (pyproject.toml, AGENTS.md, .gitignore, README.md)
    _write_meta(dest)

    return dest


def _write_configs(dest: Path) -> None:
    exp_baseline_yaml = """experiment:
  name: "exp_baseline"
  seed: 42
  device: "cpu"  # "cuda" or "cpu"

model:
  family: "sensor_classifier"
  version: "1.0.0"
  in_features: 8
  hidden_dim: 32
  num_classes: 3

training:
  epochs: 5
  batch_size: 16
  learning_rate: 0.001
"""
    (dest / "configs" / "exp_baseline.yaml").write_text(exp_baseline_yaml, encoding="utf-8")


def _write_source(dest: Path) -> None:
    models_code = """import torch
import torch.nn as nn
from typing import Any
from mangosta import register_model, ModelRegistry


@register_model(family="sensor_classifier", version="1.0.0")
class SensorClassifier(nn.Module):
    def __init__(self, in_features: int = 8, hidden_dim: int = 32, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(config: dict[str, Any]) -> nn.Module:
    \"\"\"Reconstruct model architecture from configuration dictionary using ModelRegistry.\"\"\"
    model_cfg = config.get("model", {})
    family = model_cfg.get("family", "sensor_classifier")
    version = str(model_cfg.get("version", "1.0.0"))
    init_args = {k: v for k, v in model_cfg.items() if k not in ("family", "version")}
    return ModelRegistry.build(family, version=version, **init_args)
"""
    (dest / "src" / "models.py").write_text(models_code, encoding="utf-8")


def _write_entrypoints(dest: Path) -> None:
    # 1. train.py
    train_code = """import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when running as script or CLI entrypoint
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mangosta import RunTracker, save_bundle
from src.models import build_model


def get_dataloaders(batch_size: int = 16) -> tuple[DataLoader, DataLoader]:
    torch.manual_seed(42)
    x = torch.randn(200, 8)
    y = torch.randint(0, 3, (200,))
    train_loader = DataLoader(TensorDataset(x[:160], y[:160]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x[160:], y[160:]), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Mangosta Training Pipeline")
    parser.add_argument("--config", default="configs/exp_baseline.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    epochs = args.epochs or train_cfg.get("epochs", 5)
    lr = train_cfg.get("learning_rate", 0.001)
    bs = train_cfg.get("batch_size", 16)

    family = model_cfg.get("family", "sensor_classifier")
    version = str(model_cfg.get("version", "1.0.0"))
    init_args = {k: v for k, v in model_cfg.items() if k not in ("family", "version")}

    train_loader, val_loader = get_dataloaders(bs)
    model = build_model(config)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with RunTracker(experiment=exp_cfg.get("name", "baseline"), root_dir="experiments") as run:
        run.log_params(train_cfg)

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            run.log_metrics({"epoch_loss": avg_loss}, step=epoch)
            print(f"Epoch {epoch}/{epochs} complete | Loss: {avg_loss:.4f}")

        # Checkpoint bundle (smart default: detects optimizer -> is_checkpoint=True)
        ckpt_path = save_bundle(
            output_path="checkpoints/model_checkpoint.bundle",
            model=model,
            sample_input=torch.randn(1, 8),
            optimizer=optimizer,
            epoch=epochs,
            family=family,
            version=version,
            init_args=init_args,
        )

        # Final production model (smart default: no optimizer -> is_checkpoint=False)
        final_path = save_bundle(
            output_path="checkpoints/model_final.bundle",
            model=model,
            sample_input=torch.randn(1, 8),
            family=family,
            version=version,
            init_args=init_args,
        )

        print(f"Training finished. Checkpoint: {ckpt_path.name} | Production: {final_path.name}")


if __name__ == "__main__":
    main()
"""
    (dest / "entrypoints" / "train.py").write_text(train_code, encoding="utf-8")

    # 2. infer.py
    infer_code = """import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when running as script or CLI entrypoint
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from mangosta import load_bundle
import src.models  # trigger registration


def main():
    parser = argparse.ArgumentParser(description="Mangosta Inference")
    parser.add_argument("--bundle", default="checkpoints/model_final.bundle", help="Path to .bundle")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    print(f"Loaded bundle: {bundle.family} v{bundle.version}")

    # Forward inference with automatic shape and dtype contract validation
    sample = torch.randn(2, 8)
    preds = bundle.predict(sample)
    print("Inference successful. Predictions shape:", list(preds.shape))


if __name__ == "__main__":
    main()
"""
    (dest / "entrypoints" / "infer.py").write_text(infer_code, encoding="utf-8")

    # 3. benchmark.py
    benchmark_code = """import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when running as script or CLI entrypoint
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import torch
from mangosta import load_bundle
import src.models  # trigger registration


def main():
    parser = argparse.ArgumentParser(description="Mangosta Latency Benchmark")
    parser.add_argument("--bundle", default="checkpoints/model_final.bundle", help="Path to .bundle")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=50, help="Benchmark iterations")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    dummy = torch.randn(1, 8)

    # Warmup
    for _ in range(args.warmup):
        bundle.model(dummy)

    # Benchmark timing loop
    t0 = time.perf_counter()
    for _ in range(args.runs):
        bundle.model(dummy)
    elapsed = time.perf_counter() - t0

    avg_ms = (elapsed * 1000.0) / args.runs
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"Model: {bundle.family} v{bundle.version}")
    print(f"Latency: {avg_ms:.3f} ms/sample | Throughput: {fps:.1f} FPS")


if __name__ == "__main__":
    main()
"""
    (dest / "entrypoints" / "benchmark.py").write_text(benchmark_code, encoding="utf-8")


def _write_meta(dest: Path) -> None:
    project_slug = dest.name.replace(" ", "_").replace("-", "_")

    # pyproject.toml
    pyproject_content = f"""[project]
name = "{project_slug}"
version = "0.1.0"
description = "PyTorch project powered by Mangosta"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "mangosta",
    "torch>=2.0.0",
    "pyyaml>=6.0",
    "numpy>=2.0.0",
]

[project.scripts]
train = "entrypoints.train:main"
infer = "entrypoints.infer:main"
benchmark = "entrypoints.benchmark:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src", "entrypoints"]
"""
    (dest / "pyproject.toml").write_text(pyproject_content, encoding="utf-8")

    # .agents/skills/mangosta/SKILL.md
    # Load canonical skill from packaged data or repository source
    skill_content: str | None = None
    try:
        import importlib.resources as pkg_resources

        skill_file = pkg_resources.files("mangosta") / "skills" / "mangosta" / "SKILL.md"
        skill_content = skill_file.read_text(encoding="utf-8")
    except Exception:
        pass

    if skill_content is None:
        repo_skill = Path(__file__).resolve().parent.parent.parent.parent / "skills" / "mangosta" / "SKILL.md"
        if repo_skill.is_file():
            skill_content = repo_skill.read_text(encoding="utf-8")

    if skill_content is not None:
        (dest / ".agents" / "skills" / "mangosta" / "SKILL.md").write_text(skill_content, encoding="utf-8")
    else:
        # Fallback inline instructions if package resource is unavailable
        agent_instructions = """# Agent Instructions: Mangosta Project Conventions

This project uses **Mangosta** for PyTorch model lifecycle management, experiment tracking, and artifact packaging.
Adhere to the following rules when reading, modifying, or executing code in this repository:

## 1. Operating Rules
- **No Background Daemons or External Servers**: All tracking is local and offline. Never attempt to spin up Docker, MLflow servers, or background web processes.
- **Tracking Standard**: Always track runs using `with RunTracker(experiment=..., root_dir="experiments") as run:`.
- **Metrics and Logging**:
  - Log scalar metrics: `run.log_metrics({"train/loss": loss.item()}, step=epoch)`.
  - Log parameters: `run.log_params({"lr": 1e-3, "optimizer": "AdamW"})`.
  - Log temporal validation samples: `run.log_validation_sample(sample_id="sample_01", prediction=pred_np, input_image=input_np, ground_truth=gt_np, step=epoch)`.
- **Model Registration**:
  - Always register architectures with `@register_model(family="<name>", version="<semver>")`.
  - Never bypass the registry for production bundles. Reconstruct models with `ModelRegistry.build()`.
- **Packaging with `save_bundle`**:
  - Training checkpoint: pass `optimizer=optimizer, scheduler=scheduler, epoch=epoch` to `save_bundle()`. Mangosta automatically packages training state and exact RNG seeds for bit-to-bit resumption.
  - Final inference model: omit optimizer/scheduler to produce a lightweight deployment bundle with SafeTensors weights and `ModelSchema` contract.
- **Deterministic Resumption with `resume_training`**:
  - To resume interrupted runs: `resume_state = resume_training(bundle_path, model, optimizer, scheduler, restore_rng=True)`.
- **Production Serving vs Training**:
  - For zero-PyTorch production microservices (~70MB Docker): export with `export_onnx_as_bundle(bundle_or_model, "deploy/model.onnx.bundle")` and serve with `mangosta.serving.ONNXRunner.predict(numpy_array)`.
- **Decoupled PyTorch Inference with `load_bundle`**:
  - `bundle = load_bundle("path/to.bundle")` reconstructs the model and validates inputs via `bundle.predict(sample)`.

## 2. Common Commands
- Train model: `uv run train` or `uv run python -m entrypoints.train`
- Run inference: `uv run infer`
- Benchmark latency: `uv run benchmark`
- Inspect bundle metadata (<5ms SafeTensors header parsing): `uv run mangosta inspect checkpoints/model_final.bundle --json`
- Generate offline HTML report: `uv run mangosta report experiments/<exp>/<run> -o report.html`
- Local web visualizer (zero daemons): `uv run mangosta view`
"""
        (dest / ".agents" / "skills" / "mangosta" / "SKILL.md").write_text(agent_instructions, encoding="utf-8")

    # AGENTS.md (Root agent guidelines: KISS + SOLID principles & pointers to .agents/)
    agents_root = """# AGENTS.md - Development & Architectural Guidelines

This project follows **KISS** (Keep It Simple, Stupid) and **SOLID** engineering principles, powered by [Mangosta](https://github.com/yeriel/mangosta) for PyTorch model lifecycle management.

## 1. Core Engineering Philosophy

- **KISS (Keep It Simple, Stupid)**:
  - Avoid unnecessary abstractions, background daemons, or heavy external microservices.
  - Prefer flat, local, atomic file operations (SafeTensors, JSONL, standard directories).
  - Write straightforward, readable Python over premature optimization.
  - Stick to established entrypoint patterns (`train`, `infer`, `benchmark`).

- **SOLID Principles**:
  - **Single Responsibility (SRP)**: Distinct layers for configuration (`configs/`), data & architectures (`src/`), orchestration entrypoints (`entrypoints/`), and artifact storage.
  - **Open/Closed (OCP)**: Extend model architectures via `@register_model` without altering core pipeline mechanisms.
  - **Liskov Substitution (LSP)**: All neural network modules inherit predictably from standard `torch.nn.Module`.
  - **Interface Segregation (ISP)**: Separate training logic (PyTorch native) from production serving (pure NumPy + ONNX).
  - **Dependency Inversion (DIP)**: Models are instantiated dynamically from configurations via `ModelRegistry.build()`.

## 2. Agent Context & Deep Mangosta Documentation

All comprehensive operational rules, canonical patterns, and exception remediations are located in:
- **Official Antigravity / Cursor Skill**: [`.agents/skills/mangosta/SKILL.md`](.agents/skills/mangosta/SKILL.md)

Always refer to `.agents/skills/mangosta/SKILL.md` before generating training loops, checkpointing logic, or deployment pipelines.
"""
    (dest / "AGENTS.md").write_text(agents_root, encoding="utf-8")

    # .gitignore
    gitignore = """.venv/
__pycache__/
*.py[cod]
checkpoints/
experiments/
outputs/
*.bundle
*.tmp_*
"""
    (dest / ".gitignore").write_text(gitignore, encoding="utf-8")

    # README.md
    readme = f"""# {dest.name}

PyTorch model project scaffolded with [Mangosta](https://github.com/yeriel/mangosta).

## Getting Started

### 1. Run with `uv` (Recommended)
```bash
# Train model
uv run train --config configs/exp_baseline.yaml

# Run contract-validated inference
uv run infer --bundle checkpoints/model_final.bundle

# Benchmark latency and throughput
uv run benchmark --bundle checkpoints/model_final.bundle

# Inspect bundle metadata
uv run mangosta inspect checkpoints/model_final.bundle
```

### 2. Traditional Python
```bash
python -m entrypoints.train
python -m entrypoints.infer
python -m entrypoints.benchmark
```
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")

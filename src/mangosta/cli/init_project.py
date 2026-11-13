"""Project initializer scaffolding for mangosta."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


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
  device: "auto"  # "auto", "cuda", "cpu", "mps"

model:
  family: "sensor_classifier"
  version: "1.0.0"
  init_args:
    in_features: 8
    hidden_dim: 32
    num_classes: 3

training:
  epochs: 5
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 0.0001
  amp: false
  grad_clip_norm: null
  patience: null
  monitor: "val_loss"
  monitor_mode: "min"
"""
    (dest / "configs" / "exp_baseline.yaml").write_text(exp_baseline_yaml, encoding="utf-8")


def _write_source(dest: Path) -> None:
    data_code = """import torch
from torch.utils.data import DataLoader, TensorDataset
from mangosta import MangostaConfig


def build_dataloaders(cfg: MangostaConfig) -> tuple[DataLoader, DataLoader]:
    \"\"\"Build standard PyTorch train and validation DataLoaders.\"\"\"
    torch.manual_seed(cfg.experiment.seed)
    x = torch.randn(200, 8)
    y = torch.randint(0, 3, (200,))
    bs = cfg.training.batch_size
    train_loader = DataLoader(TensorDataset(x[:160], y[:160]), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(x[160:], y[160:]), batch_size=bs, shuffle=False)
    return train_loader, val_loader
"""
    (dest / "src" / "data.py").write_text(data_code, encoding="utf-8")

    models_code = """import torch
import torch.nn as nn
from mangosta import BaseTrainer, MangostaConfig, register_model, register_trainer


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


@register_trainer(family="sensor_classifier", version="1.0.0")
class SensorTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, config: MangostaConfig | dict | None = None, **kwargs) -> None:
        super().__init__(model, config, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        return self.criterion(self.model(x), y)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)
        acc = (preds.argmax(dim=-1) == y).float().mean()
        return {"val_loss": loss, "val_acc": acc}
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

from mangosta import ModelRegistry, TrainerRegistry, load_config
import src.models  # Register models and trainers
from src.data import build_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Mangosta Training Pipeline")
    parser.add_argument("--config", default="configs/exp_baseline.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .bundle to resume")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_loader, val_loader = build_dataloaders(cfg)

    model = ModelRegistry.build_from_config(cfg)
    trainer = TrainerRegistry.build_from_config(cfg, model=model)

    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        resume_from=args.resume,
    )
    print(f"Training finished. Final production bundle: {result['final_bundle']}")


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


_TOML_HEADER_RE = re.compile(r"^\s*\[([A-Za-z0-9_.-]+)\]\s*(?:#.*)?$")


def _find_toml_section(lines: list[str], header: str) -> tuple[int, int] | None:
    """Return (header_idx, end_idx) for an exact TOML table header, or None."""
    for idx, line in enumerate(lines):
        match = _TOML_HEADER_RE.match(line)
        if match and match.group(1) == header:
            end_idx = len(lines)
            for next_idx in range(idx + 1, len(lines)):
                if _TOML_HEADER_RE.match(lines[next_idx]):
                    end_idx = next_idx
                    break
            return idx, end_idx
    return None


def _upsert_toml_section(
    text: str,
    header: str,
    entries: list[tuple[str, str]],
    replace_if_contains: str | None = None,
) -> str:
    """Ensure a TOML table exists and contains the given key/value lines."""
    lines = text.splitlines()
    section = _find_toml_section(lines, header)
    entry_lines = [f"{key} = {value}" for key, value in entries]

    if section is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"[{header}]")
        lines.extend(entry_lines)
        return "\n".join(lines) + "\n"

    header_idx, end_idx = section
    body_text = "\n".join(lines[header_idx + 1 : end_idx])

    if replace_if_contains is not None and replace_if_contains in body_text:
        replacement = list(entry_lines)
        if end_idx < len(lines):
            replacement.append("")
        lines[header_idx + 1 : end_idx] = replacement
        return "\n".join(lines).rstrip() + "\n"

    insert_idx = end_idx
    while insert_idx > header_idx + 1 and not lines[insert_idx - 1].strip():
        insert_idx -= 1

    missing: list[str] = []
    for key, value in entries:
        key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
        if not any(key_re.match(line) for line in lines[header_idx + 1 : end_idx]):
            missing.append(f"{key} = {value}")

    if missing:
        lines[insert_idx:insert_idx] = missing

    return "\n".join(lines).rstrip() + "\n"


def _upgrade_mangosta_dep(text: str) -> str:
    """Upgrade bare 'mangosta' dependency inside dependencies = [...] to 'mangosta[full]'."""

    def _replace_in_deps(match: re.Match[str]) -> str:
        block = match.group(0)
        if "mangosta[" in block:
            return block
        return re.sub(
            r'(["\'])mangosta((?:[><=!~][^"\']*)?)\1',
            r"\1mangosta[full]\2\1",
            block,
        )

    return re.sub(
        r"(?ms)^\s*dependencies\s*=\s*\[.*?\]",
        _replace_in_deps,
        text,
        count=1,
    )


def _ensure_pyproject(dest: Path, project_slug: str) -> None:
    """Create pyproject.toml if missing, or safely upgrade uv-initialized projects."""
    pyproject_path = dest / "pyproject.toml"
    if not pyproject_path.exists():
        pyproject_content = f"""[project]
name = "{project_slug}"
version = "0.1.0"
description = "PyTorch project powered by Mangosta"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "mangosta[full]",
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

[tool.uv]
package = true
"""
        pyproject_path.write_text(pyproject_content, encoding="utf-8")
        return

    # Preserve existing pyproject.toml (especially [tool.uv.sources] and dependencies)
    existing = pyproject_path.read_text(encoding="utf-8")
    updated = _upgrade_mangosta_dep(existing)
    updated = _upsert_toml_section(
        updated,
        "project.scripts",
        [
            ("train", '"entrypoints.train:main"'),
            ("infer", '"entrypoints.infer:main"'),
            ("benchmark", '"entrypoints.benchmark:main"'),
        ],
    )
    updated = _upsert_toml_section(
        updated,
        "build-system",
        [
            ("requires", '["hatchling"]'),
            ("build-backend", '"hatchling.build"'),
        ],
        replace_if_contains="uv_build",
    )
    updated = _upsert_toml_section(
        updated,
        "tool.hatch.build.targets.wheel",
        [("packages", '["src", "entrypoints"]')],
    )
    updated = _upsert_toml_section(
        updated,
        "tool.uv",
        [("package", "true")],
    )

    if updated != existing:
        pyproject_path.write_text(updated, encoding="utf-8")


def _copy_skill_tree(src: Any, dst: Path) -> None:
    """Recursively copy packaged skill resources (SKILL.md, examples/, resources/, scripts/)."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == "__pycache__" or item.name.endswith(".pyc"):
            continue
        target = dst / item.name
        if item.is_dir():
            _copy_skill_tree(item, target)
        elif item.is_file():
            target.write_bytes(item.read_bytes())


def _write_meta(dest: Path) -> None:
    project_slug = dest.name.replace(" ", "_").replace("-", "_")

    # pyproject.toml (preserves existing [tool.uv.sources] / editable dependencies)
    _ensure_pyproject(dest, project_slug)

    # .agents/skills/mangosta/ (SKILL.md + examples/ + resources/ + scripts/)
    skill_dest = dest / ".agents" / "skills" / "mangosta"
    copied = False
    try:
        import importlib.resources as pkg_resources

        skill_root = pkg_resources.files("mangosta") / "skills" / "mangosta"
        if skill_root.is_dir():
            _copy_skill_tree(skill_root, skill_dest)
            copied = True
    except Exception:
        pass

    if not copied:
        local_skill = Path(__file__).resolve().parent.parent / "skills" / "mangosta"
        if local_skill.is_dir():
            _copy_skill_tree(local_skill, skill_dest)

    # AGENTS.md (Root agent guidelines: KISS + SOLID principles & pointers to .agents/skills/mangosta/)
    agents_root = """# AGENTS.md - Development & Architectural Guidelines

This project follows **KISS** (Keep It Simple, Stupid) and **SOLID** engineering principles, powered by [Mangosta](https://github.com/yeriel/mangosta) for PyTorch model lifecycle management.

## 1. Core Engineering Philosophy

- **KISS (Keep It Simple, Stupid)**:
  - Avoid unnecessary abstractions, background daemons, or heavy external microservices.
  - Prefer flat, local, atomic file operations (SafeTensors, JSONL, `.bundle` ZIPs, YAML configs).
  - Use `BaseTrainer` and `MangostaConfig` (`load_config`) instead of reimplementing training loops, AMP, EarlyStopping, or RNG checkpointing.
  - Stick to established entrypoint patterns (`uv run train`, `uv run infer`, `uv run benchmark`).

- **SOLID Principles**:
  - **Single Responsibility (SRP)**: Distinct layers for strict config (`configs/`), data, models & trainers (`src/`), and orchestration entrypoints (`entrypoints/`).
  - **Open/Closed (OCP)**: Extend architectures and trainers via `@register_model` and `@register_trainer` without altering core pipeline mechanisms.
  - **Liskov Substitution (LSP)**: All neural network modules inherit from `torch.nn.Module`, and custom trainers inherit from `BaseTrainer`.
  - **Interface Segregation (ISP)**: Separate training logic (`mangosta[full]`) from zero-PyTorch production serving (`mangosta[serving]` + `ONNXRunner`).
  - **Dependency Inversion (DIP)**: Models and trainers are instantiated dynamically from `MangostaConfig` via `ModelRegistry.build_from_config(cfg)` and `TrainerRegistry.build_from_config(cfg, model=model)`.

## 2. Agent Context & Deep Mangosta Documentation

All operational rules, v0.4.0 APIs, and exception remediations live in:
- **Official Skill Guide**: [`.agents/skills/mangosta/SKILL.md`](.agents/skills/mangosta/SKILL.md)
- **Config Schema Reference**: [`.agents/skills/mangosta/resources/config_schema.yaml`](.agents/skills/mangosta/resources/config_schema.yaml)
- **Reference Examples**: [`.agents/skills/mangosta/examples/`](.agents/skills/mangosta/examples/)
- **Project Validator Script**: [`.agents/skills/mangosta/scripts/verify_project.py`](.agents/skills/mangosta/scripts/verify_project.py)

Always refer to `.agents/skills/mangosta/SKILL.md` before modifying configs, training loops, checkpointing, or serving pipelines.
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

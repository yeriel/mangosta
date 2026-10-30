"""Project initializer scaffolding for mangosta."""

from __future__ import annotations

from pathlib import Path


def init_project(project_name: str = ".", template: str = "default") -> Path:
    """Scaffold a complete, production-grade PyTorch project structured with mangosta."""
    dest = Path(project_name).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Directory Tree
    dirs = [
        dest / "configs" / "experiments",
        dest / "data" / "raw",
        dest / "data" / "processed",
        dest / "src" / "data",
        dest / "src" / "models" / "architectures",
        dest / "src" / "engine",
        dest / "src" / "utils",
        dest / "checkpoints",
        dest / "experiments",
        dest / "outputs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").touch(exist_ok=True)

    # 2. Package __init__.py files
    for init_dir in [
        dest / "src",
        dest / "src" / "data",
        dest / "src" / "models",
        dest / "src" / "models" / "architectures",
        dest / "src" / "engine",
        dest / "src" / "utils",
    ]:
        (init_dir / "__init__.py").touch(exist_ok=True)

    # 3. Config Files
    _write_configs(dest)

    # 4. Source Files (Models, Data, Engine, Utils)
    _write_source(dest)

    # 5. Root Entrypoints (train.py, infer.py, benchmark.py)
    _write_entrypoints(dest)

    # 6. Project meta (.gitignore, README.md)
    _write_meta(dest)

    return dest


def _write_configs(dest: Path) -> None:
    base_yaml = """system:
  seed: 42
  device: "cuda"              # "cuda" or "cpu"
  precision: "fp32"           # "fp32" or "fp16"

training_defaults:
  epochs: 10
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 0.0001
"""
    (dest / "configs" / "base.yaml").write_text(base_yaml, encoding="utf-8")

    exp_baseline_yaml = """experiment:
  name: "exp_baseline_v1"
  description: "Initial baseline training with sensor telemetry classifier"

model:
  family: "sensor_classifier"
  version: "1.0.0"
  in_features: 8
  hidden_dim: 32
  num_classes: 3

data:
  dataset_dir: "data/processed"

training:
  epochs: 5
  batch_size: 16
  learning_rate: 0.001
"""
    (dest / "configs" / "experiments" / "exp_baseline.yaml").write_text(
        exp_baseline_yaml, encoding="utf-8"
    )


def _write_source(dest: Path) -> None:
    # Model architecture
    arch_code = """import torch
import torch.nn as nn
from mangosta import register_model

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
"""
    (dest / "src" / "models" / "architectures" / "baseline.py").write_text(
        arch_code, encoding="utf-8"
    )

    # Models __init__ with dynamic build
    models_init = """import torch.nn as nn
from typing import Any
from mangosta import ModelRegistry
import src.models.architectures.baseline  # trigger registration

def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    family = model_cfg.get("family", "sensor_classifier")
    version = str(model_cfg.get("version", "1.0.0"))
    init_args = {k: v for k, v in model_cfg.items() if k not in ("family", "version")}
    return ModelRegistry.build(family, version=version, **init_args)
"""
    (dest / "src" / "models" / "__init__.py").write_text(models_init, encoding="utf-8")

    # Synthetic Dataset & DataLoader
    data_loader = """import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from mangosta import compute_sha256

def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, str]:
    data_dir = Path(config.get("data", {}).get("dataset_dir", "data/processed")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_file = data_dir / "synthetic_data.csv"

    if not csv_file.exists():
        torch.manual_seed(42)
        x = torch.randn(200, 8)
        y = torch.randint(0, 3, (200,))
        lines = ["f0,f1,f2,f3,f4,f5,f6,f7,label\\n"]
        for row, label in zip(x.tolist(), y.tolist()):
            lines.append(",".join(f"{v:.4f}" for v in row) + f",{label}\\n")
        csv_file.write_text("".join(lines), encoding="utf-8")

    # Compute SHA-256 fingerprint for data lineage audit
    data_hash = compute_sha256(csv_file)

    torch.manual_seed(42)
    inputs = torch.randn(200, 8)
    targets = torch.randint(0, 3, (200,))
    ds_train = TensorDataset(inputs[:160], targets[:160])
    ds_val = TensorDataset(inputs[160:], targets[160:])

    bs = config.get("training", {}).get("batch_size", 16)
    train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=bs, shuffle=False)
    return train_loader, val_loader, data_hash
"""
    (dest / "src" / "data" / "loader.py").write_text(data_loader, encoding="utf-8")

    # Config helper
    config_util = """from pathlib import Path
import yaml

def load_config(experiment_name: str = "exp_baseline") -> tuple[dict, Path]:
    exp_name = experiment_name if not experiment_name.endswith(".yaml") else experiment_name[:-5]
    cfg_path = Path("configs/experiments") / f"{exp_name}.yaml"
    base_path = Path("configs/base.yaml")

    config = {}
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})

    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            exp_cfg = yaml.safe_load(f) or {}
            # Deep update
            for k, v in exp_cfg.items():
                if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                    config[k].update(v)
                else:
                    config[k] = v

    return config, cfg_path
"""
    (dest / "src" / "utils" / "config.py").write_text(config_util, encoding="utf-8")


def _write_entrypoints(dest: Path) -> None:
    # 1. train.py
    train_code = """import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from mangosta import quick_track, save_bundle, resume_training
from src.utils.config import load_config
from src.data.loader import build_dataloaders
from src.models import build_model

def main():
    parser = argparse.ArgumentParser(description="Mangosta Training Pipeline")
    parser.add_argument("--config", type=str, default="exp_baseline", help="Experiment name")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .bundle")
    args = parser.parse_args()

    config, _ = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "default_exp")
    epochs = args.epochs or config.get("training", {}).get("epochs", 5)

    train_loader, val_loader, dataset_hash = build_dataloaders(config)
    model = build_model(config)
    optimizer = optim.Adam(model.parameters(), lr=config.get("training", {}).get("learning_rate", 0.001))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    if args.resume:
        state = resume_training(args.resume, model, optimizer, scheduler, restore_rng=True)
        start_epoch = state.epoch + 1
        print(f"Resumed from checkpoint: Epoch {state.epoch}, Step {state.step}")

    with quick_track(experiment=exp_name, run_name="run_01") as run:
        run.log_params(config.get("training", {}))
        run.set_dataset("data/processed/synthetic_data.csv")

        global_step = (start_epoch - 1) * len(train_loader)
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            total_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                run.log_metric("train_step_loss", loss.item(), step=global_step)
                global_step += 1

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            run.log_metric("epoch_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch}/{epochs} complete | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            # Example: save validation image / artifact at epoch end
            dummy_val_img = torch.rand(3, 32, 32)
            run.log_image(f"val_sample_epoch_{epoch}", dummy_val_img, step=epoch)

        # Save checkpoint
        save_bundle(
            output_path="checkpoints/model_checkpoint.bundle",
            model=model,
            family="sensor_classifier",
            version="1.0.0",
            init_args={"in_features": 8, "hidden_dim": 32, "num_classes": 3},
            sample_input=torch.randn(1, 8),
            is_checkpoint=True,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epochs,
            step=global_step,
            dataset_hash=dataset_hash,
        )

        # Save production bundle
        save_bundle(
            output_path="checkpoints/model_final.bundle",
            model=model,
            family="sensor_classifier",
            version="1.0.0",
            init_args={"in_features": 8, "hidden_dim": 32, "num_classes": 3},
            sample_input=torch.randn(1, 8),
            is_checkpoint=False,
            dataset_hash=dataset_hash,
        )
        print("Training complete. Artifacts saved in checkpoints/")

if __name__ == "__main__":
    main()
"""
    (dest / "train.py").write_text(train_code, encoding="utf-8")

    # 2. infer.py
    infer_code = """import argparse
import torch
from mangosta import load_bundle
import src.models.architectures.baseline  # register model

def main():
    parser = argparse.ArgumentParser(description="Mangosta Inference")
    parser.add_argument("--bundle", type=str, default="checkpoints/model_final.bundle")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    print(f"Loaded: {bundle.family} v{bundle.version}")

    # Forward with automated contract validation
    sample = torch.randn(2, 8)
    pred = bundle.predict(sample)
    print("Inference successful. Predictions shape:", list(pred.shape))

if __name__ == "__main__":
    main()
"""
    (dest / "infer.py").write_text(infer_code, encoding="utf-8")

    # 3. benchmark.py
    benchmark_code = """import argparse
import time
import torch
from mangosta import load_bundle, export_onnx
import src.models.architectures.baseline

def main():
    parser = argparse.ArgumentParser(description="Mangosta Benchmark & Export")
    parser.add_argument("--bundle", type=str, default="checkpoints/model_final.bundle")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    dummy = torch.randn(1, 8)

    # Warmup & benchmark
    for _ in range(10): bundle.model(dummy)
    t0 = time.perf_counter()
    for _ in range(50): bundle.model(dummy)
    ms = (time.perf_counter() - t0) * 1000.0 / 50

    print(f"Model: {bundle.family} v{bundle.version} | Latency: {ms:.3f} ms ({1000.0 / ms:.1f} FPS)")

    if args.export_onnx:
        out = export_onnx(args.bundle, output_path="checkpoints/model.onnx", sample_input_shape=(1, 8))
        print(f"Exported ONNX: {out}")

if __name__ == "__main__":
    main()
"""
    (dest / "benchmark.py").write_text(benchmark_code, encoding="utf-8")


def _write_meta(dest: Path) -> None:
    gitignore = """.venv/
__pycache__/
*.py[cod]
checkpoints/
experiments/
outputs/
data/processed/
"""
    (dest / ".gitignore").write_text(gitignore, encoding="utf-8")

    readme = """# Mangosta Project

Clean Architecture PyTorch project initialized with `mangosta init`.

## Commands
- Train model: `python train.py --config exp_baseline`
- Run inference: `python infer.py --bundle checkpoints/model_final.bundle`
- Benchmark: `python benchmark.py --bundle checkpoints/model_final.bundle --export-onnx`
- Visualizer: `mangosta view`
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")

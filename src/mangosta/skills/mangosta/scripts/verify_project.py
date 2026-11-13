"""Quick dry-run validator for Mangosta project config and registries."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from mangosta import ModelRegistry, TrainerRegistry, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Mangosta config, model, and trainer registration.")
    parser.add_argument("--config", default="configs/exp_baseline.yaml", help="Path to YAML config")
    parser.add_argument("--module", default="src.models", help="Module defining @register_model/@register_trainer")
    args = parser.parse_args()

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    importlib.import_module(args.module)
    cfg = load_config(args.config)
    model = ModelRegistry.build_from_config(cfg)
    trainer = TrainerRegistry.build_from_config(cfg, model=model)

    print(cfg.format_summary())
    print(f"OK: Built model '{type(model).__name__}' and trainer '{type(trainer).__name__}'.")


if __name__ == "__main__":
    main()

---
name: mangosta
description: >-
  Lightweight, zero-daemon MLOps and model lifecycle management for PyTorch:
  project scaffolding, strict YAML configuration (MangostaConfig), immutable model and trainer registration,
  white-box BaseTrainer orchestration, contract-based tensor validation, atomic experiment tracking,
  deterministic checkpointing with exact RNG restoration, slim zero-PyTorch ONNX serving,
  sub-5ms SafeTensors inspection, and ephemeral browser visualization.
  Activate whenever scaffolding, configuring, training, tracking, resuming, validating, deploying, or exporting PyTorch models.
---

# mangosta (v0.4.0): Agent Engineering & Architecture Guide

`mangosta` is a zero-daemon, local-first library for the PyTorch model lifecycle. It replaces heavy MLOps servers (MLflow, ClearML, W&B) with atomic local files (`.bundle` ZIPs, SafeTensors, JSONL), formal tensor contracts, and slim zero-PyTorch serving.

Supporting files in this skill directory:
- **`resources/config_schema.yaml`**: Complete annotated `MangostaConfig` YAML schema and defaults.
- **`examples/trainer_custom.py`**: Custom `@register_model` + `@register_trainer` + `BaseTrainer` hooks + validation sample logging.
- **`examples/onnx_serving.py`**: `export_onnx_as_bundle` and zero-PyTorch `ONNXRunner` inference.
- **`scripts/verify_project.py`**: Dry-run validator for YAML config, `ModelRegistry`, and `TrainerRegistry`.

---

## 1. Modular Architecture & Primary APIs

| Package | Responsibility | Primary APIs |
| :--- | :--- | :--- |
| **`mangosta.core`** | Typed config, deterministic seeding, model/trainer registries & tensor contracts. | `MangostaConfig`, `load_config`, `seed_everything`, `ModelRegistry`, `TrainerRegistry`, `@register_model`, `@register_trainer`, `ModelSchema`, `TensorSpec`, `infer_schema` |
| **`mangosta.engine`** | White-box PyTorch training loop with automated tracking, bundles, AMP, EarlyStopping & resume. | `BaseTrainer` |
| **`mangosta.artifacts`** | Atomic ZIP `.bundle` creation, decoupled loading & RNG state recovery. | `save_bundle`, `load_bundle`, `resume_training`, `read_bundle_manifest` |
| **`mangosta.tracking`** | Atomic JSONL step metrics, validation timelapses, HTML reports & viewer. | `RunTracker`, `quick_track`, `generate_report`, `start_viewer` |
| **`mangosta.export`** | Decoupled ONNX graph export with FP16 and dictionary output unrolling. | `export_onnx`, `export_onnx_as_bundle` |
| **`mangosta.serving`** | Slim, zero-PyTorch microservice inference engine. | `ONNXRunner`, `load_onnx_bundle` |
| **`mangosta.cli`** | Project scaffolding and ephemeral CLI tools. | `uv run mangosta init`, `inspect --json`, `report`, `view`, `export-onnx` |

All primary symbols are re-exported at top level (`from mangosta import ...`).

---

## 2. Environment & Installation (`uv` Standard)

- **Training & full development**: `uv add "mangosta[full]"` (PyTorch, SafeTensors, ONNX, ONNX Runtime).
- **Zero-PyTorch serving (~70MB containers)**: `uv add "mangosta[serving]"` (ONNX Runtime, NumPy, PyYAML).
- **Project bootstrap**: `uv run mangosta init` scaffolds `configs/`, `src/`, `entrypoints/`, `AGENTS.md`, `.agents/skills/mangosta/`, configures `hatchling` + `[project.scripts]` (`train`, `infer`, `benchmark`), upgrades `"mangosta"` to `"mangosta[full]"`, and runs `uv sync` automatically.
- **Standard entrypoints**: `uv run train`, `uv run infer`, `uv run benchmark`.

---

## 3. Agent Operational Rules

1. **Strict Configuration (`MangostaConfig`)**:
   - Load configs with `cfg = load_config("configs/exp_baseline.yaml")`.
   - Never put arbitrary keys under root, `experiment:`, or `training:` — they raise `ConfigError` with `"Did you mean '...'?"`. Put dataset/user keys under `data:` or `custom:`. See `resources/config_schema.yaml`.
2. **Never Bypass Registries**:
   - Register models with `@register_model(family="<name>", version="<semver>")` and trainers with `@register_trainer(family="<name>", version="<semver>")`.
   - Instantiate dynamically via `ModelRegistry.build_from_config(cfg)` and `TrainerRegistry.build_from_config(cfg, model=model)`.
   - Never overwrite existing `(family, version)` pairs silently (`RegistryCollisionError`); bump SemVer unless `allow_override=True` is explicitly required.
3. **Prefer `BaseTrainer` for Training Loops**:
   - Subclass `BaseTrainer` (or pass `criterion=` directly for standard `(x, y)` supervised tasks).
   - Override `configure_optimizers()`, `training_step(batch)`, `validation_step(batch)`, and—only for GANs/gradient accumulation—`optimize_step(batch)`.
   - `trainer.fit(train_loader, val_loader, resume_from=...)` automatically handles:
     - `seed_everything()`, device resolution (`auto` -> `cuda`/`mps`/`cpu`), AMP & `grad_clip_norm`.
     - Step-level (`OneCycleLR`, `CyclicLR`, or 3-tuple `(opt, sched, "step")`) and epoch-level schedulers (`ReduceLROnPlateau` tracks `training.monitor`).
     - `RunTracker` atomic logging (`experiments/<name>/<run>/`) and `EarlyStopping` (`training.patience`).
     - Three bundles in `checkpoints/`: `model_checkpoint.bundle` (every epoch with optimizer/scheduler/scaler/RNG), `model_best.bundle` (inference-only when `monitor` improves), and `model_final.bundle` (inference-only at completion).
4. **Manual Tracking (`RunTracker`) & Validation Timelapses**:
   - Outside `BaseTrainer`, always use `with RunTracker(experiment=..., run_name=...) as run:`.
   - Inside `BaseTrainer`, `self.run` is active during `fit()` so you can call `self.run.log_validation_sample(...)`.
5. **Checkpoints vs Deployment & Zero-Torch Serving**:
   - PyTorch inference: `bundle = load_bundle("checkpoints/model_final.bundle"); preds = bundle.predict(tensor)`.
   - Production microservices: export via `export_onnx_as_bundle(...)` and serve pure NumPy arrays with `load_onnx_bundle(...)` / `ONNXRunner.predict(np_array)` (zero PyTorch).
6. **Zero Background Daemons & Fast Inspection**:
   - Inspect `.bundle` files in `< 5ms` via `uv run mangosta inspect <bundle> --json` (reads SafeTensors header without loading weights).
   - Generate offline HTML reports with `uv run mangosta report <run_or_bundle> -o report.html` or launch the ephemeral local viewer with `uv run mangosta view`.

---

## 4. Canonical Minimal Workflow

```python
from mangosta import (
    BaseTrainer,
    ModelRegistry,
    TrainerRegistry,
    load_config,
    register_model,
    register_trainer,
)

# See examples/trainer_custom.py for full custom hooks & validation sample logging
cfg = load_config("configs/exp_baseline.yaml")
model = ModelRegistry.build_from_config(cfg)
trainer = TrainerRegistry.build_from_config(cfg, model=model)
result = trainer.fit(train_loader, val_loader, resume_from=None)
```

Dry-run project validation before training:
```bash
uv run python .agents/skills/mangosta/scripts/verify_project.py --config configs/exp_baseline.yaml
```

---

## 5. Structured Exception Reference & Remediation

| Exception | Root Cause | Remediation |
| :--- | :--- | :--- |
| `ConfigError` | Unknown YAML key or invalid type in `MangostaConfig`. | Fix the suggested key or move custom keys under `data:` / `custom:`. |
| `RegistryCollisionError` | Duplicate `(family, version)` registration. | Bump SemVer (e.g. `1.1.0`) or pass `allow_override=True`. |
| `ModelNotFoundError` | Family/version missing in `ModelRegistry`. | Ensure the module with `@register_model` is imported. |
| `TrainerNotFoundError` | Family/version missing in `TrainerRegistry`. | Ensure the module with `@register_trainer` is imported. |
| `SchemaValidationError` | Tensor or NumPy input violates `ModelSchema`. | Match shape/dtype against `bundle.schema.inputs`. |
| `ResumeStateError` | Bundle lacks `train_state.pt`. | Resume from `model_checkpoint.bundle` (saved with `optimizer=`). |
| `BundleCorruptedError` | Missing archive files or SHA-256 mismatch. | Re-export or restore bundle from verified source. |
| `ImportError` | PyTorch-only API called in `[serving]` env. | Use `mangosta.serving` or install `mangosta[full]`. |

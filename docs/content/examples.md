# Runnable Examples (`examples/`)

All examples in [`examples/`](https://github.com/yeriel/mangosta/tree/main/examples) are self-contained, use temporary directories (leaving zero clutter in your repo), and are numbered in the exact order a developer should explore them.

---

## Run All Examples with `uv`

```bash
# 1. Sync environment
uv sync --group dev

# 2. Run the 4 progressive tutorials
uv run python examples/01_lifecycle_training_vs_production.py
uv run python examples/02_model_checkpoint_and_deterministic_resume.py
uv run python examples/03_model_export_and_inspection.py
uv run python examples/04_visual_tracking_and_timelapses.py
```

---

## Progressive Learning Map

| # | Script | Matching Guide | Core APIs Covered |
| :-: | :--- | :--- | :--- |
| **01** | [`01_lifecycle_training_vs_production.py`](https://github.com/yeriel/mangosta/blob/main/examples/01_lifecycle_training_vs_production.py) | [Guide 01: Lifecycle](guides/01_fast_prototyping.md) | `@register_model`, `ModelRegistry.build`, `RunTracker`, `save_bundle`, `export_onnx_as_bundle`, `load_onnx_bundle` |
| **02** | [`02_model_checkpoint_and_deterministic_resume.py`](https://github.com/yeriel/mangosta/blob/main/examples/02_model_checkpoint_and_deterministic_resume.py) | [Guide 02: Checkpoints & Resume](guides/02_fault_tolerance_and_resumption.md) | `save_bundle(..., optimizer=..., scheduler=...)`, `resume_training(..., restore_rng=True)` |
| **03** | [`03_model_export_and_inspection.py`](https://github.com/yeriel/mangosta/blob/main/examples/03_model_export_and_inspection.py) | [Guide 03: Export & Inspection](guides/03_production_serving_and_contracts.md) | `export_onnx`, `output_mapping`, `export_onnx_as_bundle`, `inspect_bundle(as_json=True)`, `generate_report` |
| **04** | [`04_visual_tracking_and_timelapses.py`](https://github.com/yeriel/mangosta/blob/main/examples/04_visual_tracking_and_timelapses.py) | [Guide 04: Visual Tracking](guides/05_visualization_and_timelapses.md) | `RunTracker.log_metrics`, `RunTracker.log_validation_sample`, `mangosta view` |

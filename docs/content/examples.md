# Runnable Examples (`examples/`)

All example scripts in [`examples/`](https://github.com/yeriel/mangosta/tree/main/examples) run standalone inside temporary directories (leaving zero clutter in your repo) and match the step-by-step tutorial.

---

## Run Locally with `uv`

```bash
# Sync repository environment
uv sync --group dev

# Run the 4 tutorials
uv run python examples/01_lifecycle_training_vs_production.py
uv run python examples/02_model_checkpoint_and_deterministic_resume.py
uv run python examples/03_model_export_and_inspection.py
uv run python examples/04_visual_tracking_and_timelapses.py
```

---

## Map of Examples

| # | Script | Companion Tutorial | Key Primitives |
| :-: | :--- | :--- | :--- |
| **01** | [`01_lifecycle_training_vs_production.py`](https://github.com/yeriel/mangosta/blob/main/examples/01_lifecycle_training_vs_production.py) | [Tutorial 03](tutorial/03_training_with_basetrainer.md) / [06](tutorial/06_onnx_export_and_serving.md) | `@register_model`, `ModelRegistry.build`, `RunTracker`, `save_bundle`, `export_onnx_as_bundle`, `load_onnx_bundle` |
| **02** | [`02_model_checkpoint_and_deterministic_resume.py`](https://github.com/yeriel/mangosta/blob/main/examples/02_model_checkpoint_and_deterministic_resume.py) | [Tutorial 04: Checkpoints & Resume](tutorial/04_checkpoints_and_resume.md) | `save_bundle(..., optimizer=..., scheduler=...)`, `resume_training(..., restore_rng=True)` |
| **03** | [`03_model_export_and_inspection.py`](https://github.com/yeriel/mangosta/blob/main/examples/03_model_export_and_inspection.py) | [Tutorial 06: ONNX Export & Serving](tutorial/06_onnx_export_and_serving.md) | `export_onnx`, `output_mapping`, `export_onnx_as_bundle`, `inspect_bundle(..., as_json=True, diff_path=...)` |
| **04** | [`04_visual_tracking_and_timelapses.py`](https://github.com/yeriel/mangosta/blob/main/examples/04_visual_tracking_and_timelapses.py) | [Tutorial 05: Tracking & View](tutorial/05_visual_tracking_and_view.md) | `RunTracker.log_metrics`, `RunTracker.log_validation_sample`, `mangosta view` |

# Mangosta Step-by-Step Learning Progression

This directory contains the canonical examples for **Mangosta**.
They are numbered in logical pedagogical order to provide the smoothest possible learning curve and best developer experience (DX).

---

## Pedagogical Progression

| # | Example Script | Purpose & Key Concepts | Difficulty |
| :-: | :--- | :--- | :-: |
| **01** | [`01_lifecycle_training_vs_production.py`](./01_lifecycle_training_vs_production.py) | **The Core Lifecycle**: Understand how Mangosta separates PyTorch training from Zero-PyTorch production serving (~70MB Docker microservice). | Beginner |
| **02** | [`02_model_checkpoint_and_deterministic_resume.py`](./02_model_checkpoint_and_deterministic_resume.py) | **Checkpointing & Resumption**: How to save crash-consistent checkpoints with exact bit-to-bit RNG state recovery (PyTorch CPU/CUDA, NumPy, Python). | Intermediate |
| **03** | [`03_model_export_and_inspection.py`](./03_model_export_and_inspection.py) | **Export & Inspection**: Export to standalone ONNX (`.onnx`) or ONNX Bundles (`.onnx.bundle`) with dynamic axes, sub-5ms inspection, and offline HTML reports. | Intermediate |
| **04** | [`04_visual_tracking_and_timelapses.py`](./04_visual_tracking_and_timelapses.py) | **Visual Tracking & View**: Log temporal validation sample timelapses across steps and inspect them in the zero-daemon `mangosta view` dashboard. | Advanced |

---

## How to Run the Examples

All examples are 100% self-contained and clean up after themselves.

```bash
# Run with uv:
uv run python examples/01_lifecycle_training_vs_production.py
uv run python examples/02_model_checkpoint_and_deterministic_resume.py
uv run python examples/03_model_export_and_inspection.py
uv run python examples/04_visual_tracking_and_timelapses.py

# Or with standard python:
python examples/01_lifecycle_training_vs_production.py
```

# Canonical Examples

Mangosta provides a curated suite of numbered, progressive runnable examples in the [`examples/`](https://github.com/yeriel/mangosta/tree/main/examples) directory. They are designed to accompany developers through the learning curve with zero guesswork.

---

## Overview of Examples

```text
examples/
├── 01_lifecycle_training_vs_production.py
├── 02_model_checkpoint_and_deterministic_resume.py
├── 03_model_export_and_inspection.py
├── 04_visual_tracking_and_timelapses.py
└── README.md
```

| # | Example Script | Primary Concepts | When to Use |
| :- | :--- | :--- | :--- |
| **01** | [`01_lifecycle_training_vs_production.py`](#example-1-lifecycle-training-vs-production) | Lifecycle separation, training vs production | First script to understand how training bundles differ from lightweight production inference. |
| **02** | [`02_model_checkpoint_and_deterministic_resume.py`](#example-2-checkpointing-deterministic-resumption) | Checkpoint bundles, 4-RNG state capture, spot-instance resumption | When setting up fault-tolerant training on spot/preemptible instances. |
| **03** | [`03_model_export_and_inspection.py`](#example-3-model-export-inspection-onnx-runtime) | Model summary, sub-5ms inspection, ONNX runtime conversion | When preparing models for packaging, auditing, and edge deployment. |
| **04** | [`04_visual_tracking_and_timelapses.py`](#example-4-visual-tracking-timelapses) | Metrics tracking, validation images, `mangosta view` scrubber | When logging scalar metrics and visual artifacts for temporal analysis. |

---

## Example 1: Lifecycle (Training vs Production)

**File:** `examples/01_lifecycle_training_vs_production.py`

This script demonstrates the clear separation of concerns in Mangosta:
1. **Training Phase (Requires PyTorch)**: Architecture registration with `@register_model`, training loop, tracking scalar loss, and packing the model with `save_bundle`.
2. **Production Phase (Zero PyTorch Dependency)**: Loading the bundle with `load_bundle`, inspecting the manifest, validating tensor contracts, and executing inference with zero training overhead.

### Run It:
```bash
python examples/01_lifecycle_training_vs_production.py
```

---

## Example 2: Checkpointing & Deterministic Resumption

**File:** `examples/02_model_checkpoint_and_deterministic_resume.py`

Demonstrates Mangosta's crash-consistent fault tolerance:
1. Starts a training loop for 5 epochs.
2. At epoch 5, saves a full checkpoint bundle containing optimizer states, scheduler states, AMP scaler states, and the states of all 4 random number generators (PyTorch CPU, PyTorch CUDA, NumPy, and standard Python).
3. Simulates a crash or spot instance preemption.
4. Resumes training deterministically from epoch 5 through epoch 10 using `resume_training()`.
5. Verifies that random sequences and loss trajectories match bit-for-bit.

### Run It:
```bash
python examples/02_model_checkpoint_and_deterministic_resume.py
```

---

## Example 3: Model Export, Inspection & ONNX Runtime

**File:** `examples/03_model_export_and_inspection.py`

Demonstrates how to summarize, inspect, and export models for non-PyTorch environments:
1. Prints an architectural layer and parameter summary.
2. Exports the model to a standalone ONNX bundle via `export_onnx_as_bundle()`.
3. Demonstrates pure-Python, zero-PyTorch binary header inspection in `< 5ms`.
4. Executes ONNX inference using `load_onnx_bundle()`.

### Run It:
```bash
python examples/03_model_export_and_inspection.py
```

---

## Example 4: Visual Tracking & Timelapses

**File:** `examples/04_visual_tracking_and_timelapses.py`

Demonstrates visual experiment monitoring and temporal validation timelapses:
1. Simulates a training run generating synthetic image predictions across 10 epochs.
2. Logs scalar metrics (`loss`, `accuracy`) and validation matplotlib figures via `tracker.log_image("predictions", fig, step=epoch)`.
3. Closes the tracker cleanly.
4. Provides instructions to launch `mangosta view` to explore the interactive timeline scrubber.

### Run It:
```bash
python examples/04_visual_tracking_and_timelapses.py
mangosta view
```
Open `http://127.0.0.1:8501` to scrub through the epoch-by-epoch images in your browser.

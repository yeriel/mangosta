# Checkpoints & RNG Resume

**Goal:** Recover interrupted training jobs **bit-to-bit identically**—including PyTorch CPU/CUDA, NumPy, and Python random states—without risking `pickle` corruption.

---

## 1. The Three `.bundle` Artifacts

Every `.bundle` is an atomic ZIP archive written via a temporary file + `os.replace`, so a crash mid-write never corrupts your last checkpoint.

| Artifact | How It Is Saved | ZIP Contents | Purpose |
| :--- | :--- | :--- | :--- |
| **1. Training Checkpoint** (`.bundle`) | `save_bundle(..., optimizer=opt, scheduler=sched)` | `manifest.json`, `model.safetensors`, `train_state.pt`, optional `config.yaml` | Crash recovery and bit-to-bit resumption (`model_checkpoint.bundle`). |
| **2. Inference Model** (`.bundle`) | `save_bundle(..., sample_input=x)` *(no optimizer)* | `manifest.json`, `model.safetensors`, optional `config.yaml` | Lightweight PyTorch sharing and evaluation (`model_best.bundle`, `model_final.bundle`). |
| **3. Production ONNX** (`.onnx.bundle`) | `export_onnx_as_bundle(...)` | `manifest.json`, `model.onnx`, optional `config.yaml` | Slim `~70MB` production serving without PyTorch. |

---

## 2. Resuming with `BaseTrainer` (One Flag)

If a spot GPU preempts or you interrupt with `Ctrl+C`, pass `resume_from=` (or `--resume` in `uv run train`):

```python
result = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    resume_from="checkpoints/model_checkpoint.bundle",
)
```

Or via your generated CLI script:

```bash
uv run train --config configs/exp_baseline.yaml --epochs 10 --resume checkpoints/model_checkpoint.bundle
```

Mangosta automatically restores:

- Model weights (`model.safetensors` verified via SHA-256)
- `optimizer.state_dict()`
- `scheduler.state_dict()` (if provided)
- `scaler.state_dict()` (when AMP is enabled)
- **4 RNG states**: PyTorch CPU, all PyTorch CUDA devices, NumPy `np.random`, and Python `random`

---

## 3. Manual Checkpointing & Resumption (Without `BaseTrainer`)

When writing a custom training script, pair `save_bundle()` with `resume_training()`:

```python
import torch
from mangosta import resume_training, save_bundle

# Save checkpoint mid-training
save_bundle(
    output_path="checkpoints/epoch_05.bundle",
    model=model,
    family="tabular_classifier",
    version="1.0.0",
    init_args={"in_features": 8, "num_classes": 2},
    sample_input=torch.randn(1, 8),       # (1)!
    optimizer=optimizer,                  # (2)!
    scheduler=scheduler,
    epoch=5,
    step=100,
    metrics={"val_loss": 0.241},
)
```

1. Infers `ModelSchema` automatically so downstream serving validates input shapes and dtypes.
2. Passing `optimizer` (or `scheduler` / `scaler`) automatically enables checkpoint mode (`is_checkpoint=True`) and snapshots all 4 RNG states.

To resume later:

```python
state = resume_training(
    bundle_path="checkpoints/epoch_05.bundle",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    restore_rng=True,
)
print(f"Resuming from epoch {state.epoch + 1} (step {state.step})")
```

!!! warning "Inference Bundles Cannot Resume Optimizer State"
    Calling `resume_training()` on an inference-only bundle (`model_final.bundle`) raises `ResumeStateError` because `train_state.pt` is intentionally omitted to keep production weights clean.

---

**Companion script:** [`examples/02_model_checkpoint_and_deterministic_resume.py`](https://github.com/yeriel/mangosta/blob/main/examples/02_model_checkpoint_and_deterministic_resume.py)

**Next step:** [Tracking & `mangosta view`](05_visual_tracking_and_view.md) — visualize curves, compare models side-by-side, and scrub validation predictions.

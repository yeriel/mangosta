# Guide 02: Checkpoints & Deterministic Resume

> **Companion script:** [`examples/02_model_checkpoint_and_deterministic_resume.py`](https://github.com/yeriel/mangosta/blob/main/examples/02_model_checkpoint_and_deterministic_resume.py)  
> **Run it locally:** `uv run python examples/02_model_checkpoint_and_deterministic_resume.py`

When a training run is interrupted (spot instance preemption, SLURM timeout, or `Ctrl+C`), standard PyTorch checkpoints often fail to resume deterministically because they lose the state of random number generators (RNG) and learning rate schedulers.

Mangosta guarantees **100% bit-to-bit repeatable resumption**.

---

## Step 1: Save a Crash-Consistent Checkpoint

When you pass `optimizer` (and optionally `scheduler` or `scaler`) to `save_bundle`, Mangosta automatically marks `is_checkpoint=True` and captures:
- Model weights in `model.safetensors` (Zero-Pickle + SHA-256).
- Optimizer, scheduler, and AMP `GradScaler` states.
- All 4 RNG states: **PyTorch CPU**, **PyTorch CUDA (all GPUs)**, **NumPy**, and **Python `random`**.

```python
import torch
from mangosta import save_bundle

save_bundle(
    output_path="checkpoints/checkpoint_epoch2.bundle",
    model=model,
    family="resumable_classifier",
    version="1.0.0",
    init_args={"in_features": 16, "hidden_dim": 32, "num_classes": 4},
    sample_input=torch.randn(1, 16),
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=2,
    step=6,
    metrics={"checkpoint_loss": float(loss.item())},
)
```

> **Atomic Write Guarantee:** `save_bundle` writes to a temporary file `.tmp_<pid>_<uuid>` first and executes an atomic `os.replace` only after all bytes are flushed. A power cut mid-save will never corrupt your previous checkpoint.

---

## Step 2: Resume Training Bit-to-Bit (`resume_training`)

After a crash or restart, instantiate fresh instances of your model, optimizer, and scheduler, and call `resume_training(..., restore_rng=True)`:

```python
import torch.optim as optim
from mangosta import ModelRegistry, resume_training

# 1. Build fresh instances
fresh_model = ModelRegistry.build("resumable_classifier", version="1.0.0")
fresh_opt = optim.Adam(fresh_model.parameters(), lr=1e-3)
fresh_sched = optim.lr_scheduler.StepLR(fresh_opt, step_size=2, gamma=0.5)

# 2. Restore weights, optimizer, scheduler, and all 4 RNG generators
state = resume_training(
    bundle_path="checkpoints/checkpoint_epoch2.bundle",
    model=fresh_model,
    optimizer=fresh_opt,
    scheduler=fresh_sched,
    restore_rng=True,
)

print(f"Resumed at epoch {state.epoch}, step {state.step} (RNG restored: {state.rng_restored})")

# 3. Continue training from the next epoch
for epoch in range(state.epoch + 1, total_epochs + 1):
    ...
```

---

## What `ResumeState` Returns

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `state.epoch` | `int \| None` | Last completed epoch saved in the checkpoint (`state.epoch + 1` is the next epoch). |
| `state.step` | `int \| None` | Global training step saved in the checkpoint. |
| `state.metrics` | `dict[str, float]` | Validation/training metrics recorded when the checkpoint was saved. |
| `state.rng_restored` | `bool` | `True` when CPU, CUDA, NumPy, and Python RNG states were restored. |

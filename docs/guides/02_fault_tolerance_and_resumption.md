# Guide 2: Fault Tolerance & Deterministic Resumption

In long-running PyTorch training runs (especially on spot instances, SLURM clusters, or shared GPU nodes), preemption and unexpected interruptions (`SIGINT`, `SIGTERM`, spot reclamation) are common.

Standard PyTorch checkpoints often fail to guarantee deterministic resumption because they omit:
1. Learning rate scheduler state and warmup counters.
2. AMP (Automatic Mixed Precision) `GradScaler` internal loss scale values.
3. Random Number Generator (RNG) states across PyTorch CPU, PyTorch CUDA (all devices), Python's `random`, and NumPy's `np.random`.

`mangosta` solves this by treating checkpoint bundles as atomic, complete state snapshots.

---

## 1. Saving a Fault-Tolerant Checkpoint

When saving a checkpoint, pass your optimizer, scheduler, and optional scaler directly to `save_bundle`:

```python
import torch
from mangosta import save_bundle

# In your training loop:
save_bundle(
    output_path=f"checkpoints/epoch_{epoch}.bundle",
    model=model,
    family="mlp_classifier",
    version="1.0.0",
    sample_input=torch.randn(1, 8),
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,                  # Optional torch.cuda.amp.GradScaler
    epoch=epoch,
    step=global_step,
    metrics={"val_loss": val_loss, "val_acc": val_acc},
    config="configs/baseline.yaml",
)
```

### What Happens Internally:
- Model weights are written in SafeTensors format (zero pickle, memory-mapped read speed).
- `_capture_rng_states()` snapshots:
  - `torch.get_rng_state()`
  - `torch.cuda.get_rng_state_all()` (for multi-GPU CUDA nodes)
  - `random.getstate()`
  - `np.random.get_state()`
- The bundle is written to a temporary file `.tmp_<pid>_<uuid>` and atomically replaced (`os.replace`), guaranteeing that a mid-write crash never corrupts existing checkpoints.

---

## 2. Resuming Training Deterministically

When restarting after an interruption, initialize fresh instances of your model, optimizer, and scheduler, and invoke `resume_training`:

```python
import torch
from mangosta import resume_training, ModelRegistry

# 1. Reconstruct model architecture from registry
model = ModelRegistry.build("mlp_classifier", version="1.0.0", in_features=8, hidden_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

# 2. Restore weights, optimizer, scheduler, and all RNG generators
state = resume_training(
    bundle_path="checkpoints/epoch_4.bundle",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    restore_rng=True,
    map_location="cuda" if torch.cuda.is_available() else "cpu",
)

# 3. Resume seamlessly at the exact next step
start_epoch = (state.epoch or 0) + 1
start_step = state.step or 0

print(f"Resumed from epoch {state.epoch}, step {state.step}. Prior metrics: {state.metrics}")
```

### Bit-to-Bit Determinism Guarantee:
Because all RNG generators are restored to their exact state at checkpoint time, dropout masks, data shuffling iterators, and stochastic gradient updates will match the uninterrupted run bit-for-bit.

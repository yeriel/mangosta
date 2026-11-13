# Training with `BaseTrainer`

**Goal:** Write only your forward pass, loss, and metrics while `BaseTrainer` automates device placement, AMP, gradient clipping, LR schedulers, `EarlyStopping`, tracking, and `.bundle` outputs.

---

## 1. Option A: Standard Supervised Training (`criterion=`)

For standard `(inputs, targets)` batches, you do not even need to subclass `BaseTrainer`:

```python
import torch.nn as nn
from mangosta import BaseTrainer

trainer = BaseTrainer(
    model=model,
    config=cfg,
    criterion=nn.CrossEntropyLoss(),  # (1)!
)
result = trainer.fit(train_loader=train_loader, val_loader=val_loader)
```

1. Uses `AdamW` by default, reads `epochs`, `learning_rate`, `amp`, `grad_clip_norm`, and `patience` from `cfg.training`, and logs `train_loss`, `val_loss`, and `lr` automatically.

---

## 2. Option B: Subclassing `BaseTrainer` (Recommended)

Override only the hooks you want to customize—your PyTorch code remains completely visible ("white-box"):

```python
import torch
import torch.nn as nn
from mangosta import BaseTrainer, register_trainer

@register_trainer(family="sensor_classifier", version="1.0.0")
class SensorTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, config=None, **kwargs) -> None:
        super().__init__(model, config, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):  # (1)!
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2
        )
        return optimizer, scheduler

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch  # (2)!
        return self.criterion(self.model(x), y)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        return {"val_loss": loss, "val_acc": acc}
```

1. Return an `Optimizer`, `(Optimizer, Scheduler)`, or a 3-tuple `(Optimizer, Scheduler, "step")` for batch-level schedulers.
2. Batches are already moved recursively to `self.device` before `training_step` and `validation_step` run.

---

## 3. Scheduler Modes (Step vs. Epoch)

| What `configure_optimizers()` Returns | When Mangosta Calls `.step()` |
| :--- | :--- |
| `optimizer` | No scheduler |
| `(optimizer, CosineAnnealingLR(...))` | Once per epoch |
| `(optimizer, ReduceLROnPlateau(...))` | Once per epoch with the value of `training.monitor` |
| `(optimizer, OneCycleLR(...))` or `CyclicLR(...)` | Automatically **every batch** |
| `(optimizer, custom_sched, "step")` | Explicitly **every batch** |

---

## 4. Logging Validation Images Inside `BaseTrainer`

During `fit()`, `self.run` holds the active `RunTracker`, so you can log visual validation comparisons in `validation_step()`:

```python
def validation_step(self, batch):
    x, y = batch
    pred = self.model(x)
    loss = self.criterion(pred, y)

    if self.run is not None:
        self.run.log_validation_sample(
            sample_id="val_sample_0",
            prediction=pred[0].detach().cpu(),
            input_image=x[0].detach().cpu(),
            ground_truth=y[0].detach().cpu(),
            step=self.current_epoch,
        )
    return {"val_loss": loss}
```

---

## 5. Custom Optimization Loops (GANs & Gradient Accumulation)

If you need multiple optimizers or manual `.backward()`, override `optimize_step(self, batch)` directly and return a `dict[str, float]` of step metrics:

```python
def optimize_step(self, batch) -> dict[str, float]:
    # Your custom zero_grad / backward / step logic here
    return {"loss": float(loss.item())}
```

---

## 6. What `trainer.fit(...)` Produces

Calling `result = trainer.fit(train_loader, val_loader)` returns:

```python
{
    "final_bundle": "/abs/path/to/checkpoints/model_final.bundle",
    "best_metric": 0.1423,
    "epochs_completed": 5,
    "history": [{"train_loss": ..., "val_loss": ..., "lr": ...}, ...],
}
```

And writes on disk:

- **`experiments/<experiment.name>/<run_name>/`** — `run.json`, `metrics.jsonl`, and `artifacts/`
- **`checkpoints/model_checkpoint.bundle`** — updated every epoch with optimizer, scheduler, scaler, and 4 RNG states
- **`checkpoints/model_best.bundle`** — inference-only bundle saved when `training.monitor` improves
- **`checkpoints/model_final.bundle`** — inference-only bundle saved at the end of training

---

**Next step:** [Checkpoints & RNG Resume](04_checkpoints_and_resume.md) — understand the three `.bundle` modes and bit-to-bit resumption.

# API Reference: Engine (`mangosta.engine`)

`mangosta.engine` provides the white-box `BaseTrainer` orchestrator connecting `MangostaConfig`, `RunTracker`, `EarlyStopping`, atomic checkpointing, and deterministic resumption.

---

## `BaseTrainer`

```python
class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: MangostaConfig | dict[str, Any] | None = None,
        criterion: nn.Module | Callable[[Any, Any], torch.Tensor] | None = None,
        device: str | torch.device | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None
```

### Instance Attributes

| Attribute | Description |
| :--- | :--- |
| `self.cfg` | Validated `MangostaConfig`. |
| `self.device` | Resolved `torch.device` (`auto` -> `cuda` / `mps` / `cpu`, with warning fallback if CUDA requested when unavailable). |
| `self.model` | `nn.Module` already moved to `self.device`. |
| `self.optimizer` / `self.scheduler` / `self.scaler` | Optimizer, optional scheduler, and AMP `GradScaler`. |
| `self.run` | Active `RunTracker` inside `fit()` (`None` outside `fit()`). |
| `self.current_epoch` / `self.global_step` | Current training counters. |

---

### Overridable Hooks

#### `configure_optimizers(self) -> Any`
Returns an `Optimizer`, `(Optimizer, Scheduler)`, or `(Optimizer, Scheduler, "step")`. Defaults to `AdamW` using `cfg.training.learning_rate` and `cfg.training.weight_decay`.

#### `training_step(self, batch: Any) -> torch.Tensor | dict[str, torch.Tensor]`
Computes loss (or dict of losses) for one batch. Required unless `criterion=` was passed to `__init__`.

#### `validation_step(self, batch: Any) -> torch.Tensor | dict[str, torch.Tensor | float] | None`
Computes validation metrics per batch under `torch.no_grad()` and `model.eval()`.

#### `optimize_step(self, batch: Any) -> dict[str, float]`
Full per-batch execution (`zero_grad`, `autocast`, `training_step`, `scaler.scale(loss).backward()`, optional `clip_grad_norm_`, `scaler.step`, step-scheduler `.step()`). Override only for GANs or gradient accumulation.

---

### `fit`

```python
def fit(
    self,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any] | None = None,
    epochs: int | None = None,
    resume_from: str | Path | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    experiments_dir: str | Path = "experiments",
    sample_input: Any | None = None,
    verbose: bool = True,
) -> dict[str, Any]
```

Executes the full lifecycle and returns:

```python
{
    "final_bundle": str,         # Path to checkpoints/model_final.bundle
    "best_metric": float | None, # Best observed cfg.training.monitor value
    "epochs_completed": int,
    "history": list[dict[str, float]],
}
```

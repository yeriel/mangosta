# Config & Registries

**Goal:** Eliminate silent YAML typos and fragile `pickle` checkpoints using `MangostaConfig`, `@register_model`, and `@register_trainer`.

---

## 1. Strict Configuration (`MangostaConfig`)

Mangosta divides configuration into **3 strict sections** (`experiment`, `model`, `training`) and **2 free-form sections** (`data`, `custom`):

```yaml title="configs/exp_baseline.yaml"
experiment:
  name: "exp_baseline"
  seed: 42
  device: "auto"           # (1)!

model:
  family: "sensor_classifier"
  version: "1.0.0"
  init_args:               # (2)!
    in_features: 8
    hidden_dim: 32
    num_classes: 3

training:
  epochs: 5
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 0.0001
  amp: false
  grad_clip_norm: null
  patience: null           # (3)!
  monitor: "val_loss"
  monitor_mode: "min"

data:                      # (4)!
  dataset_dir: "./data"
  num_workers: 2

custom:
  notes: "Initial baseline"
```

1. Allowed values: `"auto"` (`cuda` -> `mps` -> `cpu`), `"cuda"`, `"mps"`, or `"cpu"`.
2. Passed directly as keyword arguments (`**init_args`) when building the model.
3. Integer enabling `EarlyStopping` after N epochs without improvement on `monitor`, or `null` to disable.
4. Put dataset paths, augmentation flags, or arbitrary project metadata under `data:` or `custom:`.

### Catch Typos Immediately

Load any YAML file (or dict) with `load_config()`:

```python
from mangosta import load_config

cfg = load_config("configs/exp_baseline.yaml")
print(cfg.format_summary())
```

If someone writes `lrearning_rate: 0.001` inside `training:`, `load_config()` fails immediately before training starts:

```text
ConfigError: Unknown parameter 'lrearning_rate' in section 'training' (configs/exp_baseline.yaml). Did you mean 'learning_rate'?
Valid parameters for 'training' are: ['amp', 'batch_size', 'epochs', 'grad_clip_norm', 'learning_rate', 'monitor', 'monitor_mode', 'patience', 'weight_decay']
```

---

## 2. Deterministic Seeding (`seed_everything`)

When using `BaseTrainer`, seeding runs automatically. Outside `BaseTrainer`, call:

```python
from mangosta import seed_everything

seed_everything(cfg.experiment.seed)  # Sets Python, PYTHONHASHSEED, NumPy, PyTorch CPU & CUDA
```

---

## 3. Immutable Registries (`@register_model` & `@register_trainer`)

Decorate your architectures and trainers with a `family` and **SemVer** string:

```python title="src/models.py"
import torch
import torch.nn as nn
from mangosta import BaseTrainer, MangostaConfig, register_model, register_trainer

@register_model(family="sensor_classifier", version="1.0.0")
class SensorClassifier(nn.Module):
    def __init__(self, in_features: int = 8, hidden_dim: int = 32, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@register_trainer(family="sensor_classifier", version="1.0.0")
class SensorTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, config: MangostaConfig | dict | None = None, **kwargs) -> None:
        super().__init__(model, config, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch):
        x, y = batch
        return self.criterion(self.model(x), y)
```

### Build Dynamically from Config

Notice how entrypoints never hardcode model classes or constructor arguments:

```python
from mangosta import ModelRegistry, TrainerRegistry, load_config
import src.models  # (1)!

cfg = load_config("configs/exp_baseline.yaml")
model = ModelRegistry.build_from_config(cfg)
trainer = TrainerRegistry.build_from_config(cfg, model=model)
```

1. Ensure the module containing `@register_model` / `@register_trainer` is imported so decorators populate the registries.

### Key Registry Guarantees

- **No Silent Overwrites:** Registering the same `(family, version)` twice raises `RegistryCollisionError`. Bump `"1.1.0"` or pass `allow_override=True` explicitly.
- **Latest Version Resolution:** Calling `ModelRegistry.build("sensor_classifier")` without `version=` resolves the highest semantic version automatically.
- **Zero-Pickle Bundle Loading:** Later, `load_bundle("model_final.bundle")` reads `family`, `version`, and `init_args` from `manifest.json` and reconstructs your module cleanly before loading SafeTensors weights.

---

**Next step:** [Training with `BaseTrainer`](03_training_with_basetrainer.md) — write clean training and validation steps while Mangosta automates AMP, schedulers, tracking, and bundle outputs.

# API Reference: Core (`mangosta.core`)

`mangosta.core` contains typed configuration (`MangostaConfig`), deterministic seeding, SemVer registries (`ModelRegistry`, `TrainerRegistry`), and I/O tensor contracts (`ModelSchema`).

---

## 1. Configuration & Seeding

### `load_config`

```python
def load_config(path_or_dict: str | Path | dict[str, Any] | MangostaConfig) -> MangostaConfig
```

Loads a YAML file or dict into a validated `MangostaConfig`. Unknown keys under root, `experiment`, or `training` raise `ConfigError` with `"Did you mean '...'?"` suggestions.

### `MangostaConfig`

| Section | Class / Type | Fields & Defaults |
| :--- | :--- | :--- |
| `experiment` | `ExperimentConfig` | `name: str = "baseline"`, `seed: int = 42`, `device: str = "auto"` (`"auto"`, `"cpu"`, `"cuda"`, `"mps"`) |
| `model` | `ModelConfig` | `family: str = ""`, `version: str = "1.0.0"`, `init_args: dict[str, Any] = {}` |
| `training` | `TrainingConfig` | `epochs: int = 10`, `batch_size: int = 16`, `learning_rate: float = 1e-3`, `weight_decay: float = 0.0`, `amp: bool = False`, `grad_clip_norm: float \| None = None`, `patience: int \| None = None`, `monitor: str = "val_loss"`, `monitor_mode: str = "min"` |
| `data` | `dict[str, Any]` | Free-form dataset configuration |
| `custom` | `dict[str, Any]` | Free-form user metadata |

Methods: `to_dict() -> dict[str, Any]`, `format_summary(resolved_device: str | None = None) -> str`.

### `seed_everything`

```python
def seed_everything(seed: int = 42) -> None
```

Fixes seeds across Python `random`, `PYTHONHASHSEED`, NumPy, PyTorch CPU, and all CUDA GPUs (when PyTorch is installed).

---

## 2. Registries

### `@register_model` & `ModelRegistry`

```python
def register_model(
    family: str,
    version: str,
    allow_override: bool = False,
) -> Callable[[type[nn.Module]], type[nn.Module]]
```

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `get_model_cls` | `(family: str, version: str \| None = None) -> tuple[type, str]` | Returns `(cls, resolved_version)`. Picks highest SemVer if `version=None`. |
| `build` | `(family: str, version: str \| None = None, **init_args: Any) -> nn.Module` | Instantiates model by coordinates. |
| `build_from_config` | `(config: MangostaConfig \| dict \| str \| Path) -> nn.Module` | Instantiates using `cfg.model.family`, `version`, and `init_args`. |
| `list_models` | `() -> list[dict[str, str]]` | Lists registered families, versions, class names, and modules. |

### `@register_trainer` & `TrainerRegistry`

```python
def register_trainer(
    family: str,
    version: str = "1.0.0",
    allow_override: bool = False,
) -> Callable[[C], C]
```

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `get_trainer_cls` | `(family: str, version: str \| None = None) -> tuple[type, str]` | Resolves exact or highest SemVer trainer subclass. |
| `build` | `(family: str, version: str \| None = None, **kwargs: Any) -> BaseTrainer` | Instantiates trainer directly. |
| `build_from_config` | `(config: Any, model: Any, **kwargs: Any) -> BaseTrainer` | Instantiates using `cfg.model.family` and `cfg.model.version`, passing `model=model, config=cfg`. |
| `list_trainers` | `() -> list[dict[str, str]]` | Lists registered trainer classes. |

### `register_component` & `get_component`

```python
def register_component(kind: str, name: str, allow_override: bool = False) -> Callable[[C], C]
```

Decorator and lookup (`ModelRegistry.get_component(kind, name)`) for auxiliary factories (losses, optimizers, schedulers).

---

## 3. Schema Contracts (`mangosta.core.schema`)

### `TensorSpec`

```python
@dataclass
class TensorSpec:
    shape: list[int | None]   # None or -1 represents a dynamic dimension
    dtype: str                # Canonical dtype e.g. "float32", "int64"
    name: str = "tensor"
```

Method: `matches(tensor: Any, allow_batch_dynamic: bool = True) -> tuple[bool, str | None]`.

### `ModelSchema`

```python
@dataclass
class ModelSchema:
    inputs: list[TensorSpec] | dict[str, TensorSpec]
    outputs: list[TensorSpec] | dict[str, TensorSpec]
    input_structure: str = "tensor"   # "tensor", "sequence", or "mapping"
    output_structure: str = "tensor"  # "tensor", "sequence", or "mapping"
    metadata: dict[str, Any] = field(default_factory=dict)
```

| Method | Behavior |
| :--- | :--- |
| `validate_input(data)` / `validate_output(data)` | Raises `SchemaValidationError` on mismatch. |
| `check_input(data)` / `check_output(data)` | Returns `(is_valid: bool, error_message: str \| None)`. |
| `is_valid_input(data)` / `is_valid_output(data)` | Returns `bool`. |
| `to_dict()` / `from_dict(data)` | JSON-serializable round-trip. |

### `infer_schema` & `infer_schema_from_sample`

```python
def infer_schema(model: nn.Module, sample_input: Any, dynamic_batch_dim: int | None = 0) -> ModelSchema
def infer_schema_from_sample(sample_input: Any, sample_output: Any, dynamic_batch_dim: int | None = 0) -> ModelSchema
```

- `infer_schema` runs one `torch.inference_mode()` pass while preserving `model.training`.
- `infer_schema_from_sample` works without PyTorch using NumPy arrays or Tensor-like objects.

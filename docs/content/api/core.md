# API Reference: Core (`mangosta.core`)

The `mangosta.core` module defines the foundational primitives for architecture registration, deterministic version resolution, and architectural decoupling.

---

## `@register_model`

```python
def register_model(
    family: str,
    version: str,
    allow_override: bool = False,
) -> Callable[[type[nn.Module]], type[nn.Module]]:
```

Decorator to register a PyTorch `nn.Module` class under a model family and semantic version.

### Parameters
- **`family`**: Architectural family identifier (e.g. `"classifier"`, `"vit"`, `"unet"`).
- **`version`**: Semantic version string (e.g. `"1.0.0"`, `"2.1.0"`).
- **`allow_override`**: If `False` (default), attempting to register the exact same `(family, version)` raises `RegistryCollisionError` with actionable remediation hints.

### Example
```python
from mangosta import register_model
import torch.nn as nn

@register_model(family="mlp", version="1.0.0")
class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 10, out_features: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)
```

---

## `ModelRegistry`

Thread-safe registry managing registered model classes and auxiliary lifecycle components.

### Methods

#### `register_model(family: str, version: str, model_cls: type, allow_override: bool = False) -> None`
Register an architecture programmatically without decorators.

#### `get_model_cls(family: str, version: str | None = None) -> tuple[type, str]`
Retrieve a registered architecture class:
- If `version` is specified (e.g. `"1.0.0"`), retrieves that exact version.
- If `version` is `None` or omitted, resolves to the latest semantic version available.
- Returns `(model_class, resolved_version)`.
- Raises `ModelNotFoundError` if family is unknown or version is missing.

#### `build(family: str, version: str | None = None, **init_args: Any) -> nn.Module`
Instantiate a registered architecture by family name, forwarding `init_args` to its constructor.

#### `list_families() -> list[str]`
List all currently registered model family names.

#### `list_versions(family: str) -> list[str]`
List all semantic versions available for a given family, sorted in ascending semver order.

---

## `register_component` & `get_component`

```python
def register_component(kind: str, name: str, component: Any, allow_override: bool = False) -> None
def get_component(kind: str, name: str) -> Any
```

Register and retrieve auxiliary training components (e.g., loss functions, optimizers, learning rate schedulers, or data augmentations).

### Parameters
- **`kind`**: Type of component (e.g., `"loss"`, `"optimizer"`, `"scheduler"`).
- **`name`**: Identifier name (e.g., `"focal_loss"`, `"cosine_warmup"`).
- **`component`**: Callable class or factory function.

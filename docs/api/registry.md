# API Reference: Registry (`mangosta.core.registry`)

The registry module provides deterministic, version-controlled architecture registration. It decouples model definitions from persistence files, allowing bundles to be serialized without pickling arbitrary Python bytecode.

---

## `@register_model`

```python
def register_model(
    family: str,
    version: str,
    allow_override: bool = False,
) -> Callable[[T], T]:
```

Decorator to register a PyTorch `nn.Module` class under a model family and semantic version.

### Parameters
- **`family`**: Architectural family name (e.g. `"classifier"`, `"segmentation_unet"`).
- **`version`**: Semantic version string (e.g. `"1.0.0"`, `"2.1.0"`).
- **`allow_override`**: If `False` (default), attempting to register the exact same `(family, version)` raises `RegistryCollisionError` to prevent silent architectural overrides.

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

Global thread-safe registry tracking registered model architectures and auxiliary components.

### Class Methods

#### `register_model(family: str, version: str, model_cls: type, allow_override: bool = False) -> None`
Register a model class programmatically without using decorators.

#### `get_model_cls(family: str, version: str | None = None) -> tuple[type, str]`
Look up a model class.
- If `version` is specified, retrieves the exact version matching.
- If `version` is omitted or `None`, resolves the latest semantic version available.
- Returns `(model_class, resolved_version_string)`.
- Raises `ModelNotFoundError` with actionable hints if family is missing or version is invalid.

#### `build(family: str, version: str | None = None, **init_args: Any) -> Any`
Instantiate a registered model by family and version, forwarding `**init_args` to its `__init__` constructor.

#### `register_component(kind: str, name: str, component: Any, allow_override: bool = False) -> None`
Register auxiliary training components (such as custom loss functions, optimizers, or learning rate schedulers).

#### `get_component(kind: str, name: str) -> Any`
Retrieve a previously registered auxiliary component.

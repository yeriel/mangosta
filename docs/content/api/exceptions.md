# API Reference: Exceptions (`mangosta.exceptions`)

Mangosta raises structured, domain-specific exceptions rather than raw `KeyError` or `RuntimeError` failures.

---

## Inheritance Hierarchy

```text
Exception
└── MangostaError
    ├── RegistryError
    │   ├── RegistryCollisionError
    │   ├── ModelNotFoundError
    │   ├── TrainerNotFoundError
    │   └── ComponentNotFoundError
    ├── ConfigError
    ├── SchemaError
    │   └── SchemaValidationError
    └── BundleError
        ├── BundleCorruptedError
        └── ResumeStateError
```

---

## Reference Table

| Exception | When Raised | Actionable Fix |
| :--- | :--- | :--- |
| `MangostaError` | Base class for all library errors. | Catch to handle any Mangosta failure generically. |
| `ConfigError` | Invalid YAML, wrong field type, or unknown key in `MangostaConfig`. | Follow the `"Did you mean '...'?"` suggestion or move custom fields under `data:` / `custom:`. |
| `RegistryCollisionError` | Duplicate `(family, version)` in `ModelRegistry` or `TrainerRegistry`, or duplicate component. | Bump semantic version (`"1.1.0"`) or pass `allow_override=True`. |
| `ModelNotFoundError` | Requested model family or version missing in `ModelRegistry`. | Import the module defining `@register_model(...)`. |
| `TrainerNotFoundError` | Requested trainer family or version missing in `TrainerRegistry`. | Import the module defining `@register_trainer(...)`. |
| `ComponentNotFoundError` | Requested auxiliary component missing in `ModelRegistry`. | Register it with `@register_component(kind, name)` before lookup. |
| `SchemaError` | Invalid schema structure or definition. | Check `ModelSchema` input/output structure (`tensor`, `sequence`, `mapping`). |
| `SchemaValidationError` | Tensor or NumPy array fails shape, rank, or dtype validation in `predict()`. | Match `runner.schema.inputs` shape and dtype. |
| `BundleError` | Base class for archive errors. | Check bundle file path and permissions. |
| `BundleCorruptedError` | Invalid ZIP, missing `manifest.json` / weights, or SHA-256 mismatch. | Re-save or re-export the bundle from a verified checkpoint. |
| `ResumeStateError` | `resume_training()` called on a bundle missing `train_state.pt`. | Use a checkpoint bundle saved with `optimizer=` (e.g. `model_checkpoint.bundle`). |

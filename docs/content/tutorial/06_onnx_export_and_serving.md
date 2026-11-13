# ONNX Export & Serving

**Goal:** Ship trained PyTorch models into `~70MB` production containers using pure NumPy + ONNX Runtime, with strict input/output contract validation.

---

## 1. Export to `.onnx` or `.onnx.bundle`

Requires `mangosta[full]` (includes PyTorch and ONNX):

=== "From Python"

    ```python
    from mangosta import export_onnx_as_bundle

    # Automatically infers input shape from the bundle's embedded ModelSchema
    export_onnx_as_bundle(
        model_or_bundle="checkpoints/model_final.bundle",
        output_path="deploy/model.onnx.bundle",
    )
    ```

    For models returning a `dict[str, torch.Tensor]` (e.g., `{"logits": ..., "bbox": ...}`), pass `output_mapping` to unroll outputs in a fixed order:

    ```python
    export_onnx_as_bundle(
        model_or_bundle="detector.bundle",
        output_path="deploy/detector.onnx.bundle",
        output_mapping=["logits", "bbox"],
        input_names=["image"],
        output_names=["pred_logits", "pred_bbox"],
        float16=False,
    )
    ```

=== "From CLI"

    ```bash
    # Auto-infers dummy input shape from ModelSchema when --input-shape is omitted
    uv run mangosta export-onnx checkpoints/model_final.bundle \
        -o deploy/model.onnx.bundle \
        -m src.models
    ```

    Pass `--input-shape 1 3 224 224` explicitly if the source bundle was saved without `sample_input`, or `--fp16` for Float16 export.

---

## 2. Inspect & Diff Bundles in `<5ms`

Before deploying, inspect or compare bundles in terminal **without importing PyTorch**:

```bash
# Single bundle summary or machine-readable JSON
uv run mangosta inspect deploy/model.onnx.bundle --json

# Compare two bundles side-by-side (params, metrics, schema, layers)
uv run mangosta inspect checkpoints/model_best.bundle --diff checkpoints/model_final.bundle
```

---

## 3. Serve in Production Without PyTorch (`mangosta[serving]`)

Inside your production microservice (`uv add "mangosta[serving]"`), PyTorch is completely absent. `load_onnx_bundle()` verifies the SHA-256 checksum of `model.onnx`, mounts the graph into ONNX Runtime directly from memory, and enforces `ModelSchema` on every request:

```python
import numpy as np
from mangosta import load_onnx_bundle

runner = load_onnx_bundle("deploy/model.onnx.bundle")  # (1)!

batch = np.random.randn(4, 8).astype(np.float32)
predictions = runner.predict(batch)                   # (2)!
```

1. Uses only `onnxruntime` and NumPy—works cleanly in a `~70MB` Python slim image.
2. Validates `shape` (`[None, 8]`) and `dtype` (`float32`), executes inference, and reconstructs dict/tuple outputs automatically when `output_mapping` or `ModelSchema` is present.

!!! tip "Universal `load_bundle()`"
    You can also call `load_bundle("deploy/model.onnx.bundle")` — Mangosta automatically detects ONNX format and delegates to `load_onnx_bundle()`.

---

**Next step:** Explore [CLI & AI Agents](../cli_and_agents.md) or browse the [API Reference](../api/core.md).

# 100% Agent-Friendly & CLI Reference

> Mangosta is designed for both human engineers and autonomous AI coding agents (Antigravity, Claude Code, Cursor, Codex). Every CLI command and exception is structured to eliminate token waste, memory overhead, and guesswork.

---

## 1. Sub-5ms Zero-PyTorch Inspection (`inspect --json`)

When an AI agent needs to check a `.bundle` checkpoint, calling `torch.load()` in Python takes seconds and allocates gigabytes of RAM/VRAM.

Mangosta's `inspect --json` reads only `manifest.json` and the 8-byte little-endian (`struct.unpack("<Q")`) SafeTensors header directly from the ZIP stream—**without importing PyTorch or decompressing weights**:

```bash
uv run mangosta inspect checkpoints/epoch_3.bundle --json
```

```json
{
  "file": {
    "name": "epoch_3.bundle",
    "path": "/workspace/checkpoints/epoch_3.bundle",
    "size_bytes": 14208
  },
  "family": "tabular_classifier",
  "version": "1.0.0",
  "format": "safetensors",
  "weights_size_bytes": 2432,
  "weights_sha256": "4b9e4a3b8c7e2f1d...",
  "is_checkpoint": true,
  "epoch": 3,
  "step": 15,
  "schema": {
    "inputs": [{"name": "input_0", "shape": [null, 8], "dtype": "float32"}],
    "outputs": [{"name": "output_0", "shape": [null, 2], "dtype": "float32"}],
    "input_structure": "tensor",
    "output_structure": "tensor"
  },
  "metrics": {"val_loss": 0.281},
  "files": ["manifest.json", "model.safetensors", "train_state.pt"],
  "tensors": [
    {"name": "net.0.weight", "shape": [32, 8], "dtype": "F32"},
    {"name": "net.0.bias", "shape": [32], "dtype": "F32"}
  ]
}
```

---

## 2. Embedded Agent Context (`.agents/mangosta.md` & `SKILL.md`)

1. **Project Scaffolding (`mangosta init`)**:
   Running `uv run mangosta init my_project` automatically generates `.agents/mangosta.md` inside the project with all architectural rules, so any AI agent opening the repository knows immediately how to register models, track runs, and save bundles.
2. **Native Antigravity Skill (`skills/mangosta/SKILL.md`)**:
   Ships directly in the repository with canonical copy-paste patterns for training, bit-to-bit resumption, and zero-PyTorch ONNX serving.

---

## 3. Actionable Exception Hierarchy

Instead of generic `RuntimeError` or `KeyError` tracebacks, Mangosta raises specific, self-explanatory exceptions from `mangosta.exceptions` with remediation guidance in the message:

| Exception | When It Is Raised | How to Fix It |
| :--- | :--- | :--- |
| `RegistryCollisionError` | Registering an existing `(family, version)` pair twice. | Bump the semantic version (`"1.1.0"`) or pass `allow_override=True`. |
| `ModelNotFoundError` | `ModelRegistry` cannot find the requested family/version. | Import the module where `@register_model` is defined before loading. |
| `SchemaValidationError` | Input tensor/NumPy array shape or `dtype` violates `ModelSchema`. | Match the expected shape and `dtype` shown in `runner.schema.inputs`. |
| `BundleCorruptedError` | Missing archive files or SHA-256 checksum mismatch on `model.safetensors` / `model.onnx`. | Re-export the bundle from a verified checkpoint. |
| `ResumeStateError` | Calling `resume_training()` on an inference bundle that lacks `train_state.pt`. | Pass `optimizer=` when calling `save_bundle()` to save a resumable checkpoint. |

---

## 4. Complete CLI Reference

All CLI commands are invoked via `uv run mangosta <command>`:

| Command & Flags | Description |
| :--- | :--- |
| `mangosta init [name] [--template default]` | Scaffold a Clean Architecture PyTorch project (`configs/`, `src/`, `entrypoints/`, `.agents/mangosta.md`). |
| `mangosta inspect <bundle> [--json]` | Inspect bundle manifest, `ModelSchema` contract, and SafeTensors layer shapes in `< 5ms`. |
| `mangosta view [target] [--port 8501] [--no-browser]` | Start the zero-daemon local browser dashboard for `target` directory or `.bundle`. |
| `mangosta report <target> [-o report.html]` | Generate a standalone, offline interactive HTML report for a run directory or `.bundle`. |
| `mangosta export-onnx <bundle> -o <out> [--input-shape ...] [--output-mapping ...] [--fp16] [--opset 18] [-m module]` | Export a `.bundle` directly to standalone `.onnx` or `.onnx.bundle`. |

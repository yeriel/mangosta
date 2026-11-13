# CLI & AI Agents

Mangosta's CLI keeps the surface area minimal (**4 commands + `--version`**) and pairs fast zero-PyTorch JSON outputs with project scaffolding designed for coding agents (Antigravity, Cursor, Claude Code, Codex).

---

## 1. Complete CLI Reference

Invoke commands via `uv run mangosta <command>`:

| Command & Options | Description |
| :--- | :--- |
| `mangosta --version` / `-V` | Print installed Mangosta version. |
| `mangosta init [name]` | Scaffold a Clean Architecture project (`configs/`, `src/`, `entrypoints/`, `AGENTS.md`, `.agents/skills/mangosta/`) and run `uv sync`. |
| `mangosta inspect <bundle> [--diff <other>] [--config] [--json]` | Inspect or diff `.bundle` / `.onnx.bundle` archives in `<5ms` **without importing PyTorch**. |
| `mangosta view [target] [--port 8501] [--no-browser]` | Start the ephemeral local browser dashboard (`Metrics & Runs`, `Validation`, `Bundles`, `Metadata`). |
| `mangosta export-onnx <bundle> -o <out> [--input-shape ...] [--output-mapping ...] [--fp16] [--opset 18] [-m module] [--json]` | Export a `.bundle` to standalone `.onnx` or `.onnx.bundle`. Infers `--input-shape` from `ModelSchema` automatically when omitted. |

---

## 2. Sub-5ms Inspection & Diff (`inspect`)

When CI checks or AI agents need to inspect a checkpoint or compare two models, calling `torch.load()` wastes seconds and gigabytes of memory.

`mangosta inspect` parses only `manifest.json`, `config.yaml`, and the 8-byte little-endian (`struct.unpack("<Q")`) SafeTensors binary header directly from the ZIP archive—**zero PyTorch imports, zero tensor allocations**.

=== "Inspect JSON"

    ```bash
    uv run mangosta inspect checkpoints/model_best.bundle --json
    ```

    ```json
    {
      "file": {
        "name": "model_best.bundle",
        "path": "/workspace/checkpoints/model_best.bundle",
        "size_bytes": 14208
      },
      "family": "sensor_classifier",
      "version": "1.0.0",
      "format": "safetensors",
      "weights_size_bytes": 2432,
      "weights_sha256": "4b9e4a3b8c7e2f1d...",
      "is_checkpoint": false,
      "epoch": 5,
      "step": 50,
      "total_params": 387,
      "schema": {
        "inputs": [{"name": "input", "shape": [null, 8], "dtype": "float32"}],
        "outputs": [{"name": "output", "shape": [null, 3], "dtype": "float32"}],
        "input_structure": "tensor",
        "output_structure": "tensor"
      },
      "metrics": {"train_loss": 0.91, "val_loss": 0.88},
      "files": ["manifest.json", "config.yaml", "model.safetensors"],
      "tensors": [
        {"name": "net.0.weight", "shape": [32, 8], "dtype": "F32", "param_count": 256, "size_bytes": 1024}
      ]
    }
    ```

=== "Diff Two Bundles"

    ```bash
    uv run mangosta inspect checkpoints/model_best.bundle --diff checkpoints/model_final.bundle --json
    ```

    Compares metadata, `total_params`, recorded metrics, `ModelSchema` contracts, and layer shapes/dtypes across any two `.bundle` or `.onnx.bundle` archives.

---

## 3. Embedded AI Agent Guidance (`AGENTS.md` + Skill)

Running `uv run mangosta init` creates two complementary agent assets inside every project:

1. **`AGENTS.md`**: Concise project-root instructions enforcing KISS and SOLID separation across `configs/`, `src/`, and `entrypoints/`.
2. **`.agents/skills/mangosta/`**:
   - `SKILL.md`: Complete v0.4.0 API & exception reference.
   - `resources/config_schema.yaml`: Full documented `MangostaConfig` YAML template.
   - `examples/`: Canonical `BaseTrainer` and `ONNXRunner` snippets.
   - `scripts/verify_project.py`: Fast dry-run validator (`uv run python .agents/skills/mangosta/scripts/verify_project.py --config configs/exp_baseline.yaml`).

---

## 4. Actionable Exceptions

Instead of opaque tracebacks, Mangosta raises typed exceptions from `mangosta.exceptions` with concrete hints:

| Exception | Cause | Fix |
| :--- | :--- | :--- |
| `ConfigError` | Invalid type or typo in `MangostaConfig` YAML. | Fix the suggested key (`Did you mean '...'?`) or move custom fields under `data:` / `custom:`. |
| `RegistryCollisionError` | Duplicate `(family, version)` registration. | Bump SemVer (`"1.1.0"`) or pass `allow_override=True`. |
| `ModelNotFoundError` / `TrainerNotFoundError` | Family or version not registered yet. | Import the module defining `@register_model` / `@register_trainer`. |
| `SchemaValidationError` | Input or output violates `ModelSchema`. | Match shape and dtype in `bundle.schema.inputs`. |
| `BundleCorruptedError` | Missing archive files or SHA-256 mismatch. | Re-export from a verified checkpoint. |
| `ResumeStateError` | `resume_training()` called on an inference bundle. | Resume from `model_checkpoint.bundle` (saved with `optimizer=`). |

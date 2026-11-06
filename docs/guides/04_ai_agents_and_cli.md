# Guide 4: Autonomous AI Agents & Machine-Readable CLI

Modern software engineering workflows increasingly leverage autonomous AI coding assistants (such as Antigravity, Cursor, Claude Code, and GitHub Copilot).

Traditional MLOps tools are hostile to AI agents:
- They require background daemons or web sockets that agents cannot easily inspect or control.
- Inspecting checkpoints requires importing PyTorch or running Python scripts, generating heavy terminal output and token exhaustion.
- Terminal outputs are styled with ANSI escape codes and ASCII boxes, confusing language models.

`mangosta` is designed to be **agent-native**.

---

## 1. Machine-Readable Inspection (`mangosta inspect --json`)

AI agents can inspect `.bundle` files in sub-milliseconds without importing PyTorch or allocating GPU memory:

```bash
mangosta inspect checkpoints/epoch_10.bundle --json
```

### Output JSON Payload:
```json
{
  "file": "checkpoints/epoch_10.bundle",
  "family": "mlp_classifier",
  "version": "1.0.0",
  "format": "safetensors",
  "weights_size_bytes": 14208,
  "weights_sha256": "4b9e4a3b8c7e2f1d9a0b...",
  "is_checkpoint": true,
  "epoch": 10,
  "step": 1250,
  "schema": {
    "inputs": [{"name": "input", "shape": [null, 8], "dtype": "float32"}],
    "outputs": [{"name": "output", "shape": [null, 2], "dtype": "float32"}],
    "input_structure": "tensor",
    "output_structure": "tensor"
  },
  "metrics": {
    "val_loss": 0.281,
    "val_acc": 0.945
  },
  "files": [
    "manifest.json",
    "model.safetensors",
    "train_state.pt",
    "config.yaml"
  ],
  "tensors": [
    {"name": "net.0.weight", "shape": [32, 8], "dtype": "F32"},
    {"name": "net.0.bias", "shape": [32], "dtype": "F32"},
    {"name": "net.2.weight", "shape": [2, 32], "dtype": "F32"},
    {"name": "net.2.bias", "shape": [2], "dtype": "F32"}
  ]
}
```

### Why this is powerful for agents:
- **Zero Token Waste**: Concise, structured JSON directly ingestible by tool-calling agents.
- **Header Parsing**: Uses raw little-endian binary header parsing (`struct.unpack("<Q")`) to extract layer names and shapes in `< 5ms`.
- **Exit Codes**: Returns standard exit codes (`0` on success, `1` on error with JSON error payload on `stderr`).

---

## 2. Agent Guidelines (`.agents/mangosta.md`)

When running `mangosta init`, an agent guidelines file is created at `.agents/mangosta.md`. AI assistants automatically discover this file and adhere to:
1. Always registering architectures with `@register_model`.
2. Saving and loading models via `save_bundle` and `load_bundle` rather than raw `torch.save(model)`.
3. Validating tensor shapes using `ModelSchema` rather than informal assertions.
4. Using `--json` flags for programmatic CLI tasks.

---

## 3. Ephemeral Server Management

When an agent needs to inspect a training run or launch a report:
- `mangosta report <run_or_bundle> -o report.html`: Generates an offline HTML artifact without leaving any running processes.
- `mangosta view <target> --no-browser`: Launches the zero-daemon HTTP server without trying to spawn GUI windows in headless CI or SSH sessions.

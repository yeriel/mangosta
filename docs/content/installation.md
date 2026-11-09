# Installation & Flags

Mangosta uses **`uv`** as the standard environment and package manager.

Because Mangosta is designed for both **PyTorch training** and **Zero-PyTorch production microservices**, its dependencies are split into lightweight feature flags (extras). This ensures a production container never has to download `2.5GB+` of PyTorch just to run inference.

---

## 1. Standard Setup with `uv` (Recommended)

Always create an isolated `uv` environment in your project before installing Mangosta:

```bash
# 1. Initialize project or virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install Mangosta according to your environment's role:
```

=== "Full Toolkit (`[all]`)"
    Use this on your development workstation or laptop when you want **training, ONNX export, and local serving** all in one place:
    ```bash
    uv add "mangosta[all]"
    ```

=== "Training Only (`[train]`)"
    Use this on GPU training nodes where you need **PyTorch + SafeTensors** for training, tracking, and checkpointing:
    ```bash
    uv add "mangosta[train]"
    # Alias also supported: uv add "mangosta[torch]"
    ```

=== "Production Serving (`[serving]`)"
    Use this inside production Docker microservices. **Does NOT install PyTorch** (`~70MB` image instead of `~2.5GB`):
    ```bash
    uv add "mangosta[serving]"
    ```

=== "ONNX Export (`[onnx]`)"
    Add this when exporting `.bundle` models to `.onnx` or `.onnx.bundle` with graph verification:
    ```bash
    uv add "mangosta[train,onnx]"
    ```

---

## 2. Complete Installation Flags Matrix

Here is every installation flag defined in `pyproject.toml` and what it installs:

| Flag / Extra | Command | Installed Packages | When to Use |
| :--- | :--- | :--- | :--- |
| **Base** *(no flags)* | `uv add mangosta` | `numpy`, `pyyaml`, `packaging` | Minimal contract validation (`ModelSchema`), CLI inspection (`mangosta inspect`), and manifest reading. |
| **`[train]`** / **`[torch]`** | `uv add "mangosta[train]"` | `torch>=2.0`, `safetensors>=0.4` | Training models, `@register_model`, `RunTracker`, `save_bundle`, and `resume_training`. |
| **`[serving]`** | `uv add "mangosta[serving]"` | `onnxruntime>=1.16`, `numpy`, `pyyaml`, `packaging` | **Zero-PyTorch** production inference with `load_onnx_bundle` and `ONNXRunner`. |
| **`[onnx]`** | `uv add "mangosta[onnx]"` | `onnx>=1.14` | Exporting PyTorch models via `export_onnx` and `export_onnx_as_bundle`. |
| **`[all]`** | `uv add "mangosta[all]"` | `torch`, `safetensors`, `onnx`, `onnxruntime` | Complete end-to-end workflow (Training + Export + Serving). |
| **`[docs]`** | `uv add "mangosta[docs]"` | `zensical>=0.1.0` | Building and previewing the documentation locally (`cd docs && uv tool run zensical serve`). |
| **`--group dev`** | `uv sync --group dev` | `pytest`, `pytest-cov`, `mypy`, `ruff`, `onnxscript` | Contributing to Mangosta, running unit tests, and strict type checking. |

---

## 3. Selecting PyTorch CPU vs. CUDA in `uv`

By default on Linux, PyTorch installs CUDA wheels. If you are on a CPU-only laptop or CI runner, you can configure `uv` in your `pyproject.toml` to pull lightweight CPU wheels (just like Mangosta's own repository does):

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cpu" }
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Or install directly via `uv pip` passing the index URL:

```bash
uv pip install "mangosta[all]" --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## 4. Working From This Repository (Contributors & Examples)

If you cloned the `mangosta` repository to run the [`examples/`](examples.md) or tests:

```bash
# Create environment and sync all dev + training + ONNX dependencies
uv sync --group dev

# Run any example inside the uv environment
uv run python examples/01_lifecycle_training_vs_production.py

# Verify CLI
uv run mangosta --help
```

---

## 5. Offline / Air-Gapped Installation

To deploy in environments without internet access:

```bash
# 1. On a machine with internet: download wheels
mkdir -p ./wheels
uv pip compile pyproject.toml --extra all -o requirements.txt
pip download -r requirements.txt -d ./wheels
pip download . --no-deps -d ./wheels

# 2. On the air-gapped machine: install offline into uv venv
uv venv
uv pip install --no-index --find-links=./wheels "mangosta[all]"
```

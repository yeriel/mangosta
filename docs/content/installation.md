# Installation

Mangosta uses **`uv`** as its standard package and virtual environment manager.

Its dependencies are intentionally split so production inference containers do **not** pull PyTorch (`~2.5GB+`).

---

## 1. Choose Your Environment Role

=== "Full Toolkit (`[full]`)"

    Use on your development machine or GPU training node when you need **PyTorch training, SafeTensors bundles, ONNX export, and local serving**:

    ```bash
    uv venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    uv add "mangosta[full]"
    ```

=== "Production Serving (`[serving]`)"

    Use inside production Docker microservices. **Does NOT install PyTorch** (`~70MB` image instead of `~2.5GB`):

    ```bash
    uv add "mangosta[serving]"
    ```

    Runs `.onnx.bundle` archives via `load_onnx_bundle()` and `ONNXRunner` using pure NumPy + ONNX Runtime.

=== "Core / CLI Only"

    Use in lightweight CI runners for YAML config validation (`load_config`), schema checks, and `<5ms` bundle inspection (`mangosta inspect`):

    ```bash
    uv add mangosta
    ```

---

## 2. Dependency Matrix

Every dependency group defined in `pyproject.toml`:

| Extra / Group | Command | Installed Packages | When to Use |
| :--- | :--- | :--- | :--- |
| **Base** *(default)* | `uv add mangosta` | `numpy`, `pyyaml`, `packaging` | `MangostaConfig`, `ModelSchema`, `read_bundle_manifest`, `inspect_bundle_metadata`, `mangosta inspect`. |
| **`[full]`** | `uv add "mangosta[full]"` | `torch>=2.0`, `safetensors>=0.4`, `onnx>=1.14`, `onnxruntime>=1.16` | Training (`BaseTrainer`), registries, `RunTracker`, `.bundle` checkpointing/resume, and ONNX export. |
| **`[serving]`** | `uv add "mangosta[serving]"` | `numpy>=2.0`, `onnxruntime>=1.16`, `pyyaml>=6.0`, `packaging>=23.0` | **Zero-PyTorch** production inference via `ONNXRunner` / `load_onnx_bundle`. |
| **`--group dev`** *(repo only)* | `uv sync --group dev` | `torch`, `safetensors`, `onnx`, `onnxruntime`, `onnxscript`, `mypy`, `ruff` | Contributing to Mangosta, running type checks, and executing repository examples. |

---

## 3. PyTorch CPU vs. CUDA Wheels with `uv`

On Linux, `torch` pulls CUDA wheels by default. For CPU-only laptops or CI runners, point `uv` to the official PyTorch CPU index in your project's `pyproject.toml`:

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

Or install directly via `uv pip`:

```bash
uv pip install "mangosta[full]" --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## 4. Offline / Air-Gapped Installation

To deploy Mangosta inside an environment without internet access:

```bash
# 1. On an online machine: compile and download all wheels
mkdir -p ./wheels
uv pip compile pyproject.toml --extra full -o requirements.txt
pip download -r requirements.txt -d ./wheels
pip download . --no-deps -d ./wheels

# 2. On the air-gapped machine: install strictly from local wheels
uv venv
uv pip install --no-index --find-links=./wheels "mangosta[full]"
```

---

## 5. Building These Docs Locally (Self-Contained `docs/`)

Documentation tooling (Zensical) lives strictly inside `docs/pyproject.toml` and is never mixed with `mangosta`'s runtime dependencies:

```bash
uv --directory docs sync
uv --directory docs run zensical serve
```

Then open `http://127.0.0.1:8000` in your browser.

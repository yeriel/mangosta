# Installation

Mangosta is lightweight, modular, and fast to install. It can run in minimal environments with standard PyTorch or with optional extras for ONNX acceleration and production microservice serving.

---

## Quick Install

=== "pip"
    ```bash
    pip install mangosta
    ```

=== "uv"
    ```bash
    uv add mangosta
    ```

---

## Optional Extras

Mangosta provides modular installation targets to keep core dependencies lean:

```bash
# Production HTTP serving (FastAPI, Uvicorn, Pydantic)
pip install "mangosta[serving]"

# ONNX export and high-performance CPU/GPU runtime
pip install "mangosta[onnx]"

# Development, testing, and static analysis tools
pip install "mangosta[dev]"

# Documentation tools (Zensical)
pip install "mangosta[docs]"

# Install everything
pip install "mangosta[all]"
```

### Dependency Breakdown

| Extra Target | Included Packages | Purpose |
| :--- | :--- | :--- |
| **Core** | `torch`, `safetensors`, `pyyaml`, `click` | Core registry, bundle packaging, SafeTensors IO, CLI |
| **`[serving]`** | `fastapi`, `uvicorn`, `pydantic` | Microservice generation and production contract endpoints |
| **`[onnx]`** | `onnx`, `onnxruntime` | ONNX model export, validation, and zero-PyTorch inference |
| **`[dev]`** | `pytest`, `pytest-cov`, `mypy`, `ruff` | Test suite, coverage reporting, and strict type verification |
| **`[docs]`** | `zensical` | Fast documentation static site generator |

---

## Air-Gapped & Offline Installation

Mangosta is designed to operate seamlessly in secure enclaves and air-gapped clusters without outbound internet access.

### 1. Download Wheels on an Internet-Connected Machine

Create a local wheelhouse containing Mangosta and all its dependencies:

=== "pip"
    ```bash
    mkdir -p ./mangosta_wheels
    pip download "mangosta[all]" -d ./mangosta_wheels
    ```

=== "uv"
    ```bash
    mkdir -p ./mangosta_wheels
    uv pip download "mangosta[all]" -d ./mangosta_wheels
    ```

### 2. Transfer to Air-Gapped Host

Archive the directory and transfer via approved secure media (e.g., USB drive, secure bastion, internal artifactory):

```bash
tar -czvf mangosta_wheels.tar.gz ./mangosta_wheels
# Transfer mangosta_wheels.tar.gz to air-gapped machine
tar -xzvf mangosta_wheels.tar.gz
```

### 3. Install Offline Without Network Access

Install directly from the local directory:

```bash
pip install --no-index --find-links=./mangosta_wheels mangosta
```

Verify that no outbound network sockets are attempted:

```bash
python -c "import mangosta; print('Mangosta version:', mangosta.__version__)"
mangosta --version
```

---

## Verification

Run the CLI help command to confirm that your installation and entry points are functioning correctly:

```bash
mangosta --help
```

You should see the command group:
```text
Usage: mangosta [OPTIONS] COMMAND [ARGS]...

  Mangosta CLI: Lightweight MLOps & Model Lifecycle Management.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  init     Scaffold a new production-ready project.
  inspect  Inspect a model bundle or checkpoint file.
  report   Generate an execution report for runs or experiments.
  serve    Start a production FastAPI server for a bundle.
  view     Launch local visual analytics dashboard.
```

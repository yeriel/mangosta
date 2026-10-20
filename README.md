# 🥭 mangosta

**Lightweight MLOps and Model Lifecycle Management for PyTorch.**

> *"Simple por defecto, estricto bajo demanda"* — Usable en 3 líneas de código para prototipos rápidos, pero extensible mediante contratos formales, fault-tolerance y auditoría de linaje para producción y agentes de IA.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![uv](https://img.shields.io/badge/packaging-uv-purple.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Características Principales

- 🚫 **Cero servicios web o demonios**: Sin MLflow, ClearML ni W&B. Persistencia local directa, atómica y crash-consistent en disco (`JSONL` y ZIPs).
- 🏷️ **Registro Inmutable (`mangosta.registry`)**: Decorador `@register_model` con prevención estricta de colisiones accidentales y resolución dinámica SemVer.
- 📐 **Contratos de Tensores y Firmas (`mangosta.schema`)**: Inferencia automática de formas (batch dinámico `-1`/`None`) y dtypes mediante forward pass no destructivo, con validación estricta en inferencia.
- 📦 **Empaquetado Autocontenido `.bundle` (`mangosta.bundle`)**: Formato ZIP atómico que encapsula pesos, manifiesto, recetas YAML y estado completo de entrenamiento (`train_state.pt`) con estados RNG de Python, PyTorch (CPU/CUDA) y NumPy.
- 🔄 **Fault Tolerance y Reanudación Exacta (`mangosta.loader`)**: `resume_training` para continuar entrenamientos interrumpidos de forma 100% determinista.
- 🚀 **Carga Universal Desacoplada (`load_bundle`)**: Reconstrucción de la arquitectura desde el registro sin necesidad de importar la clase del modelo en scripts de inferencia.
- 📊 **Visualización Serverless (`mangosta.report` & CLI)**: Reportes HTML estáticos con gráficos interactivos offline, visualizador efímero en navegador (`mangosta view`) y visor en terminal (`mangosta inspect`).

---

## ⚡ Quickstart en 3 Líneas

### 1. Tracking local atómico
```python
from mangosta import quick_track

with quick_track(experiment="mnist", run_name="baseline") as run:
    run.log_params({"lr": 1e-3, "batch_size": 64})
    run.log_metric("loss", 0.42, step=1)
```

### 2. Registrar arquitectura
```python
import torch.nn as nn
from mangosta import register_model

@register_model(family="mlp", version="1.0.0")
class MLP(nn.Module):
    def __init__(self, in_dim=10, out_dim=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)
```

### 3. Guardar y Cargar un `.bundle`
```python
import torch
from mangosta import save_bundle, load_bundle

# Guardar empaquetado con contrato inferido automáticamente
save_bundle("model.bundle", model=model, family="mlp", version="1.0.0", sample_input=torch.randn(1, 10))

# Cargar en producción sin conocer la clase del modelo
bundle = load_bundle("model.bundle")
predictions = bundle.predict(torch.randn(4, 10)) # Valida firma antes de ejecutar
```

---

## 🛠️ CLI de Terminal

`mangosta` incluye una herramienta de línea de comandos lista para usar:

```bash
# Inspeccionar un archivo .bundle en consola
mangosta inspect artifacts/model.bundle

# Generar un reporte interactivo HTML autocontenido (offline, sin servidores)
mangosta report artifacts/model.bundle -o report.html
mangosta report experiments/run_01 -o report.html

# Iniciar un visor web local efímero (se detiene con Ctrl+C, cero demonios en segundo plano)
mangosta view ./experiments
```

---

## 🔁 Tolerancia a Fallos y Reanudación Determinista

Para guardar un punto de control ante caídas o límites de tiempo de GPU (SLURM, spots):

```python
from mangosta import save_bundle, resume_training

# Guardar punto de control con estado completo
save_bundle(
    "ckpt.bundle",
    model=model,
    family="mlp",
    version="1.0.0",
    is_checkpoint=True,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=current_epoch,
    step=global_step,
)

# Reanudar tras una interrupción
state = resume_training("ckpt.bundle", model, optimizer, scheduler, restore_rng=True)
start_epoch = state.epoch + 1
start_step = state.step
```

---

## 💻 Desarrollo y Empaquetado con `uv`

```bash
# Sincronizar dependencias
uv sync

# Ejecutar suite de pruebas
uv run pytest -v --cov=mangosta

# Formato y linter
uv run ruff check src tests
uv run ruff format src tests

# Ejecutar demostración completa del ciclo de vida
uv run python examples/lifecycle_demo.py

# Compilar para distribución en PyPI
uv build
```

---

## 📜 Licencia

MIT License.

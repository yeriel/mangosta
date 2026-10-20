/usage# Spec: mangosta

Librería Python para la gestión del ciclo de vida de modelos en PyTorch: experiment tracking local atómico, registro inmutable de arquitecturas, empaquetado autocontenido (`.bundle`) con tolerancia a fallos (fault tolerance/checkpointing) y carga universal para inferencia.

---

## 1. Objective

Desarrollar `mangosta`, una librería Python lista para empaquetado y publicación (PEP 517/621 compatible con `uv` y distribuible en PyPI), diseñada bajo el principio:
> **"Simple por defecto, estricto bajo demanda"**: usable en 3 líneas de código para prototipos rápidos, pero extensible mediante contratos formales y YAML para flujos avanzados o agentes autónomos de IA.

### Capacidades Principales
1. **Registro Inmutable (`mangosta.registry`)**: Registro estricto e inmutable de modelos `nn.Module` y componentes auxiliares (losses, optimizadores, etc.) por familia y versión SemVer, con instanciación dinámica.
2. **Inferencia de Firmas y Contratos (`mangosta.schema`)**: Detección no invasiva de firmas I/O (formas y dtypes, batch dinámico) mediante forward pass de muestra y validación estricta en inferencia.
3. **Experiment Tracking Local Atómico (`mangosta.tracker`)**: Registro de parámetros y métricas paso a paso sin servidores demonio ni dependencias de nube (MLflow/W&B-free). Almacenamiento local atómico en JSONL/SQLite y fingerprinting SHA-256 de datasets.
4. **Empaquetado Autocontenido y Fault Tolerance (`mangosta.bundle`)**: Formato de archivo `.bundle` (ZIP atómico) que encapsula pesos, manifiesto, receta YAML y opcionalmente el estado completo de entrenamiento (`train_state.pt`: optimizador, scheduler, scaler AMP, epoch, step, RNG states de Python, PyTorch y NumPy).
5. **Carga Universal (`mangosta.loader`)**: Reconstrucción desacoplada de la arquitectura desde el registro sin importar el código del modelo en el script de inferencia, y utilidades de reanudación exacta de entrenamiento (`resume_training`).
6. **Visualización e Inspección Serverless (`mangosta.report` & CLI)**: Generación de reportes HTML estáticos y autocontenidos (curvas interactivas offline y metadatos sin servidores web), inspección rápida en terminal (`mangosta inspect`) y visualizador efímero en navegador (`mangosta view`) usando únicamente `http.server` estándar de Python (cero demonios).
7. **Exportación Flexible a ONNX (`mangosta.exporter` & `mangosta export-onnx`)**: Exportación desacoplada a formato ONNX estándar con soporte para mapeo automático de salidas de diccionarios a tuplas (`output_mapping`), control de precisión (`float16`/`float32`), ejes dinámicos, opset configurable y verificación de grafo con dependencia opcional `mangosta[onnx]`.

---

## 2. Capability Map & Arquitectura de Módulos

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  mangosta (API & CLI)                                  │
└──────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────┘
       │              │              │              │              │              │
       ▼              ▼              ▼              ▼              ▼              ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌─────────────┐┌────────────┐┌───────────┐
│   registry   ││    schema    ││   tracker    ││   bundle    ││   report   ││  exporter │
│  (Catálogo   ││ (Contratos   ││  (Tracking   ││ (Empacado   ││  (Reporte  ││  (ONNX /   │
│  Inmutable)  ││   E/S)       ││   Atómico)   ││   y ZIP)    ││  HTML/TUI) ││ Producción│
└──────┬───────┘└──────┬───────┘└──────┬───────┘└──────┬──────┘└─────┬──────┘└─────┬─────┘
       │               │               │               │             │             │
       └───────────────┼───────────────┴───────────────┼─────────────┘             │
                       ▼                               ▼                           │
                 ┌─────────────┐                 ┌─────────────┐                   │
                 │   loader    │                 │  checkpoint │                   │
                 │(Inferencia) │                 │ (Reanudado) │                   │
                 └─────────────┘                 └─────────────┘                   ▼
                                                                           ┌───────────────┐
                                                                           │  model.onnx   │
                                                                           └───────────────┘
```

| Módulo ID | Responsabilidad | Dependencias internas |
|---|---|---|
| `schema` | Inspección de tensores I/O, serialización de firmas y validación | — |
| `registry` | Registro inmutable de arquitecturas `nn.Module` y factorías | — |
| `tracker` | Tracking local en JSONL/SQLite, fingerprinting de datos y context manager `quick_track` | — |
| `bundle` | Empaquetado de artefactos `.bundle` (ZIP atómico), guardado de `state_dict`, `train_state` y `manifest.json` | `schema` |
| `loader` | Carga de bundles (`load_bundle`), instanciación vía `registry`, validación I/O y `resume_training` | `registry`, `schema`, `bundle` |
| `report` | Generación de reporte HTML estático autocontenido y visualización de curvas/artefactos | `tracker`, `bundle` |
| `exporter` | Exportación a ONNX desacoplada, mapeo de salidas diccionarios/tuplas, FP16/FP32 y validación | `loader`, `bundle`, `schema` |
| `cli` | Comandos de terminal: `mangosta inspect`, `mangosta report`, `mangosta view`, `mangosta export-onnx` | `loader`, `report`, `bundle`, `exporter` |

---

## 3. Tech Stack & Herramientas

- **Python**: `>=3.10, <3.14` (soporte garantizado para PyTorch actual en Linux/macOS/Windows).
- **Gestor de Entorno y Paquetes**: `uv` (>=0.11).
- **Build Backend**: `hatchling` (conforme a PEP 517 / PEP 621 en `pyproject.toml`).
- **Dependencias Base (Ultra-ligeras)**:
  - `torch >= 2.0.0`
  - `pyyaml >= 6.0`
  - `packaging >= 23.0`
- **Dependencias Opcionales**:
  - `onnx >= 1.14.0` (para verificación del grafo con `onnx.checker` en `mangosta[onnx]`)
- **Dependencias de Desarrollo**:
  - `pytest >= 8.0`
  - `pytest-cov`
  - `ruff` (linter y formateador ultrarrápido)

---

## 4. Comandos de Desarrollo (`uv`)

```bash
# Crear entorno virtual y sincronizar dependencias
uv sync

# Ejecutar tests unitarios y de integración con cobertura
uv run pytest -v --cov=mangosta --cov-report=term-missing

# Linting y verificación de formato
uv run ruff check src tests
uv run ruff format --check src tests

# Corregir formato automáticamente
uv run ruff format src tests

# Ejecutar el script de demostración de ciclo de vida completo
uv run python examples/lifecycle_demo.py

# Construir wheel y sdist para distribución
uv build
```

---

## 5. Estructura del Proyecto

```
mangosta/
├── pyproject.toml              # Configuración moderna PEP 621 + hatchling + uv
├── README.md                   # Documentación, quickstart en 3 líneas y API reference
├── src/
│   └── mangosta/
│       ├── __init__.py         # API pública de primer nivel
│       ├── exceptions.py       # Excepciones estructuradas de la librería
│       ├── registry.py         # ModelRegistry y decorador @register_model
│       ├── schema.py           # ModelSchema, TensorSpec e inferencia I/O
│       ├── tracker.py          # RunTracker, quick_track y hash de datasets
│       ├── bundle.py           # Creación y empaquetado de archivos .bundle
│       ├── loader.py           # LoadedBundle, load_bundle y resume_training
│       ├── report.py           # Generador de reportes HTML estáticos y visualización
│       └── cli.py              # CLI: inspect, report, view (servidor efímero)
├── tests/
│   ├── conftest.py             # Fixtures compartidas de PyTorch (modelos simples, datos)
│   ├── test_registry.py        # Tests de inmutabilidad, colisiones y resolución SemVer
│   ├── test_schema.py          # Tests de inferencia de firmas y validación estricta
│   ├── test_tracker.py         # Tests de persistencia atómica en JSONL y fingerprinting
│   ├── test_bundle.py          # Tests de creación de bundle, integridad ZIP y train_state
│   ├── test_loader.py          # Tests de carga universal, predicción y resume_training
│   ├── test_report.py          # Tests de generación de reporte HTML y curvas
│   └── test_cli.py             # Tests de comandos de terminal
└── examples/
    └── lifecycle_demo.py       # Ejemplo end-to-end: train -> checkpoint -> resume -> load -> infer -> report
```

---

## 6. Especificación Detallada de Módulos

### 6.1 `mangosta.registry`
- **Inmutabilidad**: Si se intenta registrar una arquitectura con una tupla `(family, version)` ya existente, se arroja `RegistryCollisionError`. Se provee `allow_override=False` por defecto.
- **Decorador**: `@register_model(family="resnet", version="1.0.0")`.
- **Despachador**: `ModelRegistry.build(family, version=None, **init_args) -> nn.Module`. Si `version` no se especifica, toma la última versión SemVer disponible de esa familia.
- **Componentes variables**: Soporte de registro para losses, optimizers y schedulers mediante `@register_component(kind="loss", name="focal_loss")`.

### 6.2 `mangosta.schema`
- **Inferencia**: `infer_schema(model: nn.Module, sample_input: Any) -> ModelSchema`.
  - Evalúa en modo seguro con `torch.inference_mode()` y preserva el estado de entrenamiento previo del modelo (`model.training`).
  - Soporta `sample_input` como `Tensor`, `Tuple[Tensor, ...]`, o `Dict[str, Tensor]`.
  - Convierte la primera dimensión en dimensión dinámica (`-1` o `None`) si se detecta batching.
- **Validación en Inferencia**: `schema.validate_input(input_data)` genera mensajes de error detallados como:
  `SchemaValidationError: Input tensor 'x' expected shape [-1, 3, 224, 224] with dtype torch.float32, got shape [1, 1, 224, 224] with dtype torch.float64.`

### 6.3 `mangosta.tracker`
- **Escritura Atómica**:
  - `metrics.jsonl`: escritura con `flush()` y `os.fsync()`, o bloqueo de archivo para evitar corrupción en cortes abruptos.
  - Opcional SQLite estructurado para queries locales rápidas.
- **Fingerprinting**: `compute_sha256(path: Union[str, Path]) -> str` procesa archivos por bloques (evitando cargar todo en RAM) y directorios computando un hash Merkle ordenado.
- **Context Manager**:
  ```python
  with quick_track(experiment="mnist_exp", run_name="baseline") as run:
      run.log_params({"lr": 1e-3, "batch_size": 64})
      run.log_metric("train_loss", 0.42, step=1)
      run.set_dataset(dataset_path)
  ```

### 6.4 `mangosta.bundle` y `mangosta.loader`
- **Estructura del archivo `.bundle`**:
  - `weights.pt`: `model.state_dict()`.
  - `manifest.json`: metadatos JSON legibles (familia, versión, argumentos de inicialización, hash de dataset, schema de entrada/salida, métricas de corrida, fecha, versiones de PyTorch/Python).
  - `config.yaml`: configuración de entrenamiento serializada (opcional).
  - `train_state.pt` (sólo si `is_checkpoint=True`):
    - `optimizer_state_dict`
    - `scheduler_state_dict`
    - `scaler_state_dict` (torch.cuda.amp.GradScaler si está activo)
    - `epoch`, `step`
    - `rng_states`: dict con estados de `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()` (si CUDA está disponible), `random.getstate()` y `np.random.get_state()` (si NumPy está instalado).
- **Garantía Atómica**: El archivo ZIP se escribe primero como `<target>.bundle.tmp` y luego se reemplaza atómicamente con `os.replace`.
- **Carga Universal**:
  ```python
  bundle = load_bundle("artifacts/model.bundle")
  predictions = bundle.predict(sample_tensor)  # Valida schema automáticamente antes de forward
  ```
- **Reanudación Exacta**:
  ```python
  state = resume_training("artifacts/checkpoint.bundle", model, optimizer, scheduler, scaler)
  # Restaura pesos en model, estados en optimizer, scheduler, scaler y generadores RNG
  start_epoch = state.epoch + 1
  start_step = state.step
  ```

### 6.6 `mangosta.exporter` y Exportación ONNX
- **Función Principal**:
  ```python
  export_onnx(
      model_or_bundle: Union[nn.Module, str, Path],
      output_path: Union[str, Path],
      weights: Optional[Union[str, Path]] = None,
      sample_input: Optional[Any] = None,
      sample_input_shape: Optional[Tuple[int, ...]] = None,
      output_mapping: Optional[Sequence[str]] = None,
      input_names: Optional[Sequence[str]] = None,
      output_names: Optional[Sequence[str]] = None,
      dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
      float16: bool = False,
      opset_version: int = 18,
      do_constant_folding: bool = True,
      device: Optional[Union[str, torch.device]] = None,
      verify: bool = True,
  ) -> Path
  ```
- **Mapeo Automático Diccionario -> Tupla**:
  Si el modelo retorna un `dict` y se provee `output_mapping=["semantic_mask", "radius_map", "gradient_map"]`, se envuelve automáticamente el modelo en un wrapper interno transparente para retornar la tupla en ese orden estricto.
- **Carga Polimórfica de Pesos**:
  Si se pasa `weights`:
  - Soporta archivos `.pth`/`.pt` estándar buscando llaves comunes (`"model_state_dict"`, `"state_dict"`, o pesos directos).
  - Soporta archivos `.bundle` de mangosta extrayendo `weights.pt`.
- **Verificación Opcional**:
  Si `verify=True`, intenta importar `onnx` y ejecuta `onnx.checker.check_model`. Si `onnx` no está instalado, emite un aviso informativo amigable sin fallar la exportación.
- **CLI**:
  `mangosta export-onnx <bundle_or_weights> -o model.onnx [--fp16] [--opset 18]`

---

## 7. Boundaries

- **Always**:
  - Usar tipado estricto (`typing` / Python 3.10+ annotations) y docstrings informativos.
  - Escribir archivos de forma atómica en disco (`tempfile` + `os.replace`).
  - Preservar reproducibilidad de RNG (Torch CPU/CUDA, Python random, NumPy).
  - Validar compatibilidad en tests unitarios usando PyTorch en CPU.
- **Ask first**:
  - Agregar dependencias pesadas más allá de las esenciales.
  - Modificar el formato de almacenamiento del archivo `.bundle`.
- **Never**:
  - Requerir servidores externos o conexiones a internet para tracking o almacenamiento.
  - Permitir sobreescritura silenciosa de modelos en `ModelRegistry`.
  - Cargar tensores con código inseguro o dependencias arbitrarias no confiables.

---

## 8. Success Criteria

1. **Suite de pruebas**: 100% de tests pasando con `uv run pytest`.
2. **Empaquetado limpio**: `uv build` genera tanto `.whl` como `.tar.gz` sin advertencias, listo para PyPI.
3. **Demo funcional integral**: `examples/lifecycle_demo.py` ejecuta el flujo completo:
   - Registro de arquitectura.
   - Tracking rápido con `quick_track`.
   - Interrupción deliberada a mitad de entrenamiento y guardado de checkpoint con estado RNG.
   - Reanudación con `resume_training` verificando continuidad determinista.
   - Exportación de `.bundle` final de inferencia.
   - Carga con `load_bundle` y predicción validada por `schema`.
   - Generación de reporte HTML estático e interactivo con curvas de aprendizaje.
4. **Inspección en terminal**: `mangosta inspect` y `mangosta report` funcionando sin fallos ni requerir red.
5. **Respeto a las restricciones**: Cero servicios web o demonios permanentes, archivos locales atómicos y compatibilidad nativa con `uv`.

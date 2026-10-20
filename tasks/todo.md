# Tasks: mangosta

- [x] Task 1: Configurar proyecto con `uv`, `pyproject.toml` y excepciones base
  - Acceptance: `pyproject.toml` configurado con `hatchling`, dependencias base (`torch`, `pyyaml`, `packaging`), y `src/mangosta/exceptions.py` definido. `uv sync` exitoso.
  - Verify: `uv sync`
  - Files: `pyproject.toml`, `src/mangosta/__init__.py`, `src/mangosta/exceptions.py`

- [x] Task 2: Implementar módulo `mangosta.schema` (inspección de firmas I/O y validación)
  - Acceptance: Inferencia de formas con batch dinámico (`-1`/`None`), soporte de tensores, tuplas y diccionarios, validación en inferencia con excepciones claras.
  - Verify: `uv run pytest tests/test_schema.py -v`
  - Files: `src/mangosta/schema.py`, `tests/test_schema.py`, `tests/conftest.py`

- [x] Task 3: Implementar módulo `mangosta.registry` (registro inmutable de arquitecturas)
  - Acceptance: `@register_model`, detección de colisiones (`RegistryCollisionError`), despachador dinámico con resolución SemVer, registro de componentes opcionales.
  - Verify: `uv run pytest tests/test_registry.py -v`
  - Files: `src/mangosta/registry.py`, `tests/test_registry.py`

- [x] Task 4: Implementar módulo `mangosta.tracker` (tracking local atómico y fingerprinting)
  - Acceptance: `RunTracker` con guardado atómico en `metrics.jsonl`, `compute_sha256` para archivos y carpetas, y context manager `quick_track`.
  - Verify: `uv run pytest tests/test_tracker.py -v`
  - Files: `src/mangosta/tracker.py`, `tests/test_tracker.py`

- [x] Task 5: Implementar módulo `mangosta.bundle` (empaquetado ZIP atómico con fault tolerance)
  - Acceptance: Generación atómica de `.bundle`, empaquetado de `weights.pt`, `manifest.json`, `config.yaml` y `train_state.pt` con RNG states (Torch CPU/CUDA, Python, NumPy).
  - Verify: `uv run pytest tests/test_bundle.py -v`
  - Files: `src/mangosta/bundle.py`, `tests/test_bundle.py`

- [x] Task 6: Implementar módulo `mangosta.loader` (carga universal y reanudación exacta)
  - Acceptance: `load_bundle` reconstruye el modelo vía registry y valida inputs con `schema.validate_input`; `resume_training` restaura pesos, optimizador, scheduler, scaler y estados RNG.
  - Verify: `uv run pytest tests/test_loader.py -v`
  - Files: `src/mangosta/loader.py`, `tests/test_loader.py`

- [x] Task 7: Implementar módulo `mangosta.report` y CLI (`inspect`, `report`, `view`)
  - Acceptance: `generate_report` produce HTML interactivo offline con curvas y metadatos. CLI `mangosta inspect`, `mangosta report` y servidor efímero `mangosta view` funcionan correctamente.
  - Verify: `uv run pytest tests/test_report.py tests/test_cli.py -v`
  - Files: `src/mangosta/report.py`, `src/mangosta/cli.py`, `tests/test_report.py`, `tests/test_cli.py`

- [x] Task 8: Implementar ejemplo integral `lifecycle_demo.py`, documentación `README.md` y verificar build de PyPI
  - Acceptance: `lifecycle_demo.py` ejecuta el ciclo completo (train -> checkpoint -> resume determinista -> bundle final -> inference -> report). `uv build` compila wheel y tarball. Todos los tests al 100%.
  - Verify: `uv run python examples/lifecycle_demo.py && uv run pytest && uv build`
  - Files: `examples/lifecycle_demo.py`, `README.md`, `pyproject.toml`

- [x] Task 9: Implementar módulo `mangosta.exporter` (exportación flexible a ONNX con mapeo de diccionarios)
  - Acceptance: `export_onnx` soporta modelos directos o bundles, mapeo de salidas dict a tupla (`output_mapping`), FP16/FP32, `input_names`, `output_names`, `dynamic_axes`, verificación opcional con `onnx.checker` y comando CLI `mangosta export-onnx`.
  - Verify: `uv run pytest tests/test_exporter.py -v`
  - Files: `src/mangosta/exporter.py`, `src/mangosta/cli.py`, `src/mangosta/__init__.py`, `pyproject.toml`, `tests/test_exporter.py`

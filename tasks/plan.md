# Technical Implementation Plan: mangosta

## 1. Overview & Architecture

`mangosta` es una librería Python para el ciclo de vida de modelos PyTorch con foco en:
- Registro inmutable de arquitecturas con resolución SemVer.
- Inferencia de contratos de tensores (formas/tipos con batch dinámico) y validación estricta.
- Experiment tracking local atómico (sin servicios demonio).
- Empaquetado autocontenido `.bundle` (ZIP atómico) con soporte para fault tolerance (RNG states, optimizer, scheduler, scaler).
- Carga universal desacoplada (`load_bundle`) y reanudación determinista (`resume_training`).
- Visualización serverless (reporte HTML interactivo offline, CLI `inspect`, `report`, `view`).

## 2. Dependency Graph & Implementation Order

```
[01: Scaffold & pyproject.toml]
           │
           ├───► [02: mangosta.schema]
           │           │
           ├───► [03: mangosta.registry]
           │           │
           ├───► [04: mangosta.tracker]
           │           │
           └───────────┼──────────────┐
                       ▼              ▼
                 [05: bundle]    [06: loader]
                       │              │
                       └──────┬───────┘
                              ▼
                        [07: report & cli]
                              │
                              ▼
                        [08: e2e demo & build]
```

### Orden de Construcción Secuencial:
1. **Fase 1: Fundaciones y Configuración `uv`** (`pyproject.toml`, estructura `src/mangosta`, excepciones comunes).
2. **Fase 2: Esquemas y Contratos de Tensores (`schema`)** (inspección no invasiva con `torch.inference_mode`, validación estricta de tensores/diccionarios/tuplas).
3. **Fase 3: Registro Inmutable de Arquitecturas (`registry`)** (decorador, prevención de colisiones, factoría dinámica con SemVer).
4. **Fase 4: Tracking Local Atómico y Linaje (`tracker`)** (escritura atómica JSONL, hashing SHA-256 de datasets/archivos, context manager `quick_track`).
5. **Fase 5: Empaquetado Autocontenido (`bundle`)** (formato ZIP atómico, `weights.pt`, `manifest.json`, `train_state.pt` con RNG CPU/CUDA/Python/NumPy).
6. **Fase 6: Carga Universal y Reanudación (`loader`)** (`load_bundle`, `LoadedBundle.predict`, `resume_training`).
7. **Fase 7: Visualización Serverless y CLI (`report` & `cli`)** (generador de HTML interactivo offline sin dependencias de red, comandos `inspect`, `report`, `view` efímero).
8. **Fase 8: Integración Final, Demo y Empaquetado** (`examples/lifecycle_demo.py`, `uv build`, verificación de cobertura de tests).

## 3. Risks & Mitigations

- **Riesgo 1: Incompatibilidad de dtypes y dispositivos en validación de firmas**.
  *Mitigación*: En `schema`, normalizar nombres de tipos canónicos (`"float32"`, `"int64"`, etc.) y permitir que el validador acepte dispositivos diferentes si el usuario traslada tensores a GPU/CPU.
- **Riesgo 2: Corrupción de archivos de checkpoints en apagados repentinos**.
  *Mitigación*: Escribir el ZIP en un archivo temporal con sufijo `.bundle.tmp` y ejecutar `os.replace` atómico a nivel de sistema operativo.
- **Riesgo 3: Desincronización de estados RNG al reanudar**.
  *Mitigación*: Serializar explícitamente `torch.get_rng_state()`, estados de CUDA si está disponible (`torch.cuda.get_rng_state_all()`), `random.getstate()` y `np.random.get_state()` si NumPy está instalado, y restaurarlos en orden estricto en `resume_training`.
- **Riesgo 4: Dependencia de red para visualización HTML**.
  *Mitigación*: Embeber el motor de visualización (SVG/Canvas y CSS ligero) directamente inline en el archivo HTML, asegurando que funcione sin conexión a internet y sin requerir servidores ni CDN.

## 4. Verification Checkpoints

- **Checkpoint 1 (Core)**: `tests/test_schema.py` y `tests/test_registry.py` pasando con 100% de éxito.
- **Checkpoint 2 (Storage & Lineage)**: `tests/test_tracker.py` y `tests/test_bundle.py` validando atomicidad, linaje SHA-256 y persistencia de RNG.
- **Checkpoint 3 (Runtime & Fault Tolerance)**: `tests/test_loader.py` comprobando que `resume_training` produce exactamente la misma trayectoria de entrenamiento que una ejecución continua.
- **Checkpoint 4 (UI & CLI)**: `tests/test_report.py` y `tests/test_cli.py` verificando generación de HTML estático y comandos CLI.
- **Checkpoint 5 (Release Ready)**: `examples/lifecycle_demo.py` ejecuta el flujo completo de extremo a extremo y `uv build` compila los artefactos de distribución sin errores.

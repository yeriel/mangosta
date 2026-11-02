# MANGOSTA SPECIFICATION: RUTA HACIA LA VERSIÓN ESTABLE 1.0.0
> **Documento:** Especificación Técnica, Principios de Diseño y Plan de Hardening  
> **Alcance:** Local Workstation / Single-GPU (PyTorch)  
> **Filosofía:** Biblioteca pura (no framework invasivo), cero bases de datos, cero demonios, máxima ergonomía para el desarrollador en solitario.  
> **Estado:** Aprobado para Ejecución (Rama: `dev`)  

---

## 1. Visión y Principios Rectores

Mangosta es una biblioteca liviana para la gestión del ciclo de vida y MLOps local de modelos PyTorch, diseñada bajo una premisa fundamental: **cero demonios en segundo plano, cero dependencias de infraestructura, cero bases de datos externas (sin SQLite, PostgreSQL ni RocksDB) y cero sobrecarga cognitiva**.

Para alcanzar la versión estable (`1.0.0`), el mandato técnico es:
1. **Biblioteca, no Framework Invasivo:** Mangosta provee herramientas atómicas (`save_bundle`, `RunTracker`, `load_bundle`, `inspect`), sin forzar al usuario a heredar de clases monolíticas ni encarcelar su código en una estructura rígida.
2. **Cero Bases de Datos (Flat-Files Transparentes):** Toda persistencia se realiza en archivos planos auditables con herramientas estándar del sistema operativo (`cat`, `grep`, `unzip`, Python `json`, Pandas):
   * Contenedores `.bundle` en formato ZIP estándar con `model.safetensors` y `manifest.json`.
   * Métricas en streaming secuencial `metrics.jsonl`.
   * Metadatos de corrida en `run.json`.
3. **Hardening sobre lo existente:** Blindar I/O en disco ante fallos inesperados (OOM, SIGINT, SIGKILL), eliminar la dependencia de `pickle` mediante **SafeTensors**, optimizar el flushing de I/O para no frenar el entrenamiento y certificar la reproducibilidad matemática bit-a-bit de checkpoints.
4. **Ergonomía Extrema para el Desarrollador en Solitario ("Smart Defaults"):** Convención sobre configuración. Las operaciones más frecuentes deben resolverse en una sola línea de código con valores por defecto inteligentes.
5. **Congelamiento de API (SemVer):** Definir una frontera estricta entre API pública y submódulos internos, garantizando tipado estricto al 100%.

---

## 2. Análisis Senior: Libertad vs. Convención en la Industria ("El Camino Pavimentado")

### 2.1. El Dilema: ¿Mucha libertad rompe el flujo de trabajo de Mangosta?
En la ingeniería de MLOps moderna existen dos extremos fallidos:
* **La Tiranía del Framework (ej. PyTorch Lightning rígido o Cookiecutters de 25 carpetas):** Obligan al usuario a estructurar cada clase en carpetas predefinidas (`models/architectures/baseline/v1/`). En iteraciones en solitario, esto genera fricción masiva y el desarrollador termina esquivando la herramienta.
* **El Caos Irrestricto:** Si no hay guía, el usuario guarda checkpoints en cualquier carpeta, mezcla artefactos pesados en el repositorio de Git y sufre errores de importación (`ModuleNotFoundError` al ejecutar `uv run`).

### 2.2. La Solución Estándar de la Industria: "Camino Pavimentado" (Paved Road) con Contratos en las Fronteras
La industria (Hugging Face `accelerate`, FastAPI, Vite, Rust `cargo`) resuelve este dilema separando dos niveles de responsabilidad:

1. **El código fuente es libre:** A Mangosta no le importa si tu modelo vive en `src/models.py`, en `model.py` en la raíz o dentro de un notebook interactivo. Solo exige que sea una instancia de `torch.nn.Module`.
2. **Las salidas y artefactos son estrictos (Contratos en las Fronteras):**
   * Los checkpoints se guardan como archivos inmutables `.bundle`.
   * Los experimentos se estructuran en `experiments/<experimento>/<corrida>/`.
   * Las herramientas de inspección y visualización (`mangosta inspect`, `mangosta view`, `mangosta report`) operan sobre estas convenciones.
3. **Barandillas de Seguridad (Guardrails) en el Scaffolding:**
   * La plantilla generada por `mangosta init` no es una camisa de fuerza, sino un **camino pavimentado** funcional desde el segundo cero:
     * Genera un `pyproject.toml` válido que permite ejecutar `uv run train` sin importar desde dónde invoques la terminal.
     * Genera un `.gitignore` robusto que evita subir accidentalmente modelos pesados a Git.
     * Utiliza **creación perezosa (lazy creation)**: si el usuario no necesita guardar salidas de prueba, no verá carpetas vacías estorbando en su explorador de archivos.

---

## 3. Las 5 Simplificaciones de Ergonomía (DX) para Iteración Rápida

### 3.1. "Smart Defaults" en `save_bundle`
Eliminar la obligación de pasar argumentos redundantes en prototipado rápido:
* Si `family` no se especifica, se auto-infiere: `model.__class__.__name__.lower()`.
* Si `version` no se especifica, por defecto es `"0.1.0"`.
* Si se proporciona `optimizer`, `scheduler` o `scaler`, **se infiere automáticamente `is_checkpoint = True`**.
* Si se proporciona `sample_input`, se infiere el contrato `schema` sin configuración extra.
* *Resultado:* Guardar un checkpoint pasa de 8 líneas a **1 línea**:
  ```python
  save_bundle("checkpoints/best.bundle", model, optimizer=optimizer, epoch=epoch)
  ```

### 3.2. Registro en Lote con `log_metrics(dict)` en `RunTracker`
* Permitir registrar múltiples métricas en un único paso:
  ```python
  run.log_metrics({"loss": loss.item(), "acc": acc, "lr": current_lr}, step=epoch)
  ```
* Evita bucles manuales de `log_metric`, minimiza escrituras y reduce la fricción en el bucle de entrenamiento.

### 3.3. Scaffolding Minimalista y Limpio (`mangosta init`)
Estructura reducida a lo esencial:
```text
mi-proyecto/
├── configs/
│   └── exp_baseline.yaml
├── entrypoints/
│   ├── __init__.py
│   ├── train.py
│   └── infer.py
├── src/
│   ├── __init__.py
│   └── models.py
├── .gitignore
├── pyproject.toml
└── README.md
```
* **Cero directorios vacíos:** `checkpoints/` y `experiments/` se crean automáticamente en disco cuando el código realmente ejecuta un guardado.
* **Integración nativa con `uv`:** `pyproject.toml` preconfigura `[project.scripts]` para permitir `uv run train` y `uv run infer`.

### 3.4. Carga Flexible y Didáctica en `load_bundle`
* Soporte transparente tanto para flujos desacoplados con registro como para flujos directos:
  ```python
  # Modo A (Vía registro automático):
  bundle = load_bundle("checkpoints/best.bundle")

  # Modo B (Directo, pasando la arquitectura):
  bundle = load_bundle("checkpoints/best.bundle", model=MyModel())
  ```
* Si el modelo no está registrado y no se proporcionó `model`, el error debe orientar al desarrollador en la terminal con una solución inmediata en lugar de un stack trace críptico.

### 3.5. Transparencia Absoluta de Artefactos (Cero Cajas Negras)
* Un `.bundle` es un ZIP estándar sin compresión en tensores (`ZIP_STORED`), lo que permite abrirlo con cualquier herramienta del sistema o inspeccionarlo en milisegundos con `mangosta inspect <bundle>`.
* No hay bases de datos intermedias: los datos son archivos planos que sobreviven a caídas y no sufren de bloqueos de concurrencia (`database is locked`).

---

## 4. Especificación Técnica de Componentes

### 4.1. Bundle y Serialización Segura (SafeTensors)
Contenido del contenedor `.bundle`:
```text
modelo.bundle (ZIP)
├── manifest.json       # JSON: Metadatos, contrato I/O, hash SHA-256 de los pesos
├── config.yaml         # YAML: Copia inmutable de la configuración
├── model.safetensors   # SafeTensors: Pesos de PyTorch (cero riesgo de pickle)
└── train_state.pt      # PyTorch binary: Optimizador, LR scheduler y RNG (solo en checkpoints)
```

#### Protocolo de Escritura Atómica en Dos Fases:
1. Crear archivo temporal `{dest_name}.tmp_{pid}_{uuid4}` en el mismo sistema de archivos.
2. Escribir manifiesto, pesos en formato SafeTensors y estado de entrenamiento.
3. Registrar hash criptográfico SHA-256 de `model.safetensors` en `manifest.json`.
4. Ejecutar `os.replace(tmp_dest, dest)`. Si ocurre cualquier excepción en el proceso, limpiar el archivo temporal en bloque `finally`.

### 4.2. Tracking Crash-Resilient sin Cuello de Botella de I/O
* **Métricas:** Escritura en streaming a `metrics.jsonl`. `flush()` inmediato para vaciar buffers de Python, y `os.fsync()` desvinculado del batch individual: se sincroniza a disco por épocas, al cerrar el tracker (`finish()`) o cada $K$ pasos configurables ($K=50$ por defecto).
* **Manejo de Señales:** Capturar `SIGINT` (Ctrl+C) y `SIGTERM` para sincronizar `run.json` marcando el estado `"interrupted"` o `"failed"` de forma ordenada, evitando corridas zombis.
* **Colisión de Nombres:** Añadir un identificador aleatorio corto al nombre automático de la corrida (`run_{timestamp}_{hex}`) para evitar colisiones si se inician dos procesos concurrentes.

### 4.3. Determinismo y Validación de Contratos
* **Invariancia Bit-a-Bit:** Garantizar que retomar un checkpoint guardado con `is_checkpoint=True` produzca exactamente la misma pérdida y pesos que una corrida continua bajo las mismas condiciones.
* **Contratos pre-CUDA:** `LoadedBundle.predict(sample, validate=True)` comprueba formas y tipos antes de invocar la GPU para evitar fallos opacos en CUDA.

---

## 5. Hoja de Ruta Faseada (Roadmap hacia v1.0.0)

```
[ v0.2.x (Alpha) ]
       │
       ▼
[ Fase 1: Scaffolding, DX & Smart Defaults ] ──► Limpieza de `init`, lazy folders, pyproject.toml, smart defaults en save_bundle/log_metrics.
       │
       ▼
[ Fase 2: I/O Hardening & SafeTensors ] ───────► SafeTensors, hashing SHA-256, atomic rename blindado, buffer I/O sin stalls.
       │
       ▼
[ Fase 3: Suite de Determinismo & CI ] ────────► Tests bit-a-bit, simulación de fallos/SIGINT, validación de schema pre-CUDA.
       │
       ▼
[ Fase 4: Congelamiento de API & SemVer ]──────► Tipado estricto mypy, delimitación _internal, CLI inspect con --json.
       │
       ▼
[ v1.0.0 (Release Estable) ]
```

### Fase 1: Scaffolding, DX y Smart Defaults (Objetivo: v0.3.0)
- [ ] Refactorizar `src/mangosta/cli/init_project.py`:
  - [ ] Eliminar creación anticipada de carpetas vacías (`checkpoints/`, `experiments/`, `outputs/`) y sus `.gitkeep`.
  - [ ] Consolidar la plantilla minimalista: `configs/exp_baseline.yaml`, `src/models.py`, `entrypoints/train.py`, `entrypoints/infer.py`.
  - [ ] Generar `pyproject.toml` con `[project.scripts]` para `uv run train` y `uv run infer`.
  - [ ] Actualizar `.gitignore` y `README.md`.
- [ ] Implementar Smart Defaults en `src/mangosta/artifacts/bundle.py`:
  - [ ] Inferencia de `family` a partir de `model.__class__.__name__.lower()`.
  - [ ] Inferencia automática de `is_checkpoint=True` si se suministra `optimizer`, `scheduler` o `scaler`.
- [ ] Implementar `log_metrics(dict)` en `src/mangosta/tracking/tracker.py`.
- [ ] Mejorar mensajes de diagnóstico en `load_bundle` de `src/mangosta/artifacts/loader.py`.
- [ ] Actualizar y pasar suite de tests de scaffolding (`tests/test_init.py`).

### Fase 2: I/O Seguro, SafeTensors y Persistencia Resiliente (Objetivo: v0.4.0)
- [ ] Añadir `safetensors>=0.4.0` a dependencias en `pyproject.toml`.
- [ ] Migrar almacenamiento de pesos en `save_bundle` a `model.safetensors`.
- [ ] Incorporar cálculo y verificación de SHA-256 de pesos en `manifest.json`.
- [ ] Blindar escritura atómica con limpieza estricta de `.tmp` en bloques `finally`.
- [ ] Actualizar `load_bundle` y `resume_training` para leer `model.safetensors` (manteniendo retrocompatibilidad con `weights.pt`).
- [ ] Desacoplar `fsync` continuo en `RunTracker.log_metric` / `log_metrics` para evitar stalls de I/O.
- [ ] Implementar manejadores de señales `SIGINT`/`SIGTERM` en `RunTracker`.

### Fase 3: Suite de Determinismo y Verificación en CI (Objetivo: v0.5.0-beta)
- [ ] Implementar `tests/test_determinism.py`:
  - [ ] Test bit-a-bit en CPU: 10 épocas continuas vs (5 épocas + checkpoint + reanudación + 5 épocas).
  - [ ] Verificación de pérdida idéntica (`loss == loss`) y pesos (`torch.allclose(atol=0.0)`).
- [ ] Implementar `tests/test_fault_tolerance.py`:
  - [ ] Simulación de proceso interrumpido durante escritura de bundle y verificación de ausencia de archivos corruptos.
  - [ ] Test de detección y reporte de bundles con pesos alterados o hashes inválidos.

### Fase 4: Congelamiento de API, Tipado y Release Estable (Objetivo: v1.0.0)
- [ ] Actualizar `mangosta inspect <bundle>` en `src/mangosta/cli/main.py`:
  - [ ] Lectura segura de encabezados SafeTensors sin instanciar PyTorch.
  - [ ] Soporte para flag `--json`.
- [ ] Congelar la superficie de API pública en `src/mangosta/__init__.py`.
- [ ] Verificación de tipado estricto con `mypy --strict` en la API pública.
- [ ] Cobertura de tests $> 85\%$.

---

## 6. Definición de "Hecho" (Definition of Done para v1.0.0)

1. **Cero bases de datos:** Toda persistencia reside en archivos planos transparentes y auditables (ZIP, SafeTensors, JSONL, JSON).
2. **Cero carpetas vacías:** El scaffolding genera únicamente archivos vivos; los directorios de artefactos se crean de forma perezosa.
3. **Flujo de desarrollo en 1 línea:** Guardar checkpoints y registrar métricas no exige boilerplate innecesario.
4. **Determinismo comprobado:** La prueba de invariancia bit-a-bit pasa de forma reproducible en CI.
5. **Seguridad y compatibilidad:** Pesos en SafeTensors, eliminación de vulnerabilidades de `pickle` en distribución de bundles e integridad validada con SHA-256.

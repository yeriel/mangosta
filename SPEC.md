# MANGOSTA SPECIFICATION: RUTA HACIA LA VERSIÓN ESTABLE 1.0.0
> **Documento:** Especificación Técnica, Principios de Diseño y Plan de Hardening  
> **Alcance:** Local Workstation / Single-GPU (PyTorch)  
> **Filosofía:** Biblioteca pura (no framework invasivo), cero bases de datos, cero demonios, agent-native y zero-overhead en entrenamiento.  
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
3. **Agent-Native por Diseño (Supervisión de Agentes de IA):** Optimizada para flujos de trabajo contemporáneos donde el desarrollador supervisa agentes de código (Claude Code, Cursor, Antigravity, OpenCode, Aider). Provee inspección en CLI con salida estructurada `--json`, contexto operativo canónico (`AGENTS.md`) y mensajes de error auto-accionables que permiten a los agentes auto-corregir código sin intervención humana.
4. **Zero-Overhead en el Bucle de Entrenamiento:** El seguimiento y guardado no debe penalizar el rendimiento del hardware. La sobrecarga introducida por Mangosta en el bucle de entrenamiento debe ser inferior al **0.5% del tiempo total de época**.
5. **Hardening sobre lo existente:** Blindar I/O en disco ante fallos inesperados (OOM, SIGINT, SIGKILL), eliminar la dependencia de `pickle` mediante **SafeTensors**, optimizar el flushing de I/O para no frenar el entrenamiento y certificar la reproducibilidad matemática bit-a-bit de checkpoints.
6. **Ergonomía Extrema ("Smart Defaults"):** Convención sobre configuración. Las operaciones más frecuentes deben resolverse en una sola línea de código con valores por defecto inteligentes.
7. **Congelamiento de API (SemVer):** Definir una frontera estricta entre API pública y submódulos internos, garantizando tipado estricto al 100%.

---

## 2. Análisis Senior: Libertad vs. Convención en la Industria ("El Camino Pavimentado")

### 2.1. El Dilema: ¿Mucha libertad rompe el flujo de trabajo de Mangosta?
En la ingeniería de MLOps moderna existen dos extremos fallidos:
* **La Tiranía del Framework (ej. PyTorch Lightning rígido o Cookiecutters de 25 carpetas):** Obligan al usuario a estructurar cada clase en carpetas predefinidas (`models/architectures/baseline/v1/`). En iteraciones en solitario, esto genera fricción masiva y el desarrollador termina esquivando la herramienta.
* **El Caos Irrestricto:** Si no hay guía, el usuario guarda checkpoints en cualquier carpeta, mezcla artefactos pesados en el repositorio de Git y sufre errores de importación (`ModuleNotFoundError` al ejecutar `uv run`).

### 2.2. La Solución Estándar de la Industria: "Camino Pavimentado" (Paved Road) con Contratos en las Fronteras
La industria (Hugging Face `accelerate`, FastAPI, Vite, Rust `cargo`) resuelve este dilema separando dos niveles de responsabilidad:

1. **El código fuente es libre:** A Mangosta no le importa si tu modelo vive en `src/models.py`, en `model.py` en la raíz o dentro de un script suelto. Solo exige que sea una instancia de `torch.nn.Module`.
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

## 3. Integración con Agentes de Código (Agent-Native DX)

En el paradigma de desarrollo actual, el rol del ingeniero se desplaza hacia la **supervisión de agentes autónomos**. Para que un agente interactúe fluidamente con un repositorio basado en Mangosta sin alucinaciones ni fricciones, la biblioteca incorpora cuatro pilares:

### 3.1. Inspección Programática Machine-Readable (`mangosta inspect --json`)
* **Problema:** Los agentes leen la salida estándar de comandos de terminal. Si la CLI solo emite tablas formateadas con colores ANSI, el agente debe usar expresiones regulares frágiles para extraer metadatos.
* **Solución:** `mangosta inspect <bundle> --json` emite un objeto JSON estructurado con:
  * Familia, versión y commit de Git del modelo.
  * Contrato formal I/O (shapes, dtypes, nombres de tensores).
  * Hiperparámetros y métricas registradas.
  * Hashes de linaje y resumen de tensores.
  * El agente puede invocar este comando, parsear el JSON y continuar su tarea de benchmarking o inferencia con 100% de certeza.

### 3.2. Errores Auto-Accionables (Actionable Stderr)
* **Problema:** Un stack trace críptico (`RuntimeError: size mismatch at dimension 1`) causa que los agentes caigan en bucles de prueba y error, editando capas a ciegas.
* **Solución:** Las excepciones de Mangosta (`SchemaValidationError`, `ModelNotFoundError`, `BundleCorruptedError`) se redactan con diagnóstico de causa raíz y sugerencia inmediata:
  ```text
  SchemaValidationError: Input tensor mismatch on 'features'.
    Expected: shape=(None, 8), dtype=torch.float32
    Received: shape=(1, 16), dtype=torch.float32
    Fix hint: Verify dataloader feature count or update init_args in configs/exp_baseline.yaml.
  ```
  Esto permite que el agente lea el mensaje y auto-corrija el código en un único paso.

### 3.3. Scaffolding con Contexto para Agentes (`AGENTS.md`)
* `mangosta init` generará automáticamente un archivo canónico `.agents/mangosta.md` o `AGENTS.md` en la raíz del proyecto.
* Contiene el mapa de convenciones, comandos permitidos (`uv run train`, `mangosta inspect`) y directrices de arquitectura. Cuando cualquier agente (Cursor, Claude Code, Antigravity) abre el repositorio, absorbe las reglas de inmediato y genera código compatible a la primera.

### 3.4. Cero Dependencias de Red ni Puertos Bloqueantes
* Al no requerir inicio de sesión en servicios cloud (a diferencia de W&B o Comet) ni contenedores Docker (a diferencia de MLflow), los agentes pueden ejecutar entrenamientos y suites de prueba dentro de sandboxes cerrados y entornos CI/CD sin bloqueos interactivos.

---

## 4. Optimización Extrema y Zero-Overhead en el Entrenamiento

El entrenamiento de modelos es costoso en tiempo y hardware. Mangosta tiene como principio innegociable **no degradar la velocidad de cómputo de PyTorch**.

### 4.1. Eliminación del Cuello de Botella de I/O en Disco (`fsync`)
* **Diagnóstico actual:** Forzar `os.fsync()` en cada llamada de `log_metric` bloquea el hilo de ejecución de Python esperando la confirmación física del disco (1 a 15 ms por batch). En bucles de 200 batches/segundo, esto destruye la velocidad de cómputo.
* **Solución Zero-Overhead:**
  * Las métricas se escriben en el archivo `metrics.jsonl` con `flush()` en memoria de proceso (costo < 5 microsegundos).
  * `os.fsync()` físico a disco se desvincula del batch: se ejecuta únicamente:
    1. Al completar cada época.
    2. Al invocar `finish()` o cerrar el contexto `with RunTracker`.
    3. Periódicamente cada $K=50$ pasos configurables.
  * Con este desacoplamiento, el impacto de I/O en la velocidad de la época cae a **menos del 0.05%**.

### 4.2. Prevención de Stalls de Sincronización en CUDA
* En PyTorch, extraer valores escalares de tensores en GPU mediante `.item()` o `.cpu()` fuerza una barrera de sincronización en el stream de CUDA, dejando a la GPU en reposo forzado (*GPU starvation*).
* `RunTracker.log_metric()` y `log_metrics()` recibirán directamente valores ya desacoplados o flotantes nativos, y la documentación/plantillas generadas instruirán el patrón óptimo de acumulación sin bloqueos de GPU.

### 4.3. Serialización de Alta Velocidad con SafeTensors
* SafeTensors evita el paso intermedio de empaquetar objetos de Python (requerido por `pickle`), volcando tensores directamente desde memoria contigua a disco.
* El guardado de checkpoints es hasta **3x–4x más rápido** y reduce el tiempo que el bucle de entrenamiento permanece en pausa durante el guardado de artefactos intermedios.

### 4.4. Benchmark de Sobrecarga en CI
* Se incorporará un test de rendimiento automatizado (`tests/test_performance_overhead.py`):
  * Ejecuta un bucle sintético estándar de PyTorch sin Mangosta.
  * Ejecuta el mismo bucle con `RunTracker` registrando loss, métricas y artefactos.
  * **Criterio de aceptación:** La diferencia de tiempo total debe ser estrictamente menor al **0.5%**.

---

## 5. Las 5 Simplificaciones de Ergonomía (DX) para Iteración Rápida

### 5.1. "Smart Defaults" en `save_bundle`
* Si `family` no se especifica, se auto-infiere: `model.__class__.__name__.lower()`.
* Si `version` no se especifica, por defecto es `"0.1.0"`.
* Si se proporciona `optimizer`, `scheduler` o `scaler`, **se infiere automáticamente `is_checkpoint = True`**.
* Si se proporciona `sample_input`, se infiere el contrato `schema` sin configuración extra.
* Guardar un checkpoint se resuelve en **1 línea**:
  ```python
  save_bundle("checkpoints/best.bundle", model, optimizer=optimizer, epoch=epoch)
  ```

### 5.2. Registro en Lote con `log_metrics(dict)` en `RunTracker`
* Permite registrar múltiples métricas en un único paso:
  ```python
  run.log_metrics({"loss": loss.item(), "acc": acc, "lr": current_lr}, step=epoch)
  ```

### 5.3. Scaffolding Minimalista y Limpio (`mangosta init`)
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
├── .agents/
│   └── mangosta.md
├── .gitignore
├── pyproject.toml
└── README.md
```
* **Cero directorios vacíos:** `checkpoints/` y `experiments/` se crean automáticamente en tiempo de ejecución.
* **Integración nativa con `uv`:** `pyproject.toml` preconfigura `[project.scripts]` para permitir `uv run train` y `uv run infer`.

### 5.4. Carga Flexible y Didáctica en `load_bundle`
* Soporte transparente para flujos con registro automático y flujos directos:
  ```python
  # Con Registro:
  bundle = load_bundle("checkpoints/best.bundle")

  # Directo, pasando la arquitectura:
  bundle = load_bundle("checkpoints/best.bundle", model=MyModel())
  ```

### 5.5. Transparencia Absoluta de Artefactos (Cero Cajas Negras)
* Un `.bundle` es un ZIP estándar con `model.safetensors`, `manifest.json` y `train_state.pt`.
* Métricas en `metrics.jsonl` auditables con `grep`, `cat` o Pandas. Cero bases de datos.

---

## 6. Hoja de Ruta Faseada (Roadmap hacia v1.0.0)

```
[ v0.2.x (Alpha) ]
       │
       ▼
[ Fase 1: Scaffolding, DX, Smart Defaults & Agent Context ] [COMPLETADA]
  - Limpieza de `init`, lazy folders, pyproject.toml, AGENTS.md, smart defaults en save_bundle/log_metrics.
       │
       ▼
[ Fase 2: I/O Hardening, SafeTensors & Zero-Overhead ] [COMPLETADA]
  - SafeTensors, hashing SHA-256, atomic rename blindado, buffer I/O sin stalls de fsync, SIGINT/SIGTERM.
       │
       ▼
[ Fase 2.5: Formalización del Arnés de Pruebas, Auditoría Rigurosa y CI Offline ] [COMPLETADA]
  - Erradicación de auto-trampas y mocks blandos: pruebas sobre PyTorch y SafeTensors reales.
  - Auditoría exhaustiva componente por componente (init, bundle, loader, tracker, schema, export).
  - Arnés Docker Ubuntu 24.04 aislado sin red (--network none) automatizado en scripts/test_offline.sh.
       │
       ▼
[ Fase 3: Suite de Determinismo, CI & Benchmarks de Sobrecarga ]
  - Tests bit-a-bit, simulación de fallos/SIGINT, validación de schema pre-CUDA, benchmark de overhead < 0.5%.
       │
       ▼
[ Fase 4: Congelamiento de API, CLI --json & SemVer ]
  - mangosta inspect --json, tipado estricto mypy, delimitación _internal, documentación de release.
       │
       ▼
[ v1.0.0 (Release Estable) ]
```

### Fase 1: Scaffolding, DX, Smart Defaults y Contexto para Agentes (Objetivo: v0.3.0) [COMPLETADA]
- [x] Refactorizar `src/mangosta/cli/init_project.py`:
  - [x] Eliminar creación anticipada de carpetas vacías (`checkpoints/`, `experiments/`, `outputs/`) y sus `.gitkeep`.
  - [x] Consolidar la plantilla minimalista: `configs/exp_baseline.yaml`, `src/models.py`, `entrypoints/train.py`, `entrypoints/infer.py`.
  - [x] Generar `pyproject.toml` con `[project.scripts]` para `uv run train` y `uv run infer`.
  - [x] Generar `.agents/mangosta.md` con las instrucciones operativas canónicas para agentes de IA.
  - [x] Actualizar `.gitignore` y `README.md`.
- [x] Implementar Smart Defaults en `src/mangosta/artifacts/bundle.py`:
  - [x] Inferencia de `family` a partir de `model.__class__.__name__.lower()`.
  - [x] Inferencia automática de `is_checkpoint=True` si se suministra `optimizer`, `scheduler` o `scaler`.
- [x] Implementar `log_metrics(dict)` en `src/mangosta/tracking/tracker.py`.
- [x] Mejorar mensajes de diagnóstico y sugerencias accionables en `load_bundle`.
- [x] Actualizar y pasar suite de tests de scaffolding (`tests/test_init.py`).

### Fase 2: I/O Seguro, SafeTensors y Persistencia Resiliente (Objetivo: v0.4.0) [COMPLETADA]
- [x] Añadir `safetensors>=0.4.0` a dependencias en `pyproject.toml`.
- [x] Migrar almacenamiento de pesos en `save_bundle` a `model.safetensors`.
- [x] Incorporar cálculo y verificación de SHA-256 de pesos en `manifest.json`.
- [x] Blindar escritura atómica con limpieza estricta de `.tmp` en bloques `finally`.
- [x] Actualizar `load_bundle` y `resume_training` para leer `model.safetensors` (clean break: formato canónico único).
- [x] Desacoplar `fsync` continuo en `RunTracker.log_metrics` (`sync_interval=10`, `flush_disk()`).
- [x] Implementar manejadores de señales `SIGINT`/`SIGTERM` en `RunTracker` para salida limpia.

### Fase 2.5: Formalización del Arnés de Pruebas, Auditoría Rigurosa y CI Offline (Objetivo: v0.4.5) [COMPLETADA]
- [x] **Erradicar la "auto-trampa" y mocks superficiales:**
  - [x] Garantizar que las pruebas unitarias y de integración auditan la lógica real contra dependencias verdaderas (PyTorch, SafeTensors, NumPy, PyYAML).
  - [x] Prohibir mocks que simulen el comportamiento del propio código bajo test; los mocks quedan reservados exclusivamente para simular fallos de hardware externos (disco lleno, fallo de I/O en `os.replace`, señales del SO).
- [x] **Formalizar `conftest.py` y fixtures canónicos:**
  - [x] Arquitecturas de red representativas (MLP simple, red con tensores no contiguos, red con salidas múltiples en diccionario).
  - [x] Manejadores de temporales aislados y generadores de tensores sintéticos controlados por semilla.
- [x] **Auditoría Componente por Componente:**
  - [x] *Auditoría 1: Scaffolding (`tests/test_init.py`):* Validar generación sintáctica de `pyproject.toml` (compatible con Hatchling y `uv`), reglas en `.agents/mangosta.md` y estructura de archivos vivos sin carpetas vacías.
  - [x] *Auditoría 2: Bundles & Pesos (`tests/test_bundle.py`):* Forzar tensores no contiguos en memoria (`.transpose()`) para comprobar normalización obligatoria antes de SafeTensors; auditar hash SHA-256 exacto y verificar limpieza atómica en `finally`.
  - [x] *Auditoría 3: Carga, Integridad y Desacoplamiento (`tests/test_loader.py`):* Comprobar que alteraciones de 1 solo bit en `model.safetensors` o `manifest.json` detienen la carga inmediatamente antes de instanciar pesos; validar que `load_bundle` con modelo explícito no toca el registro global; probar reanudación determinista de checkpoint.
  - [x] *Auditoría 4: Tracking, I/O y Señales (`tests/test_tracker.py`):* Auditar que `f.flush()` vuelca inmediatamente en userspace sin `os.fsync`, comprobar cadencia exacta de `sync_interval=10`, auditar integridad de `metrics.jsonl` ante `SIGINT`/`SIGTERM` y llamadas en hilos secundarios.
  - [x] *Auditoría 5: Contratos y Exportación (`tests/test_schema.py`, `tests/test_exporter.py`):* Auditar validación de schema (rechazo de shapes/tipos inválidos antes de entrar a CUDA) y exportación ONNX directamente desde bundles SafeTensors.
- [x] **Estandarización del Arnés de Validación Offline (Docker Ubuntu 24.04):**
  - [x] Integrar script `scripts/test_offline.sh` para levantar el contenedor con `--network none`, ejecutar la suite completa y emitir reporte de auditoría en 1 comando.
  - [x] Documentar el procedimiento de auditoría en el README de testing.

### Fase 3: Suite de Determinismo, CI y Benchmark de Rendimiento (Objetivo: v0.5.0-beta)
- [ ] Implementar `tests/test_determinism.py`:
  - [ ] Test bit-a-bit en CPU: 10 épocas continuas vs (5 épocas + checkpoint + reanudación + 5 épocas).
  - [ ] Verificación de pérdida idéntica (`loss == loss`) y pesos (`torch.allclose(atol=0.0)`).
- [ ] Implementar `tests/test_fault_tolerance.py`:
  - [ ] Simulación de proceso interrumpido durante escritura de bundle y verificación de ausencia de archivos corruptos.
  - [ ] Test de detección y reporte de bundles con pesos alterados o hashes inválidos.
- [ ] Implementar `tests/test_performance_overhead.py`:
  - [ ] Medición automatizada de latencia: verificar que `RunTracker` agrega menos del **0.5% de sobrecarga** al bucle de entrenamiento.

### Fase 4: Congelamiento de API, CLI Machine-Readable, Documentación y Release Estable (Objetivo: v1.0.0)
- [ ] Actualizar `mangosta inspect <bundle>` en `src/mangosta/cli/main.py`:
  - [ ] Lectura segura de encabezados SafeTensors sin instanciar PyTorch.
  - [ ] Soporte para flag `--json` (salida estructurada para agentes y scripts).
- [ ] Congelar la superficie de API pública en `src/mangosta/__init__.py`.
- [ ] Verificación de tipado estricto con `mypy --strict` en la API pública.
- [ ] Cobertura de tests $> 85\%$.
- [ ] **Documentación Integral y Guías de Uso de Primera Clase:**
  - [ ] Reescritura del `README.md` principal: Quickstart en 3 minutos, tabla comparativa técnica (vs MLflow, vs W&B, vs vanilla PyTorch), arquitectura de archivos planos y ejemplos listos para copiar y pegar.
  - [ ] Referencia completa de API pública documentada (`docs/api/`): docstrings detallados, tipos de entrada/salida y snippets ejecutables para cada componente (`RunTracker`, `save_bundle`, `load_bundle`, `resume_training`, `export_onnx`, `ModelRegistry`, `infer_schema`).
  - [ ] Recetarios y Guías Prácticas de Flujo Real (`docs/guides/`):
    - *Guía 1: Iteración rápida en solitario (prototipado + tracking con zero-overhead).*
    - *Guía 2: Tolerancia a fallos: cómo guardar checkpoints atómicos y reanudar bit-a-bit tras un crash.*
    - *Guía 3: Contratos de tensores e inferencia segura en producción (LoadedBundle + ONNX).*
    - *Guía 4: Supervisión de Agentes de IA: flujos con Cursor, Claude Code y Antigravity usando CLI --json y .agents/mangosta.md.*
  - [ ] Manual de CLI con ejemplos de terminal para `mangosta init`, `inspect`, `view`, `report` y `export-onnx`.

---

## 7. Definición de "Hecho" (Definition of Done para v1.0.0)

1. **Agent-Native comprobado:** Los agentes pueden inspeccionar bundles vía `--json` y cuentan con `.agents/mangosta.md` en proyectos inicializados.
2. **Zero-Overhead medido:** La sobrecarga en el bucle de entrenamiento es estrictamente menor al 0.5% en benchmarks de CI.
3. **Cero bases de datos:** Toda persistencia reside en archivos planos transparentes y auditables (ZIP, SafeTensors, JSONL, JSON).
4. **Cero carpetas vacías:** El scaffolding genera únicamente archivos vivos; los directorios de artefactos se crean de forma perezosa.
5. **Flujo de desarrollo en 1 línea:** Guardar checkpoints y registrar métricas no exige boilerplate innecesario.
6. **Determinismo comprobado:** La prueba de invariancia bit-a-bit pasa de forma reproducible en CI.
7. **Seguridad y compatibilidad:** Pesos en SafeTensors, eliminación de vulnerabilidades de `pickle` e integridad validada con SHA-256.
8. **Documentación de Primera Clase:** README modernizado, referencia completa de API con ejemplos y recetarios paso a paso de flujos de trabajo reales.

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
[ Fase 1: Scaffolding, DX, Smart Defaults & Agent Context ]
  - Limpieza de `init`, lazy folders, pyproject.toml, AGENTS.md, smart defaults en save_bundle/log_metrics.
       │
       ▼
[ Fase 2: I/O Hardening, SafeTensors & Zero-Overhead ]
  - SafeTensors, hashing SHA-256, atomic rename blindado, buffer I/O sin stalls de fsync, SIGINT/SIGTERM.
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

### Fase 1: Scaffolding, DX, Smart Defaults y Contexto para Agentes (Objetivo: v0.3.0)
- [ ] Refactorizar `src/mangosta/cli/init_project.py`:
  - [ ] Eliminar creación anticipada de carpetas vacías (`checkpoints/`, `experiments/`, `outputs/`) y sus `.gitkeep`.
  - [ ] Consolidar la plantilla minimalista: `configs/exp_baseline.yaml`, `src/models.py`, `entrypoints/train.py`, `entrypoints/infer.py`.
  - [ ] Generar `pyproject.toml` con `[project.scripts]` para `uv run train` y `uv run infer`.
  - [ ] Generar `.agents/mangosta.md` con las instrucciones operativas canónicas para agentes de IA.
  - [ ] Actualizar `.gitignore` y `README.md`.
- [ ] Implementar Smart Defaults en `src/mangosta/artifacts/bundle.py`:
  - [ ] Inferencia de `family` a partir de `model.__class__.__name__.lower()`.
  - [ ] Inferencia automática de `is_checkpoint=True` si se suministra `optimizer`, `scheduler` o `scaler`.
- [ ] Implementar `log_metrics(dict)` en `src/mangosta/tracking/tracker.py`.
- [ ] Mejorar mensajes de diagnóstico y sugerencias accionables en `load_bundle`.
- [ ] Actualizar y pasar suite de tests de scaffolding (`tests/test_init.py`).

### Fase 2: I/O Seguro, SafeTensors y Persistencia Resiliente (Objetivo: v0.4.0)
- [ ] Añadir `safetensors>=0.4.0` a dependencias en `pyproject.toml`.
- [ ] Migrar almacenamiento de pesos en `save_bundle` a `model.safetensors`.
- [ ] Incorporar cálculo y verificación de SHA-256 de pesos en `manifest.json`.
- [ ] Blindar escritura atómica con limpieza estricta de `.tmp` en bloques `finally`.
- [ ] Actualizar `load_bundle` y `resume_training` para leer `model.safetensors` (manteniendo retrocompatibilidad con `weights.pt`).
- [ ] Desacoplar `fsync` continuo en `RunTracker.log_metric` / `log_metrics` para garantizar zero-overhead en el entrenamiento.
- [ ] Implementar manejadores de señales `SIGINT`/`SIGTERM` en `RunTracker` para salida limpia.

### Fase 3: Suite de Determinismo, CI y Benchmark de Rendimiento (Objetivo: v0.5.0-beta)
- [ ] Implementar `tests/test_determinism.py`:
  - [ ] Test bit-a-bit en CPU: 10 épocas continuas vs (5 épocas + checkpoint + reanudación + 5 épocas).
  - [ ] Verificación de pérdida idéntica (`loss == loss`) y pesos (`torch.allclose(atol=0.0)`).
- [ ] Implementar `tests/test_fault_tolerance.py`:
  - [ ] Simulación de proceso interrumpido durante escritura de bundle y verificación de ausencia de archivos corruptos.
  - [ ] Test de detección y reporte de bundles con pesos alterados o hashes inválidos.
- [ ] Implementar `tests/test_performance_overhead.py`:
  - [ ] Medición automatizada de latencia: verificar que `RunTracker` agrega menos del **0.5% de sobrecarga** al bucle de entrenamiento.

### Fase 4: Congelamiento de API, CLI Machine-Readable y Release Estable (Objetivo: v1.0.0)
- [ ] Actualizar `mangosta inspect <bundle>` en `src/mangosta/cli/main.py`:
  - [ ] Lectura segura de encabezados SafeTensors sin instanciar PyTorch.
  - [ ] Soporte para flag `--json` (salida estructurada para agentes y scripts).
- [ ] Congelar la superficie de API pública en `src/mangosta/__init__.py`.
- [ ] Verificación de tipado estricto con `mypy --strict` en la API pública.
- [ ] Cobertura de tests $> 85\%$.

---

## 7. Definición de "Hecho" (Definition of Done para v1.0.0)

1. **Agent-Native comprobado:** Los agentes pueden inspeccionar bundles vía `--json` y cuentan con `.agents/mangosta.md` en proyectos inicializados.
2. **Zero-Overhead medido:** La sobrecarga en el bucle de entrenamiento es estrictamente menor al 0.5% en benchmarks de CI.
3. **Cero bases de datos:** Toda persistencia reside en archivos planos transparentes y auditables (ZIP, SafeTensors, JSONL, JSON).
4. **Cero carpetas vacías:** El scaffolding genera únicamente archivos vivos; los directorios de artefactos se crean de forma perezosa.
5. **Flujo de desarrollo en 1 línea:** Guardar checkpoints y registrar métricas no exige boilerplate innecesario.
6. **Determinismo comprobado:** La prueba de invariancia bit-a-bit pasa de forma reproducible en CI.
7. **Seguridad y compatibilidad:** Pesos en SafeTensors, eliminación de vulnerabilidades de `pickle` e integridad validada con SHA-256.

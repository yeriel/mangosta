# MANGOSTA SPECIFICATION: RUTA HACIA LA VERSIÓN ESTABLE 1.0.0
> **Documento:** Especificación Técnica y Plan de Hardening  
> **Alcance:** Local Workstation / Single-GPU (PyTorch)  
> **Objetivo:** Estabilidad, tolerancia a fallos, reproducibilidad matemática e inmutabilidad de API pública.  
> **Estado:** Aprobado para Ejecución (Rama: `dev`)  

---

## 1. Visión y Principios Rectores

Mangosta es una biblioteca liviana para la gestión del ciclo de vida y MLOps local de modelos PyTorch, diseñada bajo una premisa fundamental: **cero demonios en segundo plano, cero dependencias de infraestructura en la nube y cero sobrecarga cognitiva**.

Para alcanzar la versión estable (`1.0.0`), el mandato técnico es:
1. **No agregar nuevas features:** Congelar el conjunto de funcionalidades existentes y no expandir el alcance hacia entrenamiento distribuido (DDP/multi-nodo) ni arquitecturas complejas de servidor.
2. **Hardening sobre lo existente:** Blindar I/O en disco ante fallos inesperados (OOM, SIGINT, SIGKILL), garantizar serialización segura sin dependencias arbitrarias de código (reemplazando `pickle` por `safetensors`), y certificar la reproducibilidad bit-a-bit de checkpoints.
3. **Ergonomía y claridad:** Mantener el andamiaje (`init`) limpio y sin carpetas vacías artificiales, asegurando compatibilidad nativa con flujos modernos basados en `uv`.
4. **Congelamiento de API (SemVer):** Definir una frontera estricta entre API pública y submódulos internos, garantizando tipado estricto al 100%.

---

## 2. Auditoría Técnica Senior de la Propuesta

A continuación se presenta el análisis técnico de los puntos identificados en la propuesta inicial, contrastándolos con la realidad actual del código fuente:

### 2.1. Scaffolding del Proyecto (`mangosta init`)
* **Diagnóstico actual:** La implementación actual (`src/mangosta/cli/init_project.py`) crea de forma forzada carpetas vacías como `checkpoints/`, `experiments/` y `outputs/` con `.gitkeep`. Además, genera los ejecutables en la raíz (`train.py`, `infer.py`, `benchmark.py`) y **no genera un `pyproject.toml`** para el proyecto del usuario.
* **Evaluación Senior:** 
  * La adopción de **creación perezosa (lazy creation)** es obligatoria: ningún directorio de artefactos debe existir hasta que se invoque una rutina de guardado o tracking.
  * Mover los scripts a `entrypoints/` (`entrypoints/train.py`, etc.) limpia la raíz.
  * *Punto crítico de integración con `uv`:* Para que comandos como `uv run train` funcionen mediante `[project.scripts]`, el `pyproject.toml` generado debe declarar el paquete editable o incluir `entrypoints` en el empaquetado del build-backend (ej. hatchling). De lo contrario, `uv` arrojará `ModuleNotFoundError`. El `init` debe generar un `pyproject.toml` autocontenido y válido.

### 2.2. Robustez de I/O y Fallos de Proceso
* **Diagnóstico actual:**
  * `save_bundle` utiliza escritura temporal y `os.replace`, lo cual previene archivos `.bundle` a medio escribir. Sin embargo, no hay limpieza de archivos `.tmp` huérfanos si el proceso muere durante el empaquetado.
  * `RunTracker.log_metric()` ejecuta `with open(..., "a")`, `flush()` y `os.fsync()` en **cada llamada**.
* **Evaluación Senior:**
  * En un bucle de entrenamiento estándar de PyTorch (ej. 100-500 iteraciones por segundo), forzar un `fsync` en cada batch satura la cola de I/O del kernel y degrada el rendimiento de entrenamiento entre un 20% y 50% en discos locales o NVMe.
  * *Solución requerida:* Implementar flushing por micro-lotes o por intervalos de tiempo/pasos (ej. cada $N$ pasos o por época), manteniendo un `flush` inmediato y dejando el `fsync` estricto para eventos de epoch, guardado de checkpoint o finalización del contexto (`__exit__`).
  * *Manejo de señales:* Si el entrenamiento recibe `SIGTERM` o `SIGINT` (Ctrl+C), el tracker debe registrar el estado `"interrupted"` o `"failed"` de forma determinista para que la corrida no quede perpetuamente como `"running"` en `run.json`.

### 2.3. Formato de Bundle y Serialización Segura (SafeTensors)
* **Diagnóstico actual:** `save_bundle` serializa los pesos con `torch.save(model.state_dict(), weights_buffer)` dentro de un archivo `.zip` como `weights.pt`.
* **Evaluación Senior:**
  * `torch.save` depende de `pickle`, lo que representa un vector de vulnerabilidad ante inyección de código arbitrario y genera acoplamiento indeseado con clases de Python al momento de inspeccionar artefactos.
  * Migrar los pesos a formato **SafeTensors** (`model.safetensors`) es el estándar moderno de la industria: es inmune a ejecución de código, rápido y tipado.
  * *Detalle de arquitectura:* Dado que SafeTensors dentro de un ZIP no puede aprovechar memoria mapeada (`mmap`) con zero-copy si está comprimido, el empaquetado del bundle debe guardar `model.safetensors` usando compresión nula (`zipfile.ZIP_STORED`) o cargar los bytes directamente mediante `safetensors.torch.load(bytes)`.
  * La información de optimizador y estado de entrenamiento (`train_state.pt`) continuará aislada y solo se requerirá cuando `is_checkpoint=True`.
  * En el `manifest.json` se debe registrar el hash criptográfico SHA-256 de los pesos para validar la integridad del bundle antes de cargarlo a memoria.

### 2.4. Suite de Determinismo Bit-a-Bit en CI
* **Diagnóstico actual:** Existen pruebas de snapshot y restauración de RNG, pero no un test de integración extremo que certifique paridad matemática absoluta entre una corrida continua y una corrida interrumpida y reanudada.
* **Evaluación Senior:**
  * Para que la suite corra de forma confiable en GitHub Actions (CPU) sin requerir GPUs costosas:
    * El test de determinismo debe ejecutarse sobre CPU configurando explícitamente `torch.use_deterministic_algorithms(True)` si aplica.
    * Debe verificarse que tras 10 épocas continuas vs (5 épocas + `save_bundle` + reinicio de modelo/optimizador/RNG + `resume_training` + 5 épocas), se cumpla:
      $$\text{Loss}_{\text{continua}} == \text{Loss}_{\text{reanudada}} \quad (\text{atol} = 0.0 \text{ o error de coma flotante } \le 10^{-7})$$
      $$\text{weights}_{\text{continua}} == \text{weights}_{\text{reanudada}} \quad (\text{torch.allclose})$$
    * El test debe cubrir la semilla de Python `random`, `numpy`, PyTorch CPU y PyTorch CUDA (cuando esté disponible).

### 2.5. Superficie de API Pública (SemVer 1.0.0)
* **Diagnóstico actual:** La API expuesta en `src/mangosta/__init__.py` ya es relativamente compacta, pero no existe una clara separación física entre utilidades de soporte interno y componentes de usuario.
* **Evaluación Senior:**
  * Se deben congelar los contratos públicos. Todo módulo auxiliar que no sea parte de la interfaz oficial debe residir en `mangosta._internal` o llevar prefijo privado `_`.
  * Pasar `ruff check` sin advertencias y validar cobertura de tipos estricta (`mypy --strict`).

---

## 3. Especificación Técnica de los Componentes

### Especificación A: Scaffolding Robusto y Minimalista (`mangosta init`)
La estructura generada por `mangosta init [nombre-proyecto]` debe ser:

```text
mi-proyecto/
├── configs/
│   ├── base.yaml
│   └── experiments/
│       └── exp_baseline.yaml
├── entrypoints/
│   ├── __init__.py
│   ├── train.py
│   ├── infer.py
│   └── benchmark.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── architectures/
│   │       ├── __init__.py
│   │       └── baseline.py
│   └── utils/
│       ├── __init__.py
│       └── config.py
├── .gitignore
├── pyproject.toml
└── README.md
```

#### Reglas de Scaffolding:
1. **Cero directorios vacíos:** No crear `checkpoints/`, `experiments/` ni `outputs/`. Se generarán perezosamente en tiempo de ejecución.
2. **`pyproject.toml` preconfigurado:** Incluir definición de proyecto con `dependencies = ["mangosta", "torch", "pyyaml"]` y la sección:
   ```toml
   [project.scripts]
   train = "entrypoints.train:main"
   infer = "entrypoints.infer:main"
   benchmark = "entrypoints.benchmark:main"
   ```
3. **Resolución de módulos limpia:** `entrypoints/` incluirá `__init__.py` y cada script configurará la resolución adecuada para que funcione indistintamente como `uv run train`, `uv run python -m entrypoints.train` o `python entrypoints/train.py`.

---

### Especificación B: Formato de Bundle Seguro y Tolerante a Fallos

El archivo `.bundle` es un contenedor ZIP inmutable con la siguiente estructura interna:

```text
modelo.bundle (ZIP)
├── manifest.json       # Esquema tipado, hashes, metadatos y contrato I/O
├── config.yaml         # Copia inmutable de la configuración de entrenamiento
├── model.safetensors   # Pesos de PyTorch en formato SafeTensors (cero pickle)
└── train_state.pt      # Estado de optimizador, scheduler y RNG (solo en checkpoints)
```

#### Protocolo de Escritura Atómica en Dos Fases:
1. Crear el archivo temporal en el mismo sistema de archivos: `{dest_name}.tmp_{pid}_{uuid4}`.
2. Escribir `manifest.json`, `config.yaml`, `model.safetensors` (usando `safetensors.torch.save`) y opcionalmente `train_state.pt`.
3. Validar integridad de escritura mediante cálculo del hash SHA-256 de los pesos dentro del manifiesto.
4. Aplicar `os.replace(tmp_dest, final_dest)`. Si ocurre una excepción previa, limpiar el archivo temporal en bloque `finally`.

#### Carga Desacoplada e Inspección:
* `mangosta inspect <bundle>` debe leer **únicamente** `manifest.json` y el encabezado de `model.safetensors` sin instanciar la clase de PyTorch ni ejecutar código en Python.
* `load_bundle()` verificará el hash de los pesos contra el manifiesto antes de reconstruir el grafo en memoria.

---

### Especificación C: Tracking Crash-Resilient sin Cuellos de Botella de I/O

El componente `RunTracker`:
1. **Estrategia de escritura de métricas:**
   * Las líneas en `metrics.jsonl` se escriben de manera secuencial en formato JSON-Lines.
   * `flush()` inmediato en cada iteración para garantizar que no queden datos en los buffers de Python.
   * `os.fsync()` configurable: ejecutado en cada época, al invocar `finish()`, o cada $K$ pasos (por defecto $K=50$ en pasos intra-época), evitando degradar el rendimiento del bucle de entrenamiento.
2. **Manejo de terminación de proceso:**
   * Implementación de manejadores seguros para `SIGINT` y `SIGTERM` que finalicen la corrida actual con estado `"interrupted"` y sincronicen el archivo `run.json`.
3. **Aislamiento de identificador de corrida:**
   * Incluir un sufijo aleatorio corto o PID en el nombre automático de la corrida si no se especifica uno, previniendo colisiones de directorios en ejecuciones concurrentes en la misma máquina.

---

### Especificación D: Validación de Contratos y Determinismo

1. **Invariancia Bit-a-Bit de Checkpoints:**
   * La biblioteca proveerá garantías matemáticas: retomar un checkpoint guardado con `is_checkpoint=True` debe continuar el camino de optimización exactamente idéntico al de una ejecución sin interrupciones bajo el mismo seed.
   * Cobertura de generadores RNG: Python (`random`), NumPy (`np.random`), PyTorch CPU (`torch.random`), PyTorch CUDA (`torch.cuda.random`) si está disponible.
2. **Contrato de Inferencia Previa al Grafo:**
   * `LoadedBundle.predict(sample, validate=True)` debe interceptar errores de dimensión, canal o tipo de dato (`dtype`) antes de transferir datos al dispositivo o invocar CUDA, emitiendo excepciones claras (`SchemaValidationError`).

---

## 4. Hoja de Ruta Faseada (Roadmap hacia v1.0.0)

```
[ v0.2.x (Alpha) ]
       │
       ▼
[ Fase 1: Scaffolding & DX ]  ────────► Limpieza de `init`, lazy folders, pyproject.toml en plantillas.
       │
       ▼
[ Fase 2: I/O Hardening & SafeTensors ] ► SafeTensors en bundles, hashing SHA-256, atomic rename blindado, buffer I/O.
       │
       ▼
[ Fase 3: Suite de Determinismo & CI ] ──► Tests de invariancia bit-a-bit, simulación de fallos/SIGINT, validación de schema.
       │
       ▼
[ Fase 4: Congelamiento de API & SemVer ]► Tipado estricto mypy, delimitación _internal, CLI inspect optimizada.
       │
       ▼
[ v1.0.0 (Release Estable) ]
```

### Fase 1: Scaffolding y Limpieza de Deuda Técnica (Objetivo: v0.3.0)
- [ ] Refactorizar `src/mangosta/cli/init_project.py`:
  - [ ] Eliminar la creación anticipada de `checkpoints/`, `experiments/`, `outputs/` y sus `.gitkeep`.
  - [ ] Reestructurar scripts hacia la carpeta `entrypoints/` (`train.py`, `infer.py`, `benchmark.py`).
  - [ ] Generar un `pyproject.toml` válido para el proyecto inicializado con `[project.scripts]`.
  - [ ] Actualizar `.gitignore` y `README.md` generados.
- [ ] Adaptar y ampliar tests unitarios en `tests/test_init.py` y `tests/test_cli.py`.

### Fase 2: I/O Seguro, SafeTensors y Persistencia Resiliente (Objetivo: v0.4.0)
- [ ] Agregar dependencia `safetensors>=0.4.0` a `pyproject.toml`.
- [ ] Refactorizar `src/mangosta/artifacts/bundle.py`:
  - [ ] Reemplazar `weights.pt` por `model.safetensors` utilizando la serialización de `safetensors.torch`.
  - [ ] Incorporar cálculo y verificación de SHA-256 de los pesos en `manifest.json`.
  - [ ] Blindar la rutina de creación atómica con captura de excepciones y eliminación de archivos `.tmp` huérfanos.
- [ ] Refactorizar `src/mangosta/artifacts/loader.py`:
  - [ ] Cargar pesos desde `model.safetensors` validando previamente la firma contra el manifiesto.
  - [ ] Mantener compatibilidad de lectura hacia atrás para bundles legacy con `weights.pt` con aviso de depreciación.
- [ ] Refactorizar `src/mangosta/tracking/tracker.py`:
  - [ ] Optimizar la frecuencia de `fsync` para evitar bloqueos en bucles rápidos de entrenamiento.
  - [ ] Captura de señales de interrupción (`SIGINT`/`SIGTERM`) para cierre ordenado del estado de corrida.

### Fase 3: Suite de Determinismo y Verificación en CI (Objetivo: v0.5.0-beta)
- [ ] Implementar `tests/test_determinism.py`:
  - [ ] Test de invariancia bit-a-bit (corrida continua de 10 épocas vs 5 épocas + guardado + interrupción + carga + 5 épocas restantes).
  - [ ] Verificación de pérdida idéntica (`loss == loss`) y pesos con `torch.allclose(..., atol=0.0)`.
- [ ] Implementar `tests/test_fault_tolerance.py`:
  - [ ] Simulación de interrupción abrupta de I/O y comprobación de no corrupción de checkpoints previos.
  - [ ] Validación de captura de excepciones en contratos de inferencia errónea (dimensiones cruzadas, tipos de datos inválidos).

### Fase 4: Congelamiento de API, Tipado y Release Estable (Objetivo: v1.0.0)
- [ ] Enriquecer `mangosta inspect <bundle>` en `src/mangosta/cli/main.py`:
  - [ ] Inspección segura de tensores y metadatos vía SafeTensors sin instanciar modelos en PyTorch.
  - [ ] Soporte para salida en formato `--json` para scripts de automatización.
- [ ] Definir fronteras de API pública estricta en `src/mangosta/__init__.py`.
- [ ] Mypy / Pyright: tipado estricto en el 100% de la superficie pública sin `type: ignore` injustificados.
- [ ] Cobertura de pruebas unitarias y de integración > 85%.
- [ ] Documentación de API y guía de migración para v1.0.0.

---

## 5. Definición de "Hecho" (Definition of Done para v1.0.0)

Para declarar formalmente la versión `1.0.0`, el proyecto debe satisfacer todos los siguientes criterios:
1. **Cero regresiones ni fallos de persistencia:** Todos los tests de la suite (`pytest`) pasan con éxito.
2. **Determinismo validado:** El test de invariancia bit-a-bit se ejecuta y aprueba en cada ejecución de CI.
3. **Seguridad de artefactos:** Los bundles generados no ejecutan `pickle` para los pesos del modelo.
4. **Scaffolding limpio:** `mangosta init` no genera carpetas vacías ni archivos huérfanos.
5. **Tipado y Calidad de Código:** `ruff check` y validación de tipos sin advertencias.

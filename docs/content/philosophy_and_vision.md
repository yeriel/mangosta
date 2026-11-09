# Philosophy & Vision

> "Simple by default, strict on demand."  
> — The Mangosta Manifesto

Machine learning operations has accumulated significant architectural bloat. What began as a need to record hyperparameters and persist weights has metastasized into distributed microservice topologies, database servers, proprietary cloud services, and heavy dockerized runtimes.

**Mangosta rejects this unnecessary complexity.** It reimagines MLOps through the lens of the **Unix Philosophy**, bringing clarity, speed, security, and deterministic reliability back to the PyTorch ecosystem.

---

## The Unix Philosophy in MLOps

Mangosta adheres strictly to classical software engineering virtues:

### 1. Plain Files Over Opaque Databases
- Everything in Mangosta is a readable file on the filesystem:
  - Model bundles are standard ZIP archives containing a `manifest.json` and `model.safetensors`.
  - Experiment logs are append-only `metrics.jsonl` files.
  - Artifacts are organized hierarchically under standard directory trees.
- You can inspect, version-control with Git LFS, archive with `tar`, rsync across clusters, or inspect with `unzip -l` and `jq` using standard operating system utilities.

### 2. Composable, Machine-Readable Streams
- Programs should write text streams that other programs can parse.
- Mangosta provides pure JSON outputs via `--json` across all CLI commands. There is no ASCII art or markdown noise when interacting programmatically.
- AI agents, shell pipelines, and orchestrators (Slurm, Kubernetes, Bash) can inspect models and experiment metrics in under 5 milliseconds.

### 3. Zero-Daemon Architecture
- Traditional platforms require background servers (`mlflow server`), local database engines (SQLite locks in distributed file shares), or cloud account credentials before logging a single epoch.
- Mangosta requires **zero background processes**. When your training script finishes, no orphaned process remains.
- The visual analytics dashboard (`mangosta view`) is an on-demand, read-only static/HTTP viewer that launches when you want to view results and closes cleanly when you exit.

---

## Core Tenets

```mermaid
graph TD
    A["Mangosta Design Pillars"] --> B["Security & Integrity"]
    A --> C["Crash-Consistency"]
    A --> D["Air-Gapped & Anti-Bloat"]
    A --> E["Agent-First Ergonomics"]

    B --> B1["Zero-Pickle SafeTensors"]
    B --> B2["SHA-256 Manifest Verification"]

    C --> C1["Atomic Filesystem Renames"]
    C --> C2["Deterministic 4-RNG Resumption"]

    D --> D1["Zero Cloud Dependencies"]
    D --> D2["Pure Python Standard Library Core"]

    E --> E1["< 5ms SafeTensors Header Parser"]
    E --> E2["Structured Error Codes with Remediation Hints"]

    classDef default fill:#1e293b,stroke:#3b82f6,stroke-width:1px,color:#f8fafc;
```

### 1. Zero-Pickle Security by Default
Python's `pickle` serialization format (the historical default for `torch.save`) executes arbitrary bytecode upon deserialization. Loading untrusted `.pt` or `.pth` files is a critical remote code execution (RCE) vector.

Mangosta enforces **Zero-Pickle execution**:
- Model weights are stored exclusively using Hugging Face SafeTensors (`model.safetensors`).
- Every archive computes a SHA-256 digest at write time, verified upon loading.
- Model architectures are decoupled from serialization and registered declaratively with `@register_model`.

### 2. Crash-Consistent Fault Tolerance
Distributed training on preemptible or spot instances frequently suffers from abrupt SIGKILL terminations. In standard PyTorch code, if a process dies while writing a checkpoint, the resulting checkpoint file is corrupted and unusable.

Mangosta guarantees crash-consistency:
- All bundle saves write to a process-unique temporary file (`.tmp_<pid>_<uuid>`).
- Only after weights, manifests, schemas, and auxiliary states are completely flushed and closed does Mangosta perform an atomic filesystem rename (`os.replace`).
- If your job crashes mid-write, your previous checkpoint remains valid and pristine.

### 3. Air-Gapped and Sovereign First
Mangosta is designed from the ground up for environments with zero internet connectivity:
- Hospital on-premise clusters subject to patient data sovereignty regulations (HIPAA/GDPR).
- Air-gapped defense and government research environments.
- Edge robotic devices running offline.
- High-performance computing (HPC) environments with strict egress firewalls.

Mangosta has zero mandatory cloud telemetry, zero remote tracking pings, and does not require outbound internet access at any stage of its lifecycle.

---

## What Teams Build with Mangosta

### Solo ML Researchers & PhD Students
- Instant project scaffolding (`mangosta init`).
- Zero cognitive overhead: track experiments locally, view curves with `mangosta view`, and share reproducible `.bundle` files with collaborators.

### Enterprise Computer Vision & NLP Teams
- Standardize internal model repositories without deploying heavy infrastructure.
- Model weights verified via SHA-256 checksums to eliminate model supply chain tampering.
- Seamless conversion to ONNX bundles (`export_onnx_as_bundle`) for multi-platform hardware acceleration.

### Autonomous AI Coding Agents
- Coding agents (e.g., Antigravity, Claude Code, Cursor, Codex) can autonomously train, inspect, debug, and benchmark machine learning models.
- Fast binary inspection (< 5ms) avoids costly token waste and GPU memory consumption.
- Actionable error messages provide structured repair strategies without hallucination.

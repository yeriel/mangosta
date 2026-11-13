# First Project (`mangosta init`)

**Goal:** Scaffold a clean, runnable PyTorch project in one command and verify every layer works before writing custom code.

---

## 1. Create a New Project

Run `mangosta init` inside a new folder name (or `.` inside an existing directory):

```bash
uv run mangosta init my_vision_project
cd my_vision_project
```

When `uv` is available on your `PATH`, `mangosta init` automatically:

1. Scaffolds a Clean Architecture layout (`configs/`, `src/`, `entrypoints/`).
2. Upgrades bare `"mangosta"` to `"mangosta[full]"` in `pyproject.toml`.
3. Configures `hatchling` and registers `train`, `infer`, and `benchmark` as project scripts.
4. Copies the official AI skill into `.agents/skills/mangosta/` and generates `AGENTS.md` with KISS & SOLID guidelines.
5. Runs `uv sync` so your virtual environment is ready immediately.

---

## 2. Explore the Generated Layout

```text
my_vision_project/
├── AGENTS.md                        # Root KISS/SOLID rules pointing AI agents to the local skill
├── pyproject.toml                   # Configured with mangosta[full] & train/infer/benchmark scripts
├── README.md
├── configs/
│   └── exp_baseline.yaml            # Typed MangostaConfig hyperparameters & training settings
├── src/
│   ├── data.py                      # Pure PyTorch DataLoader builder
│   └── models.py                    # @register_model + @register_trainer classes
├── entrypoints/
│   ├── train.py                     # CLI training entrypoint (`uv run train`)
│   ├── infer.py                     # Schema-validated bundle inference (`uv run infer`)
│   └── benchmark.py                 # Latency & throughput check (`uv run benchmark`)
└── .agents/
    └── skills/mangosta/
        ├── SKILL.md                 # Full v0.4.0 reference for coding agents
        ├── resources/config_schema.yaml
        ├── examples/
        └── scripts/verify_project.py
```

!!! info "Zero Clutter Upfront"
    `experiments/` and `checkpoints/` are **not** created during `init`. They appear automatically only when you run your first training job.

---

## 3. Validate & Run in 3 Commands

Always dry-run your config and registry wiring before launching a long training job:

```bash
# 1. Validate YAML keys and verify ModelRegistry / TrainerRegistry can build your classes
uv run python .agents/skills/mangosta/scripts/verify_project.py --config configs/exp_baseline.yaml

# 2. Run training (logs to experiments/ and writes checkpoints/model_*.bundle)
uv run train --config configs/exp_baseline.yaml

# 3. Inspect results in the browser or run inference
uv run mangosta view
uv run infer --bundle checkpoints/model_final.bundle
```

---

## Recap

You now have a working PyTorch pipeline backed by `MangostaConfig`, registries, `BaseTrainer`, atomic tracking, and `.bundle` outputs.

**Next step:** [Config & Registries](02_config_and_registries.md) — learn how strict YAML validation and semantic-versioned registries work.

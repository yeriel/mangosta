# Tracking & `mangosta view`

**Goal:** Compare models, inspect hyperparameters, scrub validation images epoch-by-epoch, and diff `.bundle` architectures in an ephemeral local dashboard—with zero background servers.

---

## 1. Launch `mangosta view`

Point `mangosta view` at your project root, experiments folder, or a `.bundle`:

```bash
# Scan current workspace (default port 8501)
uv run mangosta view

# Specific folder or port
uv run mangosta view experiments --port 8502

# Remote SSH / headless server without launching a browser window
uv run mangosta view --no-browser
```

Open **`http://localhost:8501`**. Press `Ctrl+C` in the terminal to stop the server cleanly with zero orphaned processes.

---

## 2. What You Can Do in the Dashboard

The Crisp Swiss Light interface organizes your workspace into 4 tabs:

1. **`Metrics & Runs`**
   - **Compare Models & Runs**: Click **`Compare Runs`** in the toolbar, **`Latest 2` / `All`** in the sidebar, or check 2+ runs manually to unlock:
     - The **Model & Run Comparison** table (`Run`, `Model`, `Status`, `Best Monitored`, latest metrics with best-value highlighting, `Epoch / Step`, `Duration`).
     - Overlaid interactive SVG curves with **Linear / Log** scale toggle, **EMA smoothing slider**, and step crosshair tooltips.
     - **`MangostaConfig Parameters`** diff table with **`Differences only`** filter.
   - **Live Watch**: Automatically polls active runs every `2.5s` when runs are running.
2. **`Validation`**
   - Side-by-side **`Input`** vs **`Ground Truth`** vs **`Prediction`** cards for each logged `sample_id`.
   - Temporal **Epoch Scrubber** + **Play Timelapse** button + interactive **Lightbox** (mouse-wheel zoom and drag-to-pan).
3. **`Bundles`**
   - Discovers every `.bundle` and `.onnx.bundle` in your workspace and inspects SafeTensors layers without importing PyTorch (`total_params`, shapes, dtypes, `ModelSchema`, metrics).
   - Click **`Compare All`** (or check 2+ bundles) for a **side-by-side Model & Bundle Comparison**: architecture metadata, recorded metrics, `ModelSchema` contract diff, and SafeTensors layer diff.
4. **`Metadata`**
   - Provenance for the active run: Git commit/branch/dirty state, dataset SHA-256, and Python/PyTorch environment info.

---

## 3. Manual Tracking (`RunTracker`) Outside `BaseTrainer`

When you are not using `BaseTrainer`, wrap your custom loop with `RunTracker`:

```python
import math
import numpy as np
from mangosta import RunTracker

with RunTracker(experiment="segmentation", run_name="unet_lr1e-3", root_dir="experiments") as run:
    run.log_params({"learning_rate": 1e-3, "optimizer": "AdamW"})
    input_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    gt_mask = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    for epoch in range(1, 5):
        loss = 0.85 * math.exp(-0.45 * epoch)
        run.log_metrics({"train_loss": loss, "val_loss": loss + 0.04}, step=epoch)  # (1)!

        pred_mask = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        run.log_validation_sample(  # (2)!
            sample_id="patient_01",
            prediction=pred_mask,
            input_image=input_img,
            ground_truth=gt_mask,
            step=epoch,
        )
```

1. Metrics starting with `train_` / `val_` (or `train/` / `val/`) auto-pair on shared charts in `mangosta view`.
2. Saves step-versioned PNGs inside `artifacts/` so the **Validation** tab builds an epoch-by-epoch timelapse automatically.

### Flat-File Storage Layout

```text
experiments/
└── segmentation/
    └── unet_lr1e-3/
        ├── run.json       # Params, git commit, dataset SHA-256, status, timestamps
        ├── metrics.jsonl  # Append-only step metrics
        └── artifacts/     # Logged files & validation sample images
```

---

**Next step:** [ONNX Export & Serving](06_onnx_export_and_serving.md) — export models to ONNX with auto-inferred shapes and serve them in `~70MB` containers without PyTorch.

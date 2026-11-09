# Guide 04: Visual Tracking & `mangosta view`

> **Companion script:** [`examples/04_visual_tracking_and_timelapses.py`](https://github.com/yeriel/mangosta/blob/main/examples/04_visual_tracking_and_timelapses.py)  
> **Run it locally:** `uv run python examples/04_visual_tracking_and_timelapses.py`

Mangosta includes an ephemeral, zero-daemon visual dashboard (`mangosta view`) built on Python's standard `http.server`. It lets you compare multi-run loss curves and scrub epoch-by-epoch through validation sample predictions (`Input` vs. `Ground Truth` vs. `Prediction`).

---

## Step 1: Log Metrics & Validation Sample Timelapses

Use `RunTracker.log_validation_sample()` inside your validation loop to record how your model's predictions evolve on real samples across epochs:

```python
import math
import numpy as np
from mangosta import RunTracker

with RunTracker(
    experiment="medical_imaging",
    run_name="baseline_unet_lr1e-3",
    root_dir="experiments",
) as tracker:
    tracker.log_params({
        "learning_rate": 1e-3,
        "optimizer": "AdamW",
        "batch_size": 16,
    })

    # Synthetic 128x128 RGB validation input and ground truth mask
    input_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    gt_mask = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    for epoch in range(4):
        loss = 0.85 * math.exp(-0.45 * epoch) + 0.05
        dice = 0.40 + 0.52 * (1.0 - math.exp(-0.6 * epoch))

        # 1. Log scalar curves (grouped automatically by train/* and val/*)
        tracker.log_metrics({
            "train/loss": float(loss),
            "val/dice_score": float(dice),
            "val/loss": float(loss + 0.04),
        }, step=epoch)

        # 2. Log step-versioned validation sample for the temporal scrubber
        pred_mask = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        tracker.log_validation_sample(
            sample_id="patient_01_lesion",
            prediction=pred_mask,
            input_image=input_img,
            ground_truth=gt_mask,
            step=epoch,
        )
```

### Flat-File Layout on Disk
`RunTracker` stores everything as plain, inspectable files:

```text
experiments/
└── medical_imaging/
    └── baseline_unet_lr1e-3/
        ├── run.json
        ├── metrics.jsonl
        └── artifacts/
            ├── patient_01_lesion_input.png
            ├── patient_01_lesion_gt.png
            ├── patient_01_lesion_pred_step_0000.png
            ├── patient_01_lesion_pred_step_0001.png
            └── ...
```

---

## Step 2: Launch `mangosta view`

Point `mangosta view` to your workspace, experiment directory, or a specific `.bundle`:

```bash
# Scan current directory (default port 8501)
uv run mangosta view

# Or point to a specific folder or port
uv run mangosta view experiments --port 8501

# Run on a remote SSH server without opening a local browser window
uv run mangosta view experiments --no-browser
```

Open **`http://127.0.0.1:8501`** in your browser. Press `Ctrl+C` in the terminal to stop the server cleanly with zero leftover processes.

---

## What You Can Do in the Dashboard

- **Grouped Train & Val Charts**: View all `train/*` metrics together and all `val/*` metrics together with adjustable EMA smoothing.
- **Multi-Run Overlay**: Select multiple runs in the sidebar to compare hyperparameter configurations side-by-side.
- **Temporal Validation Scrubber**: Drag the epoch slider to watch `Prediction` converge toward `Ground Truth` step-by-step alongside the `Input` image.
- **SafeTensors Bundle Inspector**: Inspect `.bundle` files discovered in the workspace directly from the UI.

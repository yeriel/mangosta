# Guide 5: Visual Tracking & Validation Timelapses

Mangosta provides an integrated, zero-daemon visual analytics dashboard: `mangosta view`. It enables researchers and engineers to monitor training metrics, compare experiment runs, and scrub through visual validation outputs across training epochs.

---

## 1. Launching the Dashboard

From the root of your project (where your `runs/` directory is located), run:

```bash
mangosta view
```

### Options
```text
Usage: mangosta view [OPTIONS]

  Launch the local visual analytics dashboard.

Options:
  --port INTEGER  Port to bind dashboard server to. [default: 8501]
  --host TEXT     Host IP to bind to. [default: 127.0.0.1]
  --logdir PATH   Path to root directory containing experiment runs. [default: runs]
  --help          Show this message and exit.
```

The server starts instantly in the foreground and outputs:
```text
Mangosta View active at: http://127.0.0.1:8501
Serving runs from: runs
Press Ctrl+C to terminate.
```

---

## 2. Tracking Metrics & Images During Training

Log scalar metrics and visual artifacts using `RunTracker`:

```python
import torch
import matplotlib.pyplot as plt
from mangosta import RunTracker

with RunTracker(experiment="generative", run_name="latent_vae_01") as tracker:
    for epoch in range(1, 11):
        # Simulated training metrics
        train_loss = 1.0 / epoch
        val_loss = 1.2 / epoch
        
        tracker.log_metrics(
            {"loss": train_loss, "val_loss": val_loss, "lr": 1e-3 * (0.95 ** epoch)},
            step=epoch,
        )

        # Generate and log visual validation image
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(torch.rand(28, 28).numpy(), cmap="inferno")
        ax.set_title(f"Reconstruction - Epoch {epoch}")
        ax.axis("off")

        tracker.log_image("reconstructions", fig, step=epoch)
        plt.close(fig)
```

### Filesystem Structure
Tracking writes plain, inspectable files into `runs/`:
```text
runs/
└── generative/
    └── latent_vae_01/
        ├── metrics.jsonl
        ├── summary.json
        └── images/
            └── reconstructions/
                ├── epoch_0001.png
                ├── epoch_0002.png
                └── ...
```

---

## 3. Dashboard Features

### Real-Time Metric Curves
- Dynamic SVG charts render training loss, validation loss, accuracy, and learning rates.
- Multi-run overlay: select checkboxes next to different runs to visually compare baseline vs candidate experiments on identical axes.

### Temporal Validation Scrubber (Timelapses)
- When image tags are detected in `images/<tag>/`, the dashboard displays an interactive timeline scrubber.
- Drag the slider or use keyboard arrows to step epoch-by-epoch through images.
- Observe generative sample quality, segmentation mask refinements, or adversarial perturbations over time.

### Minimalist Industrial Craft Design
- High-contrast slate dark theme optimized for ML engineering workflows.
- Zero extraneous visual clutter or emoji decorations.
- Fully responsive layout with live auto-refresh while training loops are active.

---

## 4. Headless & Remote Usage

To view training progress on a remote cluster or SSH server:

1. Launch on the remote host binding to `127.0.0.1`:
   ```bash
   mangosta view --port 8501
   ```
2. Forward the port to your local machine via SSH:
   ```bash
   ssh -N -L 8501:127.0.0.1:8501 user@remote-gpu-host
   ```
3. Open `http://localhost:8501` in your local browser.

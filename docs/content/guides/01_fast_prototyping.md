# Guide 1: Fast Prototyping for Solo Researchers

This guide walks through creating an end-to-end PyTorch workflow in less than 5 minutes using `mangosta`'s zero-infrastructure philosophy: no cloud setup, no daemon services, and zero configuration boilerplate.

---

## 1. Project Initialization

Generate a clean, production-ready directory structure:

```bash
mangosta init my_classifier
cd my_classifier
```

The scaffolding creates:
- `configs/baseline.yaml`: Clean experiment configuration.
- `src/models/`: Modular architecture definition.
- `entrypoints/train.py`: Ready-to-run training loop.
- `.agents/mangosta.md`: Formal context guidelines for AI coding agents.

---

## 2. Model Registration & Clean Architecture

Define your neural network in `src/models/classifier.py` and decorate it with `@register_model`:

```python
import torch
import torch.nn as nn
from mangosta import register_model

@register_model(family="mlp_classifier", version="1.0.0")
class MLPClassifier(nn.Module):
    def __init__(self, in_features: int = 8, hidden_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

Why this matters:
- Decorating with `@register_model` decouples the architecture from persistence code.
- You never need to use `pickle` or store Python code strings inside checkpoint files.

---

## 3. Experiment Tracking with `RunTracker`

Log hyperparameters, training steps, and validation artifacts directly to disk:

```python
import torch
from mangosta import RunTracker
from src.models.classifier import MLPClassifier

model = MLPClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

with RunTracker(experiment="quickstart", run_name="baseline_exp") as run:
    # 1. Log configuration parameters
    run.log_params({"lr": 1e-3, "batch_size": 32, "architecture": "mlp_classifier"})

    # 2. Training loop
    for epoch in range(10):
        # Simulated batch
        inputs = torch.randn(32, 8)
        targets = torch.randint(0, 2, (32,))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Step-level metric logging
        run.log_metric("train/loss", loss.item(), step=epoch)

    # 3. Log a validation artifact (e.g. sample predictions or plots)
    run.log_image("sample_prediction", torch.randn(3, 64, 64), step=10)
```

---

## 4. Ephemeral Zero-Daemon Visualization

To inspect curves and validation artifacts, simply launch the built-in ephemeral viewer:

```bash
mangosta view
```

- Spawns Python's native `http.server` on `localhost:8501`.
- Opens your browser automatically.
- No background workers, no lingering Redis/Postgres services, and stops cleanly when pressing `Ctrl+C`.

To generate a standalone, offline HTML report to share with colleagues:

```bash
mangosta report runs/quickstart/baseline_exp -o baseline_report.html
```
The report is a single HTML file containing embedded interactive charts and zero external CDN links.

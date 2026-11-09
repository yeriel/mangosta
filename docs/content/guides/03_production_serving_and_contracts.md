# Guide 03: Export, `<5ms` Inspection & Reports

> **Companion script:** [`examples/03_model_export_and_inspection.py`](https://github.com/yeriel/mangosta/blob/main/examples/03_model_export_and_inspection.py)  
> **Run it locally:** `uv run python examples/03_model_export_and_inspection.py`

This guide covers how to export models (including multi-output dictionary models) to standalone `.onnx` or contract-verified `.onnx.bundle` archives, inspect bundles in `< 5ms`, and generate offline HTML reports.

---

## Step 1: Multi-Output Models & `output_mapping`

Real-world vision and NLP models often return a `dict[str, torch.Tensor]` (e.g., `{"logits": ..., "bbox": ...}`). Mangosta's `export_onnx` and `export_onnx_as_bundle` accept `output_mapping` to cleanly unroll dictionary outputs into positional ONNX graph outputs:

```python
import torch
import torch.nn as nn
from mangosta import ModelRegistry, register_model, save_bundle

@register_model(family="multitask_detector", version="1.0.0")
class MultiTaskDetector(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 16) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier_head = nn.Linear(feature_dim, 2)
        self.bbox_head = nn.Linear(feature_dim, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {"logits": self.classifier_head(feat), "bbox": self.bbox_head(feat)}

model = ModelRegistry.build("multitask_detector", version="1.0.0")
sample_input = torch.randn(1, 3, 64, 64)

save_bundle(
    output_path="detector.bundle",
    model=model,
    family="multitask_detector",
    version="1.0.0",
    sample_input=sample_input,
    metrics={"val_iou": 0.884, "val_acc": 0.942},
)
```

---

## Step 2: Export to `.onnx` or `.onnx.bundle`

Requires `uv add "mangosta[onnx]"`:

```python
from mangosta import export_onnx, export_onnx_as_bundle

# 1. Export standalone .onnx with dynamic batch axes and graph verification
export_onnx(
    model_or_bundle="detector.bundle",
    output_path="detector.onnx",
    sample_input_shape=(1, 3, 64, 64),
    output_mapping=["logits", "bbox"],
    input_names=["input_tensor"],
    output_names=["pred_logits", "pred_bbox"],
    dynamic_axes={
        "input_tensor": {0: "batch_size"},
        "pred_logits": {0: "batch_size"},
        "pred_bbox": {0: "batch_size"},
    },
    float16=False,
    verify=True,
)

# 2. Export self-contained .onnx.bundle (includes manifest.json + ModelSchema + model.onnx)
export_onnx_as_bundle(
    bundle_or_model="detector.bundle",
    output_path="detector.onnx.bundle",
    sample_input=sample_input,
    output_mapping=["logits", "bbox"],
    input_names=["input_tensor"],
    output_names=["pred_logits", "pred_bbox"],
)
```

Or directly from the CLI:

```bash
uv run mangosta export-onnx detector.bundle \
    -o detector.onnx.bundle \
    --input-shape 1 3 64 64 \
    --output-mapping logits bbox \
    -m my_project.models
```

---

## Step 3: Sub-5ms Inspection (`mangosta inspect`)

Inspect any `.bundle` or `.onnx.bundle` without importing PyTorch or loading weights into memory. Mangosta reads the 8-byte little-endian SafeTensors binary header directly from the ZIP stream:

```bash
# Human-readable summary in terminal
uv run mangosta inspect detector.bundle

# Machine-readable JSON (ideal for CI/CD and AI agents)
uv run mangosta inspect detector.bundle --json
```

---

## Step 4: Offline Interactive HTML Reports (`mangosta report`)

Generate a standalone, single-file `report.html` summarizing a training run or model bundle (shareable via Slack or email, zero servers required):

```bash
uv run mangosta report detector.bundle -o detector_report.html
```

Or from Python:

```python
from mangosta import generate_report

generate_report("detector.bundle", "detector_report.html")
```

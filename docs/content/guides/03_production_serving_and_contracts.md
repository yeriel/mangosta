# Guide 3: Production Serving & Tensor Contracts

Deploying PyTorch models directly into production often introduces serious operational headaches:
1. **Container Bloat**: Docker images carrying PyTorch and CUDA toolchains typically weigh `2.5GB - 6GB`.
2. **Cold Start Latency**: Importing PyTorch and compiling model graphs slows down Kubernetes pod scaling and serverless invocations.
3. **Data Drift & Silent Corruption**: Production inference receiving incorrect tensor dimensions or integer types instead of floats causes silent degradation or cryptic C++ segmentation faults.

`mangosta` solves this via **ONNX Bundles (`.onnx.bundle`)** and **Slim Serving (`mangosta[serving]`)**.

---

## 1. Exporting an ONNX Bundle

Export your trained `.bundle` to an optimized `.onnx.bundle` package:

```bash
mangosta export-onnx checkpoints/epoch_10.bundle \
    -o exports/model.onnx.bundle \
    --input-shape 1 8 \
    --fp16 \
    -m src.models.classifier
```

Or via Python:

```python
import torch
from mangosta import export_onnx

export_onnx(
    model_or_bundle="checkpoints/epoch_10.bundle",
    output_path="exports/model.onnx.bundle",
    sample_input_shape=(1, 8),
    float16=True,
    as_bundle=True,
)
```

The resulting `.onnx.bundle` is a standard ZIP file containing:
- `manifest.json`: Cryptographic SHA-256 fingerprint, model family, version, and formal `schema`.
- `model.onnx`: Standalone ONNX computation graph.
- `config.yaml`: Frozen hyperparameters and pre-processing contracts.

---

## 2. Zero-PyTorch Slim Serving

In your production API container (e.g. FastAPI, Triton, or Flask), install only the slim dependencies:

```bash
pip install "mangosta[serving]"
# or
pip install mangosta onnxruntime
```

Notice: **PyTorch is NOT installed**. The container size drops from **~3GB down to ~70MB**.

### Serving Code (`server.py`):

```python
from fastapi import FastAPI, HTTPException
import numpy as np
from mangosta import ONNXRunner
from mangosta.exceptions import SchemaValidationError

app = FastAPI(title="ML Microservice")

# Load directly from the verified ONNX bundle archive
runner = ONNXRunner("exports/model.onnx.bundle")

@app.post("/predict")
def predict(features: list[float]):
    input_data = np.array([features], dtype=np.float32)

    try:
        # Executes inference with automatic tensor contract validation
        predictions = runner.predict(input_data)
        return {"predictions": predictions.tolist()}
    except SchemaValidationError as e:
        raise HTTPException(status_code=422, detail=f"Data contract violation: {e}")
```

### Automatic Protections:
- **Shape & Dtype Verification**: If a client sends an input with 7 features instead of 8, or passes an invalid data type, `runner.predict()` rejects it before calling the C++ inference engine.
- **Output Container Preservation**: If your original model returned a dictionary (`{"logits": ..., "probs": ...}`), `runner.predict()` automatically formats the output back into the expected dictionary.

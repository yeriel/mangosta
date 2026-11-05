"""Verification script running inside ultra-slim container without PyTorch installed."""

from __future__ import annotations

import sys
import numpy as np

# 1. Assert PyTorch is NOT installed
try:
    import torch  # type: ignore
    raise AssertionError("CRITICAL: PyTorch is present in slim serving container! Expected ZERO torch.")
except ImportError:
    pass

# 2. Import mangosta - must succeed cleanly without torch
import mangosta
from mangosta.artifacts import load_bundle
from mangosta.core.schema import ModelSchema, TensorSpec, infer_schema_from_sample
from mangosta.exceptions import SchemaValidationError
from mangosta.serving import LoadedONNXBundle, ONNXRunner

assert "torch" not in sys.modules, "torch was unexpectedly imported by mangosta!"
print("✓ mangosta imported successfully in zero-PyTorch environment.")

# 3. Verify accessing PyTorch-only methods gives actionable error
try:
    mangosta.save_bundle  # type: ignore
    raise AssertionError("Accessing mangosta.save_bundle should have raised ImportError")
except ImportError as e:
    assert "requires PyTorch" in str(e)
    print("✓ Actionable hint on torch-only symbols confirmed.")

# 4. Infer schema pure NumPy
spec_in = np.zeros((1, 8), dtype=np.float32)
spec_out = np.zeros((1, 2), dtype=np.float32)
schema = infer_schema_from_sample(spec_in, spec_out)
assert schema.input_structure == "tensor"
print("✓ Schema inferred purely from NumPy sample arrays.")

# 5. Load pre-generated ONNX bundle
bundle_path = "/app/test_bundle.onnx.bundle"
loaded = load_bundle(bundle_path)
assert isinstance(loaded, LoadedONNXBundle)
print(f"✓ Loaded ONNX bundle '{bundle_path}' (family={loaded.family}).")

# 6. Verify schema enforcement
bad_input = np.random.randn(1, 10).astype(np.float32)
try:
    loaded.predict(bad_input)
    raise AssertionError("SchemaValidationError expected for bad input shape")
except SchemaValidationError:
    print("✓ Strict schema contract rejected invalid input shape in production.")

# 7. Execute valid inference
valid_input = np.random.randn(2, 8).astype(np.float32)
output = loaded.predict(valid_input)
assert output.shape == (2, 2)
assert output.dtype == np.float32
print(f"✓ Production inference executed successfully: output shape {output.shape}, dtype {output.dtype}.")

print("=" * 60)
print("ZERO-PYTORCH SLIM SERVING VERIFIED SUCCESSFULLY!")
print("=" * 60)

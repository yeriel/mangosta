"""Ultra-slim production inference runtime and contract enforcement for ONNX bundles."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    ort = None  # type: ignore[assignment]
    HAS_ONNXRUNTIME = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    HAS_YAML = False

from mangosta.core.schema import ModelSchema
from mangosta.exceptions import BundleCorruptedError


def _require_onnxruntime() -> None:
    """Ensure onnxruntime is installed before executing inference."""
    if not HAS_ONNXRUNTIME or ort is None:
        raise ImportError(
            "onnxruntime is required for ONNX serving.\n"
            "Install it via: `pip install 'mangosta[serving]'` or `pip install onnxruntime`."
        )


def _to_numpy(val: Any) -> Any:
    """Convert a tensor or array-like object to a NumPy ndarray without requiring PyTorch."""
    if hasattr(val, "detach") and hasattr(val, "cpu"):
        # PyTorch Tensor conversion without top-level torch dependency
        return val.detach().cpu().numpy()
    if HAS_NUMPY and np is not None and isinstance(val, np.ndarray):
        return val
    if hasattr(val, "__array__"):
        if HAS_NUMPY and np is not None:
            return np.asarray(val)
        return val
    return val


class LoadedONNXBundle:
    """Encapsulates a loaded ONNX model bundle ready for high-performance production serving."""

    def __init__(
        self,
        session: Any,
        manifest: dict[str, Any] | None = None,
        schema: ModelSchema | None = None,
        config: dict[str, Any] | str | None = None,
        bundle_path: Path | None = None,
    ) -> None:
        _require_onnxruntime()
        self.session = session
        self.manifest = manifest or {}
        self.schema = schema
        self.config = config
        self.bundle_path = bundle_path

        # Cache input and output node names from ONNX session
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    @property
    def family(self) -> str:
        return str(self.manifest.get("family", "unknown"))

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "unknown"))

    @property
    def metadata(self) -> dict[str, Any]:
        meta = self.manifest.get("metadata", {})
        return dict(meta) if isinstance(meta, dict) else {}

    @property
    def model(self) -> Any:
        """Polymorphic alias to the underlying ONNX inference session."""
        return self.session

    def __repr__(self) -> str:
        in_names = f", inputs={self.input_names}" if self.input_names else ""
        out_names = f", outputs={self.output_names}" if self.output_names else ""
        return f"LoadedONNXBundle(family='{self.family}', version='{self.version}', format='onnx'{in_names}{out_names})"

    def predict(self, input_data: Any, validate: bool = True) -> Any:
        """Execute forward inference with strict data contract validation against ModelSchema.

        Args:
            input_data: A NumPy ndarray, sequence of arrays, or dict of named arrays.
            validate: When True (default), strictly validates shapes, dtypes, and names
                      against ModelSchema before calling the C++ ONNX Runtime.

        Returns:
            NumPy array, tuple of arrays, or dict of arrays matching the schema contract.
        """
        # 1. Enforce input contract
        if validate and self.schema is not None:
            self.schema.validate_input(input_data)

        # 2. Build input feed dictionary for ONNX Runtime
        input_feed: dict[str, Any] = {}
        if isinstance(input_data, Mapping):
            for k, v in input_data.items():
                input_feed[k] = _to_numpy(v)
        elif isinstance(input_data, (list, tuple)):
            if len(input_data) != len(self.input_names):
                raise ValueError(
                    f"Model expects {len(self.input_names)} inputs ({self.input_names}), "
                    f"but received {len(input_data)} items."
                )
            for name, item in zip(self.input_names, input_data):
                input_feed[name] = _to_numpy(item)
        else:
            # Single array/tensor
            if not self.input_names:
                raise RuntimeError("ONNX model has no input nodes defined.")
            input_feed[self.input_names[0]] = _to_numpy(input_data)

        # 3. Execute ONNX Runtime inference
        raw_outputs = self.session.run(None, input_feed)

        # 4. Format outputs based on schema contract or session metadata
        if self.schema is not None:
            if self.schema.output_structure == "tensor":
                output = raw_outputs[0]
            elif self.schema.output_structure == "sequence":
                output = tuple(raw_outputs)
            elif self.schema.output_structure == "mapping":
                if isinstance(self.schema.outputs, dict):
                    keys = list(self.schema.outputs.keys())
                else:
                    keys = self.output_names
                output = dict(zip(keys, raw_outputs))
            else:
                output = raw_outputs[0] if len(raw_outputs) == 1 else tuple(raw_outputs)

            if validate:
                self.schema.validate_output(output)
            return output

        # If no schema was specified in manifest
        if len(raw_outputs) == 1:
            return raw_outputs[0]
        return tuple(raw_outputs)


class ONNXRunner(LoadedONNXBundle):
    """Production ONNX runner for standalone .onnx files or .onnx.bundle archives."""

    def __init__(
        self,
        model_path_or_bundle: str | Path,
        schema: ModelSchema | None = None,
        config: dict[str, Any] | str | None = None,
        providers: list[str] | None = None,
    ) -> None:
        path = Path(model_path_or_bundle).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model or bundle file not found: {model_path_or_bundle}")

        # Check if bundle archive
        if path.suffix == ".bundle" or path.name.endswith(".onnx.bundle") or zipfile.is_zipfile(path):
            loaded = load_onnx_bundle(path, providers=providers)
            super().__init__(
                session=loaded.session,
                manifest=loaded.manifest,
                schema=schema or loaded.schema,
                config=config or loaded.config,
                bundle_path=path,
            )
        else:
            # Standalone .onnx file
            _require_onnxruntime()
            providers_list = providers or ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(path), providers=providers_list)
            super().__init__(
                session=session,
                manifest={"weights_format": "onnx"},
                schema=schema,
                config=config,
                bundle_path=path,
            )

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        providers: list[str] | None = None,
    ) -> LoadedONNXBundle:
        """Load an ONNX bundle archive directly."""
        return load_onnx_bundle(bundle_path, providers=providers)


def load_onnx_bundle(
    bundle_path: str | Path,
    providers: list[str] | None = None,
) -> LoadedONNXBundle:
    """Load an ONNX bundle archive (.onnx.bundle) with cryptographic validation and schema contract."""
    _require_onnxruntime()
    path = Path(bundle_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ONNX bundle file not found: {bundle_path}")

    try:
        with zipfile.ZipFile(path, "r") as zf:
            namelist = zf.namelist()
            if "manifest.json" not in namelist:
                raise BundleCorruptedError(f"Missing 'manifest.json' in ONNX bundle: {path.name}")
            if "model.onnx" not in namelist:
                raise BundleCorruptedError(f"Missing 'model.onnx' in ONNX bundle: {path.name}")

            manifest_bytes = zf.read("manifest.json")
            manifest = json.loads(manifest_bytes.decode("utf-8"))

            onnx_bytes = zf.read("model.onnx")

            # Cryptographic integrity check
            expected_sha256 = manifest.get("model_sha256") or manifest.get("weights_sha256")
            if expected_sha256:
                actual_sha256 = hashlib.sha256(onnx_bytes).hexdigest()
                if actual_sha256 != expected_sha256:
                    raise BundleCorruptedError(
                        f"ONNX bundle cryptographic integrity check failed for '{path.name}'.\n"
                        f"Expected SHA-256: {expected_sha256}\n"
                        f"Actual SHA-256:   {actual_sha256}\n"
                        "The ONNX model file has been corrupted or tampered with."
                    )

            # Reconstruct schema contract if present
            schema: ModelSchema | None = None
            if manifest.get("schema") is not None:
                schema = ModelSchema.from_dict(manifest["schema"])

            # Optional config.yaml
            config: dict[str, Any] | str | None = None
            if "config.yaml" in namelist:
                cfg_raw = zf.read("config.yaml").decode("utf-8")
                if HAS_YAML and yaml is not None:
                    try:
                        config = yaml.safe_load(cfg_raw)
                    except Exception:
                        config = cfg_raw
                else:
                    config = cfg_raw

            # Create ONNX Runtime session directly from verified in-memory bytes
            providers_list = providers or ["CPUExecutionProvider"]
            session = ort.InferenceSession(onnx_bytes, providers=providers_list)

            return LoadedONNXBundle(
                session=session,
                manifest=manifest,
                schema=schema,
                config=config,
                bundle_path=path,
            )

    except zipfile.BadZipFile as e:
        raise BundleCorruptedError(f"Invalid or corrupted ZIP archive: {bundle_path}") from e

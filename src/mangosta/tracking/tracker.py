"""Local atomic experiment tracking, dataset fingerprinting, and run lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def compute_sha256(path: str | Path, chunk_size: int = 65536) -> str:
    """Compute an atomic SHA-256 fingerprint for a single file or a directory tree.

    For directories, files are walked and sorted by relative path, hashing each
    relative path and its file contents to create a deterministic Merkle-like root hash.
    """
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    hasher = hashlib.sha256()

    if target.is_file():
        with open(target, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    # Directory hashing
    for file_path in sorted(target.rglob("*")):
        if file_path.is_file():
            # Include relative path in hash to capture file renames/moves
            rel_path = file_path.relative_to(target).as_posix()
            hasher.update(rel_path.encode("utf-8"))
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)

    return hasher.hexdigest()


def _get_git_commit(cwd: Path | None = None) -> str | None:
    """Attempt to get current git commit hash non-invasively."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


class RunTracker:
    """Local, atomic experiment run logger with zero daemon or web dependencies."""

    def __init__(
        self,
        experiment: str = "default",
        run_name: str | None = None,
        root_dir: str | Path = "experiments",
    ) -> None:
        self.experiment = experiment.strip().replace(" ", "_")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name.strip().replace(" ", "_") if run_name else f"run_{timestamp}"

        self.root_dir = Path(root_dir).resolve()
        self.run_dir = self.root_dir / self.experiment / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.metadata_file = self.run_dir / "run.json"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.params: dict[str, Any] = {}
        self.artifacts: list[dict[str, Any]] = []
        self.dataset_hash: str | None = None
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.end_time: str | None = None
        self.status = "running"

        self._metrics_cache: dict[str, list[dict[str, Any]]] = {}
        self._step_counter = 0

        self._save_metadata()

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter or configuration value."""
        self.params[key] = value
        self._save_metadata()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple hyperparameters or configurations at once."""
        self.params.update(params)
        self._save_metadata()

    def set_dataset(self, dataset_path: str | Path) -> str:
        """Compute SHA-256 fingerprint of the dataset and link it to this run."""
        digest = compute_sha256(dataset_path)
        self.dataset_hash = digest
        self._save_metadata()
        return digest

    def log_metric(
        self,
        name: str,
        value: float | int,
        step: int | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Atomically record a metric data point to metrics.jsonl.

        Guarantees crash-consistency with immediate flush and fsync.
        """
        if step is None:
            step = self._step_counter
            self._step_counter += 1

        now = timestamp if timestamp is not None else time.time()
        record = {
            "name": name,
            "value": float(value),
            "step": int(step),
            "timestamp": now,
        }

        # Keep in-memory cache for reports
        if name not in self._metrics_cache:
            self._metrics_cache[name] = []
        self._metrics_cache[name].append(record)

        # Atomic append to disk with fsync
        line = json.dumps(record) + "\n"
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def log_artifact(self, source_path: str | Path, name: str | None = None) -> Path:
        """Atomically copy a file to the run's artifacts directory and record in run.json."""
        src = Path(source_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Artifact file not found: {source_path}")

        artifact_name = name or src.name
        dest = self.artifacts_dir / artifact_name

        import shutil

        shutil.copy2(src, dest)

        entry = {
            "name": artifact_name,
            "type": "file",
            "size_bytes": dest.stat().st_size,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        self.artifacts.append(entry)
        self._save_metadata()
        return dest

    def log_image(
        self,
        name: str,
        image: Any,
        step: int | None = None,
    ) -> Path:
        """Save and log a validation/prediction image (tensor, ndarray, or file path)."""
        base_name = name
        if isinstance(image, (str, Path)) and Path(image).is_file():
            safe_name = base_name if "." in base_name else f"{base_name}{Path(image).suffix}"
            dest = self.artifacts_dir / safe_name
            import shutil

            shutil.copy2(image, dest)
            actual_dest = dest
        else:
            safe_name = (
                base_name
                if base_name.endswith((".png", ".jpg", ".jpeg", ".ppm"))
                else f"{base_name}.png"
            )
            dest = self.artifacts_dir / safe_name
            actual_dest = self._save_image_array_or_tensor(image, dest)

        entry = {
            "name": actual_dest.name,
            "type": "image",
            "step": step,
            "size_bytes": actual_dest.stat().st_size if actual_dest.exists() else 0,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        self.artifacts.append(entry)
        self._save_metadata()
        return actual_dest

    def _save_image_array_or_tensor(self, image: Any, dest: Path) -> Path:
        """Save tensor or ndarray as an image file. Returns actual destination path."""
        import torch

        # Handle PyTorch Tensor (CPU or GPU)
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu().float()
            # If batch dimension (e.g. [1, C, H, W]), squeeze
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)

            # If CHW (channels first), convert to HWC for saving
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                t = t.permute(1, 2, 0)
            if t.ndim == 3 and t.shape[-1] == 1:
                t = t.squeeze(-1)

            # Normalize if in range [0, 1]
            if t.max() <= 1.0 and t.min() >= 0.0:
                t = t * 255.0
            arr = t.clamp(0, 255).byte().numpy()
        else:
            import numpy as np

            arr = np.asarray(image)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0 and arr.min() >= 0.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Write pure Python uncompressed PNG
        try:
            self._write_png_pure(arr, dest)
            return dest
        except Exception:
            # Fallback to PPM
            return self._write_ppm_fallback(arr, dest)

    def _write_png_pure(self, arr: Any, dest: Path) -> None:
        """Write uncompressed PNG without external libraries (zlib standard library only)."""
        import struct
        import zlib

        if arr.ndim == 2:
            height, width = arr.shape
            color_type = 0  # Grayscale
            raw_data = bytearray()
            for row in arr:
                raw_data.append(0)  # filter type None
                raw_data.extend(row.tobytes())
        elif arr.ndim == 3 and arr.shape[2] == 3:
            height, width, _ = arr.shape
            color_type = 2  # RGB
            raw_data = bytearray()
            for row in arr:
                raw_data.append(0)  # filter type None
                raw_data.extend(row.tobytes())
        elif arr.ndim == 3 and arr.shape[2] == 4:
            height, width, _ = arr.shape
            color_type = 6  # RGBA
            raw_data = bytearray()
            for row in arr:
                raw_data.append(0)  # filter type None
                raw_data.extend(row.tobytes())
        else:
            raise ValueError(f"Unsupported image shape for PNG: {arr.shape}")

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", zlib.crc32(tag + data))
            )

        png_bytes = bytearray(b"\x89PNG\r\n\x1a\n")
        # IHDR
        ihdr_data = struct.pack("!IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        png_bytes.extend(chunk(b"IHDR", ihdr_data))
        # IDAT
        compressed = zlib.compress(bytes(raw_data), level=6)
        png_bytes.extend(chunk(b"IDAT", compressed))
        # IEND
        png_bytes.extend(chunk(b"IEND", b""))

        dest.write_bytes(bytes(png_bytes))

    def _write_ppm_fallback(self, arr: Any, dest: Path) -> Path:
        """Fallback writer for PPM format."""
        import numpy as np

        ppm_dest = dest.with_suffix(".ppm")
        if arr.ndim == 2:
            h, w = arr.shape
            header = f"P5\n{w} {h}\n255\n".encode("ascii")
            ppm_dest.write_bytes(header + arr.tobytes())
        elif arr.ndim == 3 and arr.shape[2] == 3:
            h, w, _ = arr.shape
            header = f"P6\n{w} {h}\n255\n".encode("ascii")
            ppm_dest.write_bytes(header + arr.tobytes())
        else:
            gray = arr.mean(axis=-1).astype(np.uint8) if arr.ndim == 3 else arr.astype(np.uint8)
            h, w = gray.shape
            header = f"P5\n{w} {h}\n255\n".encode("ascii")
            ppm_dest.write_bytes(header + gray.tobytes())
        return ppm_dest

    def get_metrics(self) -> dict[str, list[dict[str, Any]]]:
        """Return all logged metrics structured by name."""
        if not self._metrics_cache and self.metrics_file.exists():
            # Read from file if not populated
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        name = rec["name"]
                        if name not in self._metrics_cache:
                            self._metrics_cache[name] = []
                        self._metrics_cache[name].append(rec)
        return self._metrics_cache

    def finish(self, status: str = "completed") -> None:
        """Finalize the run and mark status in run.json."""
        self.status = status
        self.end_time = datetime.now(timezone.utc).isoformat()
        self._save_metadata()

    def _save_metadata(self) -> None:
        """Atomically write run metadata to run.json via tmp file and os.replace."""
        import torch

        metadata = {
            "experiment": self.experiment,
            "run_name": self.run_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "params": self.params,
            "artifacts": self.artifacts,
            "dataset_hash": self.dataset_hash,
            "git_commit": _get_git_commit(self.run_dir),
            "env": {
                "python_version": platform.python_version(),
                "torch_version": getattr(torch, "__version__", "unknown"),
                "system": platform.system(),
                "node": platform.node(),
                "machine": platform.machine(),
            },
        }

        tmp_file = self.metadata_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self.metadata_file)

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        status = "failed" if exc_type is not None else "completed"
        self.finish(status=status)


@contextmanager
def quick_track(
    run_name: str | None = None,
    experiment: str = "default",
    root_dir: str | Path = "experiments",
) -> Iterator[RunTracker]:
    """Lightweight 3-line context manager for quick experiment tracking.

    Example:
        with quick_track(run_name="baseline_exp") as run:
            run.log_params({"lr": 0.001, "batch_size": 32})
            run.log_metric("loss", 0.42, step=0)
    """
    tracker = RunTracker(experiment=experiment, run_name=run_name, root_dir=root_dir)
    try:
        yield tracker
    except Exception:
        tracker.finish(status="failed")
        raise
    else:
        tracker.finish(status="completed")

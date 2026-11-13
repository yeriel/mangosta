"""Ephemeral, zero-daemon interactive visualizer and REST server for mangosta experiments.

Provides an "Industrial Dark / Craft DX" local dashboard featuring:
- Multi-run metric comparison with EMA smoothing and hover crosshairs.
- Validation data inspection with epoch scrubber (timelapse), A/B/GT side-by-side comparison, and zoomable lightbox.
- SafeTensors / ONNX bundle inspector and hyperparameter divergence diffing.
- Live Watch auto-refreshing polling mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
import http.server
import importlib.metadata
import json
import math
import re
import socketserver
import struct
import urllib.parse
import webbrowser
import zipfile
from pathlib import Path
from typing import Any

from mangosta.artifacts.bundle import read_bundle_manifest


def _get_mangosta_version() -> str:
    """Return installed mangosta version cleanly without circular imports."""
    try:
        return importlib.metadata.version("mangosta")
    except Exception:
        return "0.4.0"


def _compute_duration_seconds(
    start_time: str | None,
    end_time: str | None,
    status: str,
) -> float | None:
    """Compute elapsed seconds between ISO timestamps."""
    if not start_time:
        return None
    try:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        elif status == "running":
            end_dt = datetime.now(timezone.utc)
        else:
            return None
        return max(0.0, (end_dt - start_dt).total_seconds())
    except Exception:
        return None


def _is_within_allowed_roots(target: Path, roots: list[Path]) -> bool:
    """Return True if resolved target resides inside any allowed root directory."""
    try:
        resolved = target.resolve()
    except Exception:
        return False
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _read_json_safe(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _build_run_summary(run_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    artifacts_dir = run_dir / "artifacts"
    artifacts_count = (
        len([f for f in artifacts_dir.iterdir() if f.is_file()])
        if artifacts_dir.exists()
        else 0
    )
    status = meta.get("status", "unknown")
    start_time = meta.get("start_time")
    end_time = meta.get("end_time")
    return {
        "run_name": run_dir.name,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": _compute_duration_seconds(start_time, end_time, status),
        "dataset_hash": meta.get("dataset_hash"),
        "git_commit": meta.get("git_commit"),
        "artifacts_count": artifacts_count,
        "_sort_mtime": run_dir.stat().st_mtime if run_dir.exists() else 0.0,
    }


def _sort_runs_newest_first(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort runs newest-first by start_time, falling back to mtime and run_name."""
    def _key(r: dict[str, Any]) -> tuple[str, float, str]:
        return (str(r.get("start_time") or ""), float(r.get("_sort_mtime") or 0.0), str(r.get("run_name") or ""))

    ordered = sorted(runs, key=_key, reverse=True)
    for r in ordered:
        r.pop("_sort_mtime", None)
    return ordered


def _scan_experiments(root_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Scan experiments directory and return available experiments and runs.

    Resilient to whether root_dir is a root runs folder, an experiment folder, or a single run folder.
    """
    results: dict[str, list[dict[str, Any]]] = {}
    if not root_dir.exists():
        return results

    # Case 1: root_dir is itself a single run directory
    if (root_dir / "run.json").exists() or (root_dir / "metrics.jsonl").exists():
        meta = _read_json_safe(root_dir / "run.json")
        exp_name = meta.get("experiment", "default")
        runs = [_build_run_summary(root_dir, meta)]
        return {exp_name: runs}

    # Case 2: Inspect subdirectories
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue

        # Check if child is a run directory directly (root_dir is an experiment directory)
        if (child / "run.json").exists() or (child / "metrics.jsonl").exists():
            meta = _read_json_safe(child / "run.json")
            exp_name = root_dir.name
            if exp_name not in results:
                results[exp_name] = []
            results[exp_name].append(_build_run_summary(child, meta))
        else:
            # child is an experiment directory containing runs
            runs = []
            for run_dir in sorted(child.iterdir()):
                if not run_dir.is_dir() or run_dir.name.startswith("."):
                    continue
                if (run_dir / "run.json").exists() or (run_dir / "metrics.jsonl").exists():
                    meta = _read_json_safe(run_dir / "run.json")
                    runs.append(_build_run_summary(run_dir, meta))
            if runs:
                results[child.name] = _sort_runs_newest_first(runs)

    # Case 2b: Sort any direct-experiment runs
    for exp_key in list(results.keys()):
        results[exp_key] = _sort_runs_newest_first(results[exp_key])

    # Case 3: If nothing found yet, check common experiment subdirectories like experiments/ or runs/
    if not results:
        for sub_name in ("experiments", "runs", "outputs"):
            sub_path = root_dir / sub_name
            if sub_path.is_dir() and sub_path != root_dir:
                sub_res = _scan_experiments(sub_path)
                if sub_res:
                    results.update(sub_res)

    return results


def _load_run_details(root_dir: Path, experiment: str, run_name: str) -> dict[str, Any] | None:
    """Load complete run details including parameters, metrics history, and artifacts."""
    run_dir = root_dir / experiment / run_name
    if not run_dir.exists():
        for sub_name in ("experiments", "runs", "outputs"):
            candidate = root_dir / sub_name / experiment / run_name
            if candidate.exists():
                run_dir = candidate
                break
    if not run_dir.exists():
        if (root_dir / run_name).exists() and ((root_dir / run_name / "run.json").exists() or (root_dir / run_name / "metrics.jsonl").exists()):
            run_dir = root_dir / run_name
        elif root_dir.name == run_name and ((root_dir / "run.json").exists() or (root_dir / "metrics.jsonl").exists()):
            run_dir = root_dir
        else:
            return None

    meta_file = run_dir / "run.json"
    metrics_file = run_dir / "metrics.jsonl"
    artifacts_dir = run_dir / "artifacts"

    meta = _read_json_safe(meta_file)

    metrics: dict[str, list[dict[str, Any]]] = {}
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        name = rec.get("name", "metric")
                        if name not in metrics:
                            metrics[name] = []
                        metrics[name].append(rec)
                    except Exception:
                        continue

    # Artifacts list
    raw_artifacts: list[dict[str, Any]] = meta.get("artifacts", [])
    if not raw_artifacts and artifacts_dir.exists():
        raw_artifacts = []
        for art_file in sorted(artifacts_dir.iterdir()):
            if art_file.is_file():
                stem = art_file.stem
                step_val = None
                m = re.search(r"_step_(\d+)", stem)
                if m:
                    step_val = int(m.group(1))

                raw_artifacts.append({
                    "name": art_file.name,
                    "base_name": re.sub(r"_step_\d+", "", stem),
                    "type": "image" if art_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".ppm") else "file",
                    "step": step_val,
                    "size_bytes": art_file.stat().st_size,
                })

    # Enrich and standardize artifacts metadata
    enriched_artifacts: list[dict[str, Any]] = []
    for a in raw_artifacts:
        item = dict(a)
        name_str = item.get("name", "")
        stem = Path(name_str).stem

        # Extract step if missing
        if item.get("step") is None:
            m = re.search(r"_step_(\d+)", stem)
            if m:
                item["step"] = int(m.group(1))

        # Extract sample_id and tag/role
        if "sample_id" not in item:
            clean_id = re.sub(r"(_pred|_gt|_input)?(_step_\d+)?", "", stem)
            item["sample_id"] = clean_id or stem

        if "tag" not in item:
            lower = name_str.lower()
            if "_pred" in lower or "pred_" in lower:
                item["tag"] = "prediction"
            elif "_gt" in lower or "gt_" in lower or "_ground_truth" in lower:
                item["tag"] = "ground_truth"
            elif "_input" in lower or "input_" in lower:
                item["tag"] = "input"
            else:
                item["tag"] = "image" if item.get("type") == "image" else "file"

        enriched_artifacts.append(item)

    status = meta.get("status", "unknown")
    start_time = meta.get("start_time")
    end_time = meta.get("end_time")

    return {
        "experiment": experiment or meta.get("experiment", root_dir.name),
        "run_name": run_name or run_dir.name,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": _compute_duration_seconds(start_time, end_time, status),
        "params": meta.get("params", {}),
        "dataset_hash": meta.get("dataset_hash"),
        "git_commit": meta.get("git_commit"),
        "env": meta.get("env", {}),
        "metrics": metrics,
        "artifacts": enriched_artifacts,
    }


def _scan_bundles(root_dir: Path) -> list[dict[str, Any]]:
    """Scan workspace for .bundle and .onnx.bundle files up to 3 levels deep."""
    bundles: list[dict[str, Any]] = []
    if not root_dir.exists():
        return bundles

    candidates: list[Path] = []
    search_dirs = [root_dir]
    if root_dir.parent.exists() and root_dir.name in ("runs", "experiments", "outputs"):
        search_dirs.append(root_dir.parent)

    for base in search_dirs:
        try:
            candidates.extend(base.glob("*.bundle"))
            candidates.extend(base.glob("*/*.bundle"))
            candidates.extend(base.glob("*/*/*.bundle"))
        except Exception:
            pass

    seen_paths: set[str] = set()
    for p in sorted(candidates):
        resolved = str(p.resolve())
        if resolved in seen_paths or not p.is_file() or any(part.startswith(".") for part in p.parts):
            continue
        seen_paths.add(resolved)

        rel_path: str = p.name
        try:
            rel_path = str(p.relative_to(root_dir))
        except ValueError:
            if root_dir.parent.exists():
                try:
                    rel_path = str(p.relative_to(root_dir.parent))
                except ValueError:
                    rel_path = p.name

        size_bytes = p.stat().st_size
        family = "unknown"
        version = "unknown"
        format_type = "onnx" if p.name.endswith(".onnx.bundle") else "safetensors"
        is_checkpoint = False
        epoch: int | None = None
        created_at: str | None = None

        try:
            manifest = read_bundle_manifest(p)
            family = manifest.get("family", family)
            version = manifest.get("version", version)
            format_type = manifest.get("weights_format", format_type)
            is_checkpoint = bool(manifest.get("is_checkpoint", False))
            epoch = manifest.get("epoch")
            created_at = manifest.get("created_at")
        except Exception:
            pass

        bundles.append({
            "path": rel_path,
            "filename": p.name,
            "family": family,
            "version": version,
            "format": format_type,
            "is_checkpoint": is_checkpoint,
            "epoch": epoch,
            "created_at": created_at,
            "size_bytes": size_bytes,
        })
    return bundles


def _load_bundle_details(bundle_path: Path) -> dict[str, Any] | None:
    """Load detailed bundle inspection payload including layer shapes and SafeTensors header."""
    if not bundle_path.is_file():
        return None
    try:
        total_bytes = bundle_path.stat().st_size
        with zipfile.ZipFile(bundle_path, "r") as zf:
            namelist = zf.namelist()
            manifest_raw = zf.read("manifest.json").decode("utf-8")
            manifest = json.loads(manifest_raw)

            tensors: list[dict[str, Any]] = []
            total_params = 0
            if "model.safetensors" in namelist:
                with zf.open("model.safetensors") as sf:
                    header_len_bytes = sf.read(8)
                    if len(header_len_bytes) == 8:
                        header_len = struct.unpack("<Q", header_len_bytes)[0]
                        header_json_bytes = sf.read(header_len)
                        raw_tensors = json.loads(header_json_bytes.decode("utf-8"))
                        for k, v in raw_tensors.items():
                            if k != "__metadata__" and isinstance(v, dict):
                                shape = v.get("shape", [])
                                param_count = math.prod(shape) if shape else 1
                                offsets = v.get("data_offsets", [])
                                tensor_bytes = (
                                    max(0, int(offsets[1]) - int(offsets[0]))
                                    if isinstance(offsets, list) and len(offsets) == 2
                                    else 0
                                )
                                total_params += param_count
                                tensors.append({
                                    "name": k,
                                    "shape": shape,
                                    "dtype": v.get("dtype", "unknown"),
                                    "param_count": param_count,
                                    "size_bytes": tensor_bytes,
                                })

            config_content = None
            if "config.yaml" in namelist:
                config_content = zf.read("config.yaml").decode("utf-8")

            return {
                "file": bundle_path.name,
                "path": str(bundle_path),
                "size_bytes": total_bytes,
                "family": manifest.get("family", "unknown"),
                "version": manifest.get("version", "unknown"),
                "format": manifest.get("weights_format", "unknown"),
                "weights_sha256": manifest.get("weights_sha256") or manifest.get("model_sha256"),
                "is_checkpoint": manifest.get("is_checkpoint", False),
                "epoch": manifest.get("epoch"),
                "step": manifest.get("step"),
                "created_at": manifest.get("created_at"),
                "init_args": manifest.get("init_args", {}),
                "metrics": manifest.get("metrics", {}),
                "schema": manifest.get("schema"),
                "env": manifest.get("env", {}),
                "config": config_content,
                "total_params": total_params,
                "tensors": tensors,
                "files": [f for f in namelist if not f.endswith("/")],
            }
    except Exception as e:
        return {"error": str(e), "file": bundle_path.name}


def create_viewer_handler(root_dir: Path) -> type[http.server.BaseHTTPRequestHandler]:
    """Factory creating an HTTP request handler for the interactive visualizer."""
    resolved_root = root_dir.resolve()
    allowed_roots = [resolved_root]
    if resolved_root.parent.exists() and resolved_root.name in ("runs", "experiments", "outputs"):
        allowed_roots.append(resolved_root.parent.resolve())

    class MangostaViewerHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Suppress normal HTTP logging for clean terminal
            pass

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)

            # API: Server & version metadata
            if path == "/api/info":
                self._send_json({"version": _get_mangosta_version(), "root": str(resolved_root)})
                return

            # API: List all experiments & runs
            if path == "/api/experiments":
                data = _scan_experiments(resolved_root)
                self._send_json(data)
                return

            # API: Single run details
            if path == "/api/run":
                exp = query.get("exp", [""])[0]
                run = query.get("run", [""])[0]
                run_details = _load_run_details(resolved_root, exp, run)
                if run_details is None:
                    self.send_error(404, "Run not found")
                else:
                    self._send_json(run_details)
                return

            # API: Multi-run details batch
            if path == "/api/runs":
                runs_query = query.get("runs", [""])[0]
                runs_list: list[dict[str, Any]] = []
                for item in runs_query.split(","):
                    if not item.strip():
                        continue
                    if ":" in item:
                        exp_name, r_name = item.split(":", 1)
                    elif "/" in item:
                        exp_name, r_name = item.split("/", 1)
                    else:
                        exp_name, r_name = "", item
                    detail = _load_run_details(resolved_root, exp_name.strip(), r_name.strip())
                    if detail:
                        runs_list.append(detail)
                self._send_json({"runs": runs_list})
                return

            # API: List bundles
            if path == "/api/bundles":
                bundles = _scan_bundles(resolved_root)
                self._send_json({"bundles": bundles})
                return

            # API: Bundle details
            if path == "/api/bundle":
                bundle_rel = query.get("path", [""])[0]
                candidates = [
                    (resolved_root / bundle_rel),
                    (resolved_root.parent / bundle_rel),
                    Path(bundle_rel),
                ]
                target_bundle: Path | None = None
                for c in candidates:
                    if c.is_file():
                        target_bundle = c.resolve()
                        break
                if target_bundle is None:
                    fname = Path(bundle_rel).name
                    matches = [
                        m for m in list(resolved_root.glob(f"**/{fname}")) + list(resolved_root.parent.glob(f"**/{fname}"))
                        if m.is_file()
                    ]
                    if matches:
                        target_bundle = matches[0].resolve()

                if target_bundle is None or not target_bundle.is_file():
                    self.send_error(404, "Bundle not found")
                    return

                if not _is_within_allowed_roots(target_bundle, allowed_roots):
                    self.send_error(403, "Access denied")
                    return

                details = _load_bundle_details(target_bundle)
                if details is None:
                    self.send_error(404, "Bundle not found")
                else:
                    self._send_json(details)
                return

            # API: Artifact file delivery
            if path == "/api/artifact":
                exp = query.get("exp", [""])[0]
                run = query.get("run", [""])[0]
                filename = query.get("file", [""])[0]

                artifact_path = resolved_root / exp / run / "artifacts" / filename
                if not artifact_path.is_file():
                    alt1 = resolved_root / "experiments" / exp / run / "artifacts" / filename
                    alt2 = resolved_root / "runs" / exp / run / "artifacts" / filename
                    alt3 = resolved_root / run / "artifacts" / filename
                    alt4 = resolved_root / "artifacts" / filename
                    if alt1.is_file():
                        artifact_path = alt1
                    elif alt2.is_file():
                        artifact_path = alt2
                    elif alt3.is_file():
                        artifact_path = alt3
                    elif alt4.is_file():
                        artifact_path = alt4
                    else:
                        self.send_error(404, "Artifact file not found")
                        return

                if not _is_within_allowed_roots(artifact_path, allowed_roots):
                    self.send_error(403, "Access denied")
                    return

                mime = "application/octet-stream"
                ext = artifact_path.suffix.lower()
                if ext in (".png",):
                    mime = "image/png"
                elif ext in (".jpg", ".jpeg"):
                    mime = "image/jpeg"
                elif ext in (".json",):
                    mime = "application/json"
                elif ext in (".txt", ".log", ".csv"):
                    mime = "text/plain"

                data_bytes = artifact_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data_bytes)))
                self.end_headers()
                self.wfile.write(data_bytes)
                return

            # Default: Serve SPA HTML dashboard
            html_content = _get_dashboard_html()
            html_bytes = html_content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        def _send_json(self, data: Any) -> None:
            raw = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return MangostaViewerHandler


def start_viewer(
    target_path: str | Path = "experiments",
    port: int = 8501,
    open_browser: bool = True,
) -> None:
    """Launch the interactive zero-daemon visualizer server."""
    root = Path(target_path).resolve()
    handler_cls = create_viewer_handler(root)

    print("\n" + "=" * 60)
    print(f"MANGOSTA LOCAL EXPERIMENT VIEWER (v{_get_mangosta_version()})")
    print("=" * 60)
    print(f"Tracking directory: {root}")
    print(f"Dashboard URL:      http://localhost:{port}")
    print("Press Ctrl+C to terminate the viewer.")
    print("=" * 60 + "\n")

    if open_browser:
        webbrowser.open(f"http://localhost:{port}")

    # Allow port reuse immediately
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler_cls) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down mangosta viewer cleanly.")


def _get_dashboard_html() -> str:
    """Return the complete, self-contained single-page application HTML/JS/CSS."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>mangosta view</title>
  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-alt: #f1f5f9;
      --border: #e2e8f0;
      --border-strong: #cbd5e1;
      --text: #0f172a;
      --text-muted: #475569;
      --text-dim: #64748b;
      --accent: #4f46e5;
      --accent-soft: rgba(79, 70, 229, 0.08);
      --series-1: #4f46e5;
      --series-2: #0284c7;
      --ok: #059669;
      --bad: #e11d48;
      --warn: #d97706;
      --font-sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --font-mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-sans);
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
      font-variant-numeric: tabular-nums;
      -webkit-font-smoothing: antialiased;
    }

    /* Subtle Light Scrollbars */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

    /* Top Navigation Bar */
    header {
      height: 48px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      flex-shrink: 0;
      z-index: 10;
      gap: 12px;
    }
    .header-left { display: flex; align-items: center; gap: 10px; }
    .sidebar-toggle {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 5px;
      padding: 4px 8px;
      font-size: 0.82rem;
      cursor: pointer;
    }
    .sidebar-toggle:hover { background: var(--surface-alt); color: var(--text); }
    .brand {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      font-size: 0.96rem;
      letter-spacing: -0.02em;
    }
    .brand-mark { width: 8px; height: 8px; border-radius: 2px; background: var(--accent); }
    .brand-ver {
      font-size: 0.7rem;
      font-family: var(--font-mono);
      background: var(--surface-alt);
      color: var(--text-muted);
      border: 1px solid var(--border);
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 600;
    }

    .nav-tabs { display: flex; gap: 4px; }
    .tab-btn {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.12s ease;
    }
    .tab-btn:hover { color: var(--text); background: var(--surface-alt); }
    .tab-btn.active {
      color: var(--accent);
      background: var(--accent-soft);
      border-color: rgba(79, 70, 229, 0.25);
      font-weight: 600;
    }
    .tab-count {
      font-family: var(--font-mono);
      font-size: 0.7rem;
      color: var(--text-dim);
    }
    .tab-btn.active .tab-count { color: var(--accent); }

    .top-actions { display: flex; align-items: center; gap: 8px; }
    .live-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.72rem;
      font-family: var(--font-mono);
      font-weight: 600;
      padding: 4px 9px;
      border-radius: 5px;
      background: rgba(2, 132, 199, 0.08);
      color: var(--series-2);
      border: 1px solid rgba(2, 132, 199, 0.28);
      cursor: pointer;
      user-select: none;
    }
    .live-pulse {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--series-2);
      animation: pulse 2s infinite;
    }
    .live-badge.off {
      background: var(--surface-alt);
      color: var(--text-dim);
      border-color: var(--border);
    }
    .live-badge.off .live-pulse { background: var(--text-dim); animation: none; }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.35; }
    }

    /* Main Layout */
    .app-body {
      display: flex;
      flex: 1;
      height: calc(100vh - 48px);
      overflow: hidden;
    }

    /* Sidebar */
    #sidebar {
      width: 290px;
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }
    #sidebar.collapsed { display: none; }
    .sidebar-search {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .sidebar-search input {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 6px 10px;
      color: var(--text);
      font-size: 0.82rem;
      outline: none;
    }
    .sidebar-search input:focus { border-color: var(--accent); }
    .compare-bar {
      display: none;
      align-items: center;
      justify-content: space-between;
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: var(--accent);
      background: var(--accent-soft);
      padding: 4px 8px;
      border-radius: 4px;
    }
    .compare-bar.visible { display: flex; }
    .compare-clear {
      background: transparent;
      border: none;
      color: var(--accent);
      font-weight: 700;
      cursor: pointer;
      font-size: 0.72rem;
      padding: 0 4px;
    }
    #runs-list { flex: 1; overflow-y: auto; padding: 8px; }
    .exp-group { margin-bottom: 12px; }
    .exp-title {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dim);
      font-weight: 700;
      padding: 4px 8px;
      display: flex;
      justify-content: space-between;
    }
    .run-item {
      padding: 7px 9px;
      border-radius: 5px;
      cursor: pointer;
      margin-bottom: 3px;
      background: var(--surface);
      border: 1px solid transparent;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: background 0.1s ease, border-color 0.1s ease;
    }
    .run-item:hover { background: var(--surface-alt); border-color: var(--border); }
    .run-item.active {
      background: var(--accent-soft);
      border-color: var(--accent);
    }
    .run-checkbox { accent-color: var(--accent); cursor: pointer; flex-shrink: 0; }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .status-dot.completed { background: var(--ok); }
    .status-dot.failed { background: var(--bad); }
    .status-dot.interrupted { background: var(--warn); }
    .status-dot.running { background: var(--series-2); }
    .status-dot.unknown { background: var(--text-dim); }
    .run-info { flex: 1; min-width: 0; }
    .run-name {
      font-size: 0.8rem;
      font-weight: 600;
      font-family: var(--font-mono);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .run-meta {
      font-size: 0.69rem;
      font-family: var(--font-mono);
      color: var(--text-dim);
      margin-top: 2px;
      display: flex;
      gap: 8px;
    }

    /* Main Content */
    #main-content { flex: 1; overflow-y: auto; padding: 20px 24px; background: var(--bg); }
    .view-panel { display: none; }
    .view-panel.active { display: block; }

    /* KPIs & Cards */
    .grid-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 14px;
    }
    .card-label {
      font-size: 0.69rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-dim);
      font-weight: 600;
    }
    .card-val {
      font-size: 1.08rem;
      font-weight: 700;
      margin-top: 4px;
      font-family: var(--font-mono);
      color: var(--text);
    }
    .card-sub {
      font-size: 0.71rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
      margin-top: 2px;
    }
    .provenance-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 14px;
      font-family: var(--font-mono);
      font-size: 0.74rem;
      color: var(--text-muted);
      padding: 8px 12px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      margin-bottom: 16px;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      border: 1px solid var(--border);
      background: var(--surface-alt);
    }
    .status-pill.completed { color: var(--ok); border-color: rgba(5, 150, 105, 0.3); background: rgba(5, 150, 105, 0.08); }
    .status-pill.failed { color: var(--bad); border-color: rgba(225, 29, 72, 0.3); background: rgba(225, 29, 72, 0.08); }
    .status-pill.running { color: var(--series-2); border-color: rgba(2, 132, 199, 0.3); background: rgba(2, 132, 199, 0.08); }
    .status-pill.interrupted { color: var(--warn); border-color: rgba(217, 119, 6, 0.3); background: rgba(217, 119, 6, 0.08); }

    /* Section Headers & Controls */
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
      padding: 10px 14px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
    }
    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    .section-title { font-size: 0.92rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
    .control-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }

    select, input[type="text"], input[type="range"] {
      background: var(--surface);
      border: 1px solid var(--border-strong);
      color: var(--text);
      border-radius: 5px;
      font-size: 0.8rem;
      padding: 5px 9px;
      outline: none;
    }
    select:focus, input[type="text"]:focus { border-color: var(--accent); }
    button.btn {
      background: var(--surface);
      border: 1px solid var(--border-strong);
      color: var(--text);
      border-radius: 5px;
      font-size: 0.78rem;
      font-weight: 600;
      padding: 5px 10px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: background 0.12s ease;
    }
    button.btn:hover { background: var(--surface-alt); }
    button.btn.active-toggle {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
    }
    button.btn-primary { background: var(--accent); color: #ffffff; border-color: var(--accent); }
    button.btn-primary:hover { filter: brightness(1.08); }

    /* Charts Grid */
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }
    .chart-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px 16px;
      position: relative;
      min-width: 0;
      overflow: hidden;
    }
    .chart-box-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .chart-title { font-size: 0.86rem; font-weight: 600; font-family: var(--font-mono); }
    svg.chart-svg {
      width: 100%;
      height: 240px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 4px;
      overflow: hidden;
      display: block;
    }
    .chart-tooltip {
      position: absolute;
      display: none;
      background: var(--surface);
      border: 1px solid var(--border-strong);
      border-radius: 5px;
      padding: 7px 10px;
      font-family: var(--font-mono);
      font-size: 0.72rem;
      pointer-events: none;
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
      z-index: 100;
      min-width: 150px;
    }
    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 10px;
      font-size: 0.74rem;
      font-family: var(--font-mono);
    }
    .legend-item { display: inline-flex; align-items: center; gap: 5px; }
    .legend-dot { width: 8px; height: 8px; border-radius: 2px; }

    /* Validation & Bundles */
    .val-container { display: flex; gap: 16px; min-height: 500px; }
    .val-sidebar {
      width: 270px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      flex-shrink: 0;
      max-height: calc(100vh - 110px);
      overflow-y: auto;
    }
    .val-sample-item {
      padding: 8px 10px;
      border-radius: 5px;
      cursor: pointer;
      margin-bottom: 4px;
      background: var(--surface);
      border: 1px solid var(--border);
      transition: background 0.12s ease, border-color 0.12s ease;
    }
    .val-sample-item:hover { background: var(--surface-alt); }
    .val-sample-item.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .val-main { flex: 1; display: flex; flex-direction: column; gap: 14px; min-width: 0; }
    .scrubber-bar {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .scrubber-controls { display: flex; align-items: center; gap: 8px; flex: 1; }
    .scrubber-slider { flex: 1; accent-color: var(--accent); cursor: pointer; }
    .val-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }
    .val-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .val-card-header {
      padding: 8px 12px;
      font-size: 0.74rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--surface-alt);
    }
    .val-card-img-wrap {
      width: 100%;
      height: 300px;
      background: #0f172a;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: zoom-in;
      position: relative;
    }
    .val-card-img-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; }
    .zoom-hint {
      position: absolute;
      bottom: 8px;
      right: 8px;
      background: rgba(15, 23, 42, 0.75);
      color: #fff;
      font-size: 0.68rem;
      padding: 2px 6px;
      border-radius: 4px;
      pointer-events: none;
    }

    /* Lightbox */
    .lightbox-modal {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.92);
      z-index: 999;
      display: none;
      flex-direction: column;
      color: #f8fafc;
    }
    .lightbox-modal.active { display: flex; }
    .lightbox-header {
      height: 48px;
      padding: 0 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #334155;
      background: #0f172a;
    }
    .lightbox-body {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      cursor: grab;
      user-select: none;
    }
    .lightbox-body.dragging { cursor: grabbing; }
    .lightbox-img {
      max-width: 90vw;
      max-height: 85vh;
      object-fit: contain;
      transform-origin: center center;
      pointer-events: none;
    }

    /* Tables & Code */
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
      font-family: var(--font-mono);
    }
    table.data-table th, table.data-table td {
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    table.data-table th {
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: var(--surface-alt);
    }
    table.data-table tr.divergent {
      background: rgba(79, 70, 229, 0.07);
      color: var(--accent);
      font-weight: 600;
    }
    pre.code-block {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
      font-family: var(--font-mono);
      font-size: 0.78rem;
      overflow-x: auto;
      max-height: 520px;
      color: var(--text);
      line-height: 1.5;
    }
    .empty-state {
      padding: 40px 20px;
      text-align: center;
      color: var(--text-dim);
      font-size: 0.85rem;
      background: var(--surface);
      border: 1px dashed var(--border-strong);
      border-radius: 6px;
    }
    @media (max-width: 900px) {
      #sidebar { position: fixed; top: 48px; bottom: 0; left: 0; z-index: 30; box-shadow: 4px 0 18px rgba(15, 23, 42, 0.12); }
      .charts-grid { grid-template-columns: 1fr; }
      .val-container { flex-direction: column; }
      .val-sidebar { width: 100%; max-height: 220px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <button class="sidebar-toggle" onclick="toggleSidebar()" title="Toggle sidebar">☰</button>
      <div class="brand">
        <span class="brand-mark"></span>
        <span>mangosta view</span>
        <span class="brand-ver" id="app-version">v__MANGOSTA_VERSION__</span>
      </div>
    </div>

    <div class="nav-tabs">
      <button class="tab-btn active" data-tab="metrics" onclick="switchTab('metrics')">Metrics &amp; Runs</button>
      <button class="tab-btn" data-tab="val" onclick="switchTab('val')">Validation <span class="tab-count" id="val-count">(0)</span></button>
      <button class="tab-btn" data-tab="bundles" onclick="switchTab('bundles')">Bundles <span class="tab-count" id="bundle-count">(0)</span></button>
      <button class="tab-btn" data-tab="meta" onclick="switchTab('meta')">Metadata</button>
    </div>

    <div class="top-actions">
      <button class="btn" onclick="refreshData(true)" title="Refresh now (r)">Refresh</button>
      <div id="live-toggle" class="live-badge" onclick="toggleLiveWatch()" title="Auto-refresh when runs are active">
        <span class="live-pulse"></span>
        <span id="live-text">LIVE</span>
      </div>
    </div>
  </header>

  <div class="app-body">
    <!-- Left Sidebar (v0.3.1 structure) -->
    <div id="sidebar">
      <div class="sidebar-search">
        <input type="text" id="run-search" placeholder="Filter runs..." oninput="renderSidebar()" />
        <div id="compare-bar" class="compare-bar">
          <span id="compare-count">Compare runs</span>
          <div id="compare-actions" style="display:flex; gap:4px; align-items:center;">
            <button class="compare-clear" onclick="clearComparison()">Clear</button>
          </div>
        </div>
      </div>
      <div id="runs-list">Loading runs...</div>
    </div>

    <!-- Main Content Panels -->
    <div id="main-content">
      <!-- Tab 1: Metrics & Runs -->
      <div id="tab-metrics" class="view-panel active">
        <div id="metrics-top-meta" class="grid-metrics"></div>
        <div id="metrics-provenance"></div>
        <div id="metrics-compare-summary"></div>

        <div class="toolbar">
          <div class="control-row">
            <span style="font-size:0.78rem; font-weight:600; color:var(--text-muted);">View:</span>
            <select id="metric-select" onchange="renderCharts()">
              <option value="__ALL_PAIRED__">All Metrics (Train vs Val)</option>
            </select>
            <button class="btn" id="scale-linear-btn" onclick="setScale('linear')">Linear</button>
            <button class="btn" id="scale-log-btn" onclick="setScale('log')">Log</button>
            <button class="btn" id="toolbar-compare-btn" onclick="toggleQuickCompare()" style="display:none; margin-left:6px;"></button>
          </div>
          <div class="control-row">
            <span style="font-size:0.78rem; color:var(--text-muted);">EMA:</span>
            <input type="range" id="smoothing-slider" min="0" max="0.99" step="0.01" value="0.35" oninput="updateSmoothing(this.value)" style="width:120px; padding:0;" />
            <span id="smoothing-val" style="font-family:var(--font-mono); font-size:0.78rem; width:34px; color:var(--accent);">0.35</span>
          </div>
        </div>

        <div id="charts-grid" class="charts-grid"></div>

        <div class="section-header" style="margin-top:8px;">
          <div class="section-title">MangostaConfig Parameters</div>
          <label style="font-size:0.78rem; color:var(--text-muted); display:inline-flex; align-items:center; gap:6px; cursor:pointer;">
            <input type="checkbox" id="diff-only-toggle" onchange="renderParamDiffTable()" style="accent-color:var(--accent);" />
            Differences only
          </label>
        </div>
        <div class="card" style="padding:0; overflow-x:auto;">
          <table class="data-table" id="param-diff-table">
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

      <!-- Tab 2: Real Data Validation -->
      <div id="tab-val" class="view-panel">
        <div class="val-container">
          <div class="val-sidebar">
            <div class="section-title" style="font-size:0.82rem; margin-bottom:8px;">Validation Samples</div>
            <div id="val-samples-list">No image samples logged.</div>
          </div>

          <div class="val-main">
            <div class="scrubber-bar">
              <div class="scrubber-controls">
                <button class="btn" onclick="prevStep()">Prev</button>
                <button class="btn btn-primary" id="play-btn" onclick="togglePlayTimelapse()">Play Timelapse</button>
                <button class="btn" onclick="nextStep()">Next</button>
                <input type="range" id="epoch-scrubber" class="scrubber-slider" min="0" max="0" value="0" oninput="onScrubStep(this.value)" />
              </div>
              <div style="font-family:var(--font-mono); font-weight:700; font-size:0.86rem; color:var(--accent);" id="scrubber-step-indicator">
                Step: 0
              </div>
            </div>

            <div class="val-grid" id="val-grid-cards">
              <div class="empty-state">Select a validation sample to inspect temporal predictions.</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Tab 3: Bundle Inspector -->
      <div id="tab-bundles" class="view-panel">
        <div class="val-container">
          <div class="val-sidebar" style="width:290px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
              <div class="section-title" style="font-size:0.82rem; margin:0;">Discovered Bundles</div>
              <div id="bundles-compare-bar" style="display:none; align-items:center; gap:6px;">
                <button class="btn" id="bundles-compare-btn" style="padding:2px 8px; font-size:0.72rem;" onclick="compareBundles()">Compare (0)</button>
                <button class="btn" style="padding:2px 6px; font-size:0.72rem;" onclick="clearBundleSelection()">Clear</button>
              </div>
            </div>
            <div id="bundles-list">Scanning workspace...</div>
          </div>

          <div class="val-main" id="bundle-inspector-main">
            <div class="empty-state">Select a <code>.bundle</code> or <code>.onnx.bundle</code> file from the list to inspect contracts and SafeTensors layers.</div>
          </div>
        </div>
      </div>

      <!-- Tab 4: Metadata -->
      <div id="tab-meta" class="view-panel">
        <div class="section-header">
          <div class="section-title">Raw Run Metadata &amp; Environment</div>
          <button class="btn" id="copy-meta-btn" onclick="copyTextFromEl('meta-json-pre', 'copy-meta-btn')">Copy JSON</button>
        </div>
        <pre class="code-block" id="meta-json-pre">Select a run to view metadata.</pre>
      </div>
    </div>
  </div>

  <!-- Lightbox Modal -->
  <div id="lightbox" class="lightbox-modal">
    <div class="lightbox-header">
      <div id="lightbox-title" style="font-family:var(--font-mono); font-weight:600; font-size:0.85rem;">Sample Inspection</div>
      <div class="control-row">
        <button class="btn" onclick="zoomLightbox(0.25)">Zoom +</button>
        <button class="btn" onclick="zoomLightbox(-0.25)">Zoom -</button>
        <button class="btn" onclick="resetLightboxZoom()">1:1 Reset</button>
        <button class="btn btn-primary" onclick="closeLightbox()">Close (Esc)</button>
      </div>
    </div>
    <div class="lightbox-body" id="lightbox-body">
      <img id="lightbox-img" class="lightbox-img" src="" alt="high-res" />
    </div>
  </div>

  <script>
    // Global State
    let experimentsData = {};
    let orderedRunKeys = [];
    let activeExp = null;
    let activeRun = null;
    let selectedRuns = new Set();
    let loadedRunsData = {};
    let activeTab = "metrics";
    let isLiveWatch = true;
    let liveInterval = null;
    let smoothingWeight = 0.35;
    let scaleMode = "linear";
    let lastExperimentsSignature = "";
    let lastBundlesSignature = "";
    let activeBundlePath = null;
    let currentBundleData = null;
    let selectedBundles = new Set();
    let loadedBundlesCache = {};
    let discoveredBundles = [];

    // Validation Scrubber State
    let currentSampleId = null;
    let currentSampleTimeline = [];
    let currentTimelineStepIdx = 0;
    let playTimer = null;

    // Lightbox State
    let lightboxScale = 1.0;
    let lightboxTx = 0;
    let lightboxTy = 0;
    let isDraggingLightbox = false;
    let dragStartX = 0;
    let dragStartY = 0;

    const PALETTE = ["#4f46e5", "#0284c7", "#059669", "#d97706", "#e11d48", "#7c3aed", "#0d9488", "#475569"];

    function escapeHtml(val) {
      return String(val ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function formatDuration(seconds) {
      if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "N/A";
      const total = Math.round(seconds);
      if (total < 60) return `${total}s`;
      const mins = Math.floor(total / 60);
      const secs = total % 60;
      if (mins < 60) return `${String(mins).padStart(2, "0")}m ${String(secs).padStart(2, "0")}s`;
      const hrs = Math.floor(mins / 60);
      return `${hrs}h ${String(mins % 60).padStart(2, "0")}m`;
    }

    function formatBytes(bytes) {
      if (!bytes && bytes !== 0) return "N/A";
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }

    function toggleSidebar() {
      document.getElementById("sidebar").classList.toggle("collapsed");
    }

    async function init() {
      updateScaleButtons();
      await fetchExperiments();
      await fetchBundles();
      setupLiveWatch();
      setupKeyboard();
      setupLightboxInteractions();
    }

    function switchTab(tabId) {
      activeTab = tabId;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tabId));
      document.querySelectorAll(".view-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${tabId}`));
      renderActiveTab();
    }

    async function fetchExperiments(forceRender = false) {
      try {
        const res = await fetch("/api/experiments");
        const data = await res.json();
        const sig = JSON.stringify(data);
        if (!forceRender && sig === lastExperimentsSignature) return false;
        lastExperimentsSignature = sig;
        experimentsData = data;
        renderSidebar();
        return true;
      } catch (e) {
        console.error("Error fetching experiments:", e);
        return false;
      }
    }

    function renderSidebar() {
      const container = document.getElementById("runs-list");
      const searchVal = document.getElementById("run-search").value.toLowerCase();
      container.innerHTML = "";
      orderedRunKeys = [];

      const expNames = Object.keys(experimentsData);
      const totalRunCount = expNames.reduce((acc, e) => acc + (experimentsData[e] ? experimentsData[e].length : 0), 0);
      const compareBar = document.getElementById("compare-bar");
      const compareActions = document.getElementById("compare-actions");
      if (selectedRuns.size > 1) {
        compareBar.classList.add("visible");
        document.getElementById("compare-count").textContent = `Comparing ${selectedRuns.size} runs`;
        compareActions.innerHTML = `<button class="compare-clear" onclick="clearComparison()">Clear</button>`;
      } else if (totalRunCount >= 2) {
        compareBar.classList.add("visible");
        document.getElementById("compare-count").textContent = "Compare runs:";
        compareActions.innerHTML = `<button class="compare-clear" onclick="quickCompareRuns(2)">Latest 2</button><button class="compare-clear" onclick="quickCompareRuns(0)">All</button>`;
      } else {
        compareBar.classList.remove("visible");
      }

      if (expNames.length === 0) {
        container.innerHTML = "<div class='empty-state' style='padding:20px;'>No experiment runs found yet.</div>";
        return;
      }

      let firstExpRun = null;

      expNames.forEach(expName => {
        const runs = experimentsData[expName].filter(r =>
          r.run_name.toLowerCase().includes(searchVal) || expName.toLowerCase().includes(searchVal)
        );
        if (runs.length === 0) return;

        const group = document.createElement("div");
        group.className = "exp-group";
        const expCompareBtn = runs.length >= 2
          ? `<button class="compare-clear" style="font-size:0.66rem; text-transform:none; letter-spacing:0;" onclick="compareExperiment(decodeURIComponent('${encodeURIComponent(expName)}'))">Compare ${runs.length}</button>`
          : "";
        group.innerHTML = `<div class="exp-title"><span>${escapeHtml(expName)}</span><span style="display:flex; align-items:center; gap:6px;">${expCompareBtn}<span>${runs.length}</span></span></div>`;

        runs.forEach(r => {
          const key = `${expName}:${r.run_name}`;
          orderedRunKeys.push(key);
          if (!firstExpRun) firstExpRun = { exp: expName, run: r.run_name };

          const item = document.createElement("div");
          item.className = `run-item ${activeExp === expName && activeRun === r.run_name ? "active" : ""}`;
          const isChecked = selectedRuns.has(key);
          const durText = r.duration_seconds !== null && r.duration_seconds !== undefined ? formatDuration(r.duration_seconds) : r.status;
          const commitText = r.git_commit ? r.git_commit.slice(0, 7) : "";

          item.innerHTML = `
            <input type="checkbox" class="run-checkbox" ${isChecked ? "checked" : ""} />
            <span class="status-dot ${escapeHtml(r.status)}"></span>
            <div class="run-info">
              <div class="run-name">${escapeHtml(r.run_name)}</div>
              <div class="run-meta">
                <span>${escapeHtml(durText)}</span>
                <span>${r.artifacts_count} art</span>
                ${commitText ? `<span>${escapeHtml(commitText)}</span>` : ""}
              </div>
            </div>
          `;

          const cb = item.querySelector("input.run-checkbox");
          cb.onclick = (e) => onToggleRunCheck(e, expName, r.run_name);
          item.onclick = (e) => {
            if (e.target === cb) return;
            selectSingleRun(expName, r.run_name);
          };

          group.appendChild(item);
        });

        container.appendChild(group);
      });

      if (!activeRun && firstExpRun) {
        selectSingleRun(firstExpRun.exp, firstExpRun.run);
      }
    }

    function clearComparison() {
      if (activeExp && activeRun) {
        selectedRuns = new Set([`${activeExp}:${activeRun}`]);
      } else {
        selectedRuns.clear();
      }
      document.getElementById("diff-only-toggle").checked = false;
      renderSidebar();
      renderActiveTab();
    }

    async function quickCompareRuns(limit = 2) {
      if (orderedRunKeys.length < 2) return;
      const targets = limit > 0 ? orderedRunKeys.slice(0, limit) : orderedRunKeys.slice(0, 8);
      selectedRuns = new Set(targets);
      if (!activeExp || !activeRun) {
        const first = targets[0];
        const idx = first.indexOf(":");
        activeExp = first.slice(0, idx);
        activeRun = first.slice(idx + 1);
      }
      document.getElementById("diff-only-toggle").checked = true;
      await syncSelectedRunsData();
      updateTabCounts();
      renderSidebar();
      renderActiveTab();
    }

    async function compareExperiment(expName) {
      const runs = experimentsData[expName] || [];
      if (runs.length < 2) return;
      const targets = runs.slice(0, 8).map(r => `${expName}:${r.run_name}`);
      selectedRuns = new Set(targets);
      activeExp = expName;
      activeRun = runs[0].run_name;
      document.getElementById("diff-only-toggle").checked = true;
      await syncSelectedRunsData();
      updateTabCounts();
      renderSidebar();
      renderActiveTab();
    }

    function toggleQuickCompare() {
      if (selectedRuns.size > 1) {
        clearComparison();
      } else {
        quickCompareRuns(2);
      }
    }

    async function selectSingleRun(exp, run) {
      activeExp = exp;
      activeRun = run;
      const key = `${exp}:${run}`;
      if (selectedRuns.size <= 1) {
        selectedRuns.clear();
        selectedRuns.add(key);
      }
      await loadRunData(exp, run, true);
      updateTabCounts();
      renderSidebar();
      renderActiveTab();
    }

    async function onToggleRunCheck(event, exp, run) {
      event.stopPropagation();
      const key = `${exp}:${run}`;
      if (event.target.checked) {
        selectedRuns.add(key);
      } else {
        selectedRuns.delete(key);
      }
      if (selectedRuns.size === 1) {
        const [singleKey] = Array.from(selectedRuns);
        const idx = singleKey.indexOf(":");
        activeExp = singleKey.slice(0, idx);
        activeRun = singleKey.slice(idx + 1);
      }
      document.getElementById("diff-only-toggle").checked = selectedRuns.size > 1;
      await syncSelectedRunsData();
      updateTabCounts();
      renderSidebar();
      renderActiveTab();
    }

    async function loadRunData(exp, run, force = false) {
      const key = `${exp}:${run}`;
      if (!force && loadedRunsData[key]) return loadedRunsData[key];
      try {
        const res = await fetch(`/api/run?exp=${encodeURIComponent(exp)}&run=${encodeURIComponent(run)}`);
        if (!res.ok) return null;
        const data = await res.json();
        loadedRunsData[key] = data;
        return data;
      } catch (e) {
        console.error("Error loading run:", e);
        return null;
      }
    }

    async function syncSelectedRunsData() {
      const promises = Array.from(selectedRuns).map(k => {
        const idx = k.indexOf(":");
        return loadRunData(k.slice(0, idx), k.slice(idx + 1));
      });
      await Promise.all(promises);
    }

    function updateTabCounts() {
      const activeData = loadedRunsData[`${activeExp}:${activeRun}`];
      const valImages = (activeData && activeData.artifacts) ? activeData.artifacts.filter(a => a.type === "image").length : 0;
      document.getElementById("val-count").textContent = `(${valImages})`;
    }

    function renderActiveTab() {
      updateTabCounts();
      if (activeTab === "metrics") renderMetricsTab();
      else if (activeTab === "val") renderValidationTab();
      else if (activeTab === "bundles") fetchBundles();
      else if (activeTab === "meta") renderMetaTab();
    }

    // ----------------------------------------------------
    // TAB 1: METRICS & MULTI-RUN COMPARISON
    // ----------------------------------------------------
    function renderMetricsTab() {
      const activeData = loadedRunsData[`${activeExp}:${activeRun}`];
      const topMeta = document.getElementById("metrics-top-meta");
      const provEl = document.getElementById("metrics-provenance");

      if (!activeData) {
        topMeta.innerHTML = "<div class='empty-state'>Select a run from the left sidebar to inspect metrics.</div>";
        provEl.innerHTML = "";
        document.getElementById("charts-grid").innerHTML = "";
        return;
      }

      const metrics = activeData.metrics || {};
      const params = activeData.params || {};
      const monitorKey = (params.training && params.training.monitor) || (metrics.val_loss ? "val_loss" : "train_loss");
      const monSeries = metrics[monitorKey] || metrics[Object.keys(metrics)[0]] || [];
      const lastPoint = monSeries.length ? monSeries[monSeries.length - 1] : null;
      const monMode = (params.training && params.training.monitor_mode) || "min";
      const bestVal = monSeries.length
        ? (monMode === "max" ? Math.max(...monSeries.map(p => p.value)) : Math.min(...monSeries.map(p => p.value)))
        : null;

      const totalEpochsCfg = params.training && params.training.epochs ? params.training.epochs : null;
      const maxStep = Object.values(metrics).reduce((acc, arr) => Math.max(acc, ...(arr.map(p => p.step))), 0);
      const epochText = totalEpochsCfg ? `${maxStep} / ${totalEpochsCfg}` : `${maxStep}`;

      const modelFam = (params.model && params.model.family) || "model";
      const modelVer = (params.model && params.model.version) || "";
      const device = (params.experiment && params.experiment.device) || "auto";

      topMeta.innerHTML = `
        <div class="card">
          <div class="card-label">${escapeHtml(monSeries.length ? `Best ${monitorKey}` : "Primary Metric")}</div>
          <div class="card-val" style="color:var(--ok);">${bestVal !== null ? bestVal.toFixed(4) : "N/A"}</div>
          <div class="card-sub">${lastPoint ? `last: ${lastPoint.value.toFixed(4)}` : "no metrics"}</div>
        </div>
        <div class="card">
          <div class="card-label">Epoch / Step</div>
          <div class="card-val">${escapeHtml(epochText)}</div>
          <div class="card-sub">${Object.keys(metrics).length} tracked series</div>
        </div>
        <div class="card">
          <div class="card-label">Status &amp; Duration</div>
          <div class="card-val">${escapeHtml(formatDuration(activeData.duration_seconds))}</div>
          <div class="card-sub"><span class="status-pill ${escapeHtml(activeData.status)}">${escapeHtml(activeData.status)}</span></div>
        </div>
        <div class="card">
          <div class="card-label">Model &amp; Device</div>
          <div class="card-val" style="font-size:0.94rem;">${escapeHtml(modelFam)}${modelVer ? `@${escapeHtml(modelVer)}` : ""}</div>
          <div class="card-sub">device: ${escapeHtml(device)}</div>
        </div>
      `;

      const commit = activeData.git_commit ? activeData.git_commit.slice(0, 8) : "N/A";
      const dhash = activeData.dataset_hash ? activeData.dataset_hash.slice(0, 14) + "..." : "N/A";
      provEl.innerHTML = `
        <div class="provenance-bar">
          <span><strong>Run:</strong> ${escapeHtml(activeData.run_name)}</span>
          <span><strong>Commit:</strong> ${escapeHtml(commit)}</span>
          <span><strong>Dataset SHA-256:</strong> ${escapeHtml(dhash)}</span>
        </div>
      `;

      // Collect metric names across selected runs
      const metricSet = new Set();
      selectedRuns.forEach(k => {
        const d = loadedRunsData[k];
        if (d && d.metrics) Object.keys(d.metrics).forEach(m => metricSet.add(m));
      });

      const select = document.getElementById("metric-select");
      const prevVal = select.value || "__ALL_PAIRED__";
      select.innerHTML = `<option value="__ALL_PAIRED__">All Metrics (Auto-paired Train/Val)</option>`;
      Array.from(metricSet).sort().forEach(m => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        select.appendChild(opt);
      });
      select.value = Array.from(select.options).some(o => o.value === prevVal) ? prevVal : "__ALL_PAIRED__";

      const cmpBtn = document.getElementById("toolbar-compare-btn");
      if (cmpBtn) {
        if (orderedRunKeys.length >= 2) {
          cmpBtn.style.display = "inline-flex";
          if (selectedRuns.size > 1) {
            cmpBtn.textContent = `Comparing ${selectedRuns.size} (Reset)`;
            cmpBtn.classList.add("active-toggle");
          } else {
            cmpBtn.textContent = "Compare Runs";
            cmpBtn.classList.remove("active-toggle");
          }
        } else {
          cmpBtn.style.display = "none";
        }
      }

      renderRunComparisonTable();
      renderCharts();
      renderParamDiffTable();
    }

    function setScale(mode) {
      scaleMode = mode;
      updateScaleButtons();
      renderCharts();
    }

    function updateScaleButtons() {
      document.getElementById("scale-linear-btn").classList.toggle("active-toggle", scaleMode === "linear");
      document.getElementById("scale-log-btn").classList.toggle("active-toggle", scaleMode === "log");
    }

    function updateSmoothing(val) {
      smoothingWeight = parseFloat(val);
      document.getElementById("smoothing-val").textContent = smoothingWeight.toFixed(2);
      renderCharts();
    }

    function computeEMA(points, weight) {
      if (weight <= 0 || points.length === 0) return points;
      let prev = points[0].value;
      return points.map(p => {
        prev = prev * weight + p.value * (1 - weight);
        return { step: p.step, value: prev };
      });
    }

    function buildChartGroups(metricNames) {
      const selected = document.getElementById("metric-select").value;
      if (selected && selected !== "__ALL_PAIRED__") {
        return [{ title: selected, metrics: [selected] }];
      }

      const nameSet = new Set(metricNames);
      const used = new Set();
      const groups = [];

      // Auto-pair train_<base> and val_<base>
      metricNames.forEach(m => {
        if (used.has(m)) return;
        const trainMatch = m.match(/^train_(.+)$/);
        const valMatch = m.match(/^val_(.+)$/);
        const base = trainMatch ? trainMatch[1] : (valMatch ? valMatch[1] : null);
        if (base && nameSet.has(`train_${base}`) && nameSet.has(`val_${base}`)) {
          used.add(`train_${base}`);
          used.add(`val_${base}`);
          groups.push({ title: `${base} (train vs val)`, metrics: [`train_${base}`, `val_${base}`] });
        }
      });

      metricNames.forEach(m => {
        if (!used.has(m)) groups.push({ title: m, metrics: [m] });
      });

      return groups;
    }

    function renderCharts() {
      const grid = document.getElementById("charts-grid");
      const runs = Array.from(selectedRuns).map(k => loadedRunsData[k]).filter(Boolean);
      const metricSet = new Set();
      runs.forEach(r => { if (r.metrics) Object.keys(r.metrics).forEach(m => metricSet.add(m)); });

      if (metricSet.size === 0) {
        grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1;">No step metrics recorded for this run yet.</div>`;
        return;
      }

      const groups = buildChartGroups(Array.from(metricSet).sort());
      grid.innerHTML = "";

      const boxes = groups.map((grp, idx) => {
        const box = document.createElement("div");
        box.className = "chart-box";
        box.innerHTML = `
          <div class="chart-box-head">
            <span class="chart-title">${escapeHtml(grp.title)}</span>
            <span style="font-family:var(--font-mono); font-size:0.7rem; color:var(--text-dim);">${scaleMode.toUpperCase()}</span>
          </div>
          <svg class="chart-svg" id="chart-svg-${idx}"></svg>
          <div class="chart-tooltip" id="chart-tip-${idx}"></div>
          <div class="chart-legend" id="chart-leg-${idx}"></div>
        `;
        grid.appendChild(box);
        return { box, grp, idx };
      });

      boxes.forEach(({ box, grp, idx }) => {
        drawSingleChart(box, `chart-svg-${idx}`, `chart-tip-${idx}`, `chart-leg-${idx}`, runs, grp.metrics);
      });
    }

    function drawSingleChart(box, svgId, tipId, legId, runs, metricKeys) {
      const svg = document.getElementById(svgId);
      const tip = document.getElementById(tipId);
      const legend = document.getElementById(legId);

      // Build flat list of series: { label, runName, metricKey, points, color, dashed }
      const seriesList = [];
      let colorIdx = 0;
      runs.forEach(r => {
        metricKeys.forEach(mk => {
          const pts = (r.metrics && r.metrics[mk]) ? r.metrics[mk] : null;
          if (!pts || pts.length === 0) return;
          const filteredPts = scaleMode === "log" ? pts.filter(p => p.value > 0) : pts;
          if (filteredPts.length === 0) return;
          const color = PALETTE[colorIdx % PALETTE.length];
          colorIdx++;
          const label = runs.length > 1 ? `${r.run_name} · ${mk}` : mk;
          seriesList.push({
            label,
            points: filteredPts,
            color,
            dashed: metricKeys.length > 1 && mk.startsWith("val_"),
          });
        });
      });

      if (seriesList.length === 0) {
        svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="#64748b" font-family="var(--font-mono)" font-size="11">No positive points to plot in ${scaleMode} scale.</text>`;
        return;
      }

      const rect = svg.getBoundingClientRect();
      const width = Math.max(300, Math.round(rect.width || 420));
      const height = 240;
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.removeAttribute("preserveAspectRatio");
      const padL = 54, padR = 20, padT = 18, padB = 32;

      const transformY = v => (scaleMode === "log" ? Math.log10(v) : v);

      let allSteps = [];
      let allYs = [];
      seriesList.forEach(s => {
        s.points.forEach(p => {
          allSteps.push(p.step);
          allYs.push(transformY(p.value));
        });
      });

      const minS = Math.min(...allSteps);
      const maxS = Math.max(...allSteps);
      let minY = Math.min(...allYs);
      let maxY = Math.max(...allYs);
      if (minY === maxY) { minY -= 0.1; maxY += 0.1; }

      const stepRange = (maxS - minS) || 1;
      const yRange = (maxY - minY) || 1;

      const getX = s => (minS === maxS ? (padL + (width - padL - padR) / 2) : padL + ((s - minS) / stepRange) * (width - padL - padR));
      const getY = rawVal => (height - padB) - ((transformY(rawVal) - minY) / yRange) * (height - padT - padB);

      const fmtTick = yT => {
        const real = scaleMode === "log" ? Math.pow(10, yT) : yT;
        return Math.abs(real) < 0.01 || Math.abs(real) >= 1000 ? real.toExponential(1) : real.toFixed(3);
      };

      let gridSvg = `
        <line x1="${padL}" y1="${height - padB}" x2="${width - padR}" y2="${height - padB}" stroke="#cbd5e1" stroke-width="1" />
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${height - padB}" stroke="#cbd5e1" stroke-width="1" />
        <text x="${padL - 8}" y="${padT + 4}" fill="#64748b" font-family="var(--font-mono)" font-size="10" text-anchor="end">${fmtTick(maxY)}</text>
        <text x="${padL - 8}" y="${(padT + height - padB)/2 + 4}" fill="#64748b" font-family="var(--font-mono)" font-size="10" text-anchor="end">${fmtTick((maxY + minY)/2)}</text>
        <text x="${padL - 8}" y="${height - padB}" fill="#64748b" font-family="var(--font-mono)" font-size="10" text-anchor="end">${fmtTick(minY)}</text>
        <line x1="${padL}" y1="${padT}" x2="${width - padR}" y2="${padT}" stroke="#e2e8f0" stroke-dasharray="3,3" />
        <line x1="${padL}" y1="${(padT + height - padB)/2}" x2="${width - padR}" y2="${(padT + height - padB)/2}" stroke="#e2e8f0" stroke-dasharray="3,3" />
        <text x="${padL}" y="${height - padB + 18}" fill="#64748b" font-family="var(--font-mono)" font-size="10">${minS === maxS ? `step ${minS}` : `step ${minS}`}</text>
        ${minS !== maxS ? `<text x="${width - padR}" y="${height - padB + 18}" fill="#64748b" font-family="var(--font-mono)" font-size="10" text-anchor="end">step ${maxS}</text>` : ""}
      `;

      let linesSvg = "";
      let legendHtml = "";

      seriesList.forEach(s => {
        const smoothed = computeEMA(s.points, smoothingWeight);
        if (smoothingWeight > 0.05 && s.points.length > 1) {
          let rawD = "";
          s.points.forEach((p, i) => { rawD += `${i === 0 ? "M" : "L"} ${getX(p.step)} ${getY(p.value)} `; });
          linesSvg += `<path d="${rawD}" fill="none" stroke="${s.color}" stroke-width="1" stroke-opacity="0.28" />`;
        }

        if (smoothed.length > 1) {
          let d = "";
          smoothed.forEach((p, i) => { d += `${i === 0 ? "M" : "L"} ${getX(p.step)} ${getY(p.value)} `; });
          linesSvg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" ${s.dashed ? 'stroke-dasharray="5,3"' : ""} />`;
        }

        if (smoothed.length <= 15) {
          smoothed.forEach(p => {
            linesSvg += `<circle cx="${getX(p.step)}" cy="${getY(p.value)}" r="3.2" fill="${s.color}" />`;
          });
        }

        const lastVal = s.points[s.points.length - 1].value;
        legendHtml += `
          <div class="legend-item">
            <span class="legend-dot" style="background:${s.color};"></span>
            <span style="font-weight:600;">${escapeHtml(s.label)}</span>
            <span style="color:var(--text-muted);">${lastVal.toFixed(4)}</span>
          </div>
        `;
      });

      const clipId = `clip-${svgId}`;
      const plotW = Math.max(1, width - padL - padR);
      const plotH = Math.max(1, height - padT - padB);
      const defsSvg = `<defs><clipPath id="${clipId}"><rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" /></clipPath></defs>`;
      svg.innerHTML = `${defsSvg}${gridSvg}<g clip-path="url(#${clipId})">${linesSvg}</g>`;
      legend.innerHTML = legendHtml;

      svg.onmousemove = (e) => {
        const svgRect = svg.getBoundingClientRect();
        const boxRect = box.getBoundingClientRect();
        const scaleX = svgRect.width > 0 ? (width / svgRect.width) : 1;
        const mouseX = (e.clientX - svgRect.left) * scaleX;
        if (mouseX < padL || mouseX > width - padR) return;

        const targetStep = minS === maxS
          ? minS
          : Math.round(minS + ((mouseX - padL) / (width - padL - padR)) * stepRange);

        const oldCross = svg.querySelector(".crosshair-v");
        if (oldCross) oldCross.remove();
        svg.insertAdjacentHTML("beforeend", `<line class="crosshair-v" x1="${mouseX}" y1="${padT}" x2="${mouseX}" y2="${height - padB}" stroke="#4f46e5" stroke-width="1" stroke-dasharray="2,2" />`);

        let tipRows = `<div style="font-weight:700; margin-bottom:4px; color:var(--text);">Step ${targetStep}</div>`;
        seriesList.forEach(s => {
          const closest = s.points.reduce((prev, curr) => Math.abs(curr.step - targetStep) < Math.abs(prev.step - targetStep) ? curr : prev);
          tipRows += `<div style="display:flex; justify-content:space-between; gap:12px;"><span><span style="display:inline-block; width:7px; height:7px; border-radius:2px; background:${s.color}; margin-right:5px;"></span>${escapeHtml(s.label)}</span><strong>${closest.value.toFixed(4)}</strong></div>`;
        });
        tip.innerHTML = tipRows;
        tip.style.display = "block";

        const relX = e.clientX - boxRect.left;
        const relY = e.clientY - boxRect.top;
        const tipWidth = tip.offsetWidth || 160;
        const leftPos = (relX + tipWidth + 18 > boxRect.width) ? Math.max(8, relX - tipWidth - 12) : (relX + 12);
        tip.style.left = `${leftPos}px`;
        tip.style.top = `${Math.max(8, relY - 18)}px`;
      };

      svg.onmouseleave = () => {
        const oldCross = svg.querySelector(".crosshair-v");
        if (oldCross) oldCross.remove();
        tip.style.display = "none";
      };
    }

    function flattenParams(obj, prefix = "", out = {}) {
      if (!obj || typeof obj !== "object") return out;
      Object.entries(obj).forEach(([k, v]) => {
        const key = prefix ? `${prefix}.${k}` : k;
        if (v !== null && typeof v === "object" && !Array.isArray(v)) {
          const keys = Object.keys(v);
          if (keys.length === 0) {
            out[key] = "{}";
          } else {
            flattenParams(v, key, out);
          }
        } else {
          out[key] = v;
        }
      });
      return out;
    }

    function renderRunComparisonTable() {
      const container = document.getElementById("metrics-compare-summary");
      const runKeys = Array.from(selectedRuns);
      const runs = runKeys.map((k, idx) => ({ key: k, idx, data: loadedRunsData[k] })).filter(x => Boolean(x.data));
      if (runs.length < 2) {
        container.innerHTML = "";
        return;
      }

      const metricOrder = ["val_loss", "val_acc", "train_loss", "loss", "acc"];
      const allMetrics = new Set();
      runs.forEach(x => Object.keys(x.data.metrics || {}).forEach(m => allMetrics.add(m)));
      const displayMetrics = [
        ...metricOrder.filter(m => allMetrics.has(m)),
        ...Array.from(allMetrics).filter(m => !metricOrder.includes(m)).sort()
      ].slice(0, 4);

      const isLowerBetter = m => /loss|err|mae|mse|rmse|perplexity/i.test(m);
      const bestPerMetric = {};
      displayMetrics.forEach(m => {
        const vals = runs
          .map(x => {
            const pts = (x.data.metrics && x.data.metrics[m]) || [];
            return pts.length > 0 ? Number(pts[pts.length - 1].value) : null;
          })
          .filter(v => v !== null && !Number.isNaN(v));
        if (vals.length > 0) {
          bestPerMetric[m] = isLowerBetter(m) ? Math.min(...vals) : Math.max(...vals);
        }
      });

      const rowsHtml = runs.map(({ key, idx, data }) => {
        const color = PALETTE[idx % PALETTE.length];
        const params = data.params || {};
        const metrics = data.metrics || {};
        const modelInfo = params.model || {};
        const modelName = modelInfo.family ? `${modelInfo.family}${modelInfo.version ? `@${modelInfo.version}` : ""}` : "—";
        const status = String(data.status || "unknown").toLowerCase();

        const monitorKey = (params.training && params.training.monitor) || (metrics.val_loss ? "val_loss" : (metrics.train_loss ? "train_loss" : Object.keys(metrics)[0]));
        const monSeries = monitorKey ? (metrics[monitorKey] || []) : [];
        const monMode = (params.training && params.training.monitor_mode) || (monitorKey && isLowerBetter(monitorKey) ? "min" : "max");
        const bestMonVal = monSeries.length > 0
          ? (monMode === "max" ? Math.max(...monSeries.map(p => Number(p.value))) : Math.min(...monSeries.map(p => Number(p.value))))
          : null;
        const bestText = bestMonVal !== null
          ? `${escapeHtml(monitorKey)}: <strong>${bestMonVal.toFixed(4)}</strong>`
          : "—";

        const metricCells = displayMetrics.map(m => {
          const pts = metrics[m] || [];
          if (pts.length === 0) return `<td style="color:var(--text-dim);">—</td>`;
          const last = Number(pts[pts.length - 1].value);
          const isWinner = bestPerMetric[m] !== undefined && Math.abs(last - bestPerMetric[m]) < 1e-12;
          return `<td style="font-variant-numeric:tabular-nums; ${isWinner ? "color:var(--ok); font-weight:700;" : ""}">${last.toFixed(4)}</td>`;
        }).join("");

        const maxStep = Object.values(metrics).reduce((acc, arr) => Math.max(acc, ...(arr.map(p => p.step))), 0);
        const totalEpochsCfg = params.training && params.training.epochs ? params.training.epochs : null;
        const epochDisplay = totalEpochsCfg ? `${maxStep} / ${totalEpochsCfg}` : `${maxStep}`;
        const activeTag = key === `${activeExp}:${activeRun}` ? ' <span class="brand-ver" style="font-size:0.65rem;">ACTIVE</span>' : "";

        return `
          <tr>
            <td><span class="legend-dot" style="background:${color}; display:inline-block; margin-right:6px;"></span><strong>${escapeHtml(data.run_name)}</strong>${activeTag}</td>
            <td>${escapeHtml(modelName)}</td>
            <td><span class="status-pill ${escapeHtml(status)}">${escapeHtml(status)}</span></td>
            <td>${bestText}</td>
            ${metricCells}
            <td>${escapeHtml(epochDisplay)}</td>
            <td>${escapeHtml(formatDuration(data.duration_seconds))}</td>
          </tr>
        `;
      }).join("");

      container.innerHTML = `
        <div class="section-header" style="margin-top:4px;">
          <div class="section-title">Model &amp; Run Comparison (${runs.length} selected)</div>
          <button class="btn" style="padding:2px 8px; font-size:0.74rem;" onclick="clearComparison()">Exit Compare</button>
        </div>
        <div class="card" style="padding:0; overflow-x:auto; margin-bottom:16px;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Model</th>
                <th>Status</th>
                <th>Best Monitored</th>
                ${displayMetrics.map(m => `<th>Last ${escapeHtml(m)}</th>`).join("")}
                <th>Epoch / Step</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>${rowsHtml}</tbody>
          </table>
        </div>
      `;
    }

    function renderParamDiffTable() {
      const tbody = document.querySelector("#param-diff-table tbody");
      const thead = document.querySelector("#param-diff-table thead tr");
      const diffOnly = document.getElementById("diff-only-toggle").checked;
      const runs = Array.from(selectedRuns).map(k => loadedRunsData[k]).filter(Boolean);
      if (runs.length === 0) return;

      thead.innerHTML = `<th>Parameter</th>` + runs.map(r => `<th>${escapeHtml(r.run_name)}</th>`).join("");

      const flatPerRun = runs.map(r => flattenParams(r.params || {}));
      const allKeys = new Set();
      flatPerRun.forEach(fp => Object.keys(fp).forEach(k => allKeys.add(k)));

      tbody.innerHTML = "";
      if (allKeys.size === 0) {
        tbody.innerHTML = `<tr><td colspan="${runs.length + 1}" style="color:var(--text-dim); text-align:center;">No parameters logged.</td></tr>`;
        return;
      }

      let renderedRows = 0;
      Array.from(allKeys).sort().forEach(k => {
        const vals = flatPerRun.map(fp => fp[k]);
        const isDivergent = runs.length > 1 && vals.some(v => JSON.stringify(v) !== JSON.stringify(vals[0]));
        if (diffOnly && runs.length > 1 && !isDivergent) return;

        renderedRows++;
        const tr = document.createElement("tr");
        if (isDivergent) tr.className = "divergent";
        tr.innerHTML = `<td>${escapeHtml(k)}</td>` + vals.map(v => `<td>${v !== undefined ? escapeHtml(JSON.stringify(v)) : '<span style="color:var(--text-dim);">—</span>'}</td>`).join("");
        tbody.appendChild(tr);
      });

      if (renderedRows === 0) {
        tbody.innerHTML = `<tr><td colspan="${runs.length + 1}" style="color:var(--text-dim); text-align:center;">All parameters match across selected runs.</td></tr>`;
      }
    }

    // ----------------------------------------------------
    // TAB 2: REAL DATA VALIDATION & TIMELAPSE SCRUBBER
    // ----------------------------------------------------
    function renderValidationTab() {
      const activeData = loadedRunsData[`${activeExp}:${activeRun}`];
      const samplesListEl = document.getElementById("val-samples-list");
      const grid = document.getElementById("val-grid-cards");
      if (!activeData || !activeData.artifacts) {
        samplesListEl.innerHTML = "<div class='empty-state'>No run loaded.</div>";
        return;
      }

      const images = activeData.artifacts.filter(a => a.type === "image");
      const sampleGroups = {};
      images.forEach(img => {
        const sId = img.sample_id || img.base_name || "sample_default";
        if (!sampleGroups[sId]) sampleGroups[sId] = [];
        sampleGroups[sId].push(img);
      });

      const sampleIds = Object.keys(sampleGroups);
      if (sampleIds.length === 0) {
        samplesListEl.innerHTML = "<div class='empty-state' style='padding:18px;'>No validation images logged.</div>";
        grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1;">No image samples recorded for this run. For vision/segmentation tasks, log snapshots with <code>run.log_validation_sample(...)</code>.</div>`;
        return;
      }

      samplesListEl.innerHTML = "";
      sampleIds.forEach(sId => {
        const item = document.createElement("div");
        item.className = `val-sample-item ${sId === currentSampleId ? "active" : ""}`;
        item.innerHTML = `
          <div style="font-weight:600; font-family:var(--font-mono); font-size:0.8rem;">${escapeHtml(sId)}</div>
          <div style="color:var(--text-dim); font-size:0.72rem; margin-top:2px;">${sampleGroups[sId].length} snapshots</div>
        `;
        item.onclick = () => selectValidationSample(sId, sampleGroups[sId]);
        samplesListEl.appendChild(item);
      });

      const targetId = (currentSampleId && sampleGroups[currentSampleId]) ? currentSampleId : sampleIds[0];
      selectValidationSample(targetId, sampleGroups[targetId]);
    }

    function selectValidationSample(sId, artifacts) {
      currentSampleId = sId;
      const stepMap = {};
      artifacts.forEach(a => {
        const s = a.step !== null && a.step !== undefined ? a.step : 0;
        if (!stepMap[s]) stepMap[s] = {};
        stepMap[s][a.tag || "prediction"] = a;
      });

      const sortedSteps = Object.keys(stepMap).map(Number).sort((a, b) => a - b);
      currentSampleTimeline = sortedSteps.map(s => ({ step: s, ...stepMap[s] }));

      const scrubber = document.getElementById("epoch-scrubber");
      scrubber.min = 0;
      scrubber.max = Math.max(0, currentSampleTimeline.length - 1);
      currentTimelineStepIdx = Math.min(currentTimelineStepIdx, Math.max(0, currentSampleTimeline.length - 1));
      scrubber.value = currentTimelineStepIdx;
      renderTimelineStep();
    }

    function onScrubStep(idx) { currentTimelineStepIdx = parseInt(idx, 10); renderTimelineStep(); }
    function prevStep() { if (currentTimelineStepIdx > 0) { currentTimelineStepIdx--; document.getElementById("epoch-scrubber").value = currentTimelineStepIdx; renderTimelineStep(); } }
    function nextStep() { if (currentTimelineStepIdx < currentSampleTimeline.length - 1) { currentTimelineStepIdx++; document.getElementById("epoch-scrubber").value = currentTimelineStepIdx; renderTimelineStep(); } }

    function togglePlayTimelapse() {
      const btn = document.getElementById("play-btn");
      if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
        btn.textContent = "Play Timelapse";
      } else {
        btn.textContent = "Pause";
        playTimer = setInterval(() => {
          currentTimelineStepIdx = (currentTimelineStepIdx >= currentSampleTimeline.length - 1) ? 0 : currentTimelineStepIdx + 1;
          document.getElementById("epoch-scrubber").value = currentTimelineStepIdx;
          renderTimelineStep();
        }, 600);
      }
    }

    function renderTimelineStep() {
      if (!currentSampleTimeline.length) return;
      const current = currentSampleTimeline[currentTimelineStepIdx];
      document.getElementById("scrubber-step-indicator").textContent = `Step / Epoch: ${current.step}`;

      const grid = document.getElementById("val-grid-cards");
      const activeData = loadedRunsData[`${activeExp}:${activeRun}`];
      const roles = [
        { key: "input", title: "Validation Input", color: "var(--series-2)" },
        { key: "ground_truth", title: "Ground Truth", color: "var(--ok)" },
        { key: "prediction", title: `Prediction (Step ${current.step})`, color: "var(--accent)" },
      ];

      grid.innerHTML = "";
      roles.forEach(r => {
        const art = current[r.key] || current["image"];
        if (!art && r.key !== "prediction") return;
        const imgSrc = art ? `/api/artifact?exp=${encodeURIComponent(activeData.experiment)}&run=${encodeURIComponent(activeData.run_name)}&file=${encodeURIComponent(art.name)}` : "";

        const card = document.createElement("div");
        card.className = "val-card";
        card.innerHTML = `
          <div class="val-card-header">
            <span style="color:${r.color};">${escapeHtml(r.title)}</span>
          </div>
          <div class="val-card-img-wrap">
            ${imgSrc ? `<img src="${imgSrc}" alt="${escapeHtml(r.title)}" />` : `<div style="color:#94a3b8; font-size:0.8rem;">No ${escapeHtml(r.key)} logged</div>`}
            <span class="zoom-hint">Click to inspect</span>
          </div>
        `;
        if (imgSrc) {
          card.querySelector(".val-card-img-wrap").onclick = () => openLightbox(imgSrc, art.name);
        }
        grid.appendChild(card);
      });
    }

    // ----------------------------------------------------
    // TAB 3: BUNDLE INSPECTOR
    // ----------------------------------------------------
    async function fetchBundles(force = false) {
      try {
        const res = await fetch("/api/bundles");
        const data = await res.json();
        const sig = JSON.stringify(data);
        const listEl = document.getElementById("bundles-list");
        const bundles = data.bundles || [];
        discoveredBundles = bundles;
        document.getElementById("bundle-count").textContent = `(${bundles.length})`;

        if (!force && sig === lastBundlesSignature) return;
        lastBundlesSignature = sig;
        listEl.innerHTML = "";

        if (bundles.length === 0) {
          listEl.innerHTML = "<div class='empty-state' style='padding:18px;'>No <code>.bundle</code> files found in workspace.</div>";
          return;
        }

        if (!activeBundlePath || !bundles.some(b => b.path === activeBundlePath)) {
          activeBundlePath = bundles[0].path;
        }
        const validPaths = new Set(bundles.map(b => b.path));
        selectedBundles.forEach(p => { if (!validPaths.has(p)) selectedBundles.delete(p); });
        updateBundleCompareBar();

        bundles.forEach(b => {
          const item = document.createElement("div");
          item.className = `val-sample-item ${b.path === activeBundlePath ? "active" : ""}`;
          const roleBadge = b.format === "onnx" ? "ONNX" : (b.is_checkpoint ? "CKPT" : "INFER");
          const isChecked = selectedBundles.has(b.path);
          item.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; gap:6px;">
              <label style="display:flex; align-items:center; gap:6px; min-width:0; cursor:pointer; flex:1;" onclick="event.stopPropagation()">
                <input type="checkbox" ${isChecked ? "checked" : ""} onchange="toggleBundleSelection(decodeURIComponent('${encodeURIComponent(b.path)}'), this.checked)" />
                <span style="font-weight:700; font-family:var(--font-mono); font-size:0.79rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(b.filename)}</span>
              </label>
              <span class="brand-ver" style="font-size:0.65rem; flex-shrink:0;">${roleBadge}</span>
            </div>
            <div style="color:var(--text-dim); font-family:var(--font-mono); font-size:0.7rem; margin-top:4px; display:flex; justify-content:space-between;">
              <span>${escapeHtml(b.family)} v${escapeHtml(b.version)}</span>
              <span>${formatBytes(b.size_bytes)}</span>
            </div>
          `;
          item.onclick = () => {
            activeBundlePath = b.path;
            listEl.querySelectorAll(".val-sample-item").forEach(el => el.classList.remove("active"));
            item.classList.add("active");
            inspectBundle(b.path);
          };
          listEl.appendChild(item);
        });

        if (selectedBundles.size >= 2) {
          compareBundles();
        } else {
          inspectBundle(activeBundlePath);
        }
      } catch (e) {
        console.error("Error fetching bundles:", e);
      }
    }

    function updateBundleCompareBar() {
      const bar = document.getElementById("bundles-compare-bar");
      if (!bar) return;
      if (discoveredBundles.length < 2) {
        bar.style.display = "none";
        return;
      }
      bar.style.display = "flex";
      if (selectedBundles.size >= 2) {
        bar.innerHTML = `
          <button class="btn active-toggle" id="bundles-compare-btn" style="padding:2px 8px; font-size:0.72rem;" onclick="compareBundles()">Compare (${selectedBundles.size})</button>
          <button class="btn" style="padding:2px 6px; font-size:0.72rem;" onclick="clearBundleSelection()">Clear</button>
        `;
      } else {
        bar.innerHTML = `
          <button class="btn" id="bundles-compare-btn" style="padding:2px 8px; font-size:0.72rem;" onclick="compareAllBundles()">Compare All (${discoveredBundles.length})</button>
        `;
      }
    }

    function compareAllBundles() {
      if (discoveredBundles.length < 2) return;
      selectedBundles = new Set(discoveredBundles.map(b => b.path));
      document.querySelectorAll("#bundles-list input[type='checkbox']").forEach(chk => { chk.checked = true; });
      updateBundleCompareBar();
      compareBundles();
    }

    function toggleBundleSelection(bundlePath, isChecked) {
      if (isChecked) selectedBundles.add(bundlePath);
      else selectedBundles.delete(bundlePath);
      updateBundleCompareBar();
      if (selectedBundles.size >= 2) {
        compareBundles();
      } else if (activeBundlePath) {
        inspectBundle(activeBundlePath);
      }
    }

    function clearBundleSelection() {
      selectedBundles.clear();
      document.querySelectorAll("#bundles-list input[type='checkbox']").forEach(chk => { chk.checked = false; });
      updateBundleCompareBar();
      if (activeBundlePath) inspectBundle(activeBundlePath);
    }

    async function loadBundleData(bundlePath) {
      if (loadedBundlesCache[bundlePath]) return loadedBundlesCache[bundlePath];
      const res = await fetch(`/api/bundle?path=${encodeURIComponent(bundlePath)}`);
      const b = await res.json();
      loadedBundlesCache[bundlePath] = b;
      return b;
    }

    async function compareBundles() {
      const paths = Array.from(selectedBundles);
      if (paths.length < 2) {
        if (activeBundlePath) inspectBundle(activeBundlePath);
        return;
      }
      const main = document.getElementById("bundle-inspector-main");
      main.innerHTML = "<div class='empty-state'>Comparing selected bundles...</div>";

      try {
        const bundles = await Promise.all(paths.map(p => loadBundleData(p)));
        const headerCols = bundles.map(b => `<th>${escapeHtml(b.file)}</th>`).join("");

        const specRows = [
          ["Family & Version", b => `${b.family || "—"} v${b.version || "—"}`],
          ["Format / Role", b => `${(b.format || "safetensors").toUpperCase()} · ${b.is_checkpoint ? "CHECKPOINT" : "INFERENCE"}`],
          ["File Size", b => formatBytes(b.size_bytes)],
          ["Total Params", b => (b.total_params || 0).toLocaleString()],
          ["Layers", b => String(b.tensors ? b.tensors.length : 0)],
          ["Epoch / Step", b => `${b.epoch ?? "N/A"} / ${b.step ?? "N/A"}`],
          ["Weights SHA-256", b => `${(b.weights_sha256 || "N/A").slice(0, 16)}...`]
        ];
        const specHtml = specRows.map(([label, getter]) => {
          const vals = bundles.map(getter);
          const isDiff = vals.some(v => v !== vals[0]);
          return `<tr class="${isDiff ? "divergent" : ""}"><td><strong>${escapeHtml(label)}</strong></td>${vals.map(v => `<td>${escapeHtml(v)}</td>`).join("")}</tr>`;
        }).join("");

        const allMetricKeys = new Set();
        bundles.forEach(b => Object.keys(b.metrics || {}).forEach(k => allMetricKeys.add(k)));
        const metricRowsHtml = Array.from(allMetricKeys).sort().map(k => {
          const vals = bundles.map(b => (b.metrics && b.metrics[k] !== undefined) ? b.metrics[k] : null);
          const isDiff = vals.some(v => JSON.stringify(v) !== JSON.stringify(vals[0]));
          return `<tr class="${isDiff ? "divergent" : ""}"><td>${escapeHtml(k)}</td>${vals.map(v => `<td>${v === null ? '<span style="color:var(--text-dim);">—</span>' : (typeof v === "number" ? v.toFixed(6) : escapeHtml(v))}</td>`).join("")}</tr>`;
        }).join("");

        const schemaNodes = new Set();
        const schemaLookup = bundles.map(b => {
          const map = {};
          const inputs = (b.schema && b.schema.inputs) || [];
          const outputs = (b.schema && b.schema.outputs) || [];
          inputs.forEach(i => { const k = `input:${i.name}`; schemaNodes.add(k); map[k] = `[${(i.shape || []).map(d => d === null ? "dyn" : d).join(", ")}] · ${i.dtype}`; });
          outputs.forEach(o => { const k = `output:${o.name}`; schemaNodes.add(k); map[k] = `[${(o.shape || []).map(d => d === null ? "dyn" : d).join(", ")}] · ${o.dtype}`; });
          return map;
        });
        const schemaRowsHtml = Array.from(schemaNodes).sort().map(node => {
          const vals = schemaLookup.map(m => m[node] || null);
          const isDiff = vals.some(v => v !== vals[0]);
          return `<tr class="${isDiff ? "divergent" : ""}"><td>${escapeHtml(node)}</td>${vals.map(v => `<td>${v ? escapeHtml(v) : '<span style="color:var(--text-dim);">—</span>'}</td>`).join("")}</tr>`;
        }).join("");

        const allLayerNames = new Set();
        const layerLookup = bundles.map(b => {
          const map = {};
          (b.tensors || []).forEach(t => { allLayerNames.add(t.name); map[t.name] = t; });
          return map;
        });
        const layerRowsHtml = Array.from(allLayerNames).sort().map(name => {
          const vals = layerLookup.map(m => m[name] ? `[${(m[name].shape || []).join(", ")}] · ${m[name].dtype} · ${(m[name].param_count || 0).toLocaleString()}` : null);
          const isDiff = vals.some(v => v !== vals[0]);
          return `<tr class="${isDiff ? "divergent" : ""}"><td>${escapeHtml(name)}</td>${vals.map(v => `<td>${v ? escapeHtml(v) : '<span style="color:var(--text-dim);">missing</span>'}</td>`).join("")}</tr>`;
        }).join("");

        main.innerHTML = `
          <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
              <div>
                <h2 style="font-size:1.05rem; font-family:var(--font-mono);">Model &amp; Bundle Comparison (${bundles.length} selected)</h2>
                <div style="color:var(--text-muted); font-size:0.78rem; margin-top:3px;">Rows highlighted in indigo indicate differences across compared bundles.</div>
              </div>
              <button class="btn" onclick="clearBundleSelection()">Back to Single Inspector</button>
            </div>
          </div>

          <div class="card" style="padding:0; overflow:hidden;">
            <div class="section-header" style="padding:10px 14px; margin:0;"><div class="section-title" style="font-size:0.84rem;">Architecture &amp; Metadata</div></div>
            <div style="overflow-x:auto;">
              <table class="data-table"><thead><tr><th>Property</th>${headerCols}</tr></thead><tbody>${specHtml}</tbody></table>
            </div>
          </div>

          ${allMetricKeys.size > 0 ? `
            <div class="card" style="padding:0; overflow:hidden;">
              <div class="section-header" style="padding:10px 14px; margin:0;"><div class="section-title" style="font-size:0.84rem;">Recorded Metrics Comparison</div></div>
              <div style="overflow-x:auto;">
                <table class="data-table"><thead><tr><th>Metric</th>${headerCols}</tr></thead><tbody>${metricRowsHtml}</tbody></table>
              </div>
            </div>
          ` : ""}

          ${schemaNodes.size > 0 ? `
            <div class="card" style="padding:0; overflow:hidden;">
              <div class="section-header" style="padding:10px 14px; margin:0;"><div class="section-title" style="font-size:0.84rem;">ModelSchema Contract Diff</div></div>
              <div style="overflow-x:auto;">
                <table class="data-table"><thead><tr><th>Contract Node</th>${headerCols}</tr></thead><tbody>${schemaRowsHtml}</tbody></table>
              </div>
            </div>
          ` : ""}

          ${allLayerNames.size > 0 ? `
            <div class="card" style="padding:0; overflow:hidden;">
              <div class="section-header" style="padding:10px 14px; margin:0;"><div class="section-title" style="font-size:0.84rem;">SafeTensors Layer Diff (${allLayerNames.size} layers)</div></div>
              <div style="max-height:320px; overflow:auto;">
                <table class="data-table"><thead><tr><th>Layer Tensor</th>${headerCols}</tr></thead><tbody>${layerRowsHtml}</tbody></table>
              </div>
            </div>
          ` : ""}
        `;
      } catch (e) {
        main.innerHTML = `<div class='empty-state'>Error comparing bundles: ${escapeHtml(e.message)}</div>`;
      }
    }

    async function inspectBundle(bundlePath) {
      const main = document.getElementById("bundle-inspector-main");
      main.innerHTML = "<div class='empty-state'>Inspecting bundle headers...</div>";

      try {
        const b = await loadBundleData(bundlePath);
        currentBundleData = b;

        const metricEntries = Object.entries(b.metrics || {});
        const schemaInputs = (b.schema && b.schema.inputs) ? b.schema.inputs : [];
        const schemaOutputs = (b.schema && b.schema.outputs) ? b.schema.outputs : [];

        main.innerHTML = `
          <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
              <div>
                <h2 style="font-size:1.1rem; font-family:var(--font-mono);">${escapeHtml(b.file)}</h2>
                <div style="color:var(--text-muted); font-size:0.8rem; margin-top:4px; font-family:var(--font-mono);">
                  ${escapeHtml(b.family)} v${escapeHtml(b.version)} · ${(escapeHtml(b.format || "safetensors")).toUpperCase()} · ${formatBytes(b.size_bytes)}
                </div>
              </div>
              <span class="status-pill ${b.is_checkpoint ? "interrupted" : "completed"}">
                ${b.is_checkpoint ? "TRAINING CHECKPOINT" : "INFERENCE MODEL"}
              </span>
            </div>
          </div>

          <div class="grid-metrics">
            <div class="card">
              <div class="card-label">Weights SHA-256</div>
              <div class="card-val" style="font-size:0.82rem;">${escapeHtml((b.weights_sha256 || "N/A").slice(0, 20))}...</div>
            </div>
            <div class="card">
              <div class="card-label">Epoch / Step</div>
              <div class="card-val">${b.epoch ?? "N/A"} / ${b.step ?? "N/A"}</div>
            </div>
            <div class="card">
              <div class="card-label">Parameters</div>
              <div class="card-val" style="color:var(--accent);">${(b.total_params || 0).toLocaleString()}</div>
            </div>
            <div class="card">
              <div class="card-label">Layers</div>
              <div class="card-val">${b.tensors ? b.tensors.length : 0}</div>
            </div>
          </div>

          ${metricEntries.length > 0 ? `
            <div class="card" style="padding:0; overflow:hidden;">
              <div class="section-header" style="padding:10px 14px; margin:0;">
                <div class="section-title" style="font-size:0.84rem;">Recorded Bundle Metrics</div>
              </div>
              <table class="data-table">
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>
                  ${metricEntries.map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${typeof v === "number" ? v.toFixed(6) : escapeHtml(v)}</td></tr>`).join("")}
                </tbody>
              </table>
            </div>
          ` : ""}

          ${b.schema ? `
            <div class="card" style="padding:0; overflow:hidden;">
              <div class="section-header" style="padding:10px 14px; margin:0;">
                <div class="section-title" style="font-size:0.84rem;">ModelSchema Data Contract</div>
              </div>
              <table class="data-table">
                <thead><tr><th>Node</th><th>Shape</th><th>Dtype</th></tr></thead>
                <tbody>
                  ${schemaInputs.map(inp => `<tr><td>Input: ${escapeHtml(inp.name)}</td><td>[${inp.shape.map(d => d === null ? "dynamic" : d).join(", ")}]</td><td>${escapeHtml(inp.dtype)}</td></tr>`).join("")}
                  ${schemaOutputs.map(out => `<tr><td>Output: ${escapeHtml(out.name)}</td><td>[${out.shape.map(d => d === null ? "dynamic" : d).join(", ")}]</td><td>${escapeHtml(out.dtype)}</td></tr>`).join("")}
                </tbody>
              </table>
            </div>
          ` : ""}

          ${b.tensors && b.tensors.length > 0 ? `
            <div class="card" style="padding:0; overflow:hidden;">
              <div class="section-header" style="padding:10px 14px; margin:0;">
                <div class="section-title" style="font-size:0.84rem;">SafeTensors Layers (Zero-Torch Parsed)</div>
                <input type="text" placeholder="Filter layers..." oninput="filterBundleLayers(this.value)" style="width:180px; padding:4px 8px; font-size:0.75rem;" />
              </div>
              <div style="max-height:260px; overflow-y:auto;">
                <table class="data-table" id="bundle-layers-table">
                  <thead><tr><th>Layer Tensor</th><th>Shape</th><th>Dtype</th><th>Params</th><th>Size</th></tr></thead>
                  <tbody>
                    ${b.tensors.map(t => `
                      <tr>
                        <td>${escapeHtml(t.name)}</td>
                        <td>[${(t.shape || []).join(", ")}]</td>
                        <td>${escapeHtml(t.dtype)}</td>
                        <td>${(t.param_count || 0).toLocaleString()}</td>
                        <td>${formatBytes(t.size_bytes)}</td>
                      </tr>
                    `).join("")}
                  </tbody>
                </table>
              </div>
            </div>
          ` : ""}

          ${b.config ? `
            <details class="card" style="padding:10px 14px;">
              <summary style="cursor:pointer; font-weight:600; font-size:0.84rem; display:flex; justify-content:space-between; align-items:center;">
                <span>Embedded Configuration (config.yaml)</span>
                <button class="btn" id="copy-bundle-cfg-btn" onclick="event.preventDefault(); copyTextFromEl('bundle-cfg-pre', 'copy-bundle-cfg-btn')">Copy</button>
              </summary>
              <pre class="code-block" id="bundle-cfg-pre" style="margin-top:10px;">${escapeHtml(b.config)}</pre>
            </details>
          ` : ""}
        `;
      } catch (e) {
        main.innerHTML = `<div class="empty-state" style="color:var(--bad);">Error loading bundle: ${escapeHtml(e)}</div>`;
      }
    }

    function filterBundleLayers(query) {
      const q = query.toLowerCase();
      document.querySelectorAll("#bundle-layers-table tbody tr").forEach(tr => {
        const name = (tr.firstElementChild ? tr.firstElementChild.textContent : "").toLowerCase();
        tr.style.display = name.includes(q) ? "" : "none";
      });
    }

    // ----------------------------------------------------
    // TAB 4: METADATA
    // ----------------------------------------------------
    function renderMetaTab() {
      const activeData = loadedRunsData[`${activeExp}:${activeRun}`];
      const pre = document.getElementById("meta-json-pre");
      pre.textContent = activeData ? JSON.stringify(activeData, null, 2) : "Select a run to view metadata.";
    }

    function copyTextFromEl(elId, btnId) {
      const el = document.getElementById(elId);
      const btn = document.getElementById(btnId);
      if (!el || !navigator.clipboard) return;
      navigator.clipboard.writeText(el.textContent || "");
      if (btn) {
        const orig = btn.textContent;
        btn.textContent = "Copied ✓";
        setTimeout(() => { btn.textContent = orig; }, 1500);
      }
    }

    // ----------------------------------------------------
    // LIGHTBOX (ZOOM + PAN)
    // ----------------------------------------------------
    function applyLightboxTransform() {
      document.getElementById("lightbox-img").style.transform = `translate(${lightboxTx}px, ${lightboxTy}px) scale(${lightboxScale})`;
    }

    function openLightbox(src, title) {
      if (!src) return;
      document.getElementById("lightbox-img").src = src;
      document.getElementById("lightbox-title").textContent = title || "Sample Inspection";
      document.getElementById("lightbox").classList.add("active");
      resetLightboxZoom();
    }

    function closeLightbox() {
      document.getElementById("lightbox").classList.remove("active");
    }

    function zoomLightbox(delta) {
      lightboxScale = Math.max(0.5, Math.min(6.0, lightboxScale + delta));
      applyLightboxTransform();
    }

    function resetLightboxZoom() {
      lightboxScale = 1.0;
      lightboxTx = 0;
      lightboxTy = 0;
      applyLightboxTransform();
    }

    function setupLightboxInteractions() {
      const body = document.getElementById("lightbox-body");
      body.addEventListener("mousedown", (e) => {
        isDraggingLightbox = true;
        dragStartX = e.clientX - lightboxTx;
        dragStartY = e.clientY - lightboxTy;
        body.classList.add("dragging");
      });
      window.addEventListener("mousemove", (e) => {
        if (!isDraggingLightbox) return;
        lightboxTx = e.clientX - dragStartX;
        lightboxTy = e.clientY - dragStartY;
        applyLightboxTransform();
      });
      window.addEventListener("mouseup", () => {
        isDraggingLightbox = false;
        body.classList.remove("dragging");
      });
      body.addEventListener("wheel", (e) => {
        e.preventDefault();
        zoomLightbox(e.deltaY < 0 ? 0.2 : -0.2);
      }, { passive: false });
    }

    // ----------------------------------------------------
    // LIVE WATCH & KEYBOARD
    // ----------------------------------------------------
    function hasAnyRunningRun() {
      return Object.values(experimentsData).some(runs => runs.some(r => r.status === "running"));
    }

    function toggleLiveWatch() {
      isLiveWatch = !isLiveWatch;
      const badge = document.getElementById("live-toggle");
      badge.classList.toggle("off", !isLiveWatch);
      if (isLiveWatch) setupLiveWatch();
      else if (liveInterval) { clearInterval(liveInterval); liveInterval = null; }
    }

    function setupLiveWatch() {
      if (liveInterval) clearInterval(liveInterval);
      liveInterval = setInterval(async () => {
        if (!isLiveWatch) return;
        const expChanged = await fetchExperiments();
        if (activeExp && activeRun && (hasAnyRunningRun() || expChanged)) {
          await loadRunData(activeExp, activeRun, true);
          if (activeTab === "metrics") renderMetricsTab();
          else if (activeTab === "val") renderValidationTab();
          else if (activeTab === "meta") renderMetaTab();
        }
        if (activeTab === "bundles") {
          await fetchBundles();
        }
      }, 2500);
    }

    async function refreshData(force = true) {
      loadedRunsData = {};
      await fetchExperiments(force);
      await fetchBundles(force);
      if (activeExp && activeRun) await loadRunData(activeExp, activeRun, true);
      renderActiveTab();
    }

    function setupKeyboard() {
      window.addEventListener("keydown", (e) => {
        const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
        if (e.key === "Escape") { closeLightbox(); return; }
        if (tag === "input" || tag === "select" || tag === "textarea") return;

        if (e.key === "1") switchTab("metrics");
        else if (e.key === "2") switchTab("val");
        else if (e.key === "3") switchTab("bundles");
        else if (e.key === "4") switchTab("meta");
        else if (e.key === "r") refreshData(true);
        else if (activeTab === "val" && e.key === "ArrowLeft") prevStep();
        else if (activeTab === "val" && e.key === "ArrowRight") nextStep();
        else if ((e.key === "j" || e.key === "k") && orderedRunKeys.length > 0 && activeExp && activeRun) {
          const curIdx = orderedRunKeys.indexOf(`${activeExp}:${activeRun}`);
          if (curIdx !== -1) {
            const nextIdx = e.key === "j"
              ? Math.min(orderedRunKeys.length - 1, curIdx + 1)
              : Math.max(0, curIdx - 1);
            const targetKey = orderedRunKeys[nextIdx];
            const sep = targetKey.indexOf(":");
            selectSingleRun(targetKey.slice(0, sep), targetKey.slice(sep + 1));
          }
        }
      });

      let resizeTimeout;
      window.addEventListener("resize", () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => { if (activeTab === "metrics") renderCharts(); }, 150);
      });
    }

    window.onload = init;
  </script>
</body>
</html>
"""
    return html.replace("__MANGOSTA_VERSION__", _get_mangosta_version())

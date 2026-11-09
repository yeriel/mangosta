"""Ephemeral, zero-daemon interactive visualizer and REST server for mangosta experiments.

Provides an "Industrial Dark / Craft DX" local dashboard featuring:
- Multi-run metric comparison with EMA smoothing and hover crosshairs.
- Validation data inspection with epoch scrubber (timelapse), A/B/GT side-by-side comparison, and zoomable lightbox.
- SafeTensors / ONNX bundle inspector and hyperparameter divergence diffing.
- Live Watch auto-refreshing polling mode.
"""

from __future__ import annotations

import http.server
import json
import re
import socketserver
import struct
import urllib.parse
import webbrowser
import zipfile
from pathlib import Path
from typing import Any

from mangosta.artifacts.bundle import read_bundle_manifest


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
    return {
        "run_name": run_dir.name,
        "status": meta.get("status", "unknown"),
        "start_time": meta.get("start_time"),
        "dataset_hash": meta.get("dataset_hash"),
        "git_commit": meta.get("git_commit"),
        "artifacts_count": artifacts_count,
    }


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
                results[child.name] = runs

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

    return {
        "experiment": experiment or meta.get("experiment", root_dir.name),
        "run_name": run_name or run_dir.name,
        "status": meta.get("status", "unknown"),
        "start_time": meta.get("start_time"),
        "end_time": meta.get("end_time"),
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

        try:
            manifest = read_bundle_manifest(p)
            family = manifest.get("family", family)
            version = manifest.get("version", version)
            format_type = manifest.get("weights_format", format_type)
        except Exception:
            pass

        bundles.append({
            "path": rel_path,
            "filename": p.name,
            "family": family,
            "version": version,
            "format": format_type,
            "size_bytes": size_bytes,
        })
    return bundles


def _load_bundle_details(bundle_path: Path) -> dict[str, Any] | None:
    """Load detailed bundle inspection payload including layer shapes and SafeTensors header."""
    if not bundle_path.is_file():
        return None
    try:
        with zipfile.ZipFile(bundle_path, "r") as zf:
            namelist = zf.namelist()
            manifest_raw = zf.read("manifest.json").decode("utf-8")
            manifest = json.loads(manifest_raw)

            tensors: list[dict[str, Any]] = []
            if "model.safetensors" in namelist:
                with zf.open("model.safetensors") as sf:
                    header_len_bytes = sf.read(8)
                    if len(header_len_bytes) == 8:
                        header_len = struct.unpack("<Q", header_len_bytes)[0]
                        header_json_bytes = sf.read(header_len)
                        raw_tensors = json.loads(header_json_bytes.decode("utf-8"))
                        for k, v in raw_tensors.items():
                            if k != "__metadata__" and isinstance(v, dict):
                                tensors.append({
                                    "name": k,
                                    "shape": v.get("shape", []),
                                    "dtype": v.get("dtype", "unknown"),
                                })

            config_content = None
            if "config.yaml" in namelist:
                config_content = zf.read("config.yaml").decode("utf-8")

            return {
                "file": bundle_path.name,
                "path": str(bundle_path),
                "family": manifest.get("family", "unknown"),
                "version": manifest.get("version", "unknown"),
                "format": manifest.get("weights_format", "unknown"),
                "weights_sha256": manifest.get("weights_sha256") or manifest.get("model_sha256"),
                "is_checkpoint": manifest.get("is_checkpoint", False),
                "epoch": manifest.get("epoch"),
                "step": manifest.get("step"),
                "metrics": manifest.get("metrics", {}),
                "schema": manifest.get("schema"),
                "env": manifest.get("env", {}),
                "config": config_content,
                "tensors": tensors,
                "files": [f for f in namelist if not f.endswith("/")],
            }
    except Exception as e:
        return {"error": str(e), "file": bundle_path.name}


def create_viewer_handler(root_dir: Path) -> type[http.server.BaseHTTPRequestHandler]:
    """Factory creating an HTTP request handler for the interactive visualizer."""
    resolved_root = root_dir.resolve()

    class MangostaViewerHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Suppress normal HTTP logging for clean terminal
            pass

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)

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
                    Path(bundle_rel),
                    (resolved_root / bundle_rel),
                    (resolved_root.parent / bundle_rel),
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
                    # Fallback if root_dir is parent workspace or already experiment/run folder
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
    print("MANGOSTA LOCAL EXPERIMENT VIEWER (v1.0.0)")
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
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>mangosta view — Industrial Craft DX</title>
  <style>
    :root {
      --bg: #080b10;
      --sidebar: #0d121c;
      --card: #121824;
      --card-hover: #161e2e;
      --border: #1e293b;
      --border-subtle: #162030;
      --text: #f1f5f9;
      --text-muted: #8492a6;
      --accent: #06b6d4;
      --accent-glow: rgba(6, 182, 212, 0.15);
      --amber: #f59e0b;
      --emerald: #10b981;
      --purple: #a855f7;
      --rose: #f43f5e;
      --font-mono: ui-monospace, "SF Mono", "JetBrains Mono", "Geist Mono", Menlo, Consolas, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }
    /* Top Navigation Bar */
    header {
      height: 54px;
      background: var(--sidebar);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      flex-shrink: 0;
      z-index: 10;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 1.1rem;
      letter-spacing: -0.02em;
    }
    .brand span.logo { color: #ffffff; }
    .brand span.dot { color: var(--amber); }
    .brand span.tag {
      font-size: 0.7rem;
      font-family: var(--font-mono);
      background: rgba(6, 182, 212, 0.12);
      color: var(--accent);
      border: 1px solid rgba(6, 182, 212, 0.3);
      padding: 2px 7px;
      border-radius: 9999px;
      font-weight: 600;
    }
    .nav-tabs {
      display: flex;
      gap: 6px;
    }
    .tab-btn {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .tab-btn:hover { color: var(--text); background: rgba(255,255,255,0.04); }
    .tab-btn.active {
      color: #ffffff;
      background: #1e293b;
      border-color: #334155;
    }
    .top-actions {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .live-badge {
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.1);
      color: var(--emerald);
      border: 1px solid rgba(16, 185, 129, 0.25);
      cursor: pointer;
      user-select: none;
    }
    .live-pulse {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--emerald);
      box-shadow: 0 0 8px var(--emerald);
      animation: pulse 2s infinite;
    }
    .live-badge.off {
      background: rgba(148, 163, 184, 0.1);
      color: var(--text-muted);
      border-color: var(--border);
    }
    .live-badge.off .live-pulse {
      background: var(--text-muted);
      box-shadow: none;
      animation: none;
    }
    @keyframes pulse {
      0% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
      100% { opacity: 1; transform: scale(1); }
    }
    /* Layout */
    .app-body {
      display: flex;
      flex: 1;
      height: calc(100vh - 54px);
      overflow: hidden;
    }
    /* Sidebar */
    #sidebar {
      width: 320px;
      background: var(--sidebar);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }
    .sidebar-search {
      padding: 12px 14px;
      border-bottom: 1px solid var(--border-subtle);
    }
    .sidebar-search input {
      width: 100%;
      background: #080b10;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 11px;
      color: var(--text);
      font-size: 0.85rem;
      outline: none;
    }
    .sidebar-search input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-glow);
    }
    #runs-list {
      flex: 1;
      overflow-y: auto;
      padding: 10px;
    }
    .exp-group { margin-bottom: 14px; }
    .exp-title {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      font-weight: 700;
      padding: 4px 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .run-item {
      padding: 8px 10px;
      border-radius: 6px;
      cursor: pointer;
      margin-bottom: 3px;
      background: var(--card);
      border: 1px solid transparent;
      display: flex;
      align-items: center;
      gap: 10px;
      transition: all 0.12s ease;
    }
    .run-item:hover {
      background: var(--card-hover);
      border-color: var(--border);
    }
    .run-item.active {
      background: #172438;
      border-color: var(--accent);
    }
    .run-checkbox {
      accent-color: var(--accent);
      cursor: pointer;
    }
    .run-info { flex: 1; min-width: 0; }
    .run-name {
      font-size: 0.85rem;
      font-weight: 600;
      font-family: var(--font-mono);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .run-meta {
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-top: 2px;
      display: flex;
      gap: 8px;
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .status-dot.completed { background: var(--emerald); }
    .status-dot.failed { background: var(--rose); }
    .status-dot.interrupted { background: var(--amber); }
    .status-dot.running { background: var(--accent); }
    .status-dot.unknown { background: var(--text-muted); }

    /* Main Area */
    #main-content {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      background: var(--bg);
    }
    .view-panel { display: none; }
    .view-panel.active { display: block; }

    /* Cards and Grids */
    .grid-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
    }
    .card-label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      font-weight: 600;
    }
    .card-val {
      font-size: 1.15rem;
      font-weight: 700;
      margin-top: 5px;
      font-family: var(--font-mono);
      word-break: break-all;
    }
    /* Section Headers */
    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 8px;
    }
    .section-title {
      font-size: 1rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .control-row {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    select, button, input[type="range"] {
      background: #090e17;
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 6px;
      font-size: 0.85rem;
      padding: 6px 10px;
      outline: none;
    }
    button.btn {
      cursor: pointer;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s ease;
    }
    button.btn:hover { background: #1e293b; border-color: #334155; }
    button.btn-primary { background: var(--accent); color: #080b10; border-color: var(--accent); }
    button.btn-primary:hover { background: #22d3ee; }

    /* Metrics Multi-Grid Layout */
    .metrics-section {
      margin-bottom: 28px;
    }
    .group-pill {
      font-size: 0.72rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      font-family: var(--font-mono);
      letter-spacing: 0.05em;
    }
    .train-pill {
      background: rgba(6, 182, 212, 0.12);
      color: var(--accent);
      border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .val-pill {
      background: rgba(16, 185, 129, 0.12);
      color: var(--emerald);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .other-pill {
      background: rgba(168, 85, 247, 0.12);
      color: var(--purple);
      border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .charts-multi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
      gap: 16px;
    }
    .chart-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      position: relative;
      transition: border-color 0.15s ease;
    }
    .chart-card:hover {
      border-color: #334155;
    }
    .chart-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .chart-card-title {
      font-size: 0.95rem;
      font-weight: 700;
      font-family: var(--font-mono);
      color: #ffffff;
    }
    .chart-card-stats {
      display: flex;
      gap: 8px;
      font-size: 0.72rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }
    .metric-svg {
      width: 100%;
      height: 220px;
      background: #06090f;
      border-radius: 6px;
      margin-top: 6px;
      overflow: visible;
    }
    .chart-tooltip {
      position: absolute;
      display: none;
      background: rgba(13, 18, 28, 0.94);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      pointer-events: none;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
      z-index: 100;
      backdrop-filter: blur(4px);
    }
    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 14px;
      font-size: 0.8rem;
      font-family: var(--font-mono);
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }

    /* Validation & Timelapse Scrubber */
    .val-container {
      display: flex;
      gap: 20px;
      min-height: 520px;
    }
    .val-sidebar {
      width: 260px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      flex-shrink: 0;
      max-height: 700px;
      overflow-y: auto;
    }
    .val-sample-item {
      padding: 9px 12px;
      border-radius: 6px;
      cursor: pointer;
      margin-bottom: 6px;
      background: #080b10;
      border: 1px solid transparent;
      transition: all 0.15s ease;
    }
    .val-sample-item:hover { border-color: var(--border); }
    .val-sample-item.active {
      border-color: var(--amber);
      background: rgba(245, 158, 11, 0.08);
    }
    .val-main {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .scrubber-bar {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .scrubber-controls {
      display: flex;
      align-items: center;
      gap: 10px;
      flex: 1;
    }
    .scrubber-slider {
      flex: 1;
      accent-color: var(--amber);
      cursor: pointer;
    }
    .val-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }
    .val-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .val-card-header {
      padding: 10px 14px;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .val-card-img-wrap {
      width: 100%;
      height: 320px;
      background: #05070a;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: zoom-in;
      position: relative;
    }
    .val-card-img-wrap img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    .zoom-hint {
      position: absolute;
      bottom: 8px;
      right: 8px;
      background: rgba(0,0,0,0.6);
      color: #fff;
      font-size: 0.7rem;
      padding: 3px 6px;
      border-radius: 4px;
      pointer-events: none;
      opacity: 0.7;
    }

    /* Lightbox Modal */
    .lightbox-modal {
      position: fixed;
      inset: 0;
      background: rgba(3, 5, 8, 0.95);
      z-index: 999;
      display: none;
      flex-direction: column;
      backdrop-filter: blur(8px);
    }
    .lightbox-modal.active { display: flex; }
    .lightbox-header {
      height: 50px;
      padding: 0 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
    }
    .lightbox-body {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      cursor: grab;
      position: relative;
    }
    .lightbox-img {
      max-width: 90vw;
      max-height: 85vh;
      object-fit: contain;
      transition: transform 0.1s ease-out;
      transform-origin: center center;
    }

    /* Diff and Table Views */
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      font-family: var(--font-mono);
    }
    table.data-table th, table.data-table td {
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border-subtle);
    }
    table.data-table th {
      color: var(--text-muted);
      font-weight: 600;
      background: rgba(255,255,255,0.02);
    }
    table.data-table tr.divergent {
      background: rgba(245, 158, 11, 0.08);
      color: var(--amber);
      font-weight: 600;
    }
    pre.code-block {
      background: #05070a;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
      font-family: var(--font-mono);
      font-size: 0.82rem;
      overflow-x: auto;
      max-height: 480px;
      color: #93c5fd;
    }
    .empty-state {
      padding: 60px 20px;
      text-align: center;
      color: var(--text-muted);
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="logo">mangosta</span>
      <span class="dot">•</span>
      <span class="tag">view v1.0.0</span>
    </div>

    <div class="nav-tabs">
      <button class="tab-btn active" data-tab="metrics" onclick="switchTab('metrics')">Metrics &amp; Runs</button>
      <button class="tab-btn" data-tab="val" onclick="switchTab('val')">Real Data Validation</button>
      <button class="tab-btn" data-tab="bundles" onclick="switchTab('bundles')">Bundle Inspector</button>
      <button class="tab-btn" data-tab="meta" onclick="switchTab('meta')">Run Metadata</button>
    </div>

    <div class="top-actions">
      <div id="live-toggle" class="live-badge" onclick="toggleLiveWatch()">
        <span class="live-pulse"></span>
        <span id="live-text">LIVE WATCH</span>
      </div>
      <button class="btn" onclick="refreshData()">Refresh</button>
    </div>
  </header>

  <div class="app-body">
    <!-- Sidebar -->
    <div id="sidebar">
      <div class="sidebar-search">
        <input type="text" id="run-search" placeholder="Search runs..." oninput="filterRuns(this.value)" />
      </div>
      <div id="runs-list">Loading runs...</div>
    </div>

    <!-- Main Content Panels -->
    <div id="main-content">
      <!-- Tab 1: Metrics & Runs -->
      <div id="tab-metrics" class="view-panel active">
        <div id="metrics-top-meta" class="grid-metrics"></div>

        <!-- Global Chart Toolbar -->
        <div class="card" style="padding: 12px 18px; margin-bottom: 22px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
          <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-weight: 700; font-family: var(--font-mono); font-size: 0.9rem; color: #ffffff;">Metrics Dashboard</span>
            <span id="metrics-count-badge" style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);"></span>
          </div>
          <div class="control-row">
            <span style="font-size:0.8rem; color:var(--text-muted);">EMA Smoothing:</span>
            <input type="range" id="smoothing-slider" min="0" max="0.99" step="0.01" value="0.5" oninput="updateSmoothing(this.value)" style="width:140px;" />
            <span id="smoothing-val" style="font-family:var(--font-mono); font-size:0.8rem; width:35px; color: var(--accent);">0.50</span>
          </div>
        </div>

        <div id="metrics-empty-message" style="display:none;" class="empty-state">No step metrics recorded in selected runs.</div>

        <!-- Training Metrics Section -->
        <div id="train-metrics-section" class="metrics-section" style="display:none;">
          <div class="section-header" style="margin-bottom: 12px;">
            <div class="section-title" style="display: flex; align-items: center; gap: 8px;">
              <span class="group-pill train-pill">TRAIN</span>
              <span>Training Metrics</span>
              <span id="train-count-badge" style="font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono);"></span>
            </div>
          </div>
          <div id="train-charts-grid" class="charts-multi-grid"></div>
        </div>

        <!-- Validation Metrics Section -->
        <div id="val-metrics-section" class="metrics-section" style="display:none;">
          <div class="section-header" style="margin-bottom: 12px;">
            <div class="section-title" style="display: flex; align-items: center; gap: 8px;">
              <span class="group-pill val-pill">VALIDATION</span>
              <span>Validation Metrics</span>
              <span id="val-count-badge" style="font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono);"></span>
            </div>
          </div>
          <div id="val-charts-grid" class="charts-multi-grid"></div>
        </div>

        <!-- General / Other Metrics Section -->
        <div id="other-metrics-section" class="metrics-section" style="display:none;">
          <div class="section-header" style="margin-bottom: 12px;">
            <div class="section-title" style="display: flex; align-items: center; gap: 8px;">
              <span class="group-pill other-pill">GENERAL</span>
              <span>System &amp; Other Metrics</span>
              <span id="other-count-badge" style="font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono);"></span>
            </div>
          </div>
          <div id="other-charts-grid" class="charts-multi-grid"></div>
        </div>

        <div id="chart-tooltip" class="chart-tooltip"></div>

        <div class="section-header" style="margin-top:28px;">
          <div class="section-title">Hyperparameter Divergence / Comparison</div>
        </div>
        <div id="param-diff-table-wrap" class="card" style="padding:0; overflow-x:auto;">
          <table class="data-table" id="param-diff-table">
            <thead><tr><th>Hyperparameter</th><th>Value</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

      <!-- Tab 2: Validation & Real Data Timelapse -->
      <div id="tab-val" class="view-panel">
        <div class="val-container">
          <div class="val-sidebar">
            <div class="section-title" style="font-size:0.85rem; margin-bottom:8px;">Validation Samples</div>
            <div id="val-samples-list">No image samples logged for this run.</div>
          </div>

          <div class="val-main">
            <div class="scrubber-bar">
              <div class="scrubber-controls">
                <button class="btn" onclick="prevStep()">Prev</button>
                <button class="btn btn-primary" id="play-btn" onclick="togglePlayTimelapse()">Play Timelapse</button>
                <button class="btn" onclick="nextStep()">Next</button>
                <input type="range" id="epoch-scrubber" class="scrubber-slider" min="0" max="0" value="0" oninput="onScrubStep(this.value)" />
              </div>
              <div style="font-family:var(--font-mono); font-weight:700; font-size:0.95rem; color:var(--amber);" id="scrubber-step-indicator">
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
          <div class="val-sidebar" style="width:300px;">
            <div class="section-title" style="font-size:0.85rem; margin-bottom:8px;">Discovered Bundles</div>
            <div id="bundles-list">Scanning workspace...</div>
          </div>

          <div class="val-main" id="bundle-inspector-main">
            <div class="empty-state">Select a .bundle or .onnx.bundle file from the list to inspect contracts and SafeTensors layers.</div>
          </div>
        </div>
      </div>

      <!-- Tab 4: Metadata & Environment -->
      <div id="tab-meta" class="view-panel">
        <div class="section-header">
          <div class="section-title">Raw Run Metadata & Environment</div>
          <button class="btn" onclick="copyMetaJSON()">Copy JSON</button>
        </div>
        <pre class="code-block" id="meta-json-pre">Select a run to view metadata.</pre>
      </div>
    </div>
  </div>

  <!-- Lightbox Modal -->
  <div id="lightbox" class="lightbox-modal">
    <div class="lightbox-header">
      <div id="lightbox-title" style="font-family:var(--font-mono); font-weight:600;">Sample Inspection</div>
      <div class="control-row">
        <button class="btn" onclick="zoomLightbox(0.25)">Zoom +</button>
        <button class="btn" onclick="zoomLightbox(-0.25)">Zoom -</button>
        <button class="btn" onclick="resetLightboxZoom()">1:1 Reset</button>
        <button class="btn btn-primary" onclick="closeLightbox()">Close (Esc)</button>
      </div>
    </div>
    <div class="lightbox-body" id="lightbox-body" onmousedown="startDragLightbox(event)">
      <img id="lightbox-img" class="lightbox-img" src="" alt="high-res" />
    </div>
  </div>

  <script>
    // Global State
    let experimentsData = {};
    let activeExp = null;
    let activeRun = null;
    let selectedRuns = new Set(); // Multi-run comparison set
    let loadedRunsData = {}; // Cache: { "exp:run": data }
    let activeTab = "metrics";
    let isLiveWatch = true;
    let liveInterval = null;
    let smoothingWeight = 0.5;

    // Validation Scrubber State
    let currentSampleId = null;
    let currentSampleTimeline = []; // [{ step, input, gt, pred }]
    let currentTimelineStepIdx = 0;
    let playTimer = null;

    // Lightbox State
    let lightboxScale = 1.0;

    const PALETTE = ["#06b6d4", "#f59e0b", "#a855f7", "#10b981", "#f43f5e", "#3b82f6", "#ec4899", "#84cc16"];

    async function init() {
      await fetchExperiments();
      await fetchBundles();
      setupLiveWatch();
      setupKeyboard();
    }

    function switchTab(tabId) {
      activeTab = tabId;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tabId));
      document.querySelectorAll(".view-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${tabId}`));
      if (tabId === "metrics") renderMetricsTab();
      if (tabId === "val") renderValidationTab();
      if (tabId === "bundles") fetchBundles();
      if (tabId === "meta") renderMetaTab();
    }

    async function fetchExperiments() {
      try {
        const res = await fetch("/api/experiments");
        experimentsData = await res.json();
        renderSidebar();
      } catch (e) {
        console.error("Error fetching experiments:", e);
      }
    }

    function renderSidebar() {
      const container = document.getElementById("runs-list");
      const searchVal = document.getElementById("run-search").value.toLowerCase();
      container.innerHTML = "";

      const expNames = Object.keys(experimentsData);
      if (expNames.length === 0) {
        container.innerHTML = "<div class='empty-state'>No experiment runs found.</div>";
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
        group.innerHTML = `<div class="exp-title"><span>${expName}</span><span>${runs.length}</span></div>`;

        runs.forEach((r, idx) => {
          const key = `${expName}:${r.run_name}`;
          if (!firstExpRun) firstExpRun = { exp: expName, run: r.run_name };

          const item = document.createElement("div");
          item.className = `run-item ${activeExp === expName && activeRun === r.run_name ? "active" : ""}`;
          item.id = `run-item-${expName}-${r.run_name}`;

          const isChecked = selectedRuns.has(key);

          item.innerHTML = `
            <input type="checkbox" class="run-checkbox" ${isChecked ? "checked" : ""} onclick="onToggleRunCheck(event, '${expName}', '${r.run_name}')" />
            <span class="status-dot ${r.status}"></span>
            <div class="run-info">
              <div class="run-name">${r.run_name}</div>
              <div class="run-meta">
                <span>${r.artifacts_count} artifacts</span>
                <span>${(r.git_commit || '').slice(0, 7)}</span>
              </div>
            </div>
          `;

          item.onclick = (e) => {
            if (e.target.tagName.toLowerCase() === "input") return;
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

    function filterRuns(val) {
      renderSidebar();
    }

    async function selectSingleRun(exp, run) {
      activeExp = exp;
      activeRun = run;
      const key = `${exp}:${run}`;
      if (selectedRuns.size <= 1) {
        selectedRuns.clear();
        selectedRuns.add(key);
      }

      document.querySelectorAll(".run-item").forEach(el => el.classList.remove("active"));
      const el = document.getElementById(`run-item-${exp}-${run}`);
      if (el) el.classList.add("active");

      await loadRunData(exp, run);
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
        const [e, r] = singleKey.split(":");
        activeExp = e;
        activeRun = r;
      }
      await syncSelectedRunsData();
      renderActiveTab();
    }

    async function loadRunData(exp, run) {
      const key = `${exp}:${run}`;
      if (loadedRunsData[key]) return loadedRunsData[key];
      try {
        const res = await fetch(`/api/run?exp=${encodeURIComponent(exp)}&run=${encodeURIComponent(run)}`);
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
        const [e, r] = k.split(":");
        return loadRunData(e, r);
      });
      await Promise.all(promises);
    }

    function renderActiveTab() {
      if (activeTab === "metrics") renderMetricsTab();
      else if (activeTab === "val") renderValidationTab();
      else if (activeTab === "bundles") fetchBundles();
      else if (activeTab === "meta") renderMetaTab();
    }

    // ----------------------------------------------------
    // TAB 1: METRICS & MULTI-RUN COMPARISON
    // ----------------------------------------------------
    function categorizeMetric(name) {
      const lower = name.toLowerCase();
      if (lower.startsWith("train/") || lower.startsWith("train_") || lower.startsWith("train:") || lower.includes("train")) {
        return "train";
      }
      if (lower.startsWith("val/") || lower.startsWith("val_") || lower.startsWith("val:") || lower.startsWith("eval/") || lower.startsWith("eval_") || lower.includes("val") || lower.includes("eval") || lower.includes("dice") || lower.includes("iou")) {
        return "val";
      }
      if (lower === "loss" || lower === "epoch_loss") {
        return "train";
      }
      return "other";
    }

    function safeMetricId(name) {
      return name.replace(/[^a-zA-Z0-9_-]/g, "_");
    }

    function renderMetricsTab() {
      const activeData = loadedRunsData[`${activeExp}:${activeRun}`];
      const topMeta = document.getElementById("metrics-top-meta");
      if (!activeData) {
        topMeta.innerHTML = "<div class='empty-state'>Select a run to view metrics.</div>";
        return;
      }

      topMeta.innerHTML = `
        <div class="card">
          <div class="card-label">Run Name</div>
          <div class="card-val" style="font-size:1rem; color:var(--accent);">${activeData.run_name}</div>
        </div>
        <div class="card">
          <div class="card-label">Status / Duration</div>
          <div class="card-val" style="font-size:0.95rem;">${activeData.status}</div>
        </div>
        <div class="card">
          <div class="card-label">Dataset SHA-256</div>
          <div class="card-val" style="font-size:0.85rem;">${(activeData.dataset_hash || 'N/A').slice(0, 16)}...</div>
        </div>
        <div class="card">
          <div class="card-label">Git Commit</div>
          <div class="card-val" style="font-size:0.9rem;">${(activeData.git_commit || 'N/A').slice(0, 8)}</div>
        </div>
      `;

      renderAllMetricCharts();
      renderParamDiffTable();
    }

    function renderAllMetricCharts() {
      const metricSet = new Set();
      selectedRuns.forEach(k => {
        const d = loadedRunsData[k];
        if (d && d.metrics) {
          Object.keys(d.metrics).forEach(m => metricSet.add(m));
        }
      });

      const emptyMsg = document.getElementById("metrics-empty-message");
      const trainSec = document.getElementById("train-metrics-section");
      const valSec = document.getElementById("val-metrics-section");
      const otherSec = document.getElementById("other-metrics-section");

      const trainGrid = document.getElementById("train-charts-grid");
      const valGrid = document.getElementById("val-charts-grid");
      const otherGrid = document.getElementById("other-charts-grid");

      const countBadge = document.getElementById("metrics-count-badge");

      if (metricSet.size === 0) {
        if (emptyMsg) emptyMsg.style.display = "block";
        if (trainSec) trainSec.style.display = "none";
        if (valSec) valSec.style.display = "none";
        if (otherSec) otherSec.style.display = "none";
        if (countBadge) countBadge.textContent = "0 metrics";
        return;
      }

      if (emptyMsg) emptyMsg.style.display = "none";
      if (countBadge) countBadge.textContent = `${metricSet.size} metrics across ${selectedRuns.size} run(s)`;

      const allMetrics = Array.from(metricSet).sort();
      const trainMetrics = [];
      const valMetrics = [];
      const otherMetrics = [];

      allMetrics.forEach(m => {
        const cat = categorizeMetric(m);
        if (cat === "train") trainMetrics.push(m);
        else if (cat === "val") valMetrics.push(m);
        else otherMetrics.push(m);
      });

      // Render Train Section
      if (trainMetrics.length > 0) {
        trainSec.style.display = "block";
        document.getElementById("train-count-badge").textContent = `(${trainMetrics.length})`;
        renderMetricGrid(trainGrid, trainMetrics);
      } else {
        trainSec.style.display = "none";
        trainGrid.innerHTML = "";
      }

      // Render Val Section
      if (valMetrics.length > 0) {
        valSec.style.display = "block";
        document.getElementById("val-count-badge").textContent = `(${valMetrics.length})`;
        renderMetricGrid(valGrid, valMetrics);
      } else {
        valSec.style.display = "none";
        valGrid.innerHTML = "";
      }

      // Render Other Section
      if (otherMetrics.length > 0) {
        otherSec.style.display = "block";
        document.getElementById("other-count-badge").textContent = `(${otherMetrics.length})`;
        renderMetricGrid(otherGrid, otherMetrics);
      } else {
        otherSec.style.display = "none";
        otherGrid.innerHTML = "";
      }
    }

    function renderMetricGrid(gridEl, metricList) {
      gridEl.innerHTML = metricList.map(m => {
        const safeId = safeMetricId(m);
        return `
          <div class="chart-card" id="card-${safeId}">
            <div class="chart-card-header">
              <div class="chart-card-title">${m}</div>
              <div class="chart-card-stats" id="stats-${safeId}"></div>
            </div>
            <svg class="metric-svg" id="svg-${safeId}"></svg>
            <div class="chart-legend" id="legend-${safeId}"></div>
          </div>
        `;
      }).join("");

      // Render each SVG chart in this grid
      metricList.forEach(m => {
        const safeId = safeMetricId(m);
        const svg = document.getElementById(`svg-${safeId}`);
        const legend = document.getElementById(`legend-${safeId}`);
        const stats = document.getElementById(`stats-${safeId}`);
        if (svg) renderMetricSVG(m, svg, legend, stats);
      });
    }

    function updateSmoothing(val) {
      smoothingWeight = parseFloat(val);
      document.getElementById("smoothing-val").textContent = smoothingWeight.toFixed(2);
      renderAllMetricCharts();
    }

    function computeEMA(points, weight) {
      if (weight <= 0 || points.length === 0) return points;
      let prev = points[0].value;
      return points.map(p => {
        prev = prev * weight + p.value * (1 - weight);
        return { step: p.step, value: prev };
      });
    }

    function renderMetricSVG(metricName, svg, legend, stats) {
      const runs = Array.from(selectedRuns).map(k => loadedRunsData[k]).filter(d => d && d.metrics && d.metrics[metricName]);
      if (runs.length === 0) {
        svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="#64748b" font-family="var(--font-mono)" font-size="11">No data in selected runs</text>`;
        if (legend) legend.innerHTML = "";
        if (stats) stats.innerHTML = "";
        return;
      }

      const width = svg.clientWidth || 440;
      const height = 220;
      const padL = 55, padR = 20, padT = 20, padB = 30;

      // Calculate global min/max step and value across all lines
      let allSteps = [];
      let allVals = [];

      runs.forEach(r => {
        const pts = r.metrics[metricName];
        pts.forEach(p => {
          allSteps.push(p.step);
          allVals.push(p.value);
        });
      });

      const minS = Math.min(...allSteps);
      const maxS = Math.max(...allSteps);
      let minV = Math.min(...allVals);
      let maxV = Math.max(...allVals);

      if (minV === maxV) { minV -= 0.1; maxV += 0.1; }
      const stepRange = (maxS - minS) || 1;
      const valRange = (maxV - minV) || 1;

      const getX = s => padL + ((s - minS) / stepRange) * (width - padL - padR);
      const getY = v => (height - padB) - ((v - minV) / valRange) * (height - padT - padB);

      let gridSvg = `
        <line x1="${padL}" y1="${height - padB}" x2="${width - padR}" y2="${height - padB}" stroke="#1e293b" stroke-width="1.5" />
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${height - padB}" stroke="#1e293b" stroke-width="1.5" />
        <!-- Y Axis ticks -->
        <text x="${padL - 8}" y="${padT + 4}" fill="#64748b" font-family="var(--font-mono)" font-size="10" text-anchor="end">${maxV.toFixed(3)}</text>
        <text x="${padL - 8}" y="${(padT + height - padB)/2 + 4}" fill="#64748b" font-family="var(--font-mono)" font-size="10" text-anchor="end">${((maxV + minV)/2).toFixed(3)}</text>
        <text x="${padL - 8}" y="${height - padB}" fill="#64748b" font-family="var(--font-mono)" font-size="10" text-anchor="end">${minV.toFixed(3)}</text>
        <!-- Horizontal gridlines -->
        <line x1="${padL}" y1="${padT}" x2="${width - padR}" y2="${padT}" stroke="#131d2e" stroke-dasharray="3,3" />
        <line x1="${padL}" y1="${(padT + height - padB)/2}" x2="${width - padR}" y2="${(padT + height - padB)/2}" stroke="#131d2e" stroke-dasharray="3,3" />
        <!-- X Axis labels -->
        <text x="${padL}" y="${height - padB + 18}" fill="#64748b" font-family="var(--font-mono)" font-size="10">step ${minS}</text>
        <text x="${width - padR}" y="${height - padB + 18}" fill="#64748b" font-family="var(--font-mono)" font-size="10" text-anchor="end">step ${maxS}</text>
      `;

      let linesSvg = "";
      let legendHtml = "";
      let latestValues = [];

      runs.forEach((r, idx) => {
        const color = PALETTE[idx % PALETTE.length];
        const rawPoints = r.metrics[metricName];
        const smoothed = computeEMA(rawPoints, smoothingWeight);

        // Faint raw line if smoothed
        if (smoothingWeight > 0.05) {
          let rawD = "";
          rawPoints.forEach((p, i) => {
            const x = getX(p.step);
            const y = getY(p.value);
            rawD += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
          });
          linesSvg += `<path d="${rawD}" fill="none" stroke="${color}" stroke-width="1" stroke-opacity="0.25" />`;
        }

        // Main curve
        let d = "";
        smoothed.forEach((p, i) => {
          const x = getX(p.step);
          const y = getY(p.value);
          d += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
        });

        linesSvg += `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.2" />`;

        const lastVal = rawPoints[rawPoints.length - 1].value;
        latestValues.push(lastVal);
        legendHtml += `
          <div class="legend-item">
            <span class="legend-dot" style="background:${color};"></span>
            <span style="font-weight:600; color:#fff;">${r.run_name}</span>
            <span style="color:var(--text-muted); font-size:0.75rem;">last: ${lastVal.toFixed(4)}</span>
          </div>
        `;
      });

      svg.innerHTML = gridSvg + linesSvg;
      if (legend) legend.innerHTML = legendHtml;
      if (stats && latestValues.length > 0) {
        stats.innerHTML = `<span style="color:var(--accent);">curr: ${latestValues[0].toFixed(4)}</span> <span style="color:var(--text-muted);">| min: ${minV.toFixed(4)}</span>`;
      }

      // Mouse crosshair tracking
      svg.onmousemove = (e) => {
        const rect = svg.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        if (mouseX < padL || mouseX > width - padR) return;

        const ratio = (mouseX - padL) / (width - padL - padR);
        const targetStep = Math.round(minS + ratio * stepRange);

        // Render vertical crosshair line
        const crossLine = `<line id="crosshair-v-${safeMetricId(metricName)}" x1="${mouseX}" y1="${padT}" x2="${mouseX}" y2="${height - padB}" stroke="#06b6d4" stroke-width="1" stroke-dasharray="2,2" />`;
        const oldCross = svg.querySelector(`[id^="crosshair-v"]`);
        if (oldCross) oldCross.remove();
        svg.insertAdjacentHTML("beforeend", crossLine);

        // Populate Tooltip
        const tip = document.getElementById("chart-tooltip");
        let tipRows = `<strong>${metricName}</strong><div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:4px;">Step: ${targetStep}</div>`;
        runs.forEach((r, idx) => {
          const color = PALETTE[idx % PALETTE.length];
          const closest = r.metrics[metricName].reduce((prev, curr) => Math.abs(curr.step - targetStep) < Math.abs(prev.step - targetStep) ? curr : prev);
          tipRows += `<div style="display:flex; justify-content:space-between; gap:12px; align-items:center;"><span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${color}; margin-right:6px;"></span>${r.run_name}</span><strong>${closest.value.toFixed(4)}</strong></div>`;
        });
        tip.innerHTML = tipRows;
        tip.style.display = "block";
        tip.style.left = `${e.pageX + 14}px`;
        tip.style.top = `${e.pageY - 20}px`;
      };

      svg.onmouseleave = () => {
        const oldCross = svg.querySelector(`[id^="crosshair-v"]`);
        if (oldCross) oldCross.remove();
        document.getElementById("chart-tooltip").style.display = "none";
      };
    }

    function renderParamDiffTable() {
      const tbody = document.querySelector("#param-diff-table tbody");
      const thead = document.querySelector("#param-diff-table thead tr");
      const runs = Array.from(selectedRuns).map(k => loadedRunsData[k]).filter(Boolean);

      if (runs.length === 0) return;

      // Build Table Header
      thead.innerHTML = `<th>Hyperparameter</th>` + runs.map(r => `<th>${r.run_name}</th>`).join("");

      // Collect union of all param keys
      const allKeys = new Set();
      runs.forEach(r => Object.keys(r.params || {}).forEach(k => allKeys.add(k)));

      tbody.innerHTML = "";
      if (allKeys.size === 0) {
        tbody.innerHTML = `<tr><td colspan="${runs.length + 1}" style="color:var(--text-muted); text-align:center;">No parameters logged.</td></tr>`;
        return;
      }

      Array.from(allKeys).sort().forEach(k => {
        const vals = runs.map(r => r.params ? r.params[k] : undefined);
        const isDivergent = runs.length > 1 && vals.some(v => JSON.stringify(v) !== JSON.stringify(vals[0]));

        const tr = document.createElement("tr");
        if (isDivergent) tr.className = "divergent";

        tr.innerHTML = `<td>${k}</td>` + vals.map(v => `<td>${v !== undefined ? JSON.stringify(v) : '<span style="color:var(--text-muted);">null</span>'}</td>`).join("");
        tbody.appendChild(tr);
      });
    }

    // ----------------------------------------------------
    // TAB 2: REAL DATA VALIDATION & TIMELAPSE SCRUBBER
    // ----------------------------------------------------
    function renderValidationTab() {
      const activeData = loadedRunsData[`${activeExp}:${activeRun}`];
      const samplesListEl = document.getElementById("val-samples-list");
      if (!activeData || !activeData.artifacts) {
        samplesListEl.innerHTML = "<div class='empty-state'>No run loaded.</div>";
        return;
      }

      // Group image artifacts by sample_id
      const images = activeData.artifacts.filter(a => a.type === "image");
      const sampleGroups = {};

      images.forEach(img => {
        const sId = img.sample_id || img.base_name || "sample_default";
        if (!sampleGroups[sId]) sampleGroups[sId] = [];
        sampleGroups[sId].push(img);
      });

      const sampleIds = Object.keys(sampleGroups);
      if (sampleIds.length === 0) {
        samplesListEl.innerHTML = "<div class='empty-state'>No validation image artifacts found in run.</div>";
        document.getElementById("val-grid-cards").innerHTML = `<div class="empty-state">Log evaluation samples with <code>run.log_validation_sample(...)</code> or <code>run.log_image(...)</code> to inspect real visual performance.</div>`;
        return;
      }

      // Render sample selector list
      samplesListEl.innerHTML = "";
      sampleIds.forEach(sId => {
        const count = sampleGroups[sId].length;
        const item = document.createElement("div");
        item.className = `val-sample-item ${sId === currentSampleId ? "active" : ""}`;
        item.innerHTML = `
          <div style="font-weight:600; font-family:var(--font-mono); font-size:0.85rem;">${sId}</div>
          <div style="color:var(--text-muted); font-size:0.75rem; margin-top:2px;">${count} timeline snapshots</div>
        `;
        item.onclick = () => selectValidationSample(sId, sampleGroups[sId]);
        samplesListEl.appendChild(item);
      });

      if (!currentSampleId || !sampleGroups[currentSampleId]) {
        selectValidationSample(sampleIds[0], sampleGroups[sampleIds[0]]);
      } else {
        selectValidationSample(currentSampleId, sampleGroups[currentSampleId]);
      }
    }

    function selectValidationSample(sId, artifacts) {
      currentSampleId = sId;
      document.querySelectorAll(".val-sample-item").forEach(el => el.classList.remove("active"));

      // Build step-aligned timeline
      // Find all distinct steps
      const stepMap = {};
      artifacts.forEach(a => {
        const s = a.step !== null && a.step !== undefined ? a.step : 0;
        if (!stepMap[s]) stepMap[s] = {};
        stepMap[s][a.tag || "prediction"] = a;
      });

      const sortedSteps = Object.keys(stepMap).map(Number).sort((a,b) => a - b);
      currentSampleTimeline = sortedSteps.map(s => ({ step: s, ...stepMap[s] }));

      // Setup scrubber slider
      const scrubber = document.getElementById("epoch-scrubber");
      scrubber.min = 0;
      scrubber.max = Math.max(0, currentSampleTimeline.length - 1);
      currentTimelineStepIdx = Math.min(currentTimelineStepIdx, currentSampleTimeline.length - 1);
      scrubber.value = currentTimelineStepIdx;

      renderTimelineStep();
    }

    function onScrubStep(idx) {
      currentTimelineStepIdx = parseInt(idx);
      renderTimelineStep();
    }

    function prevStep() {
      if (currentTimelineStepIdx > 0) {
        currentTimelineStepIdx--;
        document.getElementById("epoch-scrubber").value = currentTimelineStepIdx;
        renderTimelineStep();
      }
    }

    function nextStep() {
      if (currentTimelineStepIdx < currentSampleTimeline.length - 1) {
        currentTimelineStepIdx++;
        document.getElementById("epoch-scrubber").value = currentTimelineStepIdx;
        renderTimelineStep();
      }
    }

    function togglePlayTimelapse() {
      const btn = document.getElementById("play-btn");
      if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
        btn.textContent = "Play Timelapse";
      } else {
        btn.textContent = "Pause";
        playTimer = setInterval(() => {
          if (currentTimelineStepIdx >= currentSampleTimeline.length - 1) {
            currentTimelineStepIdx = 0;
          } else {
            currentTimelineStepIdx++;
          }
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

      // Columns: Input, Ground Truth, Prediction
      const roles = [
        { key: "input", title: "Real Validation Input", badge: "INPUT", color: "var(--accent)" },
        { key: "ground_truth", title: "Ground Truth Target", badge: "TARGET", color: "var(--emerald)" },
        { key: "prediction", title: `Model Prediction (Step ${current.step})`, badge: "PREDICTION", color: "var(--amber)" },
      ];

      grid.innerHTML = roles.map(r => {
        const art = current[r.key] || current["image"];
        if (!art && r.key !== "prediction") return "";
        const imgSrc = art ? `/api/artifact?exp=${encodeURIComponent(activeData.experiment)}&run=${encodeURIComponent(activeData.run_name)}&file=${encodeURIComponent(art.name)}` : "";

        return `
          <div class="val-card">
            <div class="val-card-header">
              <span style="color:${r.color};">${r.title}</span>
              <span class="status-dot completed" style="background:${r.color};"></span>
            </div>
            <div class="val-card-img-wrap" onclick="openLightbox('${imgSrc}', '${art ? art.name : ''}')">
              ${imgSrc ? `<img src="${imgSrc}" alt="${r.title}" />` : `<div style="color:var(--text-muted); font-size:0.8rem;">No ${r.key} logged</div>`}
              <span class="zoom-hint">Click to zoom</span>
            </div>
          </div>
        `;
      }).join("");
    }

    // ----------------------------------------------------
    // TAB 3: BUNDLE INSPECTOR
    // ----------------------------------------------------
    async function fetchBundles() {
      try {
        const res = await fetch("/api/bundles");
        const data = await res.json();
        const listEl = document.getElementById("bundles-list");
        listEl.innerHTML = "";

        if (!data.bundles || data.bundles.length === 0) {
          listEl.innerHTML = "<div class='empty-state'>No .bundle or .onnx.bundle files found in workspace.</div>";
          return;
        }

        data.bundles.forEach((b, idx) => {
          const item = document.createElement("div");
          item.className = "val-sample-item";
          item.innerHTML = `
            <div style="font-weight:700; font-family:var(--font-mono); font-size:0.85rem; color:#fff;">${b.filename}</div>
            <div style="color:var(--text-muted); font-size:0.75rem; margin-top:3px; display:flex; justify-content:space-between;">
              <span>${b.family} (v${b.version})</span>
              <span style="color:var(--accent); font-weight:600;">${b.format.toUpperCase()}</span>
            </div>
          `;
          item.onclick = () => inspectBundle(b.path);
          listEl.appendChild(item);

          if (idx === 0) inspectBundle(b.path);
        });
      } catch (e) {
        console.error("Error fetching bundles:", e);
      }
    }

    async function inspectBundle(bundlePath) {
      const main = document.getElementById("bundle-inspector-main");
      main.innerHTML = "<div class='empty-state'>Inspecting bundle headers...</div>";

      try {
        const res = await fetch(`/api/bundle?path=${encodeURIComponent(bundlePath)}`);
        const b = await res.json();

        main.innerHTML = `
          <div class="card" style="margin-bottom:18px;">
            <div class="section-header" style="border:none; margin:0; padding:0;">
              <div>
                <h2 style="font-size:1.4rem; font-family:var(--font-mono);">${b.file}</h2>
                <div style="color:var(--text-muted); font-size:0.85rem; margin-top:4px;">
                  Family: <strong style="color:#fff;">${b.family}</strong> | Version: <strong style="color:#fff;">${b.version}</strong> | Format: <strong style="color:var(--accent);">${(b.format || 'safetensors').toUpperCase()}</strong>
                </div>
              </div>
              <span class="live-badge off" style="color:var(--amber); border-color:var(--amber);">
                ${b.is_checkpoint ? "TRAINING CHECKPOINT" : "INFERENCE MODEL"}
              </span>
            </div>
          </div>

          <div class="grid-metrics">
            <div class="card">
              <div class="card-label">SHA-256 Fingerprint</div>
              <div class="card-val" style="font-size:0.8rem;">${(b.weights_sha256 || 'N/A').slice(0, 24)}...</div>
            </div>
            <div class="card">
              <div class="card-label">Epoch / Step</div>
              <div class="card-val" style="font-size:1.0rem;">Epoch: ${b.epoch ?? 'N/A'} | Step: ${b.step ?? 'N/A'}</div>
            </div>
            <div class="card">
              <div class="card-label">Weights Layers</div>
              <div class="card-val" style="font-size:1.1rem; color:var(--accent);">${b.tensors ? b.tensors.length : 0} layers</div>
            </div>
          </div>

          ${b.schema ? `
            <div class="section-header"><div class="section-title">ModelSchema Data Contract</div></div>
            <div class="card" style="margin-bottom:20px; padding:0; overflow-x:auto;">
              <table class="data-table">
                <thead><tr><th>Node</th><th>Shape</th><th>Dtype</th></tr></thead>
                <tbody>
                  ${(b.schema.inputs || []).map(inp => `<tr><td>Input: ${inp.name}</td><td>[${inp.shape.map(d => d === null ? 'dynamic' : d).join(', ')}]</td><td>${inp.dtype}</td></tr>`).join('')}
                  ${(b.schema.outputs || []).map(out => `<tr><td>Output: ${out.name}</td><td>[${out.shape.map(d => d === null ? 'dynamic' : d).join(', ')}]</td><td>${out.dtype}</td></tr>`).join('')}
                </tbody>
              </table>
            </div>
          ` : ''}

          ${b.tensors && b.tensors.length > 0 ? `
            <div class="section-header"><div class="section-title">SafeTensors Layer Structure (Zero-Torch Parsed)</div></div>
            <div class="card" style="margin-bottom:20px; padding:0; overflow-x:auto; max-height:280px;">
              <table class="data-table">
                <thead><tr><th>Layer Tensor Name</th><th>Shape</th><th>Dtype</th></tr></thead>
                <tbody>
                  ${b.tensors.map(t => `<tr><td>${t.name}</td><td>[${t.shape.join(', ')}]</td><td>${t.dtype}</td></tr>`).join('')}
                </tbody>
              </table>
            </div>
          ` : ''}

          ${b.config ? `
            <div class="section-header"><div class="section-title">Embedded Configuration (config.yaml)</div></div>
            <pre class="code-block">${b.config}</pre>
          ` : ''}
        `;
      } catch (e) {
        main.innerHTML = `<div class="empty-state" style="color:var(--rose);">Error loading bundle: ${e}</div>`;
      }
    }

    // ----------------------------------------------------
    // TAB 4: METADATA
    // ----------------------------------------------------
    function renderMetaTab() {
      const activeData = loadedRunsData[`${activeExp}:${activeRun}`];
      const pre = document.getElementById("meta-json-pre");
      if (!activeData) {
        pre.textContent = "Select a run to view metadata.";
        return;
      }
      pre.textContent = JSON.stringify(activeData, null, 2);
    }

    function copyMetaJSON() {
      const pre = document.getElementById("meta-json-pre");
      navigator.clipboard.writeText(pre.textContent);
      alert("Run metadata copied to clipboard.");
    }

    // ----------------------------------------------------
    // LIGHTBOX MODAL (ZOOM & PAN)
    // ----------------------------------------------------
    function openLightbox(src, title) {
      if (!src) return;
      document.getElementById("lightbox-img").src = src;
      document.getElementById("lightbox-title").textContent = title || "High-Resolution Sample Inspection";
      document.getElementById("lightbox").classList.add("active");
      resetLightboxZoom();
    }

    function closeLightbox() {
      document.getElementById("lightbox").classList.remove("active");
    }

    function zoomLightbox(delta) {
      lightboxScale = Math.max(0.5, Math.min(5.0, lightboxScale + delta));
      document.getElementById("lightbox-img").style.transform = `scale(${lightboxScale})`;
    }

    function resetLightboxZoom() {
      lightboxScale = 1.0;
      const img = document.getElementById("lightbox-img");
      img.style.transform = "scale(1.0)";
    }

    function startDragLightbox(e) {
      // Allow dragging image when zoomed in
    }

    // ----------------------------------------------------
    // LIVE WATCH & UTILS
    // ----------------------------------------------------
    function toggleLiveWatch() {
      isLiveWatch = !isLiveWatch;
      const badge = document.getElementById("live-toggle");
      badge.classList.toggle("off", !isLiveWatch);
      if (isLiveWatch) {
        setupLiveWatch();
      } else if (liveInterval) {
        clearInterval(liveInterval);
        liveInterval = null;
      }
    }

    function setupLiveWatch() {
      if (liveInterval) clearInterval(liveInterval);
      liveInterval = setInterval(async () => {
        if (!isLiveWatch) return;
        await fetchExperiments();
        if (activeExp && activeRun) {
          // Invalidate cache for active run
          delete loadedRunsData[`${activeExp}:${activeRun}`];
          await loadRunData(activeExp, activeRun);
          renderActiveTab();
        }
      }, 2000);
    }

    async function refreshData() {
      loadedRunsData = {};
      await fetchExperiments();
      await fetchBundles();
      if (activeExp && activeRun) {
        await loadRunData(activeExp, activeRun);
      }
      renderActiveTab();
    }

    function setupKeyboard() {
      window.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeLightbox();
        if (activeTab === "val") {
          if (e.key === "ArrowLeft") prevStep();
          if (e.key === "ArrowRight") nextStep();
        }
      });
      let resizeTimeout;
      window.addEventListener("resize", () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
          if (activeTab === "metrics") renderAllMetricCharts();
        }, 150);
      });
    }

    window.onload = init;
  </script>
</body>
</html>
"""

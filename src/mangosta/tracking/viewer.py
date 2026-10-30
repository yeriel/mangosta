"""Ephemeral, zero-daemon interactive visualizer and REST server for mangosta experiments."""

from __future__ import annotations

import http.server
import json
import socketserver
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

from mangosta.artifacts.bundle import read_bundle_manifest


def _scan_experiments(root_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Scan experiments directory and return available experiments and runs."""
    results: dict[str, list[dict[str, Any]]] = {}
    if not root_dir.exists():
        return results

    for exp_dir in sorted(root_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        runs = []
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue

            meta_file = run_dir / "run.json"
            metrics_file = run_dir / "metrics.jsonl"
            artifacts_dir = run_dir / "artifacts"

            if meta_file.exists() or metrics_file.exists():
                meta: dict[str, Any] = {}
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                artifacts_count = (
                    len([f for f in artifacts_dir.iterdir() if f.is_file()])
                    if artifacts_dir.exists()
                    else 0
                )

                runs.append(
                    {
                        "run_name": run_dir.name,
                        "status": meta.get("status", "unknown"),
                        "start_time": meta.get("start_time"),
                        "dataset_hash": meta.get("dataset_hash"),
                        "git_commit": meta.get("git_commit"),
                        "artifacts_count": artifacts_count,
                    }
                )

        if runs:
            results[exp_dir.name] = runs

    return results


def _load_run_details(root_dir: Path, experiment: str, run_name: str) -> dict[str, Any] | None:
    """Load complete run details including parameters, metrics history, and artifacts."""
    run_dir = root_dir / experiment / run_name
    if not run_dir.exists():
        return None

    meta_file = run_dir / "run.json"
    metrics_file = run_dir / "metrics.jsonl"
    artifacts_dir = run_dir / "artifacts"

    meta: dict[str, Any] = {}
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            pass

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
    artifacts: list[dict[str, Any]] = meta.get("artifacts", [])
    if not artifacts and artifacts_dir.exists():
        artifacts = [
            {
                "name": f.name,
                "type": "image"
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".ppm")
                else "file",
                "size_bytes": f.stat().st_size,
            }
            for f in sorted(artifacts_dir.iterdir())
            if f.is_file()
        ]

    return {
        "experiment": experiment,
        "run_name": run_name,
        "status": meta.get("status", "unknown"),
        "start_time": meta.get("start_time"),
        "end_time": meta.get("end_time"),
        "params": meta.get("params", {}),
        "dataset_hash": meta.get("dataset_hash"),
        "git_commit": meta.get("git_commit"),
        "env": meta.get("env", {}),
        "metrics": metrics,
        "artifacts": artifacts,
    }


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

            # API endpoints
            if path == "/api/experiments":
                data = _scan_experiments(resolved_root)
                self._send_json(data)
                return

            if path == "/api/run":
                exp = query.get("exp", [""])[0]
                run = query.get("run", [""])[0]
                data = _load_run_details(resolved_root, exp, run)
                if data is None:
                    self.send_error(404, "Run not found")
                else:
                    self._send_json(data)
                return

            if path == "/api/artifact":
                exp = query.get("exp", [""])[0]
                run = query.get("run", [""])[0]
                filename = query.get("file", [""])[0]
                artifact_path = resolved_root / exp / run / "artifacts" / filename
                if not artifact_path.is_file():
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

            if path == "/api/bundle":
                bundle_rel = query.get("path", [""])[0]
                target_bundle = (resolved_root / bundle_rel).resolve()
                if not target_bundle.is_file():
                    self.send_error(404, "Bundle not found")
                    return
                try:
                    manifest = read_bundle_manifest(target_bundle)
                    self._send_json(manifest)
                except Exception as e:
                    self.send_error(500, f"Error reading bundle: {e}")
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
    print("MANGOSTA LOCAL EXPERIMENT VIEWER")
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
  <title>mangosta visualizer</title>
  <style>
    :root {
      --bg: #0f172a;
      --sidebar: #1e293b;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --active: #2563eb;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }
    #sidebar {
      width: 320px;
      background: var(--sidebar);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }
    .sidebar-header {
      padding: 18px;
      border-bottom: 1px solid var(--border);
    }
    .sidebar-header h1 {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text);
    }
    .sidebar-header p {
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 2px;
    }
    #runs-list {
      flex: 1;
      overflow-y: auto;
      padding: 12px;
    }
    .exp-group {
      margin-bottom: 16px;
    }
    .exp-title {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 6px;
      padding-left: 6px;
    }
    .run-item {
      padding: 10px 12px;
      border-radius: 6px;
      cursor: pointer;
      margin-bottom: 4px;
      background: #0f172a;
      border: 1px solid transparent;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 0.15s ease;
    }
    .run-item:hover {
      border-color: var(--border);
    }
    .run-item.active {
      background: #1e3a8a;
      border-color: var(--accent);
    }
    .run-name {
      font-size: 0.9rem;
      font-weight: 500;
    }
    .run-badge {
      font-size: 0.7rem;
      padding: 2px 6px;
      border-radius: 4px;
      background: #334155;
      color: var(--muted);
    }
    .run-badge.completed { background: #065f46; color: #a7f3d0; }
    .run-badge.failed { background: #991b1b; color: #fecaca; }

    #main {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
    }
    .top-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }
    .grid-cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }
    .card-label { font-size: 0.8rem; color: var(--muted); }
    .card-val { font-size: 1.15rem; font-weight: 600; margin-top: 4px; word-break: break-all; }

    .section-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .chart-box {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
    }
    select {
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
    }
    svg.chart-svg {
      width: 100%;
      height: 320px;
      background: #090d16;
      border-radius: 6px;
      margin-top: 12px;
    }

    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .gallery-item {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .gallery-item img {
      width: 100%;
      height: 200px;
      object-fit: contain;
      background: #090d16;
    }
    .gallery-meta {
      padding: 10px 12px;
      font-size: 0.85rem;
    }
    pre {
      background: #090d16;
      padding: 14px;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 0.85rem;
      overflow-x: auto;
    }
    .empty-msg {
      color: var(--muted);
      padding: 30px;
      text-align: center;
    }
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="sidebar-header">
      <h1>mangosta view</h1>
      <p>Local Experiment Dashboard</p>
    </div>
    <div id="runs-list">Loading experiments...</div>
  </div>

  <div id="main">
    <div id="content"><div class="empty-msg">Select an experiment run from the sidebar.</div></div>
  </div>

  <script>
    let currentExp = null;
    let currentRun = null;
    let currentData = null;

    async function fetchExperiments() {
      const res = await fetch("/api/experiments");
      const exps = await res.json();
      const container = document.getElementById("runs-list");
      container.innerHTML = "";

      const expNames = Object.keys(exps);
      if (expNames.length === 0) {
        container.innerHTML = "<div class='empty-msg'>No experiments found in directory.</div>";
        return;
      }

      expNames.forEach((expName, idx) => {
        const group = document.createElement("div");
        group.className = "exp-group";
        group.innerHTML = `<div class="exp-title">${expName}</div>`;

        exps[expName].forEach((r, rIdx) => {
          const item = document.createElement("div");
          item.className = "run-item";
          item.id = `item-${expName}-${r.run_name}`;
          item.innerHTML = `
            <div class="run-name">${r.run_name}</div>
            <span class="run-badge ${r.status}">${r.status}</span>
          `;
          item.onclick = () => selectRun(expName, r.run_name);
          group.appendChild(item);

          if (idx === 0 && rIdx === 0 && !currentRun) {
            selectRun(expName, r.run_name);
          }
        });
        container.appendChild(group);
      });
    }

    async function selectRun(exp, run) {
      currentExp = exp;
      currentRun = run;

      document.querySelectorAll(".run-item").forEach(el => el.classList.remove("active"));
      const activeEl = document.getElementById(`item-${exp}-${run}`);
      if (activeEl) activeEl.classList.add("active");

      const res = await fetch(`/api/run?exp=${encodeURIComponent(exp)}&run=${encodeURIComponent(run)}`);
      currentData = await res.json();
      renderRun(currentData);
    }

    function renderRun(data) {
      const content = document.getElementById("content");
      const metricNames = Object.keys(data.metrics || {});
      const images = (data.artifacts || []).filter(a => a.type === "image");

      content.innerHTML = `
        <div class="top-meta">
          <div>
            <h2 style="font-size: 1.5rem; font-weight:700;">${data.run_name}</h2>
            <div style="color:var(--muted); font-size:0.9rem; margin-top:4px;">Experiment: ${data.experiment}</div>
          </div>
          <span class="run-badge ${data.status}" style="font-size:0.9rem; padding:4px 12px;">${data.status}</span>
        </div>

        <div class="grid-cards">
          <div class="card">
            <div class="card-label">Started</div>
            <div class="card-val" style="font-size:0.95rem;">${data.start_time || 'N/A'}</div>
          </div>
          <div class="card">
            <div class="card-label">Dataset SHA-256</div>
            <div class="card-val" style="font-family:monospace; font-size:0.9rem;">${(data.dataset_hash || 'N/A').slice(0, 16)}...</div>
          </div>
          <div class="card">
            <div class="card-label">Git Commit</div>
            <div class="card-val" style="font-family:monospace; font-size:0.9rem;">${(data.git_commit || 'N/A').slice(0, 8)}</div>
          </div>
          <div class="card">
            <div class="card-label">Environment</div>
            <div class="card-val" style="font-size:0.9rem;">Py ${data.env?.python_version || '?'} | Torch ${data.env?.torch_version || '?'}</div>
          </div>
        </div>

        <div class="chart-box">
          <div class="section-title">
            <span>Metric History</span>
            <select id="metric-select" onchange="drawChart(this.value)"></select>
          </div>
          <svg id="chart-svg" class="chart-svg"></svg>
          <div id="chart-stats" style="margin-top:10px; font-size:0.85rem; color:var(--muted);"></div>
        </div>

        ${images.length > 0 ? `
          <div class="section-title">Validation & Output Artifacts (${images.length})</div>
          <div class="gallery">
            ${images.map(img => `
              <div class="gallery-item">
                <img src="/api/artifact?exp=${encodeURIComponent(data.experiment)}&run=${encodeURIComponent(data.run_name)}&file=${encodeURIComponent(img.name)}" alt="${img.name}" />
                <div class="gallery-meta">
                  <div style="font-weight:600;">${img.name}</div>
                  <div style="color:var(--muted); font-size:0.75rem; margin-top:2px;">Step: ${img.step !== undefined && img.step !== null ? img.step : 'N/A'} | ${(img.size_bytes / 1024).toFixed(1)} KB</div>
                </div>
              </div>
            `).join('')}
          </div>
        ` : ''}

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
          <div>
            <div class="section-title">Parameters</div>
            <pre>${JSON.stringify(data.params || {}, null, 2)}</pre>
          </div>
          <div>
            <div class="section-title">Artifacts Index</div>
            <pre>${JSON.stringify(data.artifacts || [], null, 2)}</pre>
          </div>
        </div>
      `;

      const select = document.getElementById("metric-select");
      if (metricNames.length === 0) {
        select.innerHTML = "<option>No metrics</option>";
        document.getElementById("chart-svg").innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="#64748b">No metric steps logged</text>`;
      } else {
        metricNames.forEach(name => {
          const opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          select.appendChild(opt);
        });
        drawChart(metricNames[0]);
      }
    }

    function drawChart(metricName) {
      if (!currentData || !currentData.metrics || !currentData.metrics[metricName]) return;
      const points = currentData.metrics[metricName];
      const svg = document.getElementById("chart-svg");
      const stats = document.getElementById("chart-stats");
      if (!points.length) return;

      const width = svg.clientWidth || 800;
      const height = 320;
      const pad = 40;

      const steps = points.map(p => p.step);
      const vals = points.map(p => p.value);
      const minS = Math.min(...steps);
      const maxS = Math.max(...steps);
      const minV = Math.min(...vals);
      const maxV = Math.max(...vals);

      const xRange = (maxS - minS) || 1;
      const yRange = (maxV - minV) || (minV === 0 ? 1 : Math.abs(minV));

      const getX = s => pad + ((s - minS) / xRange) * (width - 2 * pad);
      const getY = v => (height - pad) - ((v - minV) / yRange) * (height - 2 * pad);

      let d = "";
      points.forEach((p, i) => {
        const x = getX(p.step);
        const y = getY(p.value);
        d += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
      });

      stats.textContent = `Total steps: ${points.length} | Min: ${minV.toFixed(4)} | Max: ${maxV.toFixed(4)} | Last: ${vals[vals.length - 1].toFixed(4)}`;

      svg.innerHTML = `
        <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#334155" stroke-width="1" />
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#334155" stroke-width="1" />
        <text x="${pad}" y="${pad - 10}" fill="#94a3b8" font-size="11">${maxV.toFixed(3)}</text>
        <text x="${pad}" y="${height - pad + 20}" fill="#94a3b8" font-size="11">${minV.toFixed(3)}</text>
        <text x="${width - pad - 20}" y="${height - pad + 20}" fill="#94a3b8" font-size="11">step ${maxS}</text>
        <path d="${d}" fill="none" stroke="#38bdf8" stroke-width="2.5" />
      `;
    }

    fetchExperiments();
  </script>
</body>
</html>
"""

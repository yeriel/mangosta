"""Self-contained, serverless HTML report generator with offline interactive charts."""

from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path
from typing import Any


def _load_run_data(target_path: Path) -> dict[str, Any]:
    """Load metadata, parameters, and metrics from either a run folder or a .bundle file."""
    if target_path.is_file() and target_path.suffix == ".bundle":
        # Load from .bundle
        with zipfile.ZipFile(target_path, "r") as zf:
            manifest_raw = zf.read("manifest.json").decode("utf-8")
            manifest = json.loads(manifest_raw)

            config: Any = None
            if "config.yaml" in zf.namelist():
                config = zf.read("config.yaml").decode("utf-8")

        # Format as unified report data
        return {
            "title": f"{manifest.get('family', 'model')} v{manifest.get('version', '0.0.0')}",
            "type": "bundle",
            "is_checkpoint": manifest.get("is_checkpoint", False),
            "status": "checkpoint" if manifest.get("is_checkpoint") else "exported_bundle",
            "epoch": manifest.get("epoch", 0),
            "step": manifest.get("step", 0),
            "start_time": manifest.get("created_at"),
            "end_time": manifest.get("created_at"),
            "dataset_hash": manifest.get("dataset_hash"),
            "git_commit": manifest.get("metadata", {}).get("git_commit"),
            "params": manifest.get("init_args", {}),
            "config": config,
            "schema": manifest.get("schema"),
            "metrics": {
                k: [{"step": 0, "value": v}]
                for k, v in manifest.get("metrics", {}).items()
                if isinstance(v, (int, float))
            },
            "env": manifest.get("env", {}),
        }

    elif target_path.is_dir():
        # Load from run directory
        metadata_file = target_path / "run.json"
        metrics_file = target_path / "metrics.jsonl"

        metadata: dict[str, Any] = {}
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        metrics: dict[str, list[dict[str, Any]]] = {}
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        name = rec.get("name", "metric")
                        if name not in metrics:
                            metrics[name] = []
                        metrics[name].append(rec)

        return {
            "title": f"{metadata.get('experiment', 'Experiment')} / {metadata.get('run_name', 'Run')}",
            "type": "run",
            "is_checkpoint": False,
            "status": metadata.get("status", "unknown"),
            "epoch": None,
            "step": None,
            "start_time": metadata.get("start_time"),
            "end_time": metadata.get("end_time"),
            "dataset_hash": metadata.get("dataset_hash"),
            "git_commit": metadata.get("git_commit"),
            "params": metadata.get("params", {}),
            "config": None,
            "schema": None,
            "metrics": metrics,
            "env": metadata.get("env", {}),
        }

    else:
        raise ValueError(
            f"Target path must be an experiment run directory or a .bundle file: {target_path}"
        )


def generate_report(
    target_path: str | Path,
    output_html: str | Path = "report.html",
) -> Path:
    """Generate a self-contained, interactive HTML report with zero web server dependencies."""
    target = Path(target_path).resolve()
    out_file = Path(output_html).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data = _load_run_data(target)
    metrics_json = json.dumps(data["metrics"])
    params_json = json.dumps(data["params"], indent=2)
    schema_json = json.dumps(data["schema"], indent=2) if data["schema"] else "None"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>mangosta Report - {html.escape(data["title"])}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #f59e0b;
      --accent-hover: #d97706;
      --chart-line: #38bdf8;
      --grid: #334155;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      text-transform: uppercase;
      background-color: var(--accent);
      color: #0f172a;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .card-title {{ font-size: 0.85rem; color: var(--text-muted); margin-bottom: 4px; }}
    .card-value {{ font-size: 1.25rem; font-weight: 600; word-break: break-all; }}
    .chart-container {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    .chart-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}
    select {{
      background: #0f172a;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.95rem;
    }}
    svg.chart {{
      width: 100%;
      height: 320px;
      background: #0b1120;
      border-radius: 6px;
    }}
    pre {{
      background: #0b1120;
      padding: 14px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.88rem;
      border: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>🥭 mangosta Report</h1>
        <div style="color: var(--text-muted); font-size: 1.1rem; margin-top: 4px;">{html.escape(data["title"])}</div>
      </div>
      <span class="badge">{html.escape(str(data["status"]))}</span>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-title">Dataset SHA-256</div>
        <div class="card-value" style="font-size: 0.9rem; font-family: monospace;">{html.escape(str(data["dataset_hash"])[:16])}...</div>
      </div>
      <div class="card">
        <div class="card-title">Created / Started</div>
        <div class="card-value" style="font-size: 0.95rem;">{html.escape(str(data["start_time"]))}</div>
      </div>
      <div class="card">
        <div class="card-title">Python & PyTorch</div>
        <div class="card-value" style="font-size: 0.95rem;">Py {html.escape(str(data["env"].get("python_version")))} / Torch {html.escape(str(data["env"].get("torch_version")))}</div>
      </div>
      <div class="card">
        <div class="card-title">Git Commit</div>
        <div class="card-value" style="font-family: monospace; font-size: 0.95rem;">{html.escape(str(data.get("git_commit") or "N/A")[:8])}</div>
      </div>
    </div>

    <div class="chart-container">
      <div class="chart-header">
        <h2>📈 Metric Curves</h2>
        <select id="metricSelector" onchange="renderChart(this.value)"></select>
      </div>
      <div id="chartWrapper">
        <svg id="chartSvg" class="chart"></svg>
      </div>
      <div id="metricStats" style="margin-top: 10px; color: var(--text-muted); font-size: 0.9rem;"></div>
    </div>

    <div class="grid" style="grid-template-columns: 1fr 1fr;">
      <div class="card">
        <h3 style="margin-bottom: 12px;">⚙️ Parameters / Init Args</h3>
        <pre>{html.escape(params_json)}</pre>
      </div>
      <div class="card">
        <h3 style="margin-bottom: 12px;">📐 I/O Schema Contract</h3>
        <pre>{html.escape(schema_json)}</pre>
      </div>
    </div>
  </div>

  <script>
    const metricsData = {metrics_json};
    const selector = document.getElementById("metricSelector");
    const metricNames = Object.keys(metricsData);

    if (metricNames.length === 0) {{
      selector.innerHTML = "<option>No metrics logged</option>";
      document.getElementById("chartWrapper").innerHTML = "<p style='color:var(--text-muted); padding:20px;'>No metric points available.</p>";
    }} else {{
      metricNames.forEach(name => {{
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        selector.appendChild(opt);
      }});
      renderChart(metricNames[0]);
    }}

    function renderChart(metricName) {{
      const points = metricsData[metricName] || [];
      const svg = document.getElementById("chartSvg");
      const stats = document.getElementById("metricStats");
      if (!points.length) return;

      const width = svg.clientWidth || 800;
      const height = 320;
      const pad = 40;

      const steps = points.map(p => p.step);
      const values = points.map(p => p.value);
      const minStep = Math.min(...steps);
      const maxStep = Math.max(...steps);
      const minVal = Math.min(...values);
      const maxVal = Math.max(...values);

      stats.textContent = `Points: ${{points.length}} | Min: ${{minVal.toFixed(4)}} | Max: ${{maxVal.toFixed(4)}} | Last: ${{values[values.length - 1].toFixed(4)}}`;

      const xRange = (maxStep - minStep) || 1;
      const yRange = (maxVal - minVal) || (minVal === 0 ? 1 : Math.abs(minVal));

      const getX = s => pad + ((s - minStep) / xRange) * (width - 2 * pad);
      const getY = v => (height - pad) - ((v - minVal) / yRange) * (height - 2 * pad);

      let pathD = "";
      points.forEach((p, idx) => {{
        const x = getX(p.step);
        const y = getY(p.value);
        pathD += (idx === 0 ? `M ${{x}} ${{y}}` : ` L ${{x}} ${{y}}`);
      }});

      svg.innerHTML = `
        <line x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad}}" y2="${{height - pad}}" stroke="var(--grid)" stroke-width="1" />
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height - pad}}" stroke="var(--grid)" stroke-width="1" />
        <text x="${{pad}}" y="${{pad - 10}}" fill="var(--text-muted)" font-size="11">${{maxVal.toFixed(3)}}</text>
        <text x="${{pad}}" y="${{height - pad + 20}}" fill="var(--text-muted)" font-size="11">${{minVal.toFixed(3)}}</text>
        <text x="${{width - pad - 20}}" y="${{height - pad + 20}}" fill="var(--text-muted)" font-size="11">step ${{maxStep}}</text>
        <path d="${{pathD}}" fill="none" stroke="var(--chart-line)" stroke-width="2.5" />
      `;
    }}
  </script>
</body>
</html>
"""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return out_file

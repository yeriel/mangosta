"""Command line interface for mangosta: init, inspect, view, and export-onnx."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from mangosta.artifacts.bundle import inspect_bundle_metadata
from mangosta.cli.init_project import init_project
from mangosta.tracking.viewer import start_viewer


def _get_version() -> str:
    """Return installed mangosta version cleanly."""
    try:
        return importlib.metadata.version("mangosta")
    except Exception:
        return "0.4.0"


def _format_schema_nodes(nodes: Any) -> str:
    """Render schema input/output node specifications concisely."""
    if isinstance(nodes, list):
        parts = [
            f"{item.get('name', idx)}: {item.get('shape')} ({item.get('dtype')})"
            for idx, item in enumerate(nodes)
            if isinstance(item, dict)
        ]
        return "; ".join(parts) if parts else "none"
    if isinstance(nodes, dict):
        parts = [
            f"{k}: {v.get('shape')} ({v.get('dtype')})"
            for k, v in nodes.items()
            if isinstance(v, dict)
        ]
        return "; ".join(parts) if parts else "none"
    return "none"


def _build_inspect_payload(info: dict[str, Any]) -> dict[str, Any]:
    """Normalize metadata payload for machine-readable JSON inspection."""
    tensors = info.get("tensors") or []
    return {
        "file": {
            "name": info["file"],
            "path": info["path"],
            "size_bytes": info["size_bytes"],
        },
        "family": info.get("family"),
        "version": info.get("version"),
        "format": info.get("format"),
        "weights_size_bytes": info.get("weights_size_bytes", 0),
        "weights_sha256": info.get("weights_sha256"),
        "is_checkpoint": info.get("is_checkpoint", False),
        "epoch": info.get("epoch"),
        "step": info.get("step"),
        "dataset_hash": info.get("dataset_hash"),
        "created_at": info.get("created_at"),
        "total_params": info.get("total_params", 0),
        "schema": info.get("schema"),
        "metrics": info.get("metrics", {}),
        "init_args": info.get("init_args", {}),
        "metadata": info.get("metadata", {}),
        "config": info.get("config"),
        "files": info.get("files", []),
        "tensors": tensors if info.get("format") == "safetensors" else None,
    }


def _build_diff_payload(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare two inspected bundle metadata dicts cleanly."""
    fields = [
        "family",
        "version",
        "format",
        "size_bytes",
        "weights_size_bytes",
        "total_params",
        "is_checkpoint",
        "epoch",
        "step",
        "weights_sha256",
        "dataset_hash",
    ]
    summary_diff = {
        f: {"left": left.get(f), "right": right.get(f), "changed": left.get(f) != right.get(f)}
        for f in fields
    }

    left_metrics = left.get("metrics") or {}
    right_metrics = right.get("metrics") or {}
    metric_keys = sorted(set(left_metrics) | set(right_metrics))
    metrics_diff = {
        k: {
            "left": left_metrics.get(k),
            "right": right_metrics.get(k),
            "changed": left_metrics.get(k) != right_metrics.get(k),
        }
        for k in metric_keys
    }

    left_tensors = {t["name"]: t for t in (left.get("tensors") or [])}
    right_tensors = {t["name"]: t for t in (right.get("tensors") or [])}
    layer_names = sorted(set(left_tensors) | set(right_tensors))
    changed_layers: list[dict[str, Any]] = []
    for name in layer_names:
        lt = left_tensors.get(name)
        rt = right_tensors.get(name)
        l_spec = f"{lt.get('shape')} {lt.get('dtype')}" if lt else None
        r_spec = f"{rt.get('shape')} {rt.get('dtype')}" if rt else None
        if l_spec != r_spec:
            changed_layers.append({"name": name, "left": l_spec, "right": r_spec})

    return {
        "left": left["file"],
        "right": right["file"],
        "summary": summary_diff,
        "metrics": metrics_diff,
        "schema_changed": left.get("schema") != right.get("schema"),
        "changed_layers": changed_layers,
        "total_layers": {"left": len(left_tensors), "right": len(right_tensors)},
    }


def _print_bundle_text(info: dict[str, Any], show_config: bool = False) -> None:
    """Print human-readable bundle summary to stdout."""
    format_type = info.get("format", "unknown")
    weights_mb = (info.get("weights_size_bytes") or 0) / (1024 * 1024)
    tensors = info.get("tensors") or []

    print("\n" + "=" * 60)
    print("MANGOSTA BUNDLE INSPECTOR")
    print("=" * 60)
    print(f"File:           {info['file']} ({info['size_bytes'] / 1024:.1f} KB)")
    print(f"Family:         {info.get('family', 'N/A')}")
    print(f"Version:        {info.get('version', 'N/A')}")
    print(f"Is Checkpoint:  {info.get('is_checkpoint', False)}")
    if info.get("is_checkpoint"):
        print(f"Epoch / Step:   epoch {info.get('epoch', 0)}, step {info.get('step', 0)}")
    print(f"Dataset SHA:    {info.get('dataset_hash') or 'N/A'}")
    print(f"Created At:     {info.get('created_at') or 'N/A'}")
    print(f"Weights Format: {format_type} ({weights_mb:.2f} MB)")
    if info.get("total_params"):
        print(f"Total Params:   {info['total_params']:,}")
    if info.get("weights_sha256"):
        print(f"Weights SHA:    {info['weights_sha256']}")
    if info.get("files"):
        print(f"Files in zip:   {', '.join(info['files'])}")

    if tensors:
        print(f"\nModel Tensors / Layers ({len(tensors)} total):")
        for t in tensors[:15]:
            print(f"  - {t['name']}: shape={t['shape']}, dtype={t['dtype']} ({t['param_count']:,} params)")
        if len(tensors) > 15:
            print(f"  ... and {len(tensors) - 15} more tensors (use --json for full list)")

    schema = info.get("schema")
    if isinstance(schema, dict):
        print("\nI/O Schema Contract:")
        print(f"  Inputs ({schema.get('input_structure')}):  {_format_schema_nodes(schema.get('inputs'))}")
        print(f"  Outputs ({schema.get('output_structure')}): {_format_schema_nodes(schema.get('outputs'))}")

    metrics = info.get("metrics")
    if metrics:
        print("\nRecorded Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    if show_config and info.get("config"):
        print("\nEmbedded config.yaml:")
        print(info["config"].rstrip())

    print("=" * 60 + "\n")


def _print_bundle_diff_text(diff: dict[str, Any]) -> None:
    """Print human-readable comparison between two .bundle files."""
    print("\n" + "=" * 68)
    print(f"MANGOSTA BUNDLE DIFF: {diff['left']}  vs  {diff['right']}")
    print("=" * 68)
    for field, entry in diff["summary"].items():
        marker = "*" if entry["changed"] else " "
        print(f" {marker} {field:18s} {entry['left']}  ->  {entry['right']}")

    if diff["metrics"]:
        print("\nMetrics:")
        for k, entry in diff["metrics"].items():
            marker = "*" if entry["changed"] else " "
            print(f" {marker} {k:18s} {entry['left']}  ->  {entry['right']}")

    print(f"\nSchema changed: {diff['schema_changed']}")
    print(f"Layers count:   {diff['total_layers']['left']} vs {diff['total_layers']['right']}")
    if diff["changed_layers"]:
        print(f"Changed layers ({len(diff['changed_layers'])}):")
        for layer in diff["changed_layers"][:20]:
            print(f"  * {layer['name']}: {layer['left']} -> {layer['right']}")
        if len(diff["changed_layers"]) > 20:
            print(f"  ... and {len(diff['changed_layers']) - 20} more")
    print("=" * 68 + "\n")


def inspect_bundle(
    bundle_path: str,
    as_json: bool = False,
    diff_path: str | None = None,
    show_config: bool = False,
) -> None:
    """Inspect a .bundle archive (or diff two bundles) in terminal or JSON."""
    info = inspect_bundle_metadata(bundle_path)
    if diff_path is not None:
        other_info = inspect_bundle_metadata(diff_path)
        diff_payload = _build_diff_payload(info, other_info)
        if as_json:
            print(json.dumps(diff_payload, indent=2))
        else:
            _print_bundle_diff_text(diff_payload)
        return

    if as_json:
        print(json.dumps(_build_inspect_payload(info), indent=2))
    else:
        _print_bundle_text(info, show_config=show_config)


def run_view(target_path: str = ".", port: int = 8501, open_browser: bool = True) -> None:
    """Ephemeral zero-daemon viewer using Python's standard library http.server."""
    target = Path(target_path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target path '{target_path}' does not exist.")
    start_viewer(target, port=port, open_browser=open_browser)


def _cmd_init(args: argparse.Namespace) -> None:
    dest = init_project(args.name)
    print(f"Initialized mangosta project in '{dest}'")
    uv_bin = shutil.which("uv")
    if uv_bin is not None:
        res = subprocess.run([uv_bin, "sync"], cwd=dest, check=False)
        if res.returncode != 0:
            print("Warning: 'uv sync' did not complete; run 'uv sync' manually if needed.")


def _cmd_inspect(args: argparse.Namespace) -> None:
    inspect_bundle(
        args.bundle,
        as_json=args.json,
        diff_path=args.diff,
        show_config=args.config,
    )


def _cmd_view(args: argparse.Namespace) -> None:
    run_view(args.target, port=args.port, open_browser=not args.no_browser)


def _cmd_export_onnx(args: argparse.Namespace) -> None:
    if args.module:
        cwd_str = str(Path.cwd())
        if cwd_str not in sys.path:
            sys.path.insert(0, cwd_str)
        importlib.import_module(args.module)

    shape = tuple(args.input_shape) if args.input_shape else None
    from mangosta.export.onnx import export_onnx

    out = export_onnx(
        model_or_bundle=args.bundle,
        output_path=args.output,
        sample_input_shape=shape,
        output_mapping=args.output_mapping,
        float16=args.fp16,
        opset_version=args.opset,
    )
    if args.json:
        print(json.dumps({"status": "ok", "output": str(out)}, indent=2))
    else:
        print(f"Successfully exported model to ONNX: {out}")


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the mangosta CLI parser."""
    parser = argparse.ArgumentParser(
        prog="mangosta",
        description="Lightweight MLOps and Model Lifecycle Management for PyTorch",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    init_parser = subparsers.add_parser("init", help="Initialize a structured PyTorch project")
    init_parser.add_argument(
        "name",
        nargs="?",
        default=".",
        help="Project directory or name (default: current directory)",
    )
    init_parser.set_defaults(handler=_cmd_init)

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect or diff .bundle archives")
    inspect_parser.add_argument("bundle", help="Path to .bundle file")
    inspect_parser.add_argument(
        "--diff",
        default=None,
        help="Compare against a second .bundle file",
    )
    inspect_parser.add_argument(
        "--config",
        action="store_true",
        help="Print embedded config.yaml in text output",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON (useful for CI and AI agents)",
    )
    inspect_parser.set_defaults(handler=_cmd_inspect)

    # view
    view_parser = subparsers.add_parser("view", help="Start ephemeral local browser viewer")
    view_parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Path to runs directory, experiment folder, or .bundle file (default: current directory)",
    )
    view_parser.add_argument(
        "--port", type=int, default=8501, help="Port to serve on (default: 8501)"
    )
    view_parser.add_argument(
        "--no-browser", action="store_true", help="Do not automatically open browser"
    )
    view_parser.set_defaults(handler=_cmd_view)

    # export-onnx
    export_parser = subparsers.add_parser("export-onnx", help="Export a bundle to standalone ONNX")
    export_parser.add_argument("bundle", help="Path to .bundle file")
    export_parser.add_argument("-o", "--output", required=True, help="Target .onnx or .onnx.bundle path")
    export_parser.add_argument(
        "--fp16", action="store_true", help="Export with Float16 half precision"
    )
    export_parser.add_argument(
        "--opset", type=int, default=18, help="ONNX opset version (default: 18)"
    )
    export_parser.add_argument(
        "--output-mapping", nargs="+", help="Ordered keys to unroll dictionary output"
    )
    export_parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        help="Dummy input shape (auto-inferred from bundle ModelSchema if omitted)",
    )
    export_parser.add_argument(
        "-m",
        "--module",
        help="Python module to import containing registered models",
    )
    export_parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON",
    )
    export_parser.set_defaults(handler=_cmd_export_onnx)

    return parser


def main() -> None:
    """CLI entrypoint for mangosta."""
    parser = build_parser()
    args = parser.parse_args()
    handler: Callable[[argparse.Namespace], None] = args.handler
    try:
        handler(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

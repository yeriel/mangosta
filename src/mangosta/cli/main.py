"""Command line interface for mangosta: inspect, report, and ephemeral viewing."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

from mangosta.artifacts.bundle import read_bundle_manifest
from mangosta.cli.init_project import init_project
from mangosta.tracking.report import generate_report
from mangosta.tracking.viewer import start_viewer


def _parse_safetensors_header(header_bytes: bytes) -> dict[str, Any]:
    """Parse SafeTensors metadata header without importing PyTorch or third-party packages.

    SafeTensors format layout:
    - 8 bytes: unsigned 64-bit little-endian integer N (header length in bytes)
    - N bytes: UTF-8 encoded JSON object containing tensor shapes, dtypes, data_offsets,
      and optional __metadata__ dict.
    """
    if len(header_bytes) < 8:
        return {}
    header_len = struct.unpack("<Q", header_bytes[:8])[0]
    if len(header_bytes) < 8 + header_len:
        return {}
    json_bytes = header_bytes[8 : 8 + header_len]
    try:
        parsed = json.loads(json_bytes.decode("utf-8"))
        return dict(parsed) if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def inspect_bundle(bundle_path: str, as_json: bool = False) -> None:
    """Print formatted summary of a .bundle file to terminal or stdout as machine-readable JSON."""
    path = Path(bundle_path).resolve()
    if not path.is_file():
        if as_json:
            print(json.dumps({"error": f"Bundle file '{bundle_path}' not found."}, indent=2), file=sys.stderr)
        else:
            print(f"Error: Bundle file '{bundle_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        manifest = read_bundle_manifest(path)
    except Exception as e:
        if as_json:
            print(json.dumps({"error": f"Error reading bundle: {e}"}, indent=2), file=sys.stderr)
        else:
            print(f"Error reading bundle: {e}", file=sys.stderr)
        sys.exit(1)

    # Inspect zip contents, weights format, layers and hashes
    format_type = manifest.get("weights_format", "unknown")
    files_in_zip: list[str] = []
    weights_size_bytes = 0
    tensors_info: list[dict[str, Any]] = []

    try:
        with zipfile.ZipFile(path, "r") as zf:
            namelist = zf.namelist()
            files_in_zip = [n for n in namelist if not n.endswith("/")]

            if "model.safetensors" in namelist:
                format_type = "safetensors"
                weights_size_bytes = zf.getinfo("model.safetensors").file_size

                # Fast zero-torch read of SafeTensors header
                with zf.open("model.safetensors", "r") as sf:
                    # Read first 8 bytes for length, then read header JSON
                    header_len_bytes = sf.read(8)
                    if len(header_len_bytes) == 8:
                        header_len = struct.unpack("<Q", header_len_bytes)[0]
                        header_json_bytes = sf.read(header_len)
                        raw_header = json.loads(header_json_bytes.decode("utf-8"))
                        for tensor_name, info in raw_header.items():
                            if tensor_name == "__metadata__":
                                continue
                            if isinstance(info, dict):
                                tensors_info.append(
                                    {
                                        "name": tensor_name,
                                        "shape": info.get("shape", []),
                                        "dtype": info.get("dtype", "unknown"),
                                    }
                                )
            elif "model.onnx" in namelist:
                format_type = "onnx"
                weights_size_bytes = zf.getinfo("model.onnx").file_size
    except Exception:
        pass

    model_sha = manifest.get("model_sha256") or manifest.get("weights_sha256")

    # Machine-readable JSON output
    if as_json:
        payload = {
            "file": {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            },
            "family": manifest.get("family"),
            "version": manifest.get("version"),
            "format": format_type,
            "weights_size_bytes": weights_size_bytes,
            "weights_sha256": model_sha,
            "is_checkpoint": manifest.get("is_checkpoint", False),
            "epoch": manifest.get("epoch"),
            "step": manifest.get("step"),
            "dataset_hash": manifest.get("dataset_hash"),
            "created_at": manifest.get("created_at"),
            "schema": manifest.get("schema"),
            "metrics": manifest.get("metrics", {}),
            "init_args": manifest.get("init_args", {}),
            "metadata": manifest.get("metadata", {}),
            "files": files_in_zip,
            "tensors": tensors_info if format_type == "safetensors" else None,
        }
        print(json.dumps(payload, indent=2))
        return

    # Human-readable formatted text output
    print("\n" + "=" * 60)
    print("MANGOSTA BUNDLE INSPECTOR")
    print("=" * 60)
    print(f"File:           {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"Family:         {manifest.get('family', 'N/A')}")
    print(f"Version:        {manifest.get('version', 'N/A')}")
    print(f"Is Checkpoint:  {manifest.get('is_checkpoint', False)}")
    if manifest.get("is_checkpoint"):
        print(f"Epoch / Step:   epoch {manifest.get('epoch', 0)}, step {manifest.get('step', 0)}")
    print(f"Dataset SHA:    {manifest.get('dataset_hash') or 'N/A'}")
    print(f"Created At:     {manifest.get('created_at', 'N/A')}")

    if format_type == "safetensors":
        print(f"Weights Format: safetensors ({weights_size_bytes / (1024 * 1024):.2f} MB)")
    elif format_type == "onnx":
        print(f"Weights Format: onnx ({weights_size_bytes / (1024 * 1024):.2f} MB)")
    else:
        print(f"Weights Format: {format_type}")

    if model_sha:
        print(f"Weights SHA:    {model_sha}")
    if files_in_zip:
        print(f"Files in zip:   {', '.join(files_in_zip)}")

    # Print SafeTensors Layer Shapes if parsed
    if tensors_info:
        print(f"\nModel Tensors / Layers ({len(tensors_info)} total):")
        display_tensors = tensors_info[:15]
        for t in display_tensors:
            print(f"  - {t['name']}: shape={t['shape']}, dtype={t['dtype']}")
        if len(tensors_info) > 15:
            print(f"  ... and {len(tensors_info) - 15} more tensors (use --json for full list)")

    # Print Schema
    schema = manifest.get("schema")
    if schema:
        print("\nI/O Schema Contract:")
        print(f"  Input structure:  {schema.get('input_structure')}")
        inputs = schema.get("inputs", [])
        if isinstance(inputs, list):
            for idx, item in enumerate(inputs):
                print(
                    f"    [{idx}] {item.get('name')}: shape={item.get('shape')}, dtype={item.get('dtype')}"
                )
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                print(f"    ['{k}']: shape={v.get('shape')}, dtype={v.get('dtype')}")

        print(f"  Output structure: {schema.get('output_structure')}")
        outputs = schema.get("outputs", [])
        if isinstance(outputs, list):
            for idx, item in enumerate(outputs):
                print(
                    f"    [{idx}] {item.get('name')}: shape={item.get('shape')}, dtype={item.get('dtype')}"
                )
        elif isinstance(outputs, dict):
            for k, v in outputs.items():
                print(f"    ['{k}']: shape={v.get('shape')}, dtype={v.get('dtype')}")

    # Print Metrics
    metrics = manifest.get("metrics")
    if metrics:
        print("\nRecorded Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print("=" * 60 + "\n")


def run_report(target_path: str, output: str) -> None:
    """Generate standalone HTML report."""
    try:
        out_path = generate_report(target_path, output)
        print(f"Report successfully generated at: {out_path}")
    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        sys.exit(1)


def run_view(target_path: str = ".", port: int = 8501, open_browser: bool = True) -> None:
    """Ephemeral zero-daemon viewer using Python's standard library http.server."""
    target = Path(target_path).resolve()
    if not target.exists():
        print(f"Error: Target path '{target_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    start_viewer(target, port=port, open_browser=open_browser)


def main() -> None:
    """CLI entrypoint for mangosta."""
    parser = argparse.ArgumentParser(
        prog="mangosta",
        description="Lightweight MLOps and Model Lifecycle Management for PyTorch",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: init
    init_parser = subparsers.add_parser("init", help="Initialize a structured PyTorch project")
    init_parser.add_argument(
        "name",
        nargs="?",
        default=".",
        help="Project directory or name (default: current directory)",
    )
    init_parser.add_argument(
        "--template",
        default="default",
        help="Project template (default: default)",
    )

    # Subcommand: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a .bundle archive in terminal")
    inspect_parser.add_argument("bundle", help="Path to .bundle file")
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON (useful for CI and AI agents)",
    )

    # Subcommand: report
    report_parser = subparsers.add_parser(
        "report", help="Generate standalone interactive HTML report"
    )
    report_parser.add_argument("target", help="Path to run directory or .bundle file")
    report_parser.add_argument(
        "-o", "--output", default="report.html", help="Output HTML file path"
    )

    # Subcommand: view
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

    # Subcommand: export-onnx
    export_parser = subparsers.add_parser("export-onnx", help="Export a bundle to standalone ONNX")
    export_parser.add_argument("bundle", help="Path to .bundle file")
    export_parser.add_argument("-o", "--output", required=True, help="Target .onnx output path")
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
        "--input-shape", nargs="+", type=int, help="Dummy input shape (e.g. 1 3 224 224)"
    )
    export_parser.add_argument(
        "-m",
        "--module",
        help="Python module to import containing registered models (e.g. models or examples.lifecycle_demo)",
    )

    args = parser.parse_args()

    if args.command == "init":
        dest = init_project(args.name, template=args.template)
        print(f"Initialized mangosta project in '{dest}'")
        print("Next steps:")
        print("  uv sync")
        print("  uv run train")
    elif args.command == "inspect":
        inspect_bundle(args.bundle, as_json=args.json)
    elif args.command == "report":
        run_report(args.target, args.output)
    elif args.command == "view":
        run_view(args.target, port=args.port, open_browser=not args.no_browser)
    elif args.command == "export-onnx":
        if args.module:
            import importlib
            import sys
            from pathlib import Path

            if str(Path.cwd()) not in sys.path:
                sys.path.insert(0, str(Path.cwd()))
            importlib.import_module(args.module)

        shape = tuple(args.input_shape) if args.input_shape else (1, 4)
        from mangosta.export.onnx import export_onnx

        out = export_onnx(
            model_or_bundle=args.bundle,
            output_path=args.output,
            sample_input_shape=shape,
            output_mapping=args.output_mapping,
            float16=args.fp16,
            opset_version=args.opset,
        )
        print(f"Successfully exported model to ONNX: {out}")


if __name__ == "__main__":
    main()

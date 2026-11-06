"""Tests for mangosta viewer HTTP server and endpoints."""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pytest

from mangosta.artifacts import save_bundle
from mangosta.core.schema import infer_schema_from_sample
from mangosta.tracking import RunTracker
from mangosta.tracking.viewer import (
    _load_run_details,
    _scan_experiments,
)
import numpy as np


def test_scan_and_load_details_comprehensive(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    
    # Empty dir scan
    empty_scan = _scan_experiments(tmp_path / "nonexistent")
    assert empty_scan == {}

    # Run with metrics and artifacts
    with RunTracker(experiment="nlp", run_name="bert_01", root_dir=exp_dir) as tracker:
        tracker.log_params({"lr": 1e-4, "model": "bert"})
        tracker.log_metrics({"loss": 0.5}, step=0)
        tracker.log_metrics({"loss": 0.3}, step=1)
        
        sample_file = tmp_path / "vocab.txt"
        sample_file.write_text("hello\nworld\n", encoding="utf-8")
        tracker.log_artifact(sample_file)

    # Scan
    scan = _scan_experiments(exp_dir)
    assert "nlp" in scan
    assert len(scan["nlp"]) == 1
    assert scan["nlp"][0]["run_name"] == "bert_01"
    assert scan["nlp"][0]["status"] == "completed"
    assert scan["nlp"][0]["artifacts_count"] == 1

    # Load details
    details = _load_run_details(exp_dir, "nlp", "bert_01")
    assert details is not None
    assert details["params"]["lr"] == 1e-4
    assert len(details["metrics"]["loss"]) == 2
    assert len(details["artifacts"]) == 1

    # Nonexistent run details
    assert _load_run_details(exp_dir, "nlp", "nonexistent") is None
    assert _load_run_details(exp_dir, "nonexistent", "run") is None


def test_viewer_handler_endpoints(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    with RunTracker(experiment="cv", run_name="resnet_01", root_dir=exp_dir) as tracker:
        tracker.log_params({"lr": 0.001})
        tracker.log_metrics({"acc": 0.9}, step=0)
        art = tmp_path / "classes.txt"
        art.write_text("cat\ndog\n", encoding="utf-8")
        tracker.log_artifact(art)

    from mangosta.tracking.viewer import create_viewer_handler
    handler_cls = create_viewer_handler(exp_dir)

    class DummySocket:
        def __init__(self) -> None:
            self.output = io.BytesIO()
        def sendall(self, data: bytes) -> None:
            self.output.write(data)
        def makefile(self, mode: str, bufsize: int = -1) -> io.BytesIO:
            return io.BytesIO(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")

    # Helper to invoke GET
    def invoke_get(path_and_query: str) -> tuple[int, dict[str, str], bytes]:
        req_bytes = f"GET {path_and_query} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode("utf-8")
        in_buf = io.BytesIO(req_bytes)
        out_buf = io.BytesIO()

        class MockRequest:
            def makefile(self, mode: str, *args, **kwargs) -> io.BytesIO:
                return in_buf if "r" in mode else out_buf
            def sendall(self, data: bytes) -> None:
                out_buf.write(data)

        handler_cls(MockRequest(), ("127.0.0.1", 12345), None)  # type: ignore
        response_bytes = out_buf.getvalue()
        
        # Parse status code
        lines = response_bytes.split(b"\r\n")
        status_line = lines[0].decode("utf-8")
        status_code = int(status_line.split(" ")[1])
        return status_code, {}, response_bytes

    # 1. Test / (HTML dashboard)
    code, _, resp = invoke_get("/")
    assert code == 200
    assert b"mangosta" in resp

    # 2. Test /api/experiments
    code, _, resp = invoke_get("/api/experiments")
    assert code == 200
    assert b"cv" in resp

    # 3. Test /api/run?exp=cv&run=resnet_01
    code, _, resp = invoke_get("/api/run?exp=cv&run=resnet_01")
    assert code == 200
    assert b"resnet_01" in resp

    # 4. Test /api/run 404
    code, _, resp = invoke_get("/api/run?exp=cv&run=not_found")
    assert code == 404

    # 5. Test /api/artifact
    code, _, resp = invoke_get("/api/artifact?exp=cv&run=resnet_01&file=classes.txt")
    assert code == 200
    assert b"cat\ndog\n" in resp

    # 6. Test /api/artifact 404
    code, _, resp = invoke_get("/api/artifact?exp=cv&run=resnet_01&file=missing.txt")
    assert code == 404

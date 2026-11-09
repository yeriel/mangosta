"""Tests for Phase 5: Enhanced mangosta view, validation data inspection, and multi-run comparison."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mangosta.artifacts import save_bundle
from mangosta.tracking import RunTracker
from mangosta.tracking.viewer import (
    _load_bundle_details,
    _load_run_details,
    _scan_bundles,
    _scan_experiments,
    create_viewer_handler,
)


def _make_mock_request(handler_cls: type, path: str) -> tuple[int, dict[str, str], bytes]:
    """Helper to simulate an HTTP GET request against a BaseHTTPRequestHandler."""
    request_data = f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode("utf-8")
    mock_socket = MagicMock()
    mock_socket.makefile.side_effect = lambda mode, *args, **kwargs: (
        BytesIO(request_data) if "r" in mode else BytesIO()
    )

    rfile = BytesIO(request_data)
    wfile = BytesIO()

    # Instantiate handler with mocked streams
    handler = object.__new__(handler_cls)
    handler.rfile = rfile
    handler.wfile = wfile
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.command = "GET"
    handler.path = path
    handler.request_version = "HTTP/1.1"
    handler.headers = {}  # type: ignore[assignment]
    handler.server_version = "BaseHTTP/0.6"
    handler.sys_version = "Python/3.x"

    # Capture response
    status_code = 200
    headers: dict[str, str] = {}

    def mock_send_response(code: int, message: str | None = None) -> None:
        nonlocal status_code
        status_code = code

    def mock_send_header(keyword: str, value: str) -> None:
        headers[keyword.lower()] = value

    def mock_end_headers() -> None:
        pass

    def mock_send_error(code: int, message: str | None = None, explain: str | None = None) -> None:
        nonlocal status_code
        status_code = code

    handler.send_response = mock_send_response  # type: ignore[assignment]
    handler.send_header = mock_send_header  # type: ignore[assignment]
    handler.end_headers = mock_end_headers  # type: ignore[assignment]
    handler.send_error = mock_send_error  # type: ignore[assignment]

    handler.do_GET()
    return status_code, headers, wfile.getvalue()


def test_log_image_step_versioning(tmp_path: Path) -> None:
    """Test that log_image versions filenames across steps and avoids overwriting."""
    exp_dir = tmp_path / "runs"
    with RunTracker(experiment="val_exp", run_name="run_01", root_dir=exp_dir) as tracker:
        # Step 0
        img0 = np.zeros((16, 16, 3), dtype=np.uint8)
        p0 = tracker.log_image("sample_pred", img0, step=0)
        assert p0.name == "sample_pred_step_0000.png"
        assert p0.exists()

        # Step 1: Same base name, different step
        img1 = np.ones((16, 16, 3), dtype=np.uint8) * 255
        p1 = tracker.log_image("sample_pred", img1, step=1)
        assert p1.name == "sample_pred_step_0001.png"
        assert p1.exists()

        # Verify step 0 was not overwritten
        assert p0.exists()
        assert p0.read_bytes() != p1.read_bytes()

    # Verify metadata in run.json
    run_json = exp_dir / "val_exp" / "run_01" / "run.json"
    meta = json.loads(run_json.read_text(encoding="utf-8"))
    arts = meta["artifacts"]
    assert len(arts) == 2
    assert arts[0]["step"] == 0
    assert arts[0]["sample_id"] == "sample_pred"
    assert arts[1]["step"] == 1
    assert arts[1]["sample_id"] == "sample_pred"


def test_log_validation_sample_triplet(tmp_path: Path) -> None:
    """Test logging a real validation triplet: input, prediction, and ground truth."""
    exp_dir = tmp_path / "runs"
    with RunTracker(experiment="real_vision", run_name="run_unet", root_dir=exp_dir) as tracker:
        real_input = torch.rand(3, 32, 32)
        ground_truth = torch.ones(1, 32, 32)
        prediction = torch.zeros(1, 32, 32)

        saved = tracker.log_validation_sample(
            sample_id="patient_slice_42",
            prediction=prediction,
            input_image=real_input,
            ground_truth=ground_truth,
            step=5,
        )

        assert "input" in saved
        assert "ground_truth" in saved
        assert "prediction" in saved

        assert saved["input"].name == "patient_slice_42_input_step_0005.png"
        assert saved["ground_truth"].name == "patient_slice_42_gt_step_0005.png"
        assert saved["prediction"].name == "patient_slice_42_pred_step_0005.png"

        assert saved["input"].exists()
        assert saved["ground_truth"].exists()
        assert saved["prediction"].exists()

    run_json = exp_dir / "real_vision" / "run_unet" / "run.json"
    meta = json.loads(run_json.read_text(encoding="utf-8"))
    arts = meta["artifacts"]
    assert len(arts) == 3
    tags = [a["tag"] for a in arts]
    assert "input" in tags
    assert "ground_truth" in tags
    assert "prediction" in tags
    for a in arts:
        assert a["sample_id"] == "patient_slice_42"
        assert a["step"] == 5


def test_viewer_endpoints_runs_and_bundles(tmp_path: Path) -> None:
    """Test /api/runs (batch compare) and /api/bundles (discovery) endpoints."""
    runs_dir = tmp_path / "experiments"

    # Create Run 1
    with RunTracker(experiment="exp_a", run_name="run_01", root_dir=runs_dir) as r1:
        r1.log_params({"lr": 0.001, "batch_size": 32})
        r1.log_metrics({"loss": 0.5, "acc": 0.8}, step=0)
        r1.log_metrics({"loss": 0.3, "acc": 0.9}, step=1)
        r1.log_image("test_sample", np.zeros((10, 10, 3), dtype=np.uint8), step=0)

    # Create Run 2
    with RunTracker(experiment="exp_a", run_name="run_02", root_dir=runs_dir) as r2:
        r2.log_params({"lr": 0.01, "batch_size": 32})
        r2.log_metrics({"loss": 0.6, "acc": 0.75}, step=0)
        r2.log_metrics({"loss": 0.2, "acc": 0.92}, step=1)

    # Create a Bundle in workspace
    bundle_path = tmp_path / "checkpoints" / "model_epoch1.bundle"
    model = torch.nn.Linear(4, 2)
    save_bundle(
        output_path=bundle_path,
        model=model,
        family="linear_test",
        version="1.0.0",
        config={"hidden": 64},
    )

    handler_cls = create_viewer_handler(runs_dir)

    # 1. Test /api/experiments
    code, _, body = _make_mock_request(handler_cls, "/api/experiments")
    assert code == 200
    exps = json.loads(body.decode("utf-8"))
    assert "exp_a" in exps
    assert len(exps["exp_a"]) == 2

    # 2. Test /api/runs (batch multi-run comparison)
    code, _, body = _make_mock_request(handler_cls, "/api/runs?runs=exp_a:run_01,exp_a:run_02")
    assert code == 200
    runs_resp = json.loads(body.decode("utf-8"))
    assert len(runs_resp["runs"]) == 2
    assert runs_resp["runs"][0]["run_name"] == "run_01"
    assert runs_resp["runs"][1]["run_name"] == "run_02"
    # Verify hyperparameter difference can be spotted
    assert runs_resp["runs"][0]["params"]["lr"] == 0.001
    assert runs_resp["runs"][1]["params"]["lr"] == 0.01

    # 3. Test /api/bundles
    code, _, body = _make_mock_request(handler_cls, "/api/bundles")
    assert code == 200
    bundles_resp = json.loads(body.decode("utf-8"))
    assert len(bundles_resp["bundles"]) >= 1
    found = [b for b in bundles_resp["bundles"] if b["filename"] == "model_epoch1.bundle"]
    assert len(found) == 1
    assert found[0]["family"] == "linear_test"
    assert found[0]["format"] == "safetensors"

    # 4. Test /api/bundle inspection
    rel_bundle = found[0]["path"]
    code, _, body = _make_mock_request(handler_cls, f"/api/bundle?path={rel_bundle}")
    assert code == 200
    bundle_detail = json.loads(body.decode("utf-8"))
    assert bundle_detail["family"] == "linear_test"
    assert bundle_detail["weights_sha256"] is not None
    assert len(bundle_detail["tensors"]) > 0

    # 5. Test /api/artifact image serving
    code, headers, img_bytes = _make_mock_request(
        handler_cls,
        "/api/artifact?exp=exp_a&run=run_01&file=test_sample_step_0000.png",
    )
    assert code == 200
    assert headers["content-type"] == "image/png"
    assert len(img_bytes) > 0

    # 6. Test / dashboard SPA HTML
    code, headers, html_bytes = _make_mock_request(handler_cls, "/")
    assert code == 200
    assert "text/html" in headers["content-type"]
    assert b"mangosta" in html_bytes
    assert b"LIVE WATCH" in html_bytes
    assert b"Real Data Validation" in html_bytes


def test_viewer_error_branches(tmp_path: Path) -> None:
    """Test error handling in viewer handler."""
    handler_cls = create_viewer_handler(tmp_path)

    # Run not found
    code, _, _ = _make_mock_request(handler_cls, "/api/run?exp=ghost&run=ghost")
    assert code == 404

    # Artifact not found
    code, _, _ = _make_mock_request(handler_cls, "/api/artifact?exp=ghost&run=ghost&file=ghost.png")
    assert code == 404

    # Bundle not found
    code, _, _ = _make_mock_request(handler_cls, "/api/bundle?path=ghost.bundle")
    assert code == 404

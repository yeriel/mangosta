"""Targeted tests to push code coverage past 85% for Mangosta Phase 4."""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from mangosta.artifacts.bundle import (
    _capture_rng_states,
    _restore_rng_states,
    read_bundle_manifest,
    save_bundle,
)
from mangosta.artifacts.loader import load_bundle, resume_training
from mangosta.cli.main import _parse_safetensors_header, inspect_bundle, main, run_report, run_view
from mangosta.core.registry import register_model
from mangosta.exceptions import BundleCorruptedError, ResumeStateError
from mangosta.export.onnx import _DictOutputWrapper, _load_weights_into_model, export_onnx
from mangosta.serving.runner import LoadedONNXBundle, ONNXRunner, load_onnx_bundle
from mangosta.tracking.viewer import _load_run_details


@register_model("boost_linear", "1.0.0")
class BoostModel(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DictOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 2)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"out1": self.fc1(x), "out2": self.fc2(x)}


class TupleOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 2)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fc1(x), self.fc2(x)


class BadDictModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_parse_safetensors_header_edge_cases() -> None:
    # 1. Less than 8 bytes
    assert _parse_safetensors_header(b"123") == {}

    # 2. Header length claims more bytes than available
    hdr_too_long = struct.pack("<Q", 100) + b"{}"
    assert _parse_safetensors_header(hdr_too_long) == {}

    # 3. Invalid UTF-8 / JSON
    hdr_invalid_json = struct.pack("<Q", 5) + b"not{}"
    assert _parse_safetensors_header(hdr_invalid_json) == {}

    # 4. Valid header
    valid_data = json.dumps({"weight": {"shape": [2, 4], "dtype": "F32"}}).encode("utf-8")
    hdr_valid = struct.pack("<Q", len(valid_data)) + valid_data
    parsed = _parse_safetensors_header(hdr_valid)
    assert "weight" in parsed
    assert parsed["weight"]["shape"] == [2, 4]


def test_inspect_bundle_corrupt_files(tmp_path: Path) -> None:
    # Corrupted zip
    corrupt_file = tmp_path / "corrupt.bundle"
    corrupt_file.write_bytes(b"not a zip file")

    with pytest.raises(SystemExit):
        inspect_bundle(str(corrupt_file), as_json=False)

    with pytest.raises(SystemExit):
        inspect_bundle(str(corrupt_file), as_json=True)


def test_cli_main_init_and_export_onnx(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # CLI init
    init_dir = tmp_path / "cli_project"
    with patch("sys.argv", ["mangosta", "init", str(init_dir)]):
        main()
    assert (init_dir / "pyproject.toml").exists()

    # Save a bundle for export-onnx
    bundle_path = tmp_path / "cli_model.bundle"
    from mangosta.core.registry import ModelRegistry
    ModelRegistry.register_model("boost_linear", "1.0.0", BoostModel, allow_override=True)
    model = BoostModel()
    save_bundle(
        output_path=bundle_path,
        model=model,
        family="boost_linear",
        version="1.0.0",
        sample_input=torch.randn(1, 4),
    )

    # CLI export-onnx
    out_onnx = tmp_path / "cli_exported.onnx"
    with patch(
        "sys.argv",
        [
            "mangosta",
            "export-onnx",
            str(bundle_path),
            "-o",
            str(out_onnx),
            "--input-shape",
            "1",
            "4",
        ],
    ):
        main()

    assert out_onnx.exists()
    assert out_onnx.stat().st_size > 0


def test_cli_report_and_view_error_handling(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nonexistent"

    with pytest.raises(SystemExit):
        run_report(str(nonexistent), str(tmp_path / "report.html"))

    with pytest.raises(SystemExit):
        run_view(str(nonexistent))


def test_bundle_rng_and_config_file(tmp_path: Path) -> None:
    # Test _restore_rng_states with edge cases
    _restore_rng_states(None)  # type: ignore[arg-type]
    _restore_rng_states({})
    rng = _capture_rng_states()
    _restore_rng_states(rng)

    # Test save_bundle with config file path
    cfg_file = tmp_path / "experiment_cfg.yaml"
    cfg_file.write_text("learning_rate: 0.005\nbatch_size: 32\n", encoding="utf-8")

    bundle_path = tmp_path / "with_cfg_file.bundle"
    model = BoostModel()
    save_bundle(
        output_path=bundle_path,
        model=model,
        family="boost_linear",
        config=cfg_file,
    )

    manifest = read_bundle_manifest(bundle_path)
    assert manifest["family"] == "boost_linear"

    with zipfile.ZipFile(bundle_path, "r") as zf:
        assert "config.yaml" in zf.namelist()
        content = zf.read("config.yaml").decode("utf-8")
        assert "learning_rate: 0.005" in content


def test_bundle_manifest_errors(tmp_path: Path) -> None:
    nonexistent = tmp_path / "ghost.bundle"
    with pytest.raises(FileNotFoundError):
        read_bundle_manifest(nonexistent)

    # Missing manifest.json
    no_manifest_zip = tmp_path / "no_manifest.bundle"
    with zipfile.ZipFile(no_manifest_zip, "w") as zf:
        zf.writestr("dummy.txt", "hello")
    with pytest.raises(BundleCorruptedError):
        read_bundle_manifest(no_manifest_zip)

    # Corrupt zip
    bad_zip = tmp_path / "bad.bundle"
    bad_zip.write_bytes(b"garbage")
    with pytest.raises(BundleCorruptedError):
        read_bundle_manifest(bad_zip)


def test_loader_exceptions(tmp_path: Path) -> None:
    # File not found
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "none.bundle")

    # Missing manifest.json
    no_manifest = tmp_path / "no_manifest2.bundle"
    with zipfile.ZipFile(no_manifest, "w") as zf:
        zf.writestr("something.bin", b"data")
    with pytest.raises(BundleCorruptedError):
        load_bundle(no_manifest)

    # Missing model.safetensors
    dummy_model = BoostModel()
    no_weights = tmp_path / "no_weights.bundle"
    with zipfile.ZipFile(no_weights, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"family": "boost_linear", "weights_format": "safetensors"}))
    with pytest.raises(BundleCorruptedError):
        load_bundle(no_weights, model=dummy_model)

    # Corrupted weights sha256
    bad_hash_bundle = tmp_path / "bad_hash.bundle"
    with zipfile.ZipFile(bad_hash_bundle, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"family": "boost_linear", "weights_format": "safetensors", "weights_sha256": "abcdef"}),
        )
        zf.writestr("model.safetensors", b"corrupted bytes")
    with pytest.raises(BundleCorruptedError):
        load_bundle(bad_hash_bundle, model=dummy_model)

    # resume_training on non-checkpoint bundle
    non_chk_bundle = tmp_path / "non_chk.bundle"
    model = BoostModel()
    save_bundle(non_chk_bundle, model=model)
    with pytest.raises(ResumeStateError, match="does not contain 'train_state.pt'"):
        resume_training(non_chk_bundle, model=model)


def test_export_onnx_weights_loader_and_dict_wrapper(tmp_path: Path) -> None:
    # 1. _load_weights_into_model with missing file
    model = BoostModel()
    with pytest.raises(FileNotFoundError):
        _load_weights_into_model(model, tmp_path / "missing.pth", torch.device("cpu"))

    # 2. _load_weights_into_model with .pth containing state_dict
    pth_file = tmp_path / "weights.pth"
    torch.save({"state_dict": model.state_dict()}, pth_file)
    _load_weights_into_model(model, pth_file, torch.device("cpu"))

    # 3. _load_weights_into_model with non-dict
    pth_bad = tmp_path / "bad.pth"
    torch.save("not a state dict", pth_bad)
    with pytest.raises(ValueError, match="Expected state_dict dictionary"):
        _load_weights_into_model(model, pth_bad, torch.device("cpu"))

    # 4. _DictOutputWrapper errors
    bad_dict_model = BadDictModel()
    wrapper_bad = _DictOutputWrapper(bad_dict_model, ["key1"])
    with pytest.raises(TypeError, match="expected to return a dict"):
        wrapper_bad(torch.randn(1, 4))

    dict_model = DictOutputModel()
    wrapper_missing_key = _DictOutputWrapper(dict_model, ["nonexistent_key"])
    with pytest.raises(KeyError, match="Missing expected keys"):
        wrapper_missing_key(torch.randn(1, 4))

    # 5. Export with DictOutputModel
    onnx_dict_dest = tmp_path / "dict_model.onnx"
    export_onnx(
        model_or_bundle=dict_model,
        output_path=onnx_dict_dest,
        sample_input=torch.randn(1, 4),
        output_mapping=["out1", "out2"],
    )
    assert onnx_dict_dest.exists()

    # 6. Export without sample input or shape
    with pytest.raises(ValueError, match="Must provide either 'sample_input' or 'sample_input_shape'"):
        export_onnx(model, tmp_path / "fail.onnx")


def test_serving_runner_coverage(tmp_path: Path) -> None:
    # 1. Nonexistent bundle
    with pytest.raises(FileNotFoundError):
        load_onnx_bundle(tmp_path / "ghost.onnx.bundle")

    # 2. Bad zip archive
    corrupt_bundle = tmp_path / "corrupt.onnx.bundle"
    corrupt_bundle.write_bytes(b"not a zip")
    with pytest.raises(BundleCorruptedError):
        load_onnx_bundle(corrupt_bundle)

    # 3. Missing model.onnx
    missing_model_bundle = tmp_path / "no_model.onnx.bundle"
    with zipfile.ZipFile(missing_model_bundle, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"weights_format": "onnx"}))
    with pytest.raises(BundleCorruptedError):
        load_onnx_bundle(missing_model_bundle)

    # 4. Corrupted SHA256 in onnx bundle
    bad_hash_onnx = tmp_path / "bad_sha.onnx.bundle"
    with zipfile.ZipFile(bad_hash_onnx, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"weights_format": "onnx", "model_sha256": "invalid_hash"}),
        )
        zf.writestr("model.onnx", b"onnx content")
    with pytest.raises(BundleCorruptedError, match="integrity check failed"):
        load_onnx_bundle(bad_hash_onnx)

    # 5. Export valid bundle and test ONNXRunner methods
    model = BoostModel().eval()
    onnx_bundle_path = tmp_path / "valid.onnx.bundle"
    export_onnx(
        model_or_bundle=model,
        output_path=onnx_bundle_path,
        sample_input=torch.randn(1, 4),
        as_bundle=True,
    )

    runner = ONNXRunner(onnx_bundle_path)

    # Run with mismatched input length when validate=False
    with pytest.raises(ValueError, match="Model expects"):
        runner.predict([torch.randn(1, 4), torch.randn(1, 4)], validate=False)

    # Test ONNXRunner on standalone ONNX
    standalone_onnx = tmp_path / "standalone.onnx"
    export_onnx(
        model_or_bundle=model,
        output_path=standalone_onnx,
        sample_input=torch.randn(1, 4),
        as_bundle=False,
    )
    direct_runner = ONNXRunner(standalone_onnx)
    out = direct_runner.predict(torch.randn(1, 4).numpy())
    assert out.shape == (1, 2)

    # Test ONNXRunner with dict/mapping output model
    dict_model = DictOutputModel().eval()
    dict_bundle_path = tmp_path / "dict_model.onnx.bundle"
    export_onnx(
        model_or_bundle=dict_model,
        output_path=dict_bundle_path,
        sample_input=torch.randn(1, 4),
        output_mapping=["out1", "out2"],
        as_bundle=True,
    )
    dict_runner = ONNXRunner(dict_bundle_path)
    dict_out = dict_runner.predict(torch.randn(1, 4).numpy())
    assert isinstance(dict_out, dict)
    assert "out1" in dict_out and "out2" in dict_out

    # Test ONNXRunner with tuple/sequence output model and from_bundle classmethod
    tuple_model = TupleOutputModel().eval()
    tuple_bundle_path = tmp_path / "tuple_model.onnx.bundle"
    export_onnx(
        model_or_bundle=tuple_model,
        output_path=tuple_bundle_path,
        sample_input=torch.randn(1, 4),
        output_names=["y1", "y2"],
        family="tuple_family",
        version="2.0.0",
        config={"batch_size": 16},
        as_bundle=True,
    )
    loaded_tuple = ONNXRunner.from_bundle(tuple_bundle_path)
    tuple_out = loaded_tuple.predict(torch.randn(1, 4).numpy())
    assert isinstance(tuple_out, tuple)
    assert len(tuple_out) == 2


def test_viewer_get_run_data_fallback(tmp_path: Path) -> None:
    exp_dir = tmp_path / "runs" / "exp_test" / "run_01"
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)

    # Create dummy artifact file
    (artifacts_dir / "sample_img.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Meta without artifacts list
    meta_path = exp_dir / "run.json"
    meta_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    # Metrics file with some invalid lines
    metrics_path = exp_dir / "metrics.jsonl"
    metrics_path.write_text(
        '{"name": "loss", "step": 1, "value": 0.5}\ncorrupt json line\n',
        encoding="utf-8",
    )

    data = _load_run_details(tmp_path / "runs", "exp_test", "run_01")
    assert data is not None
    assert data["status"] == "completed"
    assert "loss" in data["metrics"]
    assert len(data["metrics"]["loss"]) == 1
    assert len(data["artifacts"]) == 1
    assert data["artifacts"][0]["name"] == "sample_img.png"
    assert data["artifacts"][0]["type"] == "image"

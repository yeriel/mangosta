"""Tests for project initializer scaffolding (mangosta init)."""

from __future__ import annotations

import py_compile
from pathlib import Path
from unittest.mock import patch

from mangosta.cli.init_project import init_project
from mangosta.cli.main import main


def test_init_project_structure(tmp_path: Path):
    target = tmp_path / "test_proj"
    out_path = init_project(str(target))

    assert out_path == target.resolve()

    # Lean config
    assert (target / "configs" / "exp_baseline.yaml").is_file()

    # Entrypoints
    assert (target / "entrypoints" / "__init__.py").is_file()
    assert (target / "entrypoints" / "train.py").is_file()
    assert (target / "entrypoints" / "infer.py").is_file()
    assert (target / "entrypoints" / "benchmark.py").is_file()

    # Source models
    assert (target / "src" / "__init__.py").is_file()
    assert (target / "src" / "models.py").is_file()

    # Meta files
    assert (target / "pyproject.toml").is_file()
    assert (target / ".agents" / "mangosta.md").is_file()
    assert (target / ".gitignore").is_file()
    assert (target / "README.md").is_file()

    # Verify zero empty artifact directories are created upfront (lazy creation)
    assert not (target / "checkpoints").exists()
    assert not (target / "experiments").exists()
    assert not (target / "outputs").exists()

    # Verify pyproject.toml contains entrypoint scripts for uv
    pyproject_text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "entrypoints.train:main" in pyproject_text
    assert "entrypoints.infer:main" in pyproject_text
    assert "entrypoints.benchmark:main" in pyproject_text
    assert "packages = [\"src\", \"entrypoints\"]" in pyproject_text

    # Verify all generated python files compile cleanly
    for py_file in target.rglob("*.py"):
        py_compile.compile(str(py_file), doraise=True)


def test_init_cli_invocation(tmp_path: Path):
    target = tmp_path / "cli_proj"
    with patch("sys.argv", ["mangosta", "init", str(target)]):
        main()

    assert (target / "entrypoints" / "train.py").is_file()
    assert (target / "configs" / "exp_baseline.yaml").is_file()
    assert (target / "pyproject.toml").is_file()
    assert (target / ".agents" / "mangosta.md").is_file()

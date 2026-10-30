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
    assert (target / "configs" / "base.yaml").is_file()
    assert (target / "configs" / "experiments" / "exp_baseline.yaml").is_file()
    assert (target / "src" / "data" / "loader.py").is_file()
    assert (target / "src" / "models" / "architectures" / "baseline.py").is_file()
    assert (target / "src" / "utils" / "config.py").is_file()
    assert (target / "train.py").is_file()
    assert (target / "infer.py").is_file()
    assert (target / "benchmark.py").is_file()
    assert (target / ".gitignore").is_file()
    assert (target / "README.md").is_file()

    # Verify all generated python files compile cleanly
    for py_file in target.rglob("*.py"):
        py_compile.compile(str(py_file), doraise=True)


def test_init_cli_invocation(tmp_path: Path):
    target = tmp_path / "cli_proj"
    with patch("sys.argv", ["mangosta", "init", str(target)]):
        main()

    assert (target / "train.py").is_file()
    assert (target / "configs" / "base.yaml").is_file()

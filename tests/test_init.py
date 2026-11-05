"""Tests for project initializer scaffolding (mangosta init)."""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path
from unittest.mock import patch

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

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

    # Rigorous TOML parsing audit
    pyproject_bytes = (target / "pyproject.toml").read_bytes()
    toml_data = tomllib.loads(pyproject_bytes.decode("utf-8"))

    # Audit scripts
    scripts = toml_data["project"]["scripts"]
    assert scripts["train"] == "entrypoints.train:main"
    assert scripts["infer"] == "entrypoints.infer:main"
    assert scripts["benchmark"] == "entrypoints.benchmark:main"

    # Audit dependencies
    deps = toml_data["project"]["dependencies"]
    assert any("torch>=" in d for d in deps)
    assert any("numpy>=2.0.0" in d for d in deps)

    # Audit wheel packaging for Hatchling
    wheel_pkgs = toml_data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert wheel_pkgs == ["src", "entrypoints"]

    # Audit agent context file
    agent_rules = (target / ".agents" / "mangosta.md").read_text(encoding="utf-8")
    assert "Agent Instructions: Mangosta Project Conventions" in agent_rules
    assert "uv run train" in agent_rules
    assert "uv run infer" in agent_rules

    # Audit .gitignore
    gitignore_text = (target / ".gitignore").read_text(encoding="utf-8")
    assert "checkpoints/" in gitignore_text
    assert "experiments/" in gitignore_text
    assert "*.bundle" in gitignore_text

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

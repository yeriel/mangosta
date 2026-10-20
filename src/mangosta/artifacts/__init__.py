"""Artifact management: atomic packaging, fault-tolerant checkpoints, and universal loading."""

from __future__ import annotations

from mangosta.artifacts.bundle import (
    read_bundle_manifest,
    save_bundle,
)
from mangosta.artifacts.loader import (
    LoadedBundle,
    ResumeState,
    load_bundle,
    resume_training,
)

__all__ = [
    "save_bundle",
    "load_bundle",
    "read_bundle_manifest",
    "LoadedBundle",
    "ResumeState",
    "resume_training",
]

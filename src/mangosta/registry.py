"""Immutable model architecture catalog and component registry with SemVer dispatch."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import torch.nn as nn
from packaging.version import InvalidVersion, Version

from mangosta.exceptions import (
    ComponentNotFoundError,
    ModelNotFoundError,
    RegistryCollisionError,
)

T = TypeVar("T", bound=type[nn.Module])
C = TypeVar("C", bound=Any)


class ModelRegistry:
    """Thread-safe catalog for neural network architectures and auxiliary components."""

    _models: dict[tuple[str, str], type[nn.Module]] = {}
    _components: dict[tuple[str, str], Any] = {}

    @classmethod
    def register_model(
        cls,
        family: str,
        version: str,
        model_cls: type[nn.Module],
        allow_override: bool = False,
    ) -> None:
        """Register a model architecture class under a family and version."""
        if not issubclass(model_cls, nn.Module):
            raise TypeError(f"Registered model must inherit from torch.nn.Module, got {model_cls}")

        key = (family.strip().lower(), version.strip())
        if key in cls._models and not allow_override:
            raise RegistryCollisionError(
                f"Model family '{family}' version '{version}' is already registered with {cls._models[key]}. "
                "Set allow_override=True if you intentionally wish to overwrite."
            )
        cls._models[key] = model_cls

    @classmethod
    def get_model_cls(cls, family: str, version: str | None = None) -> tuple[type[nn.Module], str]:
        """Look up a model class.

        If version is omitted or None, resolves the highest SemVer available.
        Returns (model_class, resolved_version).
        """
        family_clean = family.strip().lower()
        matching_versions: list[str] = [v for (f, v) in cls._models.keys() if f == family_clean]

        if not matching_versions:
            available_families = sorted({f for (f, _) in cls._models.keys()})
            raise ModelNotFoundError(
                f"Model family '{family}' not found. Available families: {available_families}"
            )

        if version is not None:
            v_clean = version.strip()
            key = (family_clean, v_clean)
            if key not in cls._models:
                raise ModelNotFoundError(
                    f"Model family '{family}' found, but version '{version}' does not exist. Available: {sorted(matching_versions)}"
                )
            return cls._models[key], v_clean

        # Resolve latest SemVer
        resolved_version = cls._resolve_latest_version(matching_versions)
        return cls._models[(family_clean, resolved_version)], resolved_version

    @classmethod
    def build(cls, family: str, version: str | None = None, **init_args: Any) -> nn.Module:
        """Dynamically instantiate a model registered under family and version."""
        model_cls, resolved_version = cls.get_model_cls(family, version)
        try:
            return model_cls(**init_args)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate model '{family}' (v{resolved_version}) with args {init_args}: {e}"
            ) from e

    @classmethod
    def register_component(
        cls, kind: str, name: str, component: Any, allow_override: bool = False
    ) -> None:
        """Register an auxiliary component (e.g. loss, optimizer factory, lr_scheduler)."""
        key = (kind.strip().lower(), name.strip().lower())
        if key in cls._components and not allow_override:
            raise RegistryCollisionError(
                f"Component of kind '{kind}' named '{name}' is already registered. "
                "Set allow_override=True to overwrite."
            )
        cls._components[key] = component

    @classmethod
    def get_component(cls, kind: str, name: str) -> Any:
        """Retrieve an auxiliary component by kind and name."""
        key = (kind.strip().lower(), name.strip().lower())
        if key not in cls._components:
            available = [n for (k, n) in cls._components.keys() if k == key[0]]
            raise ComponentNotFoundError(
                f"Component kind '{kind}' named '{name}' not found. Available for '{kind}': {available}"
            )
        return cls._components[key]

    @classmethod
    def list_models(cls) -> list[dict[str, str]]:
        """Return a list of all registered models with family, version, and class name."""
        return [
            {
                "family": family,
                "version": version,
                "class_name": model_cls.__name__,
                "module": model_cls.__module__,
            }
            for (family, version), model_cls in sorted(cls._models.items())
        ]

    @classmethod
    def clear(cls) -> None:
        """Reset all registered models and components (primarily for tests)."""
        cls._models.clear()
        cls._components.clear()

    @staticmethod
    def _resolve_latest_version(versions: list[str]) -> str:
        """Resolve the latest version string using packaging.version if possible, else string sort."""
        parsed_versions: list[tuple[Version | None, str]] = []
        for v in versions:
            try:
                parsed_versions.append((Version(v), v))
            except InvalidVersion:
                parsed_versions.append((None, v))

        # Sort: valid SemVer first by version, fallback string sort for others
        sorted_vers = sorted(
            parsed_versions,
            key=lambda item: (item[0] is not None, item[0] if item[0] is not None else item[1]),
        )
        return sorted_vers[-1][1]


def register_model(
    family: str,
    version: str,
    allow_override: bool = False,
) -> Callable[[T], T]:
    """Decorator to register a PyTorch nn.Module class in ModelRegistry.

    Example:
        @register_model(family="mlp", version="1.0.0")
        class MyMLP(nn.Module):
            ...
    """

    def decorator(cls: T) -> T:
        ModelRegistry.register_model(
            family=family,
            version=version,
            model_cls=cls,
            allow_override=allow_override,
        )
        return cls

    return decorator


def register_component(
    kind: str,
    name: str,
    allow_override: bool = False,
) -> Callable[[C], C]:
    """Decorator to register an auxiliary component (loss, optimizer, scheduler)."""

    def decorator(comp: C) -> C:
        ModelRegistry.register_component(
            kind=kind,
            name=name,
            component=comp,
            allow_override=allow_override,
        )
        return comp

    return decorator

"""Immutable model architecture catalog and component registry with SemVer dispatch."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

try:
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False

from packaging.version import InvalidVersion, Version

from mangosta.exceptions import (
    ComponentNotFoundError,
    ModelNotFoundError,
    RegistryCollisionError,
    TrainerNotFoundError,
)

if HAS_TORCH and nn is not None:
    T = TypeVar("T", bound=type[nn.Module])
else:
    T = TypeVar("T", bound=Any)  # type: ignore[misc]
C = TypeVar("C", bound=Any)


def _resolve_latest_version(versions: list[str]) -> str:
    """Resolve the latest version string using packaging.version if possible, else string sort."""
    parsed_versions: list[tuple[Version | None, str]] = []
    for v in versions:
        try:
            parsed_versions.append((Version(v), v))
        except InvalidVersion:
            parsed_versions.append((None, v))

    sorted_vers = sorted(
        parsed_versions,
        key=lambda item: (item[0] is not None, item[0] if item[0] is not None else item[1]),
    )
    return sorted_vers[-1][1]


def _lookup_registered_cls(
    catalog: dict[tuple[str, str], Any],
    family: str,
    version: str | None,
    not_found_exc: type[Exception],
    entity_label: str,
) -> tuple[Any, str]:
    """Shared SemVer & tolerant normalized lookup for ModelRegistry and TrainerRegistry."""
    family_clean = family.strip().lower()
    matching_versions: list[str] = [v for (f, v) in catalog.keys() if f == family_clean]

    if not matching_versions:
        def _norm(s: str) -> str:
            return s.replace("_", "").replace("-", "")

        target_norm = _norm(family_clean)
        for (f, _) in catalog.keys():
            if _norm(f) == target_norm:
                family_clean = f
                matching_versions = [ver for (fam, ver) in catalog.keys() if fam == family_clean]
                break

    if not matching_versions:
        available_families = sorted({f for (f, _) in catalog.keys()})
        raise not_found_exc(
            f"{entity_label} family '{family}' not found. Available families: {available_families}"
        )

    if version is not None:
        v_clean = version.strip()
        key = (family_clean, v_clean)
        if key not in catalog:
            raise not_found_exc(
                f"{entity_label} family '{family}' found, but version '{version}' does not exist. "
                f"Available: {sorted(matching_versions)}"
            )
        return catalog[key], v_clean

    resolved_version = _resolve_latest_version(matching_versions)
    return catalog[(family_clean, resolved_version)], resolved_version


class ModelRegistry:
    """Thread-safe catalog for neural network architectures and auxiliary components."""

    _models: dict[tuple[str, str], Any] = {}
    _components: dict[tuple[str, str], Any] = {}

    @classmethod
    def register_model(
        cls,
        family: str,
        version: str,
        model_cls: Any,
        allow_override: bool = False,
    ) -> None:
        """Register a model architecture class under a family and version."""
        if not HAS_TORCH or nn is None:
            raise ImportError(
                "register_model requires PyTorch. Please install torch: pip install torch"
            )
        if not issubclass(model_cls, nn.Module):
            raise TypeError(f"Registered model must inherit from torch.nn.Module, got {model_cls}")

        key = (family.strip().lower(), version.strip())
        if key in cls._models and not allow_override:
            raise RegistryCollisionError(
                f"Model family '{family}' version '{version}' is already registered with {cls._models[key]}. "
                "Set allow_override=True if you intentionally wish to overwrite."
            )
        cls._models[key] = model_cls
        try:
            setattr(model_cls, "_mangosta_family", family.strip().lower())
            setattr(model_cls, "_mangosta_version", version.strip())
        except Exception:
            pass

    @classmethod
    def get_model_cls(cls, family: str, version: str | None = None) -> tuple[Any, str]:
        """Look up a model class.

        If version is omitted or None, resolves the highest SemVer available.
        Returns (model_class, resolved_version).
        """
        return _lookup_registered_cls(
            catalog=cls._models,
            family=family,
            version=version,
            not_found_exc=ModelNotFoundError,
            entity_label="Model",
        )

    @classmethod
    def build(cls, family: str, version: str | None = None, **init_args: Any) -> Any:
        """Dynamically instantiate a model registered under family and version."""
        model_cls, resolved_version = cls.get_model_cls(family, version)
        try:
            return model_cls(**init_args)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate model '{family}' (v{resolved_version}) with args {init_args}: {e}"
            ) from e

    @classmethod
    def build_from_config(cls, config: Any) -> Any:
        """Instantiate a registered model directly from a MangostaConfig or dictionary."""
        from mangosta.core.config import load_config

        cfg = load_config(config)
        return cls.build(cfg.model.family, version=cfg.model.version, **cfg.model.init_args)

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


class TrainerRegistry:
    """Thread-safe catalog for BaseTrainer subclasses with SemVer dispatch."""

    _trainers: dict[tuple[str, str], Any] = {}

    @classmethod
    def register_trainer(
        cls,
        family: str,
        version: str = "1.0.0",
        trainer_cls: Any = None,
        allow_override: bool = False,
    ) -> None:
        """Register a BaseTrainer subclass under a family and version."""
        key = (family.strip().lower(), version.strip())
        if key in cls._trainers and not allow_override:
            raise RegistryCollisionError(
                f"Trainer family '{family}' version '{version}' is already registered with {cls._trainers[key]}. "
                "Set allow_override=True if you intentionally wish to overwrite."
            )
        cls._trainers[key] = trainer_cls
        try:
            setattr(trainer_cls, "_mangosta_family", family.strip().lower())
            setattr(trainer_cls, "_mangosta_version", version.strip())
        except Exception:
            pass

    @classmethod
    def get_trainer_cls(cls, family: str, version: str | None = None) -> tuple[Any, str]:
        """Look up a registered trainer class using SemVer & tolerant normalized matching."""
        return _lookup_registered_cls(
            catalog=cls._trainers,
            family=family,
            version=version,
            not_found_exc=TrainerNotFoundError,
            entity_label="Trainer",
        )

    @classmethod
    def build(cls, family: str, version: str | None = None, **kwargs: Any) -> Any:
        """Dynamically instantiate a registered trainer."""
        trainer_cls, resolved_version = cls.get_trainer_cls(family, version)
        try:
            return trainer_cls(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate trainer '{family}' (v{resolved_version}): {e}"
            ) from e

    @classmethod
    def build_from_config(cls, config: Any, model: Any, **kwargs: Any) -> Any:
        """Instantiate a registered trainer directly from a MangostaConfig or dictionary."""
        from mangosta.core.config import load_config

        cfg = load_config(config)
        return cls.build(
            cfg.model.family,
            version=cfg.model.version,
            model=model,
            config=cfg,
            **kwargs,
        )

    @classmethod
    def list_trainers(cls) -> list[dict[str, str]]:
        """Return a list of all registered trainers with family, version, and class name."""
        return [
            {
                "family": family,
                "version": version,
                "class_name": trainer_cls.__name__,
                "module": trainer_cls.__module__,
            }
            for (family, version), trainer_cls in sorted(cls._trainers.items())
        ]

    @classmethod
    def clear(cls) -> None:
        """Reset all registered trainers."""
        cls._trainers.clear()


def register_trainer(
    family: str,
    version: str = "1.0.0",
    allow_override: bool = False,
) -> Callable[[C], C]:
    """Decorator to register a BaseTrainer subclass in TrainerRegistry.

    Example:
        @register_trainer(family="sensor_classifier", version="1.0.0")
        class SensorTrainer(BaseTrainer):
            ...
    """

    def decorator(cls: C) -> C:
        TrainerRegistry.register_trainer(
            family=family,
            version=version,
            trainer_cls=cls,
            allow_override=allow_override,
        )
        return cls

    return decorator

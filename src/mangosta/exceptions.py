"""Custom structured exceptions for the mangosta library."""

from __future__ import annotations


class MangostaError(Exception):
    """Base exception for all errors raised by mangosta."""


class RegistryError(MangostaError):
    """Base exception for registry operations."""


class RegistryCollisionError(RegistryError):
    """Raised when attempting to register a model or component that already exists."""


class ModelNotFoundError(RegistryError):
    """Raised when a requested model architecture or version is not found in the registry."""


class ComponentNotFoundError(RegistryError):
    """Raised when a requested auxiliary component is not found in the registry."""


class SchemaError(MangostaError):
    """Base exception for schema inference and validation."""


class SchemaValidationError(SchemaError):
    """Raised when an input or output tensor fails schema validation."""


class BundleError(MangostaError):
    """Base exception for bundle packaging and loading."""


class BundleCorruptedError(BundleError):
    """Raised when a .bundle archive is corrupted, missing required files, or fails integrity checks."""


class ResumeStateError(BundleError):
    """Raised when a bundle does not contain train state or train state cannot be applied."""

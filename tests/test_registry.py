"""Unit tests for mangosta.registry: immutable architecture catalog and SemVer dispatch."""

from __future__ import annotations

import pytest

from mangosta.core.registry import ModelRegistry, register_component, register_model
from mangosta.exceptions import (
    ComponentNotFoundError,
    ModelNotFoundError,
    RegistryCollisionError,
)
from tests.conftest import DummyMLP


def test_register_and_build_model() -> None:
    @register_model(family="mlp", version="1.0.0")
    class MLPv1(DummyMLP):
        pass

    model = ModelRegistry.build("mlp", version="1.0.0", in_features=8, out_features=4)
    assert isinstance(model, MLPv1)
    assert model.in_features == 8
    assert model.out_features == 4


def test_collision_raises_error() -> None:
    @register_model(family="resnet", version="0.1.0")
    class ResNetA(DummyMLP):
        pass

    with pytest.raises(RegistryCollisionError, match="already registered"):

        @register_model(family="resnet", version="0.1.0")
        class ResNetB(DummyMLP):
            pass

    # With allow_override=True, it succeeds
    @register_model(family="resnet", version="0.1.0", allow_override=True)
    class ResNetC(DummyMLP):
        pass

    model_cls, _ = ModelRegistry.get_model_cls("resnet", "0.1.0")
    assert model_cls is ResNetC


def test_register_non_module_fails() -> None:
    with pytest.raises(TypeError, match="must inherit from torch.nn.Module"):

        @register_model(family="invalid", version="1.0.0")  # type: ignore
        class NotAModule:
            pass


def test_semver_latest_resolution() -> None:
    @register_model(family="transformer", version="1.0.0")
    class TransV1(DummyMLP):
        pass

    @register_model(family="transformer", version="1.2.0")
    class TransV2(DummyMLP):
        pass

    @register_model(family="transformer", version="1.10.0")
    class TransV10(DummyMLP):
        pass

    @register_model(family="transformer", version="2.0.0-rc.1")
    class TransV2RC(DummyMLP):
        pass

    # Without specifying version, builds highest SemVer
    cls_resolved, ver_resolved = ModelRegistry.get_model_cls("transformer")
    assert ver_resolved == "2.0.0-rc.1"
    assert cls_resolved is TransV2RC


def test_model_not_found_errors() -> None:
    with pytest.raises(ModelNotFoundError, match="Model family 'unknown' not found"):
        ModelRegistry.build("unknown")

    @register_model(family="cnn", version="1.0.0")
    class CNN(DummyMLP):
        pass

    with pytest.raises(ModelNotFoundError, match="version '2.0.0' does not exist"):
        ModelRegistry.build("cnn", version="2.0.0")


def test_register_and_get_component() -> None:
    @register_component(kind="loss", name="focal")
    def focal_loss(gamma: float = 2.0) -> str:
        return f"focal_gamma_{gamma}"

    comp = ModelRegistry.get_component("loss", "focal")
    assert comp(gamma=3.0) == "focal_gamma_3.0"

    with pytest.raises(RegistryCollisionError):
        register_component(kind="loss", name="focal")(lambda: None)

    with pytest.raises(ComponentNotFoundError):
        ModelRegistry.get_component("loss", "nonexistent")


def test_list_models() -> None:
    @register_model(family="model_a", version="1.0.0")
    class A(DummyMLP):
        pass

    models = ModelRegistry.list_models()
    assert len(models) == 1
    assert models[0]["family"] == "model_a"
    assert models[0]["version"] == "1.0.0"
    assert models[0]["class_name"] == "A"

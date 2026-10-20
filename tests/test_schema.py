"""Unit tests for mangosta.schema: contract inspection and validation."""

from __future__ import annotations

import pytest
import torch

from mangosta.exceptions import SchemaValidationError
from mangosta.schema import ModelSchema, TensorSpec, infer_schema
from tests.conftest import DummyDictInput, DummyMLP, DummyMultiInput


def test_tensor_spec_matching() -> None:
    spec = TensorSpec(name="x", shape=[None, 4], dtype="float32")
    # Matches any batch size
    t1 = torch.randn(1, 4, dtype=torch.float32)
    t8 = torch.randn(8, 4, dtype=torch.float32)
    assert spec.matches(t1)[0] is True
    assert spec.matches(t8)[0] is True

    # Fails on wrong feature dimension
    t_wrong_dim = torch.randn(8, 5, dtype=torch.float32)
    ok, err = spec.matches(t_wrong_dim)
    assert ok is False
    assert "dimension 1 mismatch" in str(err)

    # Fails on wrong dtype
    t_wrong_dtype = torch.randn(8, 4, dtype=torch.float64)
    ok, err = spec.matches(t_wrong_dtype)
    assert ok is False
    assert "expected dtype float32" in str(err)

    # Fails on non-tensor
    ok, err = spec.matches("not a tensor")  # type: ignore
    assert ok is False


def test_infer_schema_single_tensor() -> None:
    model = DummyMLP(in_features=4, out_features=2)
    model.train()

    sample = torch.randn(2, 4, dtype=torch.float32)
    schema = infer_schema(model, sample, dynamic_batch_dim=0)

    # Model training state was preserved
    assert model.training is True

    # Inputs and outputs inspected
    assert schema.input_structure == "tensor"
    assert schema.output_structure == "tensor"
    assert isinstance(schema.inputs, list)
    assert schema.inputs[0].shape == [None, 4]
    assert schema.inputs[0].dtype == "float32"

    assert isinstance(schema.outputs, list)
    assert schema.outputs[0].shape == [None, 2]
    assert schema.outputs[0].dtype == "float32"


def test_infer_schema_multi_input() -> None:
    model = DummyMultiInput(feat_dim=4)
    a = torch.randn(3, 4, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)

    schema = infer_schema(model, (a, b))
    assert schema.input_structure == "sequence"
    assert len(schema.inputs) == 2
    assert schema.inputs[0].shape == [None, 4]
    assert schema.inputs[1].shape == [None, 4]


def test_infer_schema_dict_input() -> None:
    model = DummyDictInput()
    inputs = {
        "x": torch.randn(5, 4, dtype=torch.float32),
        "mask": torch.ones(5, 2, dtype=torch.float32),
    }

    schema = infer_schema(model, inputs)
    assert schema.input_structure == "mapping"
    assert "x" in schema.inputs
    assert "mask" in schema.inputs
    assert schema.inputs["x"].shape == [None, 4]


def test_schema_validation_passes() -> None:
    model = DummyMLP(in_features=4, out_features=2)
    schema = infer_schema(model, torch.randn(1, 4))

    # Should validate without exception for batch size 10
    valid_input = torch.randn(10, 4, dtype=torch.float32)
    schema.validate_input(valid_input)


def test_schema_validation_raises_on_invalid_data() -> None:
    model = DummyMLP(in_features=4, out_features=2)
    schema = infer_schema(model, torch.randn(1, 4))

    # Wrong shape
    with pytest.raises(SchemaValidationError, match="dimension 1 mismatch"):
        schema.validate_input(torch.randn(1, 5, dtype=torch.float32))

    # Wrong dtype
    with pytest.raises(SchemaValidationError, match="expected dtype float32"):
        schema.validate_input(torch.zeros(1, 4, dtype=torch.int32))

    # Wrong type structure
    with pytest.raises(SchemaValidationError, match="must be a torch.Tensor"):
        schema.validate_input([torch.randn(1, 4)])


def test_schema_serialization_roundtrip() -> None:
    model = DummyMLP(in_features=4, out_features=2)
    schema = infer_schema(model, torch.randn(1, 4))

    serialized = schema.to_dict()
    restored = ModelSchema.from_dict(serialized)

    assert restored.input_structure == schema.input_structure
    assert restored.output_structure == schema.output_structure
    assert restored.inputs[0].shape == schema.inputs[0].shape
    assert restored.inputs[0].dtype == schema.inputs[0].dtype

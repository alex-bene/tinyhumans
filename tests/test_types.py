import pytest
import torch

from src.tinyhumans.tiny_types import AutoTensorDict, LimitedAttrTensorDictWithDefaults


def test_auto_tensor_dict():
    # Test with random tensor
    tensor = torch.rand(2, 3, 4)
    auto_dict = AutoTensorDict({"test": tensor})
    assert auto_dict.batch_size == torch.Size([2])
    assert auto_dict.device == tensor.device

    # Test with empty instance and assignment
    auto_dict = AutoTensorDict()
    auto_dict["test"] = tensor
    assert auto_dict.batch_size == torch.Size([2])
    assert auto_dict.device == tensor.device


def test_limited_attr_tensor_dict_with_defaults():
    # Test with valid attributes
    LimitedAttrTensorDictWithDefaults.valid_attr_keys = ("test",)
    LimitedAttrTensorDictWithDefaults.valid_attr_sizes = ((3, 4),)

    tensor = torch.rand(2, 3, 4)

    limited_dict = LimitedAttrTensorDictWithDefaults({"test": tensor})
    assert limited_dict["test"].shape == (2, 3, 4)

    # Test with invalid attributes
    try:
        LimitedAttrTensorDictWithDefaults({"invalid": tensor})
        pytest.fail("Should have raised KeyError")
    except KeyError:
        pass

    # Test with empty instance and assignment
    limited_dict = LimitedAttrTensorDictWithDefaults()
    limited_dict["test"] = tensor
    assert limited_dict["test"].shape == (2, 3, 4)

    # Test with invalid key assignment
    try:
        limited_dict["invalid"] = tensor
        pytest.fail("Should have raised KeyError")
    except KeyError:
        pass

    # Test attribute access
    limited_dict.test = tensor
    assert (limited_dict.test == tensor).all()

    # Test attribute access with default value
    limited_dict = LimitedAttrTensorDictWithDefaults()
    assert limited_dict.test.shape == (1, 3, 4)

    # Test attribute assignment
    limited_dict = LimitedAttrTensorDictWithDefaults()
    limited_dict.test = tensor
    assert (limited_dict.test == tensor).all()
    assert (limited_dict.test == limited_dict["test"]).all()

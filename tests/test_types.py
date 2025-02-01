from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from src.tinyhumans.tiny_types import (
    AutoTensorDict,
    FLAMEPose,
    LimitedAttrTensorDictWithDefaults,
    MANOPose,
    Pose,
    ShapeComponents,
    SMPLHPose,
    SMPLPose,
    SMPLXPose,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def check_invalid_keys(tiny_type: LimitedAttrTensorDictWithDefaults, keys: str | Sequence[str]) -> None:
    """Check if keys are valid for the class.

    Args:
        tiny_type (LimitedAttrTensorDictWithDefaults): The tiny type instance to check.
        keys (str | Sequence[str]): Key or sequence of keys to check.

    """
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        with pytest.raises(KeyError):
            tiny_type.check_keys(key)


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
    check_invalid_keys(limited_dict, "invalid")

    # Test with empty instance and assignment
    limited_dict = LimitedAttrTensorDictWithDefaults()
    limited_dict["test"] = tensor
    assert limited_dict["test"].shape == (2, 3, 4)

    # Test with invalid key assignment
    check_invalid_keys(limited_dict, "invalid")

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


def test_pose():
    # Test check_model_type with valid model type
    for model_type in ["smpl", "smplh", "smplx", "flame", "mano", None, "SmPl"]:
        Pose.check_model_type(model_type)

    # Test check_model_type with invalid model type
    check_invalid_keys(Pose, "invalid")


def test_smpl_pose():
    # Test to_tensor method
    Pose.valid_attr_keys = ("body", "hand")
    smpl_pose = Pose()
    smpl_pose["body"] = torch.rand(2, 63)
    smpl_pose["hand"] = torch.rand(2, 6)
    tensor_out = smpl_pose.to_tensor()
    assert tensor_out.shape == (2, 69)
    assert (smpl_pose.body == tensor_out[:, :63]).all()
    assert (smpl_pose.hand == tensor_out[:, 63:]).all()

    # Test with valid attributes
    tensor = torch.rand(2, 69)
    smpl_pose = SMPLPose()
    smpl_pose.from_tensor(tensor)
    assert smpl_pose.body.shape == (2, 63)
    assert smpl_pose.hand.shape == (2, 6)
    assert smpl_pose.model_type == "smpl"
    assert smpl_pose.to_tensor().shape == (2, 69)
    assert (smpl_pose.to_tensor() == tensor).all()
    smpl_pose = SMPLPose(dict(zip(smpl_pose.valid_attr_keys, tensor.split([63, 6], dim=-1))))
    assert (smpl_pose.to_tensor() == tensor).all()
    assert (smpl_pose.body == tensor[:, :63]).all()
    assert (smpl_pose.hand == tensor[:, 63:]).all()

    # Test from_tensor with smplh
    tensor = torch.rand(2, 153)
    smpl_pose = SMPLPose()
    smpl_pose.from_tensor(tensor, model_type="smplh")
    assert smpl_pose.body.shape == (2, 63)
    assert smpl_pose.hand.shape == (2, 6)
    assert smpl_pose.to_tensor().shape == (2, 69)
    assert (smpl_pose.to_tensor()[:, :66] == tensor[:, :66]).all()
    assert (smpl_pose.to_tensor()[:, 66:] == tensor[:, 108:111]).all()

    # Test from_tensor with invalid model type
    check_invalid_keys(smpl_pose, "invalid")


def test_mano_pose():
    # Test with valid attributes
    tensor = torch.rand(2, 45)
    mano_pose = MANOPose()
    mano_pose.from_tensor(tensor)
    assert mano_pose.hand.shape == (2, 45)
    assert mano_pose.model_type == "mano"
    assert (mano_pose.to_tensor() == tensor).all()

    # Test from_tensor with smplh-l
    tensor = torch.rand(2, 153)
    mano_pose = MANOPose()
    mano_pose.from_tensor(tensor, model_type="smplh-l")
    assert mano_pose.hand.shape == (2, 45)
    assert (mano_pose.to_tensor() == tensor[:, 63:108]).all()

    # Test from_tensor with smplh-r
    tensor = torch.rand(2, 153)
    mano_pose = MANOPose()
    mano_pose.from_tensor(tensor, model_type="smplh-r")
    assert mano_pose.hand.shape == (2, 45)
    assert (mano_pose.to_tensor() == tensor[:, 108:]).all()

    # Test from_tensor with invalid model type
    check_invalid_keys(mano_pose, "invalid")


def test_smplh_pose():
    # Test with valid attributes
    tensor = torch.rand(2, 153)
    smplh_pose = SMPLHPose()
    smplh_pose.from_tensor(tensor)
    assert smplh_pose.body.shape == (2, 63)
    assert smplh_pose.hand.shape == (2, 90)
    assert smplh_pose.model_type == "smplh"
    assert (smplh_pose.to_tensor() == tensor).all()
    assert (smplh_pose.body == tensor[:, :63]).all()
    assert (smplh_pose.hand == tensor[:, 63:]).all()

    # Test from_tensor with smpl
    tensor = torch.rand(2, 69)
    smplh_pose = SMPLHPose()
    smplh_pose.from_tensor(tensor, model_type="smpl")
    assert smplh_pose.body.shape == (2, 63)
    assert smplh_pose.hand.shape == (2, 90)
    assert (smplh_pose.body == tensor[:, :63]).all()
    assert (smplh_pose.hand[:, :3] == tensor[:, 63:66]).all()
    assert (smplh_pose.hand[:, 45:48] == tensor[:, 66:]).all()

    # # Test from_tensor with mano
    # tensor = torch.rand(2, 45)
    # smplh_pose = SMPLHPose()
    # smplh_pose.from_tensor(tensor, model_type="mano")
    # assert smplh_pose.hand.shape == (2, 90)

    # Test from_tensor with invalid model type
    check_invalid_keys(smplh_pose, "invalid")


def test_flame_pose():
    # Test with valid attributes
    tensor = torch.rand(2, 12)
    flame_pose = FLAMEPose()
    flame_pose.from_tensor(tensor)
    assert flame_pose.body.shape == (2, 3)
    assert flame_pose.jaw.shape == (2, 3)
    assert flame_pose.eyes.shape == (2, 6)
    assert flame_pose.model_type == "flame"
    assert (flame_pose.to_tensor() == tensor).all()
    assert (flame_pose.body == tensor[:, :3]).all()
    assert (flame_pose.jaw == tensor[:, 3:6]).all()
    assert (flame_pose.eyes == tensor[:, 6:]).all()

    # Test from_tensor with smplx
    tensor = torch.rand(2, 162)
    flame_pose = FLAMEPose()
    flame_pose.from_tensor(tensor, model_type="smplx")
    assert flame_pose.body.shape == (2, 3)
    assert flame_pose.jaw.shape == (2, 3)
    assert flame_pose.eyes.shape == (2, 6)
    assert (flame_pose.body == tensor[:, :3]).all()
    assert (flame_pose.jaw == tensor[:, 63:66]).all()
    assert (flame_pose.eyes == tensor[:, 66:72]).all()

    # Test from_tensor with invalid model type
    check_invalid_keys(flame_pose, "invalid")


def test_smplx_pose():
    # Test with valid attributes
    tensor = torch.rand(2, 162)
    smplx_pose = SMPLXPose()
    smplx_pose.from_tensor(tensor)
    assert smplx_pose.body.shape == (2, 63)
    assert smplx_pose.jaw.shape == (2, 3)
    assert smplx_pose.eyes.shape == (2, 6)
    assert smplx_pose.hand.shape == (2, 90)
    assert smplx_pose.model_type == "smplx"
    assert (smplx_pose.to_tensor() == tensor).all()
    assert (smplx_pose.body == tensor[:, :63]).all()
    assert (smplx_pose.jaw == tensor[:, 63:66]).all()
    assert (smplx_pose.eyes == tensor[:, 66:72]).all()
    assert (smplx_pose.hand == tensor[:, 72:]).all()

    # Test from_tensor with flame
    tensor = torch.rand(2, 12)
    smplx_pose = SMPLXPose()
    smplx_pose.from_tensor(tensor, model_type="flame")
    assert smplx_pose.jaw.shape == (2, 3)
    assert smplx_pose.eyes.shape == (2, 6)
    assert (smplx_pose.body[:, :3] == tensor[:, :3]).all()
    assert (smplx_pose.jaw == tensor[:, 3:6]).all()
    assert (smplx_pose.eyes == tensor[:, 6:]).all()

    # Test from_tensor with invalid model type
    check_invalid_keys(smplx_pose, "invalid")


def test_shape_components():
    # Test with valid attributes
    shape_components = ShapeComponents()
    tensor = torch.rand(2, 10)
    shape_components["betas"] = tensor
    assert shape_components.to_tensor().shape == (2, 10)
    assert (shape_components.to_tensor() == shape_components["betas"]).all()
    assert (shape_components.betas == shape_components["betas"]).all()
    shape_components.valid_attr_sizes = (10,)
    shape_components.from_tensor(tensor)
    assert (shape_components.to_tensor() == tensor).all()
    check_invalid_keys(shape_components, ["expression", "dmpls"])

    shape_components = ShapeComponents(use_expression=True)
    tensor = torch.rand(2, 15)
    shape_components["betas"], shape_components["expression"] = tensor.split([10, 5], dim=-1)
    assert shape_components.to_tensor().shape == (2, 15)
    assert (shape_components.to_tensor() == tensor).all()
    assert (shape_components.betas == shape_components["betas"]).all()
    assert (shape_components.expression == shape_components["expression"]).all()
    shape_components.valid_attr_sizes = (10, 5)
    shape_components.from_tensor(tensor)
    assert (shape_components.to_tensor() == tensor).all()
    check_invalid_keys(shape_components, "dmpls")

    shape_components = ShapeComponents(use_dmpl=True)
    tensor = torch.rand(2, 30)
    shape_components["betas"], shape_components["dmpls"] = tensor.split([10, 20], dim=-1)
    assert shape_components.to_tensor().shape == (2, 30)
    assert (shape_components.to_tensor() == tensor).all()
    assert (shape_components.betas == shape_components["betas"]).all()
    assert (shape_components.dmpls == shape_components["dmpls"]).all()
    shape_components.valid_attr_sizes = (10, 20)
    shape_components.from_tensor(tensor)
    assert (shape_components.to_tensor() == tensor).all()
    check_invalid_keys(shape_components, "expression")

    shape_components = ShapeComponents(use_expression=True, use_dmpl=True)
    tensor = torch.rand(2, 35)
    shape_components["betas"], shape_components["expression"], shape_components["dmpls"] = tensor.split(
        [10, 5, 20], dim=-1
    )
    assert shape_components.to_tensor().shape == (2, 35)
    assert (shape_components.to_tensor() == tensor).all()
    assert (shape_components.betas == shape_components["betas"]).all()
    assert (shape_components.expression == shape_components["expression"]).all()
    assert (shape_components.dmpls == shape_components["dmpls"]).all()
    shape_components.valid_attr_sizes = (10, 5, 20)
    shape_components.from_tensor(tensor)
    assert (shape_components.to_tensor() == tensor).all()

    # Test that copy retains "use_expression" and "use_dmpl" attributes
    copy: ShapeComponents = shape_components.clone()
    assert copy.use_expression == shape_components.use_expression
    assert copy.use_dmpl == shape_components.use_dmpl

from __future__ import annotations

import torch

from src.tinyhumans.modules.hps_layer import ConstantLayer, HandsLayer, HPSLayer


def test_constant_layer() -> None:
    constant_layer = ConstantLayer(output_shape=6, value=0.0)
    assert torch.allclose(constant_layer(torch.randn(2, 4)), torch.zeros(2, 6))
    constant_layer = ConstantLayer(output_shape=(3, 4), value=12.0)
    assert torch.allclose(constant_layer(torch.randn(2, 4)), torch.full((2, 3, 4), 12.0))


def test_hands_layer() -> None:
    hands_layer = HandsLayer(token_dim=128, num_hand_joints=15, rotation_representation="6D")
    assert hands_layer(torch.randn(2, 128)).shape == (2, 30 * 6)
    hands_layer = HandsLayer(token_dim=128, num_hand_joints=1, rotation_representation="axis_angle")
    assert hands_layer(torch.randn(2, 128)).shape == (2, 2 * 3)
    hands_layer = HandsLayer(
        token_dim=128, num_hand_joints=15, no_hand_joints=True, rotation_representation="quaternion"
    )
    assert hands_layer(torch.randn(2, 128)).shape == (2, 30 * 4)
    assert torch.allclose(
        hands_layer(torch.randn(2, 128))[..., (1 * 4) : (15 * 4)].view(2, 14, 4)[..., 1:], torch.zeros(2, 14, 3)
    )
    assert torch.allclose(
        hands_layer(torch.randn(2, 128))[..., (16 * 4) : (30 * 4)].view(2, 14, 4)[..., 1:], torch.zeros(2, 14, 3)
    )
    assert torch.allclose(
        hands_layer(torch.randn(2, 128))[..., (1 * 4) : (15 * 4)].view(2, 14, 4)[..., 0], torch.ones(2, 14)
    )
    assert torch.allclose(
        hands_layer(torch.randn(2, 128))[..., (16 * 4) : (30 * 4)].view(2, 14, 4)[..., 0], torch.ones(2, 14)
    )


def test_hps_layer_smpl_head() -> None:
    hps_layer = HPSLayer(
        token_dim=256, body_type="smplx", no_hand_joints=True, no_head_joints=True, rotation_representation="6D"
    )
    smpl_data = hps_layer(torch.randn(2, 256), torch.randn(2, 256), torch.randn(2, 256))
    smpl_data = hps_layer(torch.randn(2, 256))["smpl_data"]
    assert torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.head_pose, torch.zeros(2, 1, 1, 3, 3))
    hps_layer = HPSLayer(
        token_dim=256,
        body_type="smplx",
        no_hand_joints=True,
        no_head_joints=False,
        rotation_representation="axis_angle",
    )
    smpl_data = hps_layer(torch.randn(2, 256))["smpl_data"]
    assert torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert not torch.allclose(smpl_data.head_pose, torch.zeros(2, 1, 1, 3, 3))
    hps_layer = HPSLayer(
        token_dim=256,
        body_type="smplx",
        no_hand_joints=False,
        no_head_joints=True,
        rotation_representation="quaternion",
    )
    smpl_data = hps_layer(torch.randn(2, 256))["smpl_data"]
    assert not torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert not torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.head_pose, torch.zeros(2, 1, 1, 3, 3))
    hps_layer = HPSLayer(token_dim=256, body_type="smplx", no_hand_joints=False, no_head_joints=False)
    smpl_data = hps_layer(torch.randn(2, 256))["smpl_data"]
    assert not torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert not torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert not torch.allclose(smpl_data.head_pose, torch.zeros(2, 1, 1, 3, 3))

    hps_layer = HPSLayer(token_dim=256, body_type="smplh", no_hand_joints=False)
    smpl_data = hps_layer(torch.randn(2, 256))["smpl_data"]
    assert not torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert not torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    hps_layer = HPSLayer(token_dim=256, body_type="smplh", no_hand_joints=True)
    smpl_data = hps_layer(torch.randn(2, 256))["smpl_data"]
    assert torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))

    hps_layer = HPSLayer(token_dim=256, body_type="smplh", no_hand_joints=False)
    smpl_data = hps_layer(torch.randn(2, 256))["smpl_data"]
    assert not torch.allclose(smpl_data.body_pose[..., 20:21, :], torch.zeros(2, 1, 1, 1, 3))
    assert not torch.allclose(smpl_data.body_pose[..., 21:22, :], torch.zeros(2, 1, 1, 1, 3))
    hps_layer = HPSLayer(token_dim=256, body_type="smpl", no_hand_joints=True)
    smpl_data = hps_layer(torch.randn(2, 256))["smpl_data"]
    assert not torch.allclose(smpl_data.body_pose[..., 22:23, :], torch.zeros(2, 1, 1, 1, 3))
    assert not torch.allclose(smpl_data.body_pose[..., 23:24, :], torch.zeros(2, 1, 1, 1, 3))


def test_hps_layer_translation_heads() -> None:
    hps_layer = HPSLayer(token_dim=256, body_type="smpl", pose_target_convention="LogarithmicDisparitySpace")
    pose_target = hps_layer(torch.randn(2, 256))["pose_target"]
    assert pose_target.translation.shape == (2, 3)
    assert pose_target.translation_scale.shape == (2, 1)
    assert torch.allclose(pose_target.translation[:, -1], torch.zeros(2))
    assert not torch.allclose(pose_target.translation_scale, torch.ones(2, 1))

    hps_layer = HPSLayer(token_dim=256, body_type="smpl", pose_target_convention="DisparitySpace")
    pose_target = hps_layer(torch.randn(2, 256))["pose_target"]
    assert pose_target.translation.shape == (2, 3)
    assert pose_target.translation_scale.shape == (2, 1)
    assert torch.allclose(pose_target.translation[:, -1], torch.zeros(2))
    assert not torch.allclose(pose_target.translation_scale, torch.ones(2, 1))

    hps_layer = HPSLayer(token_dim=256, body_type="smpl", pose_target_convention="ApparentSize")
    pose_target = hps_layer(torch.randn(2, 256))["pose_target"]
    assert pose_target.translation.shape == (2, 3)
    assert pose_target.translation_scale.shape == (2, 1)
    assert torch.allclose(torch.norm(pose_target.translation, dim=-1), torch.ones(2))
    assert not torch.allclose(pose_target.translation_scale, torch.ones(2, 1))

    hps_layer = HPSLayer(token_dim=256, body_type="smpl", pose_target_convention="NormalizedSceneScaleAndTranslation")
    pose_target = hps_layer(torch.randn(2, 256))["pose_target"]
    assert pose_target.translation.shape == (2, 3)
    assert pose_target.translation_scale.shape == (2, 1)
    assert torch.allclose(torch.norm(pose_target.translation, dim=-1), torch.ones(2))
    assert not torch.allclose(pose_target.translation_scale, torch.ones(2, 1))

    hps_layer = HPSLayer(token_dim=256, body_type="smpl", pose_target_convention="ScaleShiftInvariantWTranslationScale")
    pose_target = hps_layer(torch.randn(2, 256))["pose_target"]
    assert pose_target.translation.shape == (2, 3)
    assert pose_target.translation_scale.shape == (2, 1)
    assert torch.allclose(torch.norm(pose_target.translation, dim=-1), torch.ones(2))
    assert not torch.allclose(pose_target.translation_scale, torch.ones(2, 1))

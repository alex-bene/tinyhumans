from __future__ import annotations

import torch

from src.tinyhumans.modules.hps_layer import HandsLayer, HPSLayer, ZeroLayer


def test_zero_layer() -> None:
    zero_layer = ZeroLayer(output_shape=6)
    assert zero_layer(torch.randn(2, 4)).shape == (2, 6)
    zero_layer = ZeroLayer(output_shape=(3, 4))
    assert zero_layer(torch.randn(2, 4)).shape == (2, 3, 4)


def test_hands_layer() -> None:
    hands_layer = HandsLayer(input_size=128, num_hand_joints=15)
    assert hands_layer(torch.randn(2, 128)).shape == (2, 30 * 3)
    hands_layer = HandsLayer(input_size=128, num_hand_joints=1)
    assert hands_layer(torch.randn(2, 128)).shape == (2, 2 * 3)
    hands_layer = HandsLayer(input_size=128, num_hand_joints=15, no_hand_joints=True)
    assert hands_layer(torch.randn(2, 128)).shape == (2, 30 * 3)
    assert torch.allclose(hands_layer(torch.randn(2, 128))[..., (1 * 3) : (15 * 3)], torch.zeros(2, 14 * 3))
    assert torch.allclose(hands_layer(torch.randn(2, 128))[..., (16 * 3) : (30 * 3)], torch.zeros(2, 14 * 3))
    hands_layer = HandsLayer(input_size=128, num_hand_joints=1, no_hand_joints=True)
    assert hands_layer(torch.randn(2, 128)).shape == (2, 2 * 3)
    assert not torch.allclose(hands_layer(torch.randn(2, 128))[..., 0:3], torch.zeros(2, 1 * 3))
    assert not torch.allclose(hands_layer(torch.randn(2, 128))[..., 3:6], torch.zeros(2, 1 * 3))


def test_hps_layer() -> None:
    hps_layer = HPSLayer(input_size=256, body_type="smplx", no_hand_joints=True, no_head_joints=True)
    smpl_data = hps_layer(torch.randn(2, 256))
    assert torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.head_pose, torch.zeros(2, 1, 1, 3, 3))
    hps_layer = HPSLayer(input_size=256, body_type="smplx", no_hand_joints=True, no_head_joints=False)
    smpl_data = hps_layer(torch.randn(2, 256))
    assert torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert not torch.allclose(smpl_data.head_pose, torch.zeros(2, 1, 1, 3, 3))
    hps_layer = HPSLayer(input_size=256, body_type="smplx", no_hand_joints=False, no_head_joints=True)
    smpl_data = hps_layer(torch.randn(2, 256))
    assert not torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert not torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.head_pose, torch.zeros(2, 1, 1, 3, 3))
    hps_layer = HPSLayer(input_size=256, body_type="smplx", no_hand_joints=False, no_head_joints=False)
    smpl_data = hps_layer(torch.randn(2, 256))
    assert not torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert not torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert not torch.allclose(smpl_data.head_pose, torch.zeros(2, 1, 1, 3, 3))

    hps_layer = HPSLayer(input_size=256, body_type="smplh", no_hand_joints=False)
    smpl_data = hps_layer(torch.randn(2, 256))
    assert not torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert not torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    hps_layer = HPSLayer(input_size=256, body_type="smplh", no_hand_joints=True)
    smpl_data = hps_layer(torch.randn(2, 256))
    assert torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))

    hps_layer = HPSLayer(input_size=256, body_type="smplh", no_hand_joints=False)
    smpl_data = hps_layer(torch.randn(2, 256))
    assert not torch.allclose(smpl_data.body_pose[..., 20:21, :], torch.zeros(2, 1, 1, 1, 3))
    assert not torch.allclose(smpl_data.body_pose[..., 21:22, :], torch.zeros(2, 1, 1, 1, 3))
    hps_layer = HPSLayer(input_size=256, body_type="smpl", no_hand_joints=True)
    smpl_data = hps_layer(torch.randn(2, 256))
    assert not torch.allclose(smpl_data.body_pose[..., 22:23, :], torch.zeros(2, 1, 1, 1, 3))
    assert not torch.allclose(smpl_data.body_pose[..., 23:24, :], torch.zeros(2, 1, 1, 1, 3))

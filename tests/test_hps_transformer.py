from __future__ import annotations

import torch

from src.tinyhumans.modules.hps_transformer import HPSTransformer


def test_hps_transformer() -> None:
    hps_model = HPSTransformer(
        latent_dim=256,
        num_shape_parameters=10,
        no_global_orientation=False,
        no_global_translation=False,
        no_hand_joints=True,
        no_head_joints=True,
        dropout=0.1,
        token_projectors_depth=3,
        body_type="smplx",
        ff_block_kwargs=None,
        pose_target_convention="LogarithmicDisparitySpace",
        rotation_representation="6d",
        separate_hands_token=False,
        nhead=8,
        num_layers=4,
        activation="relu",
        norm_first=True,
        bias=True,
        batch_first=True,
        dim_feedforward_multiplier=4,
        output_norm=torch.nn.LayerNorm,
    )
    smpl_data, pose_target = hps_model(torch.randn(2, 12, 256))
    assert torch.allclose(smpl_data.left_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.right_hand_pose[..., 1:15, :], torch.zeros(2, 1, 1, 14, 3))
    assert torch.allclose(smpl_data.head_pose, torch.zeros(2, 1, 1, 3, 3))

    assert pose_target.translation.shape == (2, 3)
    assert pose_target.translation_scale.shape == (2, 1)
    assert torch.allclose(pose_target.translation[:, -1], torch.ones(2))
    assert not torch.allclose(pose_target.translation_scale, torch.ones(2, 1))

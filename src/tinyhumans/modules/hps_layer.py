"""Module for HPS prediction."""

from typing import TYPE_CHECKING, Literal

import torch
from pytorch3d.transforms.rotation_conversions import (
    matrix_to_axis_angle,
    quaternion_to_axis_angle,
    rotation_6d_to_matrix,
)
from smplcodec import SMPLVersion
from tinytools.threeD.pose_target import PoseTarget, PoseTargetFactory
from tinytools.torch.modules import ConstantLayer, FFBlock, LocationHead
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import pad

from tinyhumans.datatypes import SMPLData
from tinyhumans.tools import inverse_stereographic_projection


class HandsLayer(nn.Module):
    """Hand prediction layer."""

    def __init__(
        self,
        token_dim: int,
        num_hand_joints: int,
        dropout: float = 0.1,
        no_hand_joints: bool = False,
        ff_block_kwargs: dict | None = None,
        rotation_representation: Literal["6D", "axis_angle", "quaternion"] = "6D",
    ) -> None:
        super().__init__()
        if num_hand_joints not in [1, 15]:
            msg = "num_hand_joints must be one of [1, 15]"
            raise ValueError(msg)
        rotation_representation = rotation_representation.strip().lower()
        if rotation_representation not in ["6d", "axis_angle", "quaternion"]:
            msg = "rotation_representation must be one of ['6D', 'axis_angle', 'quaternion']"
            raise ValueError(msg)
        self.rotation_representation = rotation_representation
        self.rotation_dim = (
            6 if self.rotation_representation == "6d" else 3 if self.rotation_representation == "axis_angle" else 4
        )

        ff_block_kwargs = {
            "input_dim": token_dim,
            "hidden_dim": 4 * token_dim,
            "bias": True,
            "dropout": dropout,
            "dropout_at_end": False,
            "mlp_type": "vanilla",
            "activation_fn": F.relu,
            "norm_first": True,
            "norm_fn": nn.LayerNorm,
            "residual": False,
        } | (ff_block_kwargs or {})
        num_pred_hand_joints = 1 + (0 if no_hand_joints else num_hand_joints - 1)
        self.left_hand_pose_head = FFBlock(output_dim=num_pred_hand_joints * self.rotation_dim, **ff_block_kwargs)
        self.right_hand_pose_head = FFBlock(output_dim=num_pred_hand_joints * self.rotation_dim, **ff_block_kwargs)
        self.num_pad_joints = num_hand_joints - 1 if no_hand_joints else 0

    if TYPE_CHECKING:

        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward hand layer."""
        left_hand_pose = self.left_hand_pose_head(hidden_states)
        left_hand_pose = pad(left_hand_pose, (0, self.num_pad_joints * self.rotation_dim), "constant", 0)
        right_hand_pose = self.right_hand_pose_head(hidden_states)
        right_hand_pose = pad(right_hand_pose, (0, self.num_pad_joints * self.rotation_dim), "constant", 0)

        return torch.cat([left_hand_pose, right_hand_pose], dim=-1)


class HPSLayer(torch.nn.Module):
    """HPS prediction layer."""

    def __init__(
        self,
        token_dim: int,
        num_shape_parameters: int = 10,
        no_global_orientation: bool = False,
        no_global_translation: bool = False,
        no_hand_joints: bool = True,
        no_head_joints: bool = True,
        dropout: float = 0.1,
        body_type: Literal["smpl", "smplh", "smplx"] = "smpl",
        ff_block_kwargs: dict | None = None,
        pose_target_convention: str = "LogarithmicDisparitySpace",
        rotation_representation: Literal["6d", "axis_angle", "quaternion"] = "6d",
    ) -> None:
        super().__init__()
        if body_type not in ["smpl", "smplh", "smplx"]:
            msg = "body_type must be one of ['smpl', 'smplh', 'smplx']"
            raise ValueError(msg)
        self.body_type = SMPLVersion.from_string(body_type)
        rotation_representation = rotation_representation.strip().lower()
        if rotation_representation not in ["6d", "axis_angle", "quaternion"]:
            msg = "rotation_representation must be one of ['6d', 'axis_angle', 'quaternion']"
            raise ValueError(msg)
        self.rotation_representation = rotation_representation
        self.rotation_dim = (
            6 if self.rotation_representation == "6d" else 3 if self.rotation_representation == "axis_angle" else 4
        )
        # Token projectors
        ff_block_kwargs = {
            "input_dim": token_dim,
            "hidden_dim": 4 * token_dim,
            "bias": True,
            "dropout": dropout,
            "dropout_at_end": False,
            "mlp_type": "vanilla",
            "activation_fn": F.relu,
            "norm_first": True,
            "norm_fn": nn.LayerNorm,
            "residual": False,
        } | (ff_block_kwargs or {})
        # Prediction heads
        ## Translation head
        self.translation_head = LocationHead(**ff_block_kwargs) if not no_global_translation else None
        self.pose_target_cls: type[PoseTarget] = PoseTargetFactory[pose_target_convention]
        ## Shape head
        self.shape_head = FFBlock(output_dim=num_shape_parameters, **ff_block_kwargs)
        ## Body pose head
        self.no_global_orientation = no_global_orientation
        num_body_joints = 21
        if not self.no_global_orientation:
            num_body_joints += 1
        self.body_pose_head = FFBlock(output_dim=num_body_joints * self.rotation_dim, **ff_block_kwargs)
        ## Head pose head
        self.head_pose_head = (
            FFBlock(output_dim=3 * self.rotation_dim, **ff_block_kwargs)
            if body_type == "smplx" and not no_head_joints
            else ConstantLayer(output_shape=3 * self.rotation_dim)
            if body_type == "smplx"
            else None
        )
        ## Hand pose head
        self.hand_pose_head = HandsLayer(
            token_dim=token_dim,
            dropout=dropout,
            num_hand_joints=1 if body_type == "smpl" else 15,
            no_hand_joints=no_hand_joints,
            ff_block_kwargs=ff_block_kwargs,
            rotation_representation=rotation_representation,
        )

    if TYPE_CHECKING:

        def __call__(
            self,
            smpl_token: torch.Tensor,
            hands_token: torch.Tensor | None = None,
            translation_token: torch.Tensor | None = None,
            scene_scale: torch.Tensor | None = None,
            scene_center: torch.Tensor | None = None,
        ) -> dict[str, SMPLData | PoseTarget | None]:
            """Type hinting fix."""
            return self.forward(
                smpl_token=smpl_token,
                hands_token=hands_token,
                translation_token=translation_token,
                scene_scale=scene_scale,
                scene_center=scene_center,
            )

    def forward(
        self,
        smpl_token: torch.Tensor,
        hands_token: torch.Tensor | None = None,
        translation_token: torch.Tensor | None = None,
        scene_scale: torch.Tensor | None = None,
        scene_center: torch.Tensor | None = None,
    ) -> dict[str, SMPLData | PoseTarget | None]:
        """Forward HPS hidden states.

        Args:
            smpl_token (torch.Tensor): The latent token for SMPL parameters of shape (..., token_dim)
            hands_token (torch.Tensor | None, optional): The latent token for hand parameters of shape (..., token_dim).
                If None, smpl_token is used. Defaults to None.
            translation_token (torch.Tensor | None, optional): The latent token for global translation of shape
                (..., token_dim). If None and global translation is predicted, smpl_token is used. Defaults to None.
            scene_scale (torch.Tensor | None, optional): The scene scale for pose target computation of shape
                (..., 1 or 3). Defaults to None.
            scene_center (torch.Tensor | None, optional): The scene center for pose target computation of shape
                (..., 3). Defaults to None.

        Returns:
            dict[str, SMPLData | PoseTarget | None]: A dictionary containing:
                - "smpl_data" (SMPLData): The predicted SMPL data
                - "pose_target" (PoseTarget | None): The predicted pose target if not no_global_translation, else None

        """
        ## Shape
        shape_parameters = self.shape_head(smpl_token)
        ## Pose
        full_pose = [self.body_pose_head(smpl_token)]
        if self.head_pose_head is not None:
            full_pose.append(self.head_pose_head(smpl_token))
        full_pose.append(self.hand_pose_head(smpl_token if hands_token is None else hands_token))
        full_pose = torch.cat(full_pose, dim=-1)
        if self.no_global_orientation:
            full_pose = pad(full_pose, (self.rotation_dim, 0), "constant", 0)
        full_pose = full_pose.reshape((*smpl_token.shape[:-1], -1, self.rotation_dim))
        full_pose = self.to_axis_angle(full_pose)
        ## Translation
        xy = None
        if self.translation_head is not None:
            xy, z = self.translation_head(translation_token if translation_token is not None else smpl_token).values()

        # Return
        return {
            "smpl_data": SMPLData.from_full_pose(
                full_pose=full_pose, shape_parameters=shape_parameters, smpl_version=self.body_type
            ),
            "pose_target": self.get_pose_target(xy, z, scene_scale=scene_scale, scene_center=scene_center),
        }

    def get_pose_target(
        self,
        xy: torch.Tensor | None,
        z: torch.Tensor,
        scene_scale: torch.Tensor | None = None,
        scene_center: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Get pose target from global translation."""
        if xy is None:
            return None
        translation_scale = None
        if self.pose_target_cls.pose_target_convention in ("LogarithmicDisparitySpace", "DisparitySpace"):
            translation_scale = z
            translation = torch.cat([xy, torch.ones_like(z)], dim=-1)
        elif self.pose_target_cls.pose_target_convention in (
            "ApparentSize",
            "NormalizedSceneScaleAndTranslation",
            "ScaleShiftInvariantWTranslationScale",
        ):
            translation_scale = z
            translation = inverse_stereographic_projection(xy)  # project to unit sphere
        else:
            translation = torch.cat([xy, z], dim=-1)
        return self.pose_target_cls(
            translation=translation,
            translation_scale=translation_scale,
            scene_scale=scene_scale,
            scene_center=scene_center,
        )

    def to_axis_angle(self, rotations: torch.Tensor) -> torch.Tensor:
        """Convert rotations to axis-angle representation."""
        if self.rotation_representation == "axis_angle":
            return rotations
        if self.rotation_representation == "quaternion":
            return quaternion_to_axis_angle(rotations)
        # if self.rotation_representation == "6d"
        rot_mats = rotation_6d_to_matrix(rotations)
        return matrix_to_axis_angle(rot_mats)

    # def initialize(self) -> None:
    #     init_depth = torch.tensor([[1 / 10.0]]) if self.inverse_depth else torch.tensor([[10.0]])
    #     init_pose = torch.cat(
    #         [torch.tensor([[1.0, 0, 0, 0, -1, 0]]), torch.tensor([[1.0, 0, 0, 0, 1, 0]]).tile(21)], dim=1
    #     )

    #     self.register_buffer("init_pose", init_pose)
    #     self.register_buffer("init_depth", init_depth)

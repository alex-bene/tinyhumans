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
from tinytools.torch import ConstantLayer, FFBlock
from tinytools.torch.modules import LocationHead
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import pad

from tinyhumans.datatypes import SMPLData
from tinyhumans.tools import inverse_stereographic_projection


class HandsLayer(nn.Module):
    """Hand prediction layer."""

    def __init__(
        self,
        num_hand_joints: int,
        ff_block_kwargs: dict,
        no_hand_joints: bool = False,
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

        self.left_hand_pose_head = FFBlock(output_dim=self.rotation_dim, **ff_block_kwargs)  # first hand joint
        self.left_hand_extra_pose_head = (
            None
            if num_hand_joints == 1
            else ConstantLayer(output_shape=(num_hand_joints - 1) * self.rotation_dim)
            if no_hand_joints
            else FFBlock(output_dim=(num_hand_joints - 1) * self.rotation_dim, **ff_block_kwargs)
        )
        self.right_hand_pose_head = FFBlock(output_dim=self.rotation_dim, **ff_block_kwargs)
        self.right_hand_extra_pose_head = (
            None
            if num_hand_joints == 1
            else ConstantLayer(output_shape=(num_hand_joints - 1) * self.rotation_dim)
            if no_hand_joints
            else FFBlock(output_dim=(num_hand_joints - 1) * self.rotation_dim, **ff_block_kwargs)
        )

    if TYPE_CHECKING:

        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward hand layer."""
        hand_pose = [self.left_hand_pose_head(hidden_states)]
        if self.left_hand_extra_pose_head is not None:
            hand_pose.append(self.left_hand_extra_pose_head(hidden_states))
        hand_pose.append(self.right_hand_pose_head(hidden_states))
        if self.right_hand_extra_pose_head is not None:
            hand_pose.append(self.right_hand_extra_pose_head(hidden_states))

        return torch.cat(hand_pose, dim=-1)


class HPSLayer(torch.nn.Module):
    """HPS prediction layer."""

    def __init__(
        self,
        latent_dim: int,
        num_shape_parameters: int = 10,
        no_global_orientation: bool = False,
        no_global_translation: bool = False,
        no_hand_joints: bool = True,
        no_head_joints: bool = True,
        dropout: float = 0.1,
        token_projectors_depth: int = 1,
        latent_transformer_kwargs: dict | None = None,
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
        # Latent transformer
        ## Latent tokens
        self.latent_tokens = nn.Embedding(2, latent_dim)  # smpl, translation TODO: add hands tokens here?
        ## Initialize latent transformer
        latent_transformer_kwargs = {
            "d_model": latent_dim,
            "nhead": 8,
            "dim_feedforward": 4 * latent_dim,
            "dropout": dropout,
            "activation": "relu",
            "batch_first": True,
            "norm_first": False,
            "bias": True,
            "num_layers": 4,
            "output_norm": nn.LayerNorm,
        } | (latent_transformer_kwargs or {})
        num_layers = latent_transformer_kwargs.pop("num_layers")
        output_norm = latent_transformer_kwargs.pop("output_norm")(latent_transformer_kwargs["d_model"])
        latent_model_layer = nn.TransformerDecoderLayer(**latent_transformer_kwargs)
        self.latent_model = nn.TransformerDecoder(latent_model_layer, num_layers=num_layers, norm=output_norm)
        # Token projectors
        ff_block_kwargs = {
            "input_dim": latent_dim,
            "hidden_dim": 4 * latent_dim,
            "bias": True,
            "dropout": dropout,
            "mlp_type": "gated",
            "activation_fn": F.silu,
            "norm_first": False,
            "norm_fn": nn.LayerNorm,
            "residual": True,
        } | (ff_block_kwargs or {})
        ## ff nets for each token
        self.smpl_token_projector = nn.Identity()
        self.translation_token_projector = nn.Identity() if not no_global_translation else None
        if token_projectors_depth > 0:
            self.smpl_token_projector = nn.Sequential(
                [FFBlock(**ff_block_kwargs) for _ in range(token_projectors_depth)]
            )
            self.translation_token_projector = (
                nn.Sequential([FFBlock(**ff_block_kwargs) for _ in range(token_projectors_depth)])
                if not no_global_translation
                else None
            )
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
            num_hand_joints=1 if body_type == "smpl" else 15,
            no_hand_joints=no_hand_joints,
            ff_block_kwargs=ff_block_kwargs,
            rotation_representation=rotation_representation,
        )

    if TYPE_CHECKING:

        def __call__(self, hidden_states: torch.Tensor) -> SMPLData:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        scene_scale: torch.Tensor | None = None,
        scene_center: torch.Tensor | None = None,
    ) -> tuple[SMPLData, PoseTarget | None]:
        """Forward HPS hidden states."""
        batch_size = hidden_states.shape[0]
        # Prepare latent tokens
        latent_tokens = self.latent_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_tokens, D)
        # Pass through latent transformer
        latent_states = self.latent_model.forward(tgt=latent_tokens, memory=hidden_states)  # (B, num_tokens, D)
        # Project each token
        smpl_token = self.smpl_token_projector(latent_states[:, 0, :])  # (B, D)
        translation_token = None
        if self.translation_token_projector is not None:
            translation_token = self.translation_token_projector(latent_states[:, 1, :])  # (B, D)
        # Prediction heads
        ## Shape
        shape_parameters = self.shape_head(smpl_token)
        ## Pose
        full_pose = [self.body_pose_head(smpl_token)]
        if self.head_pose_head is not None:
            full_pose.append(self.head_pose_head(smpl_token))
        full_pose.append(self.hand_pose_head(smpl_token))
        full_pose = torch.cat(full_pose, dim=-1)
        full_pose = self.to_axis_angle(full_pose)
        if self.no_global_orientation:
            full_pose = pad(full_pose, (3, 0), "constant", 0)
        ## Translation
        xy = None
        if self.translation_head is not None:
            xy, z = self.translation_head(translation_token).values()

        # Return SMPLData
        return SMPLData.from_full_pose(
            full_pose=full_pose, shape_parameters=shape_parameters, smpl_version=self.body_type
        ), self.get_pose_target(xy, z, scene_scale=scene_scale, scene_center=scene_center)

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

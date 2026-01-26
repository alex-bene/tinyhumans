"""HPS Decoder module."""

from typing import TYPE_CHECKING, Literal

import torch
from tinytools.threeD.pose_target import PoseTarget
from tinytools.torch import FFBlock
from torch import nn
from torch.nn import functional as F

from tinyhumans.datatypes import SMPLData

from .hps_layer import HPSLayer


class HPSDecoder(torch.nn.Module):
    """HPS Decoder model.

    A transformer-based model that predicts SMPL parameters from latent tokens.
    """

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
        token_projectors_ff_block_kwargs: dict | None = None,
        body_type: Literal["smpl", "smplh", "smplx"] = "smpl",
        pose_target_convention: str = "LogarithmicDisparitySpace",
        rotation_representation: Literal["6d", "axis_angle", "quaternion"] = "6d",
        separate_hands_token: bool = False,
        nhead: int = 8,
        num_layers: int = 4,
        activation: str = "relu",
        norm_first: bool = True,
        bias: bool = True,
        batch_first: bool = True,
        dim_feedforward_multiplier: int = 4,
        output_norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        ## Latent tokens
        num_of_tokens = 1 + (0 if no_global_translation else 1) + (1 if separate_hands_token else 0)
        self.latent_tokens = nn.Embedding(num_of_tokens, latent_dim)
        ## Initialize latent transformer
        latent_transformer_kwargs = {
            "d_model": latent_dim,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward_multiplier * latent_dim,
            "dropout": dropout,
            "activation": activation,
            "batch_first": batch_first,
            "norm_first": norm_first,
            "bias": bias,
        }
        latent_model_layer = nn.TransformerDecoderLayer(**latent_transformer_kwargs)
        output_norm = output_norm(latent_transformer_kwargs["d_model"]) if output_norm is not None else None
        self.latent_model = nn.TransformerDecoder(latent_model_layer, num_layers=num_layers, norm=output_norm)
        # Token projectors
        ## ff nets for each token
        self.smpl_token_projector = nn.Identity()
        self.translation_token_projector = nn.Identity() if not no_global_translation else None
        if token_projectors_depth > 0:
            token_projectors_ff_block_kwargs = {
                "input_dim": latent_dim,
                "hidden_dim": dim_feedforward_multiplier * latent_dim,
                "bias": bias,
                "dropout": dropout,
                "mlp_type": "gated",
                "activation_fn": F.silu,
                "norm_first": norm_first,
                "norm_fn": nn.LayerNorm,
                "residual": True,
            } | (token_projectors_ff_block_kwargs or {})
            self.smpl_token_projector = nn.Sequential(
                *(FFBlock(**token_projectors_ff_block_kwargs) for _ in range(token_projectors_depth))
            )
            self.translation_token_projector = (
                nn.Sequential(*(FFBlock(**token_projectors_ff_block_kwargs) for _ in range(token_projectors_depth)))
                if not no_global_translation
                else None
            )
        # HPS Head
        self.num_shape_parameters = num_shape_parameters
        self.no_global_translation = no_global_translation
        self.no_global_orientation = no_global_orientation
        self.no_hand_joints = no_hand_joints
        self.no_head_joints = no_head_joints
        self.hps_head = HPSLayer(
            token_dim=latent_dim,
            num_shape_parameters=num_shape_parameters,
            no_global_translation=no_global_translation,
            no_global_orientation=no_global_orientation,
            no_hand_joints=no_hand_joints,
            no_head_joints=no_head_joints,
            body_type=body_type,
            pose_target_convention=pose_target_convention,
            rotation_representation=rotation_representation,
        )
        self.body_type = self.hps_head.body_type

    if TYPE_CHECKING:

        def __call__(
            self,
            hidden_states: torch.Tensor,
            scene_scale: torch.Tensor | None = None,
            scene_center: torch.Tensor | None = None,
        ) -> dict[str, SMPLData | PoseTarget | None]:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states, scene_scale=scene_scale, scene_center=scene_center)

    def forward(
        self,
        hidden_states: torch.Tensor,
        scene_scale: torch.Tensor | None = None,
        scene_center: torch.Tensor | None = None,
    ) -> dict[str, SMPLData | PoseTarget | None]:
        """Forward HPS hidden states."""
        batch_size = hidden_states.shape[0]
        # Prepare latent tokens
        latent_tokens = self.latent_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_tokens, D)
        # Pass through latent transformer
        latent_states = self.latent_model.forward(tgt=latent_tokens, memory=hidden_states)  # (B, num_tokens, D)
        # Project each token
        idx = 0
        smpl_token = self.smpl_token_projector(latent_states[:, idx, :])  # (B, D)
        idx += 1
        translation_token = None
        if self.translation_token_projector is not None:
            translation_token = self.translation_token_projector(latent_states[:, idx, :])  # (B, D)
            idx += 1
        hands_token = smpl_token
        if latent_states.shape[1] > idx:
            hands_token = latent_states[:, idx, :]  # (B, D)
        # Prediction heads
        return self.hps_head.forward(
            smpl_token=smpl_token,
            hands_token=hands_token,
            translation_token=translation_token,
            scene_scale=scene_scale,
            scene_center=scene_center,
        )

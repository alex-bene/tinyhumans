"""BodyModel class for TinyHumans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from tinyhumans.models.base_parametric_model import BaseParametricModel
from tinyhumans.types import ShapeComponents

if TYPE_CHECKING:
    from pathlib import Path


class SMPL(BaseParametricModel):
    def __init__(
        self,
        pretrained_model_path: str | Path,
        gender: str = "neutral",
        num_betas: int = 10,
        num_dmpls: int | None = None,
        dmpl_filename: str | Path | None = None,
        do_pose_conditioned_shape: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            pretrained_model_path=pretrained_model_path,
            body_type=self.__class__.__name__,
            gender=gender,
            num_betas=num_betas,
            do_pose_conditioned_shape=do_pose_conditioned_shape,
            dtype=dtype,
        )
        self.pose_parts.update({"body", "hand"})

        # Setup shapes of default parameters
        self.body_pose_size = 63
        self.hand_pose_size = 1 * 3 * 2
        self.dmpls_size = num_dmpls if num_dmpls is not None else 0

        # DMPLs check and load and register dmpl directions
        self.use_dmpl = False
        if num_dmpls is not None and num_dmpls > 0:
            if dmpl_filename is None:
                msg = "`dmpl_filename` should be provided when using dmpls!"
                raise ValueError(msg)
            self.use_dmpl = True
            dmpl_directions = torch.from_numpy(np.load(dmpl_filename)["eigvec"][:, :, :num_dmpls]).to(dtype)
            self.shape_directions = torch.cat([self.shape_directions, dmpl_directions], dim=-1)

    def get_shape_components(
        self,
        shape_components: ShapeComponents | dict | torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> ShapeComponents:
        out = ShapeComponents(
            shape_components, use_expression=False, use_dmpl=self.use_dmpl, device=device if device else self.device
        )
        out.valid_attr_sizes = (self.betas_size, self.dmpls_size, 0)
        return out


class SMPLH(SMPL):
    # Avoid re-writing __init__
    @property
    def hand_pose_size(self) -> int:
        return 15 * 3 * 2

    @hand_pose_size.setter
    def hand_pose_size(self, value: int) -> None:
        pass


class SMPLX(SMPLH):
    def __init__(
        self,
        pretrained_model_path: str | Path,
        gender: str = "neutral",
        num_betas: int = 10,
        num_expression: int | None = None,
        do_pose_conditioned_shape: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            pretrained_model_path=pretrained_model_path,
            gender=gender,
            num_betas=num_betas,
            num_dmpls=0,
            dmpl_filename="",
            do_pose_conditioned_shape=do_pose_conditioned_shape,
            dtype=dtype,
        )
        self.pose_parts.update({"jaw", "eyes"})

        # Setup shapes of default parameters
        self.jaw_pose_size = 1 * 3
        self.eyes_pose_size = 2 * 3
        self.expression_size = num_expression if num_expression is not None else 0

        # Expression directions register
        self.use_expression = num_expression is not None
        if self.use_expression:
            ## The PCA vectors for expression conditioned displacements
            ### self.full_shape_directions.shape[-1] > 300
            expression_directions = torch.from_numpy(self.full_shape_directions[:, :, 300 : (300 + num_expression)])
            self.shape_directions = torch.cat([self.shape_directions, expression_directions.to(dtype)], dim=-1)

    def get_shape_components(
        self,
        shape_components: ShapeComponents | dict | torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> ShapeComponents:
        out = ShapeComponents(
            shape_components,
            use_expression=self.use_expression,
            use_dmpl=False,
            device=device if device else self.device,
        )
        out.valid_attr_sizes = (self.betas_size, 0, self.expression_size)
        return out

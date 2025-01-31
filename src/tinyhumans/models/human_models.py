"""Human body models based on BaseParametricModel.

This module defines specific human body model classes such as SMPL, SMPLH, and SMPLX, which inherit from the
BaseParametricModel class and provide model-specific initialization and shape component handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from tinyhumans.models.base_parametric_model import BaseParametricModel
from tinyhumans.types import ShapeComponents

if TYPE_CHECKING:
    from pathlib import Path


class SMPL(BaseParametricModel):
    """SMPL body model class.

    This class implements the SMPL body model, inheriting from BaseParametricModel. It initializes the SMPL model with
    specific parameters and configurations, including handling of DMPL shape components.

    Attributes:
        pose_parts (set[str]): Set of pose parameter groups, updated to include "body" and "hand".
        body_pose_size (int): Size of body pose parameters (63).
        hand_pose_size (int): Size of hand pose parameters (6).
        dmpls_size (int): Size of DMPL shape parameters, if used.
        use_dmpl (bool): Flag indicating whether DMPL shape components are used.

    """

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
        """Initialize SMPL model.

        Args:
            pretrained_model_path (str | Path): Path to the pretrained SMPL model parameters (.npz file).
            gender (str, optional): Gender of the model ("neutral", "male", "female"). Defaults to "neutral".
            num_betas (int, optional): Number of shape parameters (betas). Defaults to 10.
            num_dmpls (int | None, optional): Number of DMPL shape parameters. Defaults to None (no DMPL).
            dmpl_filename (str | Path | None, optional): Path to the DMPL eigenvectors file (.npz).
                Required if `num_dmpls` > 0. Defaults to None.
            do_pose_conditioned_shape (bool, optional): Whether to use pose-conditioned shape displacements.
                Defaults to True.
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        Raises:
            ValueError: If `dmpl_filename` is not provided when `num_dmpls` > 0.

        """
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
        """Get ShapeComponents object for SMPL model.

        Overrides the base class method to handle DMPL shape components.

        Args:
            shape_components (ShapeComponents | dict | torch.Tensor | None, optional): Input shape components.
                Defaults to None.
            device (torch.device | str | None, optional): Device to put the ShapeComponents object on.
                Defaults to None (uses model device).

        Returns:
            ShapeComponents: ShapeComponents object with SMPL-specific shape parameters.

        """
        out = ShapeComponents(
            shape_components, use_expression=False, use_dmpl=self.use_dmpl, device=device if device else self.device
        )
        out.valid_attr_sizes = (self.betas_size, self.dmpls_size, 0)
        return out


class SMPLH(SMPL):
    """SMPLH body model class.

    This class implements the SMPLH body model, inheriting from SMPL. It specializes the SMPL model to include hand
    pose parameters with a larger size.

    Attributes:
        hand_pose_size (int): Size of hand pose parameters for SMPLH (90).

    """

    # Avoid re-writing __init__
    @property
    def hand_pose_size(self) -> int:
        """int: Size of hand pose parameters for SMPLH (90)."""
        return 15 * 3 * 2

    @hand_pose_size.setter
    def hand_pose_size(self, value: int) -> None:
        pass


class SMPLX(SMPLH):
    """SMPLX body model class.

    This class implements the SMPLX body model, inheriting from SMPLH. It further specializes SMPLH to include
    expression and jaw/eye pose parameters.

    Attributes:
        pose_parts (set[str]): Set of pose parameter groups, updated to include "jaw" and "eyes".
        jaw_pose_size (int): Size of jaw pose parameters (3).
        eyes_pose_size (int): Size of eye pose parameters (6).
        expression_size (int): Size of expression shape parameters, if used.
        use_expression (bool): Flag indicating whether expression shape components are used.

    """

    def __init__(
        self,
        pretrained_model_path: str | Path,
        gender: str = "neutral",
        num_betas: int = 10,
        num_expression: int | None = None,
        do_pose_conditioned_shape: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize SMPLX model.

        Args:
            pretrained_model_path (str | Path): Path to the pretrained SMPLX model parameters (.npz file).
            gender (str, optional): Gender of the model ("neutral", "male", "female"). Defaults to "neutral".
            num_betas (int, optional): Number of shape parameters (betas). Defaults to 10.
            num_expression (int | None, optional): Number of expression shape parameters.
                Defaults to None (no expression).
            do_pose_conditioned_shape (bool, optional): Whether to use pose-conditioned shape displacements.
                Defaults to True.
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        """
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
            expression_directions = torch.from_numpy(self.full_shape_directions[:, :, 300 : (300 + num_expression)])
            self.shape_directions = torch.cat([self.shape_directions, expression_directions.to(dtype)], dim=-1)

    def get_shape_components(
        self,
        shape_components: ShapeComponents | dict | torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> ShapeComponents:
        """Get ShapeComponents object for SMPLX model.

        Overrides the base class method to handle expression shape components.

        Args:
            shape_components (ShapeComponents | dict | torch.Tensor | None, optional): Input shape components.
                Defaults to None.
            device (torch.device | str | None, optional): Device to put the ShapeComponents object on.
                Defaults to None (uses model device).

        Returns:
            ShapeComponents: ShapeComponents object with SMPLX-specific shape parameters.

        """
        out = ShapeComponents(
            shape_components,
            use_expression=self.use_expression,
            use_dmpl=False,
            device=device if device else self.device,
        )
        out.valid_attr_sizes = (self.betas_size, 0, self.expression_size)
        return out

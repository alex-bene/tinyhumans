"""Human body models based on BaseParametricModel.

This module defines specific human body model classes such as SMPL, SMPLH, and SMPLX, which inherit from the
BaseParametricModel class and provide model-specific initialization and shape component handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from tinyhumans.models.base_parametric_model import BaseParametricModel
from tinyhumans.types import ShapeComponents

if TYPE_CHECKING:
    from pathlib import Path


class SMPL(BaseParametricModel):
    """SMPL body model class.

    This class implements the SMPL body model, inheriting from BaseParametricModel. It initializes the SMPL model with
    specific parameters and configurations, including handling of DMPL shape components.

    Attributes:
        body_pose_size (int): Size of body pose parameters (63).
        hand_pose_size (int): Size of hand pose parameters (6).
        dmpls_size (int): Size of DMPL shape parameters, if used.
        use_dmpl (bool): Flag indicating whether DMPL shape components are used.
        betas_size (int): Size of the betas shape parameters.

    """

    body_pose_size = 21 * 3
    hand_pose_size = 1 * 3 * 2

    def __init__(
        self,
        vertices_template: Tensor,
        faces: Tensor,
        kinematic_tree_vector: Tensor,
        blending_weights: Tensor,
        num_joints: int,
        num_betas: int | None = None,
        num_dmpls: int | None = None,
        body_type: str | None = None,
        gender: str = "neutral",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize SMPL model.

        Args:
            vertices_template (torch.Tensor): Mean template vertices of body at rest.
            faces (torch.Tensor): Faces of the mesh topology.
            kinematic_tree_vector (torch.Tensor): Kinematic tree vector for the body model.
            blending_weights (torch.Tensor): Linear blend skinning weights.
            num_joints (int): Number of joints in the body model.
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to 10.
            num_dmpls (int | None, optional): Number of DMPL shape parameters. Defaults to None (no DMPL).
            body_type (str, optional): Type of the body model (e.g., "smpl", "smplh", "smplx"). Defaults to None.
            gender (str, optional): Gender of the model ("neutral", "male", "female"). Defaults to "neutral".
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        """
        # Setup shapes of default parameters
        self.use_dmpl = num_dmpls is not None and num_dmpls > 0
        self.dmpls_size = num_dmpls if self.use_dmpl else 0
        self.betas_size = num_betas if num_betas is not None and num_betas > 0 else 0

        super().__init__(
            body_type="smpl" if body_type is None else body_type,
            gender=gender,
            num_shape_coeffs=num_betas + self.dmpls_size,
            num_joints=num_joints,
            vertices_template=vertices_template,
            faces=faces,
            kinematic_tree_vector=kinematic_tree_vector,
            blending_weights=blending_weights,
            dtype=dtype,
        )

    @classmethod
    def load_shape_components(
        cls,
        model_params_dict: dict[str, Tensor],
        num_betas: int | None = None,
        num_dmpls: int | None = None,
        dmpl_filename: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Load shape components from model parameters.

        Args:
            model_params_dict (dict[str, Tensor]): Dictionary of model parameters.
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to None.
            num_dmpls (int | None, optional): Number of DMPL shape parameters. Defaults to None.
            dmpl_filename (str | Path | None, optional): Path to the DMPL eigenvectors file (.npz).
                Required if `num_dmpls` > 0. Defaults to None.
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        Returns:
            Tensor: Shape components tensor.

        Raises:
            ValueError: If `dmpl_filename` is not provided when `num_dmpls` > 0.

        """
        shape_blend_components = super().load_shape_components(model_params_dict, num_betas, dtype)

        if num_dmpls is None:
            num_dmpls = 300 - model_params_dict["shapedirs"].shape[-1]
        elif num_dmpls < 0:
            num_dmpls = 0

        if num_dmpls > 0:
            if dmpl_filename is None:
                msg = "`dmpl_filename` should be provided when using dmpls!"
                raise ValueError(msg)
            dmpl_blend_components = torch.from_numpy(np.load(dmpl_filename)["eigvec"][:, :, :num_dmpls]).to(dtype)
            # Shape + DMPL components -> Shape components (num_vertices x 3 x num_shape_coeffs)
            shape_blend_components = torch.cat([shape_blend_components, dmpl_blend_components], dim=-1)

        return shape_blend_components

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
        body_pose_size (int): Size of body pose parameters (63).
        hand_pose_size (int): Size of hand pose parameters (90).
        dmpls_size (int): Size of DMPL shape parameters, if used.
        use_dmpl (bool): Flag indicating whether DMPL shape components are used.
        betas_size (int): Size of the betas shape parameters.

    """

    hand_pose_size = 15 * 3 * 2


class SMPLX(BaseParametricModel):
    """SMPLX body model class.

    This class implements the SMPLX body model, inheriting from SMPLH. It further specializes SMPLH to include
    expression and jaw/eye pose parameters.

    Attributes:
        pose_parts (set[str]): Set of pose parameter groups, updated to include "jaw" and "eyes".
        body_pose_size (int): Size of body pose parameters (63).
        hand_pose_size (int): Size of hand pose parameters (6).
        jaw_pose_size (int): Size of jaw pose parameters (3).
        eyes_pose_size (int): Size of eye pose parameters (6).
        expression_coeffs_size (int): Size of expression shape parameters, if used.
        use_expression (bool): Flag indicating whether expression shape components are used.
        betas_size (int): Size of the betas shape parameters.

    """

    body_pose_size = 21 * 3
    hand_pose_size = 1 * 3 * 2
    jaw_pose_size = 1 * 3
    eyes_pose_size = 2 * 3

    def __init__(
        self,
        vertices_template: Tensor,
        faces: Tensor,
        kinematic_tree_vector: Tensor,
        blending_weights: Tensor,
        num_joints: int,
        num_betas: int | None = None,
        num_expression_coeffs: int | None = None,
        body_type: str | None = None,
        gender: str = "neutral",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize SMPLX model.

        Args:
            vertices_template (torch.Tensor): Mean template vertices of body at rest.
            faces (torch.Tensor): Faces of the mesh topology.
            kinematic_tree_vector (torch.Tensor): Kinematic tree vector for the body model.
            blending_weights (torch.Tensor): Linear blend skinning weights.
            num_joints (int): Number of joints in the body model.
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to 10.
            num_expression_coeffs (int | None, optional): Number of expression shape parameters.
                Defaults to None (no expression).
            body_type (str, optional): Type of the body model (e.g., "smpl", "smplh", "smplx"). Defaults to None.
            gender (str, optional): Gender of the model ("neutral", "male", "female"). Defaults to "neutral".
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        """
        # Setup shapes of default parameters
        self.betas_size = num_betas if num_betas is not None and num_betas > 0 else 0
        self.use_expression = num_expression_coeffs is not None and num_expression_coeffs > 0
        self.expression_coeffs_size = num_expression_coeffs if self.use_expression else 0

        super().__init__(
            body_type="smplx" if body_type is None else body_type,
            gender=gender,
            num_shape_coeffs=num_betas + self.expression_coeffs_size,
            num_joints=num_joints,
            vertices_template=vertices_template,
            faces=faces,
            kinematic_tree_vector=kinematic_tree_vector,
            blending_weights=blending_weights,
            dtype=dtype,
        )

    @classmethod
    def load_shape_components(
        cls,
        model_params_dict: dict[str, Tensor],
        num_betas: int | None = None,
        num_expression_coeffs: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Load shape components from model parameters.

        Args:
            model_params_dict (dict[str, Tensor]): Dictionary of model parameters.
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to None.
            num_expression_coeffs (int | None, optional): Number of expression shape parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        Returns:
            Tensor: Shape components tensor.

        """
        shape_blend_components = super().load_shape_components(model_params_dict, num_betas, dtype)

        if num_expression_coeffs is None:
            num_expression_coeffs = 300 - model_params_dict["shapedirs"].shape[-1]
        elif num_expression_coeffs < 0:
            num_expression_coeffs = 0

        # The orthonormal principal components for expression conditioned displacements
        # (num_vertices x 3 x num_expression_coeffs)
        if num_expression_coeffs > 0:
            expression_blend_components = torch.from_numpy(
                model_params_dict["shapedirs"][:, :, 300 : (300 + num_expression_coeffs)]
            ).to(dtype)
            # Shape + Expression components -> Shape components (num_vertices x 3 x num_shape_coeffs)
            shape_blend_components = torch.cat([shape_blend_components, expression_blend_components], dim=-1)

        return shape_blend_components

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
        out.valid_attr_sizes = (self.betas_size, 0, self.expression_coeffs_size)
        return out

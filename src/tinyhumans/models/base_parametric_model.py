"""Base parametric model for human bodies.

This module defines the BaseParametricModel class, which serves as an abstract base class for parametric 3D human body
models. It provides common functionalities for loading model parameters, managing shape and pose components, and
performing linear blend skinning to generate meshes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from smplx.lbs import lbs as linear_blend_skinning
from torch import nn

from tinyhumans.mesh import BodyMeshes
from tinyhumans.tiny_types import FLAMEPose, MANOPose, Pose, ShapeComponents, SMPLHPose, SMPLPose, SMPLXPose


class BaseParametricModel(nn.Module):
    """Abstract base class for parametric 3D human body models.

    This class provides common functionalities for loading model parameters, managing shape and pose components, and
    performing linear blend skinning to generate meshes. It is intended to be subclassed by specific body model classes
    such as SMPL, SMPLH, and SMPLX.

    Attributes:
        pose_parts (set[str]): Set of pose parameter groups (e.g., "body", "hand").
        gender (str): Gender of the model ("neutral", "male", "female").
        body_type (str): Type of the body model (e.g., "smpl", "smplh", "smplx").
        _pose_class (type[Pose]): Pose class associated with the body model type.
        kinematic_tree_table (torch.Tensor): Kinematic tree table for the body model.
        blending_weights (torch.Tensor): Linear blend skinning weights.
        default_vertices_template (torch.Tensor): Mean template vertices.
        joint_regressor (torch.Tensor): Regressor for joint locations given shape.
        faces (torch.Tensor): Faces of the mesh topology.
        shape_directions (torch.Tensor): Shape PCA directions.
        pose_directions (torch.Tensor, optional): Pose PCA directions, if pose-conditioned shape is enabled.

    """

    def __init__(
        self,
        pretrained_model_path: str | Path,
        body_type: str | None = None,
        gender: str = "neutral",
        num_betas: int = 10,
        do_pose_conditioned_shape: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize BaseParametricModel.

        Args:
            pretrained_model_path (str | Path): Path to the pretrained model parameters (.npz file).
            body_type (str, optional): Type of the body model (e.g., "smpl", "smplh", "smplx").
                If None, it will be inferred from the model parameters if possible. Defaults to None.
            gender (str, optional): Gender of the model ("neutral", "male", "female"). Defaults to "neutral".
            num_betas (int, optional): Number of shape parameters (betas). Defaults to 10.
            do_pose_conditioned_shape (bool, optional): Whether to use pose-conditioned shape displacements.
                Defaults to True.
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        Raises:
            ValueError: If `pretrained_model_path` is not a .npz file.
            ValueError: If required keys are missing in the model dictionary.
            ValueError: If `body_type` is not provided and cannot be inferred from model parameters.
            ValueError: If the provided `body_type` does not match the inferred body type from model parameters.

        """
        super().__init__()
        self.pose_parts = set()
        self.gender = gender

        # Load model parameters
        model_params_dict, self.body_type = self.load_model_weights(pretrained_model_path, do_pose_conditioned_shape)

        # Check body type is valid
        if body_type:
            if self.body_type is not None and body_type.lower() != self.body_type.lower():
                msg = (
                    f"Body type provided or class used ({body_type}) does not match the body type inferred from the "
                    f"model parameters ({self.body_type.value}). Using the inferred body type."
                )
                raise ValueError(msg)
            if self.body_type is None:
                SMPLHPose.check_model_type(body_type)
                self.body_type = body_type
        elif not self.body_type:
            msg = "`body_type` must be provided if `posedirs` is not in the model dictionary."
            raise ValueError(msg)

        self._pose_class = (
            SMPLPose
            if self.body_type == "smpl"
            else SMPLHPose
            if self.body_type == "smplh"
            else SMPLXPose
            if self.body_type == "smplx"
            else FLAMEPose
            if self.body_type == "flame"
            else MANOPose
            if self.body_type == "mano"
            else None
        )

        # indices of parents for each joints
        kinematic_tree_table = torch.from_numpy(model_params_dict["kintree_table"]).to(torch.long)
        # LBS weights
        blending_weights = torch.from_numpy(model_params_dict["weights"]).to(dtype)
        # Mean template vertices
        default_vertices_template = torch.from_numpy(model_params_dict["v_template"]).unsqueeze(0).to(dtype)
        # Regressor for joint locations given shape - 6890 x 24
        joint_regressor = torch.from_numpy(model_params_dict["J_regressor"]).to(dtype)
        # Faces
        faces = torch.from_numpy(model_params_dict["f"]).unsqueeze(0).to(torch.long)

        # The PCA (?) vectors for pose conditioned displacements
        pose_directions = None
        if do_pose_conditioned_shape:
            ## Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
            pose_directions = torch.from_numpy(model_params_dict["posedirs"]).to(dtype).flatten(0, 1).T

        # Shape parameters
        self.full_shape_directions = model_params_dict["shapedirs"]
        num_betas = self.full_shape_directions.shape[-1] if num_betas < 1 else num_betas
        ## The PCA vectors for shape conditioned displacements
        shape_directions = torch.from_numpy(self.full_shape_directions[:, :, :num_betas]).to(dtype)

        # Setup shapes for default parameters
        self.betas_size = num_betas
        self.root_position_size = 3
        self.root_orientation_size = 3

        # Register buffers
        self.register_buffer("kinematic_tree_table", kinematic_tree_table)
        self.kinematic_tree_table: torch.Tensor = self.kinematic_tree_table
        self.register_buffer("blending_weights", blending_weights)
        self.blending_weights: torch.Tensor = self.blending_weights
        self.register_buffer("default_vertices_template", default_vertices_template)
        self.default_vertices_template: torch.Tensor = self.default_vertices_template
        self.register_buffer("joint_regressor", joint_regressor)
        self.joint_regressor: torch.Tensor = self.joint_regressor
        self.register_buffer("faces", faces)
        self.faces: torch.Tensor = self.faces
        self.register_buffer("shape_directions", shape_directions)
        self.shape_directions: torch.Tensor = self.shape_directions
        if do_pose_conditioned_shape:
            self.register_buffer("pose_directions", pose_directions)
            self.pose_directions: torch.Tensor = self.pose_directions

    @property
    def device(self) -> torch.device:
        """torch.device: Device on which the model is loaded."""
        return self.blending_weights.device

    @property
    def dtype(self) -> torch.dtype:
        """torch.dtype: Data type of the model parameters."""
        return self.blending_weights.dtype

    def load_model_weights(
        self, pretrained_model_path: str | Path, do_pose_conditioned_shape: bool = False
    ) -> tuple[dict[str, np.ndarray], str | None]:
        """Load model weights from a pretrained model file.

        Args:
            pretrained_model_path (str | Path): Path to the pretrained model parameters (.npz file).
            do_pose_conditioned_shape (bool, optional): Whether to load pose-conditioned shape parameters.
                Defaults to False.

        Returns:
            tuple[dict[str, np.ndarray], str | None]: A tuple containing:
                - model_params_dict (dict): Dictionary of model parameters loaded from the .npz file.
                - body_type (str | None): Inferred body type from the model parameters, or None if not inferable.

        Raises:
            ValueError: If `pretrained_model_path` is not a .npz file.
            ValueError: If required keys are missing in the model dictionary.

        """
        # Load model parameters
        if Path(pretrained_model_path).suffix == ".npz":
            model_params_dict: dict[str, np.ndarray] = np.load(pretrained_model_path, encoding="latin1")
        else:
            msg = f"`pretrained_model_path` must be a .npz file: {pretrained_model_path}"
            raise ValueError(msg)

        # Check that all required keys are present in the model dictionary
        should_exist_in_dict = ["v_template", "f", "shapedirs", "J_regressor", "kintree_table", "weights"] + (
            ["posedirs"] if do_pose_conditioned_shape else []
        )
        for key in should_exist_in_dict:
            if key not in model_params_dict:
                msg = f"Key {key} not found in model dictionary read from {pretrained_model_path}"
                raise ValueError(msg)

        body_type = None
        if "posedirs" in model_params_dict:
            npose_params = model_params_dict["posedirs"].shape[2] // 3
            body_type = {12: "flame", 69: "smpl", 153: "smplh", 162: "smplx", 45: "mano"}[npose_params]

        return model_params_dict, body_type

    def get_shape_components(
        self,
        shape_components: ShapeComponents | dict | torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> ShapeComponents:
        """Get ShapeComponents object from input.

        Args:
            shape_components (ShapeComponents | dict | torch.Tensor | None, optional): Input shape components.
                Defaults to None.
            device (torch.device | str | None, optional): Device to put the ShapeComponents object on.
                Defaults to None (uses model device).

        Returns:
            ShapeComponents: ShapeComponents object.

        """
        out = ShapeComponents(
            shape_components, use_expression=False, use_dmpl=False, device=device if device else self.device
        )
        out.valid_attr_sizes = (self.betas_size,)
        return out

    def get_default(self, key: str) -> torch.Tensor:
        """Get default parameter tensor.

        If the default parameter tensor for the given key is not already registered as a buffer, it will be created
        and registered.

        Args:
            key (str): Name of the parameter (e.g., "root_position", "root_orientation", "vertices_template").

        Returns:
            torch.Tensor: Default parameter tensor of shape (1, parameter_size).

        """
        if hasattr(self, "default_" + key):
            return getattr(self, "default_" + key)

        self.register_buffer(
            "default_" + key,
            torch.zeros(1, getattr(self, key + "_size"), dtype=self.dtype, device=self.device),
            persistent=False,
        )

        return getattr(self, "default_" + key)

    def infer_batch_size(self, kwargs: dict) -> int:
        """Infer batch size from input keyword arguments.

        Args:
            kwargs (dict): Dictionary of input keyword arguments.

        Returns:
            int: Batch size inferred from the input tensors.

        Raises:
            ValueError: If input tensors have inconsistent batch sizes.

        """
        bs = max(len(value) for value in kwargs.values() if torch.is_tensor(value))

        if not all(bs == len(value) for value in kwargs.values() if torch.is_tensor(value)):
            msg = "All tensors must have the same batch size."
            raise ValueError(msg)

        return bs

    def forward(
        self,
        poses: Pose | dict | torch.Tensor | None = None,
        shape_components: ShapeComponents | dict | torch.Tensor | None = None,
        *,
        root_positions: torch.Tensor | None = None,
        root_orientations: torch.Tensor | None = None,
        vertices_templates: torch.Tensor | None = None,
        transform_poses_to_rotation_matrices: bool = True,
    ) -> BodyMeshes:
        """Forward pass of the base parametric model.

        Generates body meshes from pose and shape parameters using linear blend skinning.

        Args:
            poses (Pose | dict | torch.Tensor | None, optional): Pose parameters. Defaults to None.
            shape_components (ShapeComponents | dict | torch.Tensor | None, optional): Shape parameters.
                Defaults to None.
            root_positions (torch.Tensor | None, optional): Root positions of the bodies. Defaults to None.
            root_orientations (torch.Tensor | None, optional): Root orientations of the bodies. Defaults to None.
            vertices_templates (torch.Tensor | None, optional): Template vertices. Defaults to None.
            transform_poses_to_rotation_matrices (bool, optional): Whether to transform pose parameters to rotation
                matrices. Defaults to True.

        Returns:
            BodyMeshes: Output body meshes with vertices, faces, joints, poses, shape components, root positions,
                root orientation, and vertices template.

        """
        # Make sure the inputs are of the correct type
        poses = self._pose_class(poses)
        shape_components = self.get_shape_components(shape_components)

        # Infer batch size
        batch_size = self.infer_batch_size(poses.to_dict() | shape_components.to_dict() | locals())

        # Fill in default values if None
        root_positions = self.get_default("root_position") if root_positions is None else root_positions
        root_positions = root_positions.expand(batch_size, -1)
        root_orientation = self.get_default("root_orientation") if root_orientations is None else root_orientations
        root_orientation = root_orientation.expand(batch_size, -1)
        vertices_template = self.get_default("vertices_template") if vertices_templates is None else vertices_templates

        # Get pose components
        pose_tensor = torch.cat([root_orientation, poses.to_tensor().expand(batch_size, -1)], dim=-1)

        # Linear blend skinning
        verts, joints = linear_blend_skinning(
            betas=shape_components.to_tensor().expand(batch_size, -1),
            pose=pose_tensor,
            v_template=vertices_template.expand(batch_size, -1, -1),
            shapedirs=self.shape_directions,
            posedirs=self.pose_directions,
            J_regressor=self.joint_regressor,
            parents=self.kinematic_tree_table[0],
            lbs_weights=self.blending_weights,
            pose2rot=transform_poses_to_rotation_matrices,
        )

        return BodyMeshes(
            verts=verts + root_positions.unsqueeze(dim=1),
            faces=self.faces.expand(batch_size, -1, -1),
            joints=joints + root_positions.unsqueeze(dim=1).expand(batch_size, -1, -1),
            poses=poses,
            shape_components=shape_components,
            root_positions=root_positions,
            root_orientation=root_orientation,
            vertices_template=vertices_template,
            # bStree_table = self.kintree_table
        )

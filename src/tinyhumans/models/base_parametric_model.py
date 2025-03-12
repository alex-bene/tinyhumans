"""Base parametric model for human bodies.

This module defines the BaseParametricModel class, which serves as a base class for parametric 3D human body
models. It provides common functionalities for loading model parameters, managing shape and pose components, and
performing linear blend skinning to generate meshes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix
from torch import Tensor, nn

from tinyhumans.datatypes import (
    FLAMEPoses,
    MANOPoses,
    ParametricModelOutput,
    Poses,
    ShapeComponents,
    SMPLHPoses,
    SMPLPoses,
    SMPLXPoses,
)
from tinyhumans.models.base_model import BaseModel
from tinyhumans.tools import apply_rigid_transform


class BaseParametricModel(BaseModel):
    """Base class for parametric 3D human body models.

    This class provides common functionalities for loading model parameters, managing shape and pose components, and
    performing linear blend skinning to generate meshes. It is intended to be subclassed by specific body model classes
    such as SMPL, SMPLH, and SMPLX.

    Attributes:
        gender (str): Gender of the model ("neutral", "male", "female").
        body_type (str): Type of the body model (e.g., "smpl", "smplh", "smplx").
        _pose_class (type[Poses]): Poses class associated with the body model type.
        kinematic_tree_vector (torch.Tensor): Kinematic tree vector for the body model.
        blending_weights (torch.Tensor): Linear blend skinning weights.
        vertices_template (torch.Tensor): Mean template vertices.
        shaped_verts_to_joints_regressor (nn.Parameter): Regressor for joint locations given shape.
        pose_blend_components (nn.Parameter): Orthonormal principal components for pose conditioned displacements.
        shape_blend_components (nn.Parameter): Orthonormal principal components for shape conditioned displacements.
        shape_coeff_size (int): Number of shape coefficients.
        faces (torch.Tensor): Faces of the mesh topology.
        root_position (torch.Tensor): Default root position of the body.
        root_orientation (torch.Tensor): Default root orientation of the body.

    """

    def __init__(
        self,
        body_type: str,
        vertices_template: Tensor,
        faces: Tensor,
        kinematic_tree_vector: Tensor,
        blending_weights: Tensor,
        num_joints: int,
        num_shape_coeffs: int,
        gender: str = "neutral",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize BaseParametricModel.

        Args:
            body_type (str): Type of the body model (e.g., "smpl", "smplh", "smplx").
            vertices_template (torch.Tensor): Mean template vertices of body at rest.
            faces (torch.Tensor): Faces of the mesh topology.
            kinematic_tree_vector (torch.Tensor): Kinematic tree vector for the body model.
            blending_weights (torch.Tensor): Linear blend skinning weights.
            num_joints (int): Number of joints in the body model.
            num_shape_coeffs (int): Number of shape coefficients.
            gender (str, optional): Gender of the model ("neutral", "male", "female"). Defaults to "neutral".
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        Raises:
            NotImplementedError: If the class is instantiated directly.

        """
        if self.__class__ == BaseParametricModel:
            msg = "`BaseParametricModel` should only be instantiated through a subclass"
            raise NotImplementedError(msg)

        super().__init__()
        self.gender = gender
        num_vertices = vertices_template.shape[0]

        # Setup shapes for default parameters
        self.shape_coeff_size = num_shape_coeffs

        self.root_orientation_size = 3

        # Check body type is valid
        Poses.check_model_type(body_type)
        self.body_type = body_type

        # Set pose class to use
        self._pose_class = {
            "smpl": SMPLPoses,
            "smplh": SMPLHPoses,
            "smplx": SMPLXPoses,
            "flame": FLAMEPoses,
            "mano": MANOPoses,
        }.get(self.body_type, None)

        # Parameters of body in rest pose
        ## indices of parents for each joints (num_joints)
        self.register_buffer("kinematic_tree_vector", torch.from_numpy(kinematic_tree_vector).to(torch.long))
        self.kinematic_tree_vector: Tensor = self.kinematic_tree_vector
        ## LBS weights (num_vertices x num_joints)
        self.register_buffer("blending_weights", torch.from_numpy(blending_weights).to(dtype))
        self.blending_weights: Tensor = self.blending_weights
        ## Vertices template at rest (num_vertices x 3)
        self.register_buffer("vertices_template", torch.from_numpy(vertices_template).to(dtype))
        self.vertices_template: Tensor = self.vertices_template
        ## Faces (num_faces x 3)
        self.register_buffer("faces", torch.from_numpy(faces).to(torch.long))
        self.faces: Tensor = self.faces
        ## Default root pose
        self.register_buffer("root_position", torch.zeros(3, dtype=dtype), persistent=False)
        self.root_position: Tensor = self.root_position
        self.register_buffer("root_orientation", torch.zeros(3, dtype=dtype), persistent=False)
        self.root_orientation: Tensor = self.root_orientation

        # Learned body parameters
        ## Regressor for joint locations given shape (num_joints x num_vertices)
        self.shaped_verts_to_joints_regressor = nn.Parameter(torch.rand(num_vertices, num_joints, dtype=dtype))
        ## The orthonormal principal components for pose conditioned displacements ((num_joints-1)*9, num_vertices*3)
        ### One rotation matrix for each joint except the root joint
        self.pose_blend_components = nn.Parameter(torch.rand((num_joints - 1) * 9, num_vertices * 3, dtype=dtype))
        ## The orthonormal principal components for shape conditioned displacements (num_vertices, 3, num_shape_coeffs)
        self.shape_blend_components = nn.Parameter(torch.rand(num_vertices, num_shape_coeffs, dtype=dtype))

    @classmethod
    def _load_shape_components(
        cls,
        model_params_dict: dict[str, Tensor],
        num_betas: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Load shape components from model parameters.

        Args:
            model_params_dict (dict[str, Tensor]): Dictionary of model parameters.
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to None.
            device (torch.device | None, optional): Device to put the shape components on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        Returns:
            Tensor: Shape components tensor.

        """
        return torch.from_numpy(model_params_dict["shapedirs"][:, :, :num_betas]).to(device, dtype)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | Path,
        num_betas: int | None = None,
        device_map: str | torch.device | None = "auto",
        torch_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> BaseParametricModel:
        """Load a pre-trained model.

        Args:
            pretrained_model_path (str | Path): Path to the pretrained model parameters (.npz file).
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to None.
            device_map (str | torch.device): Device to map the model weights to.
            torch_dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.
            **kwargs: Additional keyword arguments passed to the model constructor.

        Returns:
            BaseParametricModel: A pre-trained BaseParametricModel instance.

        """
        pretrained_model_path: Path = Path(pretrained_model_path)
        if not pretrained_model_path.exists():
            msg = f"Could not find the pretrained model path: {pretrained_model_path}"
            raise ValueError(msg)

        if device_map == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        elif isinstance(device_map, str):
            device = torch.device(device_map)
        elif isinstance(device_map, torch.device) or device_map is None:
            device = device_map
        else:
            msg = f"Invalid device_map: {device_map}"
            raise ValueError(msg)

        # Load model parameters
        model_params_dict, body_type = cls._load_model_weights(pretrained_model_path)

        # Regressor for joint locations given shape (num_joints x num_vertices)
        shaped_verts_to_joints_regressor = torch.from_numpy(model_params_dict["J_regressor"]).to(device, torch_dtype)
        # The orthonormal principal components for pose conditioned displacements (num_pose_points*3, num_vertices*3)
        pose_blend_components = torch.from_numpy(model_params_dict["posedirs"]).flatten(0, 1).T.to(device, torch_dtype)
        # The orthonormal principal components for shape conditioned displacements (num_vertices, 3, num_shape_coeffs)
        shape_blend_components = cls._load_shape_components(
            model_params_dict, num_betas, dtype=torch_dtype, device=device, **kwargs
        )
        num_betas = shape_blend_components.shape[-1]

        if cls == BaseParametricModel:
            msg = "`BaseParametricModel` should only be instantiated through a subclass"
            raise NotImplementedError(msg)

        # Create model
        obj: BaseParametricModel = cls(
            body_type=body_type,
            num_betas=num_betas,
            num_joints=shaped_verts_to_joints_regressor.shape[0],
            vertices_template=model_params_dict["v_template"],
            faces=model_params_dict["f"],
            kinematic_tree_vector=model_params_dict["kintree_table"][0],
            blending_weights=model_params_dict["weights"],
            **kwargs,
        )

        # Set pretrained parameters
        obj.shaped_verts_to_joints_regressor = nn.Parameter(shaped_verts_to_joints_regressor).to(device, torch_dtype)
        obj.shape_blend_components = nn.Parameter(shape_blend_components).to(device, torch_dtype)
        obj.pose_blend_components = nn.Parameter(pose_blend_components).to(device, torch_dtype)

        # Set evaluation mode
        obj.eval()

        return obj.to(device, torch_dtype)

    @classmethod
    def _load_model_weights(cls, pretrained_model_path: str | Path) -> tuple[dict[str, np.ndarray], str | None]:
        """Load model weights from a pretrained model file.

        Args:
            pretrained_model_path (str | Path): Path to the pretrained model parameters (.npz file).

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
        should_exist_in_dict = ["v_template", "f", "shapedirs", "J_regressor", "kintree_table", "weights", "posedirs"]

        for key in should_exist_in_dict:
            if key not in model_params_dict:
                msg = f"Key {key} not found in model dictionary read from {pretrained_model_path}"
                raise ValueError(msg)

        npose_params = model_params_dict["posedirs"].shape[2] // 3
        body_type = {12: "flame", 69: "smpl", 153: "smplh", 162: "smplx", 45: "mano"}[npose_params]

        return model_params_dict, body_type

    @property
    def device(self) -> torch.device:
        """torch.device: Device on which the model is loaded."""
        return self.blending_weights.device

    @property
    def dtype(self) -> torch.dtype:
        """torch.dtype: Data type of the model parameters."""
        return self.blending_weights.dtype

    def get_shape_components(
        self, shape_components: ShapeComponents | dict | Tensor | None = None, device: torch.device | str | None = None
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
        out.valid_attr_sizes = (self.shape_coeff_size,)
        return out

    def infer_batch_size(self, kwargs: dict) -> int:
        """Infer batch size from input keyword arguments.

        Args:
            kwargs (dict): Dictionary of input keyword arguments.

        Returns:
            int: Batch size inferred from the input tensors.

        Raises:
            ValueError: If input tensors have inconsistent batch sizes.

        """
        bs = max(len(value) for value in kwargs.values() if torch.is_tensor(value) and value.ndim > 0)
        if not all(bs == len(value) for value in kwargs.values() if torch.is_tensor(value) and value.ndim > 0):
            msg = "All tensors must have the same batch size."
            raise ValueError(msg)

        return bs

    def forward(
        self,
        poses: Poses | dict | Tensor | None = None,
        shape_components: ShapeComponents | dict | Tensor | None = None,
        *,
        root_positions: Tensor | None = None,
        root_orientations: Tensor | None = None,
        poses_in_axis_angles: bool = True,
    ) -> ParametricModelOutput:
        """Forward pass of the base parametric model.

        Generates body meshes from pose and shape parameters using linear blend skinning.

        Args:
            poses (Poses | dict | torch.Tensor | None, optional): Poses parameters. Defaults to None.
            shape_components (ShapeComponents | dict | torch.Tensor | None, optional): Shape parameters.
                Defaults to None.
            root_positions (torch.Tensor | None, optional): Root positions of the bodies. Defaults to None.
            root_orientations (torch.Tensor | None, optional): Root orientations of the bodies. Defaults to None.
            poses_in_axis_angles (bool, optional): Whether the provided poses are in axis angle representations (need to
                be transformed to rotation matrices in this case). Defaults to True.

        Returns:
            BodyMeshes: Output body meshes with vertices, faces, joints, poses, shape components, root positions,
                root orientation, and vertices template.

        """
        # Make sure the inputs are of the correct type
        poses = self._pose_class(poses)
        shape_components = self.get_shape_components(shape_components, device=self.device)

        # Infer batch size
        batch_size = self.infer_batch_size(poses.to_dict() | shape_components.to_dict() | locals())

        # Fill in default values if None
        root_positions = self.root_position if root_positions is None else root_positions
        root_positions = root_positions.expand(batch_size, -1)
        root_orientation = self.root_orientation if root_orientations is None else root_orientations
        root_orientation = root_orientation.expand(batch_size, -1)

        # Get pose components
        pose_tensor = torch.cat([root_orientation, poses.to_tensor().expand(batch_size, -1)], dim=-1)

        # Linear blend skinning
        verts, joints = self.linear_blend_skinning(
            betas=shape_components.to_tensor().expand(batch_size, -1),
            pose=pose_tensor,
            vertices_template=self.vertices_template.expand(batch_size, -1, -1),
            poses_in_axis_angles=poses_in_axis_angles,
        )

        return ParametricModelOutput(
            verts=verts + root_positions.unsqueeze(dim=1),
            joints=joints + root_positions.unsqueeze(dim=1).expand(batch_size, -1, -1),
            poses=poses,
            shape_components=shape_components,
            root_positions=root_positions,
            root_orientations=root_orientation,
            vertices_template=self.vertices_template,
            # bStree_table = self.kintree_table
        )

    def linear_blend_skinning(
        self, betas: Tensor, pose: Tensor, vertices_template: Tensor, poses_in_axis_angles: bool = True
    ) -> tuple[Tensor, Tensor]:
        """Perform Linear Blend Skinning with the given shape and pose parameters.

        Args:
            betas (torch.Tensor): The tensor of shape parameters with shape (batch_size, num_betas).
            pose (torch.Tensor): The pose parameters in axis-angle format with shape (batch_size, (num_joints + 1) * 3)
                if poses_in_axis_angles is True, otherwise (batch_size, (num_joints + 1) * 9).
            vertices_template (torch.Tensor): The template (mean) mesh that will be deformed with shape
                (batch_size, num_vertices, 3).
            poses_in_axis_angles (bool, optional): Whether the provided poses are in axis angle representations (need to
                be transformed to rotation matrices in this case). Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - The vertices of the mesh after applying the shape and pose displacements with shape
                  (batch_size, num_vertices, 3).
                - The joints of the model with shape (batch_size, num_joints, 3).

        """
        # 0. Infer parameters
        batch_size = max(betas.shape[0], pose.shape[0])
        device, dtype = betas.device, betas.dtype
        ## Transform axis-angle rotations into rotation matrices if needed
        rot_mats = axis_angle_to_matrix(pose.view(-1, 3)).view([batch_size, -1, 3, 3]) if poses_in_axis_angles else pose
        ## Set the correct rotation matrices shape
        rot_mats = rot_mats.view(batch_size, -1, 3, 3)

        # 1. Add the per vertex displacement due to the shape
        ## Same as batched matrix multiplication with implicit batch size expansion
        ## e.g.: torch.bmm(betas, self.shape_directions.expand(betas.shape[0], -1, -1))
        ## (B x num_betas) x (V x 3 x num_betas) -> (B x V x 3)
        verts_shaped = vertices_template + torch.einsum("bi,jki->bjk", betas, self.shape_blend_components)

        # 2. Add the per vertex displacement due to the pose
        ident = torch.eye(3, dtype=dtype, device=device)
        ## Get the pose feature vector (relative rotation matrices for each joint (ignore global rotation) minus identity)
        pose_feature = rot_mats[:, 1:, :, :] - ident
        ## Calculate the vertices offsets
        ## (N x J x 3 x 3) x (J*3*3, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), self.pose_blend_components).view(
            batch_size, -1, 3
        )
        ## Get the vertices considering the pose contributions
        verts_posed = verts_shaped + pose_offsets

        # 3. Get the joints locations
        ## Get the joints at rest (based on the shape)
        ## (B x V x 3) x (J x V) -> B x J x 3
        joints = torch.einsum("bij,ki->bkj", [verts_shaped, self.shaped_verts_to_joints_regressor])
        ## Get the global joint location based on the joint rotations (including global orientation)
        joints_posed, joint_transforms = apply_rigid_transform(rot_mats, joints, self.kinematic_tree_vector)

        # 5. Do skinning:
        ## (B x (J + 1) x 4 x 4) x (V x (J + 1)) -> (B x V x 4 x 4)
        skinning_transforms = torch.einsum("bjik,vj->bvik", [joint_transforms, self.blending_weights])
        ## Get the homogeneous coordinate representation of the vertices
        verts_posed = F.pad(verts_posed, (0, 1), mode="constant", value=1)
        ## (B x V x 4 x 4) x (B x V x 3) -> (B x V x 3)
        verts = torch.matmul(skinning_transforms, torch.unsqueeze(verts_posed, dim=-1))[:, :, :3, 0]

        return verts, joints_posed

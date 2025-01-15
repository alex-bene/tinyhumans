"""BodyModel class for TinyHumans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from smplx.lbs import lbs as linear_blend_skinning
from torch import nn

from tinyhumans.mesh import BodyMeshes
from tinyhumans.types import FLAMEPose, MANOPose, Pose, ShapeComponents, SMPLHPose, SMPLPose, SMPLXPose


class BaseParametricModel(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str | Path,
        body_type: str | None = None,
        gender: str = "neutral",
        num_betas: int = 10,
        do_pose_conditioned_shape: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
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
        return self.blending_weights.device

    @property
    def dtype(self) -> torch.dtype:
        return self.blending_weights.dtype

    # TODO: I have not seen this at all
    def load_model_weights(self, pretrained_model_path: str | Path, do_pose_conditioned_shape: bool = False):
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
        out = ShapeComponents(
            shape_components, use_expression=False, use_dmpl=False, device=device if device else self.device
        )
        out.valid_attr_sizes = (self.betas_size,)
        return out

    def get_default(self, key: str) -> torch.Tensor:
        if hasattr(self, "default_" + key):
            return getattr(self, "default_" + key)

        self.register_buffer(
            "default_" + key,
            torch.zeros(1, getattr(self, key + "_size"), dtype=self.dtype, device=self.device),
            persistent=False,
        )

        return getattr(self, "default_" + key)

    def infer_batch_size(self, kwargs: dict) -> int:
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

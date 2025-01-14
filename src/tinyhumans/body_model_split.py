"""BodyModel class for TinyHumans."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import torch
from smplx.lbs import lbs as linear_blend_skinning
from torch import nn

from src.tinyhumans.mesh import BodyMeshes
from src.tinyhumans.types import FLAMEPose, MANOPose, ModelType, Pose, ShapeComponents, SMPLHPose, SMPLPose, SMPLXPose

if TYPE_CHECKING:
    from pathlib import Path


class BodyModel(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str | Path,
        body_type: ModelType | str | None = None,
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
            if self.body_type is not None and body_type.lower() != self.body_type.value.lower():
                msg = (
                    f"Body type provided or class used ({body_type}) does not match the body type inferred from the "
                    f"model parameters ({self.body_type.value}). Using the inferred body type."
                )
                raise ValueError(msg)
            if self.body_type is None:
                self.body_type = ModelType(body_type)
        elif not self.body_type:
            msg = "`body_type` must be provided if `posedirs` is not in the model dictionary."
            raise ValueError(msg)

        self._pose_class = (
            SMPLPose
            if self.body_type.value == "smpl"
            else SMPLHPose
            if self.body_type.value == "smplh"
            else SMPLXPose
            if self.body_type.value == "smplx"
            else FLAMEPose
            if self.body_type.value == "flame"
            else MANOPose
            if self.body_type.value == "mano"
            else None
        )
        self._shape_components_class = ShapeComponents

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
    def load_model_weights(self, pretrained_model_path: str, do_pose_conditioned_shape: bool = False):
        # Load model parameters
        if pretrained_model_path.endswith(".npz"):
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
            body_type = ModelType(
                {
                    12: "flame",
                    69: "smpl",
                    153: "smplh",
                    162: "smplx",
                    45: "mano",
                    105: "animal_horse",
                    102: "animal_dog",
                }[npose_params]
            )

        return model_params_dict, body_type

    def get_default(self, key: str) -> torch.Tensor:
        if hasattr(self, "default_" + key):
            return getattr(self, "default_" + key)

        self.register_buffer(
            "default_" + key,
            torch.zeros(1, getattr(self, key + "_size"), dtype=self.dtype, device=self.device),
            persistent=False,
        )

        return getattr(self, "default_" + key)

    @property
    def default_pose(self) -> Pose:
        if hasattr(self, "_default_pose"):
            return self._default_pose

        self._default_pose = self._pose_class()
        for part in self.pose_parts:
            setattr(self._default_pose, part, self.get_default(part + "_pose"))
        return self._default_pose

    @property
    def default_shape_components(self) -> ShapeComponents:
        if hasattr(self, "_default_shape_components"):
            return self._default_shape_components

        self._default_shape_components = self._shape_components_class(use_expression=False, use_dmpl=False)
        return self._default_shape_components

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
        # assert not (v_template is not None and betas is not None), ValueError('vtemplate and betas could not be used jointly.')

        # Make sure the inputs are of the correct type
        poses: Pose = self._pose_class.create(poses)
        shape_components = self._shape_components_class.create(shape_components)

        # Infer batch size
        batch_size = self.infer_batch_size(asdict(poses) | asdict(shape_components) | locals())

        # Fill in default values if None
        root_positions = self.get_default("root_position") if root_positions is None else root_positions
        root_positions = root_positions.expand(batch_size, -1)
        root_orientation = self.get_default("root_orientation") if root_orientations is None else root_orientations
        root_orientation = root_orientation.expand(batch_size, -1)
        vertices_template = self.get_default("vertices_template") if vertices_templates is None else vertices_templates

        # Get shape and pose components and directions
        shape_components = shape_components.to_tensor(default_shape_components=self.default_shape_components)
        pose_tensor = torch.cat([root_orientation, poses.to_tensor(default_pose=self.default_pose)], dim=-1)
        pose_tensor = pose_tensor.expand(batch_size, -1)

        # Linear blend skinning
        verts, joints = linear_blend_skinning(
            betas=shape_components,
            pose=pose_tensor,
            v_template=vertices_template.expand(batch_size, -1, -1),
            shapedirs=self.shape_directions,
            posedirs=self.pose_directions,
            J_regressor=self.joint_regressor,
            parents=self.kinematic_tree_table[0],
            lbs_weights=self.blending_weights,
            pose2rot=transform_poses_to_rotation_matrices,
        )

        # res['bStree_table'] = self.kintree_table

        return BodyMeshes(
            verts=verts + root_positions.unsqueeze(dim=1),
            faces=self.faces.expand(batch_size, -1, -1),
            joints=joints + root_positions.unsqueeze(dim=1).expand(batch_size, -1, -1),
            poses=poses,
            shape_components=shape_components,
            root_positions=root_positions,
            root_orientation=root_orientation,
            vertices_template=vertices_template,
        )


class SMPL(BodyModel):
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
        self.dmpls_size = num_dmpls

        # DMPLs check and load and register dmpl directions
        self.use_dmpl = False
        if num_dmpls is not None and num_dmpls > 0:
            if dmpl_filename is None:
                msg = "`dmpl_filename` should be provided when using dmpls!"
                raise ValueError(msg)
            self.use_dmpl = True
            dmpl_directions = torch.from_numpy(np.load(dmpl_filename)["eigvec"][:, :, :num_dmpls]).to(dtype)
            self.shape_directions = torch.cat([self.shape_directions, dmpl_directions], dim=-1)

    @property
    def default_shape_components(self) -> ShapeComponents:
        if hasattr(self, "_default_shape_components"):
            return self._default_shape_components

        self._default_shape_components = self._shape_components_class(use_expression=False, use_dmpl=self.use_dmpl)
        if self.use_dmpl:
            self._default_shape_components.dmpls = self.get_default("dmpls")

        return self._default_shape_components


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
        self.expression_size = num_expression

        # Expression directions register
        self.use_expression = num_expression is not None
        if self.use_expression:
            ## The PCA vectors for expression conditioned displacements
            ### self.full_shape_directions.shape[-1] > 300
            expression_directions = torch.from_numpy(self.full_shape_directions[:, :, 300 : (300 + num_expression)])
            self.shape_directions = torch.cat([self.shape_directions, expression_directions.to(dtype)], dim=-1)

    @property
    def default_shape_components(self) -> ShapeComponents:
        if hasattr(self, "_default_shape_components"):
            return self._default_shape_components

        self._default_shape_components = self._shape_components_class(
            use_expression=self.use_expression, use_dmpl=False
        )
        if self.use_expression:
            self._default_shape_components.dmpls = self.get_default("expression")

        return self._default_shape_components

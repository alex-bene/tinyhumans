"""BodyModel class for TinyHumans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from smplx.lbs import lbs as linear_blend_skinning
from torch import nn

from src.tinyhumans.mesh import BodyMeshes

if TYPE_CHECKING:
    from pathlib import Path


class BodyModel(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str | Path,
        body_type: str | None = None,
        gender: str = "neutral",
        num_betas: int = 10,
        num_dmpls: int | None = None,
        num_expression: int | None = None,
        dmpl_filename: str | Path | None = None,
        do_pose_conditioned_shape: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_dmpls = num_dmpls
        self.num_expression = num_expression
        self.gender = gender

        # Load model parameters
        if pretrained_model_path.endswith(".npz"):
            model_params_dict: dict[str, np.ndarray] = np.load(pretrained_model_path, encoding="latin1")
        else:
            msg = f"`pretrained_model_path` must be a .npz file: {pretrained_model_path}"
            raise ValueError(msg)

        # Check body type is valid
        if body_type and body_type.lower() not in (
            ["smpl", "smplh", "smplx", "flame", "mano", "animal_horse", "animal_dog", "animal_rat"]
        ):
            msg = (
                "`body_type` should be one of "
                '"smpl", "smplh", "smplx", "flame", "mano", "animal_horse", "animal_dog", "animal_rat" or None'
            )
            raise ValueError(msg)

        if body_type:
            self.body_type = body_type.lower()
        elif "posedirs" in model_params_dict:
            npose_params = model_params_dict["posedirs"].shape[2] // 3
            self.body_type = {
                12: "flame",
                69: "smpl",
                153: "smplh",
                162: "smplx",
                45: "mano",
                105: "animal_horse",
                102: "animal_dog",
            }[npose_params]
        else:
            msg = "`body_type` must be provided if `posedirs` is not in the model dictionary."
            raise ValueError(msg)

        # DMPLs check and load
        self.use_dmpl = False
        if num_dmpls is not None:
            if dmpl_filename is not None:
                self.use_dmpl = True
                if self.body_type in ["smplx", "mano", "animal_horse", "animal_dog"]:
                    msg = "DMPLs only work with SMPL/SMPLH models for now."
                    raise NotImplementedError(msg)
            else:
                msg = "`dmpl_filename` should be provided when using dmpls!"
                raise ValueError(msg)

        if self.use_dmpl:
            dmpl_directions = torch.from_numpy(np.load(dmpl_filename)["eigvec"][:, :, :num_dmpls]).to(dtype)

        # Check that all required keys are present in the model dictionary
        should_exist_in_dict = ["v_template", "f", "shapedirs", "J_regressor", "kintree_table", "weights"] + (
            ["posedirs"] if do_pose_conditioned_shape else []
        )
        for key in should_exist_in_dict:
            if key not in model_params_dict:
                msg = f"Key {key} not found in model dictionary read from {pretrained_model_path}"
                raise ValueError(msg)
            # bdtype = torch.long if key in ["f", "kinematic_tree_table"] else dtype
            # self.register_buffer(key, torch.from_numpy(model_params_dict[key], dtype=bdtype))

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
        if do_pose_conditioned_shape:
            ## Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
            pose_directions = torch.from_numpy(model_params_dict["posedirs"]).to(dtype).flatten(0, 1).T
        else:
            pose_directions = None

        # Shape parameters
        num_betas = model_params_dict["shapedirs"].shape[-1] if num_betas < 1 else num_betas
        ## The PCA vectors for shape conditioned displacements
        shape_directions = torch.from_numpy(model_params_dict["shapedirs"][:, :, :num_betas]).to(dtype)

        self.use_expression = self.body_type in ["smplx", "flame"] and num_expression is not None
        if self.use_expression:
            if model_params_dict["shapedirs"].shape[-1] > 300:  # noqa: PLR2004
                begin_shape_id = 300
            else:
                begin_shape_id = 10
                num_expression = model_params_dict["shapedirs"].shape[-1] - 10

            ## The PCA vectors for expression conditioned displacements
            expression_directions = torch.from_numpy(
                model_params_dict["shapedirs"][:, :, begin_shape_id : (begin_shape_id + num_expression)]
            ).to(dtype)

        # Setup shapes of default parameters
        self.default_betas_size = num_betas
        if self.use_expression:
            self.default_expression_size = num_expression
        if self.use_dmpl:
            self.default_dmpls_size = num_dmpls

        self.default_body_pose_size = (
            63
            if self.body_type in ["smpl", "smplh", "smplx"]
            else 102
            if self.body_type in ["animal_dog", "animal_rat"]
            else 105
            if self.body_type == "animal_horse"
            else 3  # flame
        )
        self.default_hand_pose_size = (
            1 * 3 * 2
            if self.body_type == "smpl"
            else 15 * 3 * 2
            if self.body_type in ["smplh", "smplx"]
            else 15 * 3  # mano
        )
        self.default_jaw_pose_size = 1 * 3
        self.default_eye_pose_size = 2 * 3

        self.default_root_location_size = 3
        self.default_root_orientation_size = 3

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
        if self.use_dmpl:
            self.register_buffer("dmpl_directions", dmpl_directions)
            self.dmpl_directions: torch.Tensor = self.dmpl_directions
        if do_pose_conditioned_shape:
            self.register_buffer("pose_directions", pose_directions)
            self.pose_directions: torch.Tensor = self.pose_directions
        if self.use_expression:
            self.register_buffer("expression_directions", expression_directions)
            self.expression_directions: torch.Tensor = self.expression_directions

    @property
    def device(self) -> torch.device:
        return self.blending_weights.device

    @property
    def dtype(self) -> torch.dtype:
        return self.blending_weights.dtype

    def get_default_value(self, key: str) -> torch.Tensor:
        # check if in attributes
        if hasattr(self, "default_" + key):
            return getattr(self, "default_" + key)

        size = getattr(self, "default_" + key + "_size")
        # if not torch.is_tensor(shape_or_value):
        self.register_buffer(
            "default_" + key, torch.zeros(1, size, dtype=self.dtype, device=self.device), persistent=False
        )

        return getattr(self, "default_" + key)

    def forward(
        self,
        betas: torch.Tensor | None = None,
        body_pose: torch.Tensor | None = None,
        *,
        hand_pose: torch.Tensor | None = None,
        jaw_pose: torch.Tensor | None = None,
        eye_pose: torch.Tensor | None = None,
        expression: torch.Tensor | None = None,
        dmpls: torch.Tensor | None = None,
        root_location: torch.Tensor | None = None,
        root_orientation: torch.Tensor | None = None,
        vertices_template: torch.Tensor | None = None,
        pose2rot: bool = True,
    ) -> BodyMeshes:
        # assert not (v_template is not None and betas is not None), ValueError('vtemplate and betas could not be used jointly.')

        # Load default values for unset variables
        check_for_default = ["betas", "root_location", "root_orientation", "vertices_template"]
        check_for_default += ["body_pose"] if self.body_type != "mano" else []
        check_for_default += ["hand_pose"] if self.body_type in ["smplx", "smplh", "smpl", "mano"] else []
        check_for_default += ["jaw_pose", "eye_pose"] if self.body_type in ["smplx", "flame"] else []
        check_for_default += ["expression"] if self.use_expression else []
        check_for_default += ["dmpls"] if self.use_dmpl else []

        ## compute batchsize by any of the provided variables
        locals_snapshot = locals()
        batch_size = max(len(locals_snapshot[key]) for key in check_for_default if locals_snapshot[key] is not None)

        ## Fill in default values
        betas = self.get_default_value("betas").expand(batch_size, -1) if betas is None else betas
        root_location = (
            self.get_default_value("root_location").expand(batch_size, -1) if root_location is None else root_location
        )
        root_orientation = (
            self.get_default_value("root_orientation").expand(batch_size, -1)
            if root_orientation is None
            else root_orientation
        )
        vertices_template = (
            self.get_default_value("vertices_template").expand(batch_size, -1, -1)
            if vertices_template is None
            else vertices_template
        )
        if self.body_type != "mano":
            body_pose = self.get_default_value("body_pose").expand(batch_size, -1) if body_pose is None else body_pose
        if self.body_type in ["smplx", "smplh", "smpl", "mano"]:
            hand_pose = self.get_default_value("hand_pose").expand(batch_size, -1) if hand_pose is None else hand_pose
        if self.body_type in ["smplx", "flame"]:
            jaw_pose = self.get_default_value("jaw_pose").expand(batch_size, -1) if jaw_pose is None else jaw_pose
            eye_pose = self.get_default_value("eye_pose").expand(batch_size, -1) if eye_pose is None else eye_pose
        if self.use_expression:
            expression = (
                self.get_default_value("expression").expand(batch_size, -1) if expression is None else expression
            )
        if self.use_dmpl:
            dmpls = self.get_default_value("dmpls").expand(batch_size, -1) if dmpls is None else dmpls
        # for name, tensor in check_for_default.items():
        #     if locals()[name] is None:
        #         default: torch.Tensor = self.get_default_value("default_" + name)
        #         tensor = default.expand(batch_size, *default.shape[1:])

        # Calculate shape components and PCA directions
        if self.use_dmpl:
            shape_components = torch.cat([betas, dmpls], dim=-1)
            shape_directions = torch.cat([self.shape_directions, self.dmpl_directions], dim=-1)
        elif self.use_expression:
            shape_components = torch.cat([betas, expression], dim=-1)
            shape_directions = torch.cat([self.shape_directions, self.expression_directions], dim=-1)
        else:
            shape_components = betas
            shape_directions = self.shape_directions

        if self.body_type in ["smplh", "smpl"]:
            # orient:3, body:63, l_hand:45 if "smplh" else 3, r_hand:45 if "smplh" else 3
            full_pose = torch.cat([root_orientation, body_pose, hand_pose], dim=-1)
            pose_parts = ["root_orientation", "body_pose", "hand_pose"]
        elif self.body_type == "smplx":
            # orient:3, body:63, jaw:3, eyel:3, eyer:3, l_hand:45, r_hand:45
            full_pose = torch.cat([root_orientation, body_pose, jaw_pose, eye_pose, hand_pose], dim=-1)
            pose_parts = ["root_orientation", "body_pose", "jaw_pose", "eye_pose", "hand_pose"]
        elif self.body_type == "flame":
            # orient:3, body:3, jaw:3, l_eye:3, r_eye:3
            full_pose = torch.cat([root_orientation, body_pose, jaw_pose, eye_pose], dim=-1)
            pose_parts = ["root_orientation", "body_pose", "jaw_pose", "eye_pose"]
        elif self.body_type in ["mano"]:
            # orient:3, hand:45
            full_pose = torch.cat([root_orientation, hand_pose], dim=-1)
            pose_parts = ["root_orientation", "hand_pose"]
        elif self.body_type in ["animal_horse", "animal_dog", "animal_rat"]:
            # orient:3, body:105 if "animal_horse" else 102
            full_pose = torch.cat([root_orientation, body_pose], dim=-1)
            pose_parts = ["root_orientation", "body_pose"]

        verts, joints = linear_blend_skinning(
            betas=shape_components,
            pose=full_pose,
            v_template=vertices_template,
            shapedirs=shape_directions,
            posedirs=self.pose_directions,
            J_regressor=self.joint_regressor,
            parents=self.kinematic_tree_table[0].long(),
            lbs_weights=self.blending_weights,
        )

        # res['bStree_table'] = self.kintree_table

        locals_snapshot = locals()
        return BodyMeshes(
            verts=verts + root_location.unsqueeze(dim=1),
            faces=self.faces.expand(batch_size, -1, -1),
            joints=joints + root_location.unsqueeze(dim=1),
            poses={key: locals_snapshot[key] for key in pose_parts},
            root_location=root_location,
            root_orientation=root_orientation,
            vertices_template=vertices_template,
            expression=expression,
            dmpls=dmpls,
            betas=betas,
        )

"""Module for HPS prediction."""

from typing import TYPE_CHECKING, Literal

import torch
from smplcodec import SMPLVersion
from torch.nn.functional import pad

from tinyhumans.datatypes import SMPLData


class HPSLayer(torch.nn.Module):
    """HPS prediction layer."""

    def __init__(
        self,
        input_size: int,
        num_shape_parameters: int = 10,
        no_global_orientation: bool = False,
        no_global_translation: bool = False,
        body_type: Literal["smpl", "smplh", "smplx"] = "smpl",
    ) -> None:
        super().__init__()
        self.no_global_orientation = no_global_orientation
        self.no_global_translation = no_global_translation
        if not self.no_global_translation:
            self.global_translation_regressor = torch.nn.Linear(input_size, 3)
        self.shape_regressor = torch.nn.Linear(input_size, num_shape_parameters)
        if body_type not in ["smpl", "smplh", "smplx"]:
            msg = "body_type must be one of ['smpl', 'smplh', 'smplx']"
            raise ValueError(msg)
        num_of_joints = {"smpl": 23, "smplh": 51, "smplx": 54}[body_type]
        self.body_type = SMPLVersion.from_string(body_type)
        if not self.no_global_orientation:
            num_of_joints += 1
        self.pose_regressor = torch.nn.Linear(input_size, num_of_joints * 3)

    if TYPE_CHECKING:

        def __call__(self, hidden_states: torch.Tensor) -> SMPLData:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> SMPLData:
        """Forward HPS hidden states."""
        # Shape
        shape_parameters = self.shape_regressor(hidden_states)

        # Pose
        full_pose = self.pose_regressor(hidden_states)
        if self.no_global_orientation:
            full_pose = pad(full_pose, (3, 0), "constant", 0)

        # Body Translation
        global_translation = None
        if not self.no_global_translation:
            global_translation = self.global_translation_regressor(hidden_states)

        # Return SMPLData
        return SMPLData.from_full_pose(
            full_pose=full_pose,
            shape_parameters=shape_parameters,
            body_translation=global_translation,
            smpl_version=self.body_type,
        )

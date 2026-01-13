"""Module for HPS prediction."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import torch
from smplcodec import SMPLVersion
from torch.nn.functional import pad

from tinyhumans.datatypes import SMPLData


class ZeroLayer(torch.nn.Module):
    """A layer that returns zeros of a specified shape."""

    def __init__(self, output_shape: int | Iterable[int]) -> None:
        super().__init__()
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        self.output_shape = output_shape

    if TYPE_CHECKING:

        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward zero layer."""
        return torch.zeros((*hidden_states.shape[:-1], *self.output_shape), device=hidden_states.device)


class HandsLayer(torch.nn.Module):
    """Hand prediction layer."""

    def __init__(self, input_size: int, num_hand_joints: int, no_hand_joints: bool = False) -> None:
        super().__init__()
        if num_hand_joints not in [1, 15]:
            msg = "num_hand_joints must be one of [1, 15]"
            raise ValueError(msg)
        self.left_hand_pose_regressor = torch.nn.Linear(input_size, 1 * 3)  # first hand joint
        self.left_hand_extra_pose_regressor = (
            None
            if num_hand_joints == 1
            else ZeroLayer(output_shape=(num_hand_joints - 1) * 3)
            if no_hand_joints
            else torch.nn.Linear(input_size, (num_hand_joints - 1) * 3)
        )
        self.right_hand_pose_regressor = torch.nn.Linear(input_size, 1 * 3)
        self.right_hand_extra_pose_regressor = (
            None
            if num_hand_joints == 1
            else ZeroLayer(output_shape=(num_hand_joints - 1) * 3)
            if no_hand_joints
            else torch.nn.Linear(input_size, (num_hand_joints - 1) * 3)
        )

    if TYPE_CHECKING:

        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward hand layer."""
        hand_pose = [self.left_hand_pose_regressor(hidden_states)]
        if self.left_hand_extra_pose_regressor is not None:
            hand_pose.append(self.left_hand_extra_pose_regressor(hidden_states))
        hand_pose.append(self.right_hand_pose_regressor(hidden_states))
        if self.right_hand_extra_pose_regressor is not None:
            hand_pose.append(self.right_hand_extra_pose_regressor(hidden_states))

        return torch.cat(hand_pose, dim=-1)


class HPSLayer(torch.nn.Module):
    """HPS prediction layer."""

    def __init__(
        self,
        input_size: int,
        num_shape_parameters: int = 10,
        no_global_orientation: bool = False,
        no_global_translation: bool = False,
        no_hand_joints: bool = True,
        no_head_joints: bool = True,
        body_type: Literal["smpl", "smplh", "smplx"] = "smpl",
    ) -> None:
        super().__init__()
        self.no_global_orientation = no_global_orientation
        self.no_global_translation = no_global_translation
        self.no_hand_joints = no_hand_joints
        self.no_head_joints = no_head_joints
        if not self.no_global_translation:
            self.global_translation_regressor = torch.nn.Linear(input_size, 3)
        self.shape_regressor = torch.nn.Linear(input_size, num_shape_parameters)
        if body_type not in ["smpl", "smplh", "smplx"]:
            msg = "body_type must be one of ['smpl', 'smplh', 'smplx']"
            raise ValueError(msg)
        self.body_type = SMPLVersion.from_string(body_type)
        # Body pose regressor
        num_body_joints = 21
        if not self.no_global_orientation:
            num_body_joints += 1
        self.body_pose_regressor = torch.nn.Linear(input_size, num_body_joints * 3)
        # Head pose regressor
        self.head_pose_regressor = (
            torch.nn.Linear(input_size, 3 * 3)
            if body_type == "smplx" and not no_head_joints
            else ZeroLayer(output_shape=3 * 3)
            if body_type == "smplx"
            else None
        )
        # Hand pose regressor
        self.hand_pose_regressor = HandsLayer(
            input_size=input_size, num_hand_joints=1 if body_type == "smpl" else 15, no_hand_joints=self.no_hand_joints
        )

    if TYPE_CHECKING:

        def __call__(self, hidden_states: torch.Tensor) -> SMPLData:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> SMPLData:
        """Forward HPS hidden states."""
        # Shape
        shape_parameters = self.shape_regressor(hidden_states)

        # Pose
        body_pose = self.body_pose_regressor(hidden_states)
        if self.no_global_orientation:
            body_pose = pad(body_pose, (3, 0), "constant", 0)
        full_pose = [body_pose]
        if self.head_pose_regressor is not None:
            full_pose.append(self.head_pose_regressor(hidden_states))
        full_pose.append(self.hand_pose_regressor(hidden_states))
        full_pose = torch.cat(full_pose, dim=-1)

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

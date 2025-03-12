"""Poses types for TinyHumans.

This module defines custom types and classes for working with human body models in TinyHumans. It includes classes for
pose parameters (Poses, SMPLPoses, SMPLHPoses, SMPLXPoses, FLAMEPoses, MANOPoses).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch

from tinyhumans.datatypes import LimitedAttrTensorDictWithDefaults

if TYPE_CHECKING:
    from torch import Tensor

    NestedKey = str | tuple["NestedKeyType", ...]  # type: ignore  # noqa: F821, PGH003


MANO_POSE_SIZE = 15
SMPL_POSE_SIZE = 21
FLAME_POSE_SIZE = 4


class Poses(LimitedAttrTensorDictWithDefaults):
    """Base class for pose parameters.

    This class extends LimitedAttrTensorDictWithDefaults and serves as a base class for different pose representations
    (e.g., SMPLPoses, SMPLHPoses). It defines valid model types and provides methods for checking model types and
    converting pose parameters to tensors.

    Attributes:
        valid_model_types (tuple[str, ...]): Tuple of valid model types (e.g., "smpl", "smplh").

    Raises:
        NotImplementedError: If model_type property is accessed directly from Poses class.
        ValueError: If an invalid model type is provided.

    """

    valid_model_types: tuple[str, ...] = ("smpl", "smplh", "smplx", "flame", "mano")

    @staticmethod
    def check_model_type(model_type: str | None) -> None:
        """Check if model_type is valid.

        Args:
            model_type (str | None): Model type to check.

        Raises:
            ValueError: if model_type is not valid.

        """
        if model_type is None:
            return

        if model_type.lower() not in Poses.valid_model_types:
            msg = f"{model_type} is not a valid model type."
            msg += f"Valid types are: {', '.join([repr(m) for m in Poses.valid_model_types])}."
            raise ValueError(msg)

    @property
    def model_type(self) -> str:
        """str: Model type inferred from class name."""
        if self.__class__.__name__ == "Poses" or not self.__class__.__name__.endswith("Poses"):
            raise NotImplementedError

        return self.__class__.__name__[:-5].lower()

    def to_tensor(self) -> Tensor:
        """Convert pose parameters to a tensor.

        Returns:
            Tensor: Concatenated tensor of pose parameters.

        """
        return torch.cat([self[attr_name] for attr_name in self.default_attr_sizes], dim=-1)

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.

        """
        raise NotImplementedError


class SMPLPoses(Poses):
    """Poses parameters for SMPL model.

    This class extends Poses and defines the pose parameters specific to the SMPL body model, including body pose and
    hand pose.
    """

    default_attr_sizes: ClassVar[dict[tuple[int, ...] | int]] = {"body": 63, "l_hand": 3, "r_hand": 3}

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load SMPL pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            ValueError: If model_type is not compatible or loading to SMPL pose is not possible.

        """
        if model_type is not None:
            model_type = model_type.lower()
            if model_type == "smplh":
                self.body, self.l_hand, self.r_hand = pose_tensor.split(
                    [SMPL_POSE_SIZE * 3, MANO_POSE_SIZE * 3, MANO_POSE_SIZE * 3], dim=-1
                )
                # TODO: which smplh index corresponds to the smpl hand index?
                self.l_hand = self.l_hand[:, :3]
                self.r_hand = self.r_hand[:, :3]
                return
            if model_type != "smpl":
                msg = f"{model_type.capitalize} pose loading to SMPL poses is not possible or does not make sense."
                msg += " Supported model types for loading are: smpl and smplh."
                raise ValueError(msg)

        self.body, self.l_hand, self.r_hand = pose_tensor.split(list(self.default_attr_sizes.values()), dim=-1)


class MANOPoses(Poses):
    """Poses parameters for MANO model.

    This class extends Poses and defines the pose parameters specific to the MANO hand model, including hand pose.
    """

    default_attr_sizes: ClassVar[dict[tuple[int, ...] | int]] = {"hand": 45}

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load MANO pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            ValueError: If model_type is not compatible or loading to MANO pose is not possible.

        """
        if model_type is not None:
            model_type = model_type.lower()
            if model_type == "smplh-l":
                self.hand = pose_tensor[:, SMPL_POSE_SIZE * 3 : (SMPL_POSE_SIZE + MANO_POSE_SIZE) * 3]
                return
            if model_type == "smplh-r":
                self.hand = pose_tensor[
                    :, (SMPL_POSE_SIZE + MANO_POSE_SIZE) * 3 : (SMPL_POSE_SIZE + 2 * MANO_POSE_SIZE) * 3
                ]
                return
            if model_type != "mano":
                msg = f"{model_type.capitalize} pose loading to MANO poses is not possible."
                msg += " Supported model types for loading are: mano and smplh-l and smplh-r."
                raise ValueError(msg)

        self.hand = pose_tensor


class SMPLHPoses(SMPLPoses):
    """Poses parameters for SMPL+H model.

    This class extends SMPLPoses and defines the pose parameters specific to the SMPL+H body model, including body pose
    and hand pose with more DoF for hands.
    """

    default_attr_sizes: ClassVar[dict[tuple[int, ...] | int]] = {"body": 63, "l_hand": 45, "r_hand": 45}

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load SMPL+H pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            ValueError: If model_type is not compatible or loading to SMPL+H pose is not possible.

        """
        if model_type is not None:
            model_type = model_type.lower()
            if model_type == "smpl":
                # TODO: which smplh index corresponds to the smpl hand index?
                self.body, l_hand, r_hand = pose_tensor.split([SMPL_POSE_SIZE * 3, 3, 3], dim=-1)
                self.l_hand = torch.zeros(
                    [len(pose_tensor), MANO_POSE_SIZE * 3], dtype=pose_tensor.dtype, device=pose_tensor.device
                )
                self.r_hand = torch.zeros(
                    [len(pose_tensor), MANO_POSE_SIZE * 3], dtype=pose_tensor.dtype, device=pose_tensor.device
                )
                self.l_hand[:, :3] = l_hand
                self.r_hand[:, :3] = r_hand
                return
            if model_type == "mano-l":
                self.l_hand = pose_tensor
                return
            if model_type == "mano-r":
                self.r_hand = pose_tensor
                return
            if model_type != "smplh":
                msg = f"{model_type.capitalize} pose loading to SMPL-H poses is not possible."
                msg += " Supported model types for loading are: smpl, smplh, mano-l and mano-r."
                raise ValueError(msg)

        self.body, self.l_hand, self.r_hand = pose_tensor.split(list(self.default_attr_sizes.values()), dim=-1)


class FLAMEPoses(Poses):
    """Poses parameters for FLAME model.

    This class extends Poses and defines the pose parameters specific to the FLAME face model, including body pose,
    jaw pose, and eye pose.
    """

    default_attr_sizes: ClassVar[dict[tuple[int, ...] | int]] = {"body": 3, "jaw": 3, "eyes": 6}

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load FLAME pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            ValueError: If model_type is not compatible or loading to FLAME pose is not possible.

        """
        if model_type is not None:
            model_type = model_type.lower()
            if model_type == "smplx":
                body, self.jaw, self.eyes, _ = pose_tensor.split(
                    [SMPL_POSE_SIZE * 3, 3, 6, 2 * MANO_POSE_SIZE * 3], dim=-1
                )
                # flame's body index is the neck which is smplx's 11 index (counting from 0 and ignoring the pelvis)
                self.body = body[:, 11 * 3 : 12 * 3]
                return
            if model_type != "flame":
                msg = f"{model_type.capitalize} pose loading to SMPL-X poses is not possible."
                raise ValueError(msg)

        self.body, self.jaw, self.eyes = pose_tensor.split(list(self.default_attr_sizes.values()), dim=-1)


class SMPLXPoses(FLAMEPoses):
    """Poses parameters for SMPL-X model.

    This class extends FLAMEPoses and defines the pose parameters specific to the SMPL-X body and face model, including
    body pose, jaw pose, eye pose, and hand pose.
    """

    default_attr_sizes: ClassVar[dict[tuple[int, ...] | int]] = (
        FLAMEPoses.default_attr_sizes | SMPLHPoses.default_attr_sizes
    )

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load SMPL-X pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            ValueError: If model_type is not compatible or loading to SMPL-X pose is not possible.

        """
        if model_type is not None:
            model_type = model_type.lower()
            if model_type == "flame":
                body, self.jaw, self.eyes = pose_tensor.split([3, 3, 6], dim=-1)
                self.body = torch.zeros(
                    [len(pose_tensor), SMPL_POSE_SIZE * 3], dtype=pose_tensor.dtype, device=pose_tensor.device
                )
                # flame's body index is the neck which is smplx's 11 index (counting from 0 and ignoring the pelvis)
                self.body[:, 11 * 3 : 12 * 3] = body
                return
            if model_type != "smplx":
                msg = f"{model_type.capitalize} pose loading to SMPL-X poses is not possible."
                raise ValueError(msg)

        self.body, self.jaw, self.eyes, self.l_hand, self.r_hand = pose_tensor.split(
            list(self.default_attr_sizes.values()), dim=-1
        )

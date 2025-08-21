"""Types for TinyHumans.

This module defines custom types and classes for working with human body models in TinyHumans. It includes base utility
classes like AutoTensorDict and LimitedAttrTensorDictWithDefaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tensordict.tensorclass import tensorclass

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class CameraIntrinsics:
    """A dataclass to hold camera intrinsic parameters."""

    yfov: float
    aspect_ratio: float
    znear: float | None = None

    def get_focal_length(self, image_width: int) -> float:
        """Get the focal length of the camera from the yfov given the image width."""
        # TODO: width or height?
        return image_width / (2 * np.tan(self.yfov / 2))


@tensorclass
class CameraData:
    """A dataclass to hold camera animation data."""

    times: Tensor
    R_cw: Tensor  # (B, T, 3, 3) Camera-from-world rotation matrices
    T_cw: Tensor  # (B, T, 3) Camera-from-world translation vectors

    # abstract the smpl_data class and validate the inputs here, unsqueeze, cast etc

    def __post_init__(self) -> None:
        """Check that the inputs are valid and set batch_size and device possible."""
        self.auto_batch_size_(2)
        if self.device is None:
            self.auto_device_()

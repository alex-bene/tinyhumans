"""Types for TinyHumans.

This module defines custom types and classes for working with human body models in TinyHumans. It includes base utility
classes like AutoTensorDict and LimitedAttrTensorDictWithDefaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tinyhumans.parsers import McsParser
from tinyhumans.tools import get_logger

from .camera import CameraData, CameraIntrinsics
from .smpl_data import SMPLData

logger = get_logger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


@dataclass
class Scene4D:
    """A dataclass to hold all data for a loaded .mcs scene."""

    num_frames: int
    smpl_data: SMPLData | None = None
    camera_data: CameraData | None = None
    camera_intrinsics: CameraIntrinsics | None = None
    video_frames: list[np.ndarray] | None = None

    @classmethod
    def from_msc_file(cls, filename: str | Path) -> Scene4D:
        """Create a Scene4D from a .msc file."""
        parsed_dict = McsParser(filename).parse()
        return cls(
            num_frames=parsed_dict["num_frames"],
            smpl_data=SMPLData(**parsed_dict["smpl_data"]),
            camera_data=CameraData(**parsed_dict["camera_data"]),
            camera_intrinsics=CameraIntrinsics(**parsed_dict["camera_intrinsics"]),
            video_frames=parsed_dict["video_frames"],
        )

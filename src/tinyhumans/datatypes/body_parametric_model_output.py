"""Body parametric model output types for TinyHumans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import trimesh
from tensordict import tensorclass

from tinyhumans.mesh import Meshes

if TYPE_CHECKING:
    from torch import Tensor


@tensorclass
class BodyParametricModelOutput:
    """Output of a body parametric model.

    This class stores the output of a body parametric model, including vertices and joints.
    It supports extracting Pytorch3D Meshes and Trimesh objects representing ecah body across samples and timesteps.

    Attributes:
        verts: Tensor of vertices of shape (batch_count, frame_count, human_count, num_verts, 3)
        joints: Tensor of joints of shape (batch_count, frame_count, human_count, num_joints, 3)

    """

    verts: Tensor
    joints: Tensor | None = None

    def __post_init__(self) -> None:
        """Check that the inputs are valid and set batch_size and device possible."""
        if self.joints is not None and self.joints.ndim != self.verts.ndim:
            msg = "joints and verts must have the same number of dimensions"
            raise ValueError(msg)
        self.auto_batch_size_(3)
        if self.device is None:
            self.auto_device_()

    def get_meshes(
        self, faces: Tensor
    ) -> list[Meshes]:  # TODO: fix this, it was written for 5 dims only but want to support 3 and 4
        """Get a list of Pytorch3D Meshes, one for each sample in the batch that includes many timesteps."""
        if faces.ndim != 2:
            msg = "faces must be 2-dimensional (num_faces, 3)"
            raise ValueError(msg)
        return [
            [Meshes(verts=verts_ij, faces=faces.expand((verts_ij.shape[0], -1, -1))) for verts_ij in verts_i]
            for verts_i in self.verts
        ]

    def get_trimeshes(self, faces: Tensor) -> list[list[trimesh.Trimesh]]:
        """Get a list of lists of Trimesh objects for each timestep and sample in the inputs.."""
        if faces.ndim != 2:
            msg = "faces must be 2-dimensional (num_faces, 3)"
            raise ValueError(msg)
        return [
            [
                [
                    trimesh.Trimesh(vertices=verts_ijk.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
                    for verts_ijk in verts_ij
                ]
                for verts_ij in verts_i
            ]
            for verts_i in self.verts
        ]

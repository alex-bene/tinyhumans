from __future__ import annotations

from typing import TYPE_CHECKING

import trimesh
from tensordict import tensorclass

from tinyhumans.mesh import Meshes

if TYPE_CHECKING:
    from torch import Tensor

    from tinyhumans.datatypes import FLAMEPoses, MANOPoses, Poses, ShapeComponents, SMPLHPoses, SMPLPoses, SMPLXPoses


@tensorclass
class ParametricModelOutput:
    verts: Tensor
    joints: Tensor | None = None
    poses: Poses | SMPLPoses | SMPLHPoses | SMPLXPoses | MANOPoses | FLAMEPoses | None = None
    shape_components: ShapeComponents | None = None
    root_orientations: Tensor | None = None
    root_positions: Tensor | None = None
    vertices_template: Tensor | None = None

    def __post_init__(self) -> None:
        self.auto_batch_size_(1)
        if self.device is None:
            self.auto_device_()

    def get_meshes(self, faces: Tensor) -> Meshes:
        batched_faces = faces.expand((self.verts.shape[0], -1, -1)) if faces.ndim == 2 else faces
        return Meshes(verts=self.verts, faces=batched_faces)

    def get_trimeshes(self, faces: Tensor) -> list[trimesh.Trimesh]:
        batched_faces = faces.expand((self.verts.shape[0], -1, -1)) if faces.ndim == 2 else faces
        return [
            trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=batched_faces[i].detach().cpu().numpy())
            for i, verts in enumerate(self.verts)
        ]

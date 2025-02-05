"""Meshes class for TinyHumans.

This module defines the Meshes and BodyMeshes classes, which extend PyTorch3D's Meshes class to provide additional
functionality for working with 3D meshes, including conversion to Trimesh objects and handling of body-specific
parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pytorch3d.renderer import TexturesAtlas, TexturesVertex
from pytorch3d.structures import Meshes as P3D_Meshes
from trimesh import Trimesh

from tinyhumans.tools import get_logger

if TYPE_CHECKING:
    from pytorch3d.renderer import Textures

    from tinyhumans.types import Pose, ShapeComponents

    MeshesArguents = list[torch.Tensor] | torch.Tensor

# Initialize a logger
logger = get_logger(__name__)


class Meshes(P3D_Meshes):
    """Extend PyTorch3D Meshes class to support additional functionality."""

    def to_trimesh(self) -> list[Trimesh]:
        """Convert PyTorch3D Meshes objects to Trimesh objects.

        This function handles different texture types (vertex colors, atlas colors) and correctly extracts vertex and
        face information for creating Trimesh objects.

        Args:
            meshes(Meshes | list[Meshes]): A Meshes object or a list of Meshes objects.

        Returns:
            list[Trimesh]: A list of Trimesh objects corresponding to the input Meshes.

        Raises:
            NotImplementedError: If the texture type is TexturesUV.

        """
        trimeshes = []

        for idx, (verts, faces) in enumerate(zip(self.verts_list(), self.faces_list())):
            textures = None if self.textures is None else self.textures[idx]
            verts_colors = face_colors = None

            if textures is not None:
                if isinstance(textures, TexturesVertex):
                    verts_colors = textures.verts_features_list()[0].detach().cpu().numpy()
                elif isinstance(textures, TexturesAtlas):
                    atlas = textures.atlas_list()[0]
                    if atlas.shape[-3:] == (1, 1, 3):
                        face_colors = atlas.view(-1, 3).detach().cpu().numpy()
                else:
                    logger.warning("TexturesUV is not yet supported. Texture sampling will be used instead.")

                if verts_colors is None and face_colors is None:
                    # source: https://github.com/facebookresearch/pytorch3d/issues/854#issuecomment-925737629
                    verts_colors_packed = torch.zeros_like(verts)
                    verts_colors_packed[faces] = textures.faces_verts_textures_packed()
                    verts_colors = verts_colors_packed.detach().cpu().numpy()

            trimeshes.append(
                Trimesh(
                    vertices=verts.detach().cpu().numpy(),
                    faces=faces.detach().cpu().numpy(),
                    vertex_colors=verts_colors,
                    face_colors=face_colors,
                    process=False,
                )
            )

        return trimeshes


class BodyMeshes(Meshes):
    """Extend the internal Meshes class to support additional functionality for body meshes."""

    def __init__(
        self,
        verts: MeshesArguents,
        faces: MeshesArguents,
        textures: Textures | None = None,
        *,
        verts_normals: MeshesArguents | None = None,
        joints: MeshesArguents | None = None,
        poses: Pose | None = None,
        shape_components: ShapeComponents | None = None,
        root_positions: MeshesArguents | None = None,
        root_orientation: MeshesArguents | None = None,
        vertices_template: MeshesArguents | None = None,
    ) -> None:
        """Initialize BodyMeshes.

        Args:
            verts (MeshesArguents): Mesh vertices.
            faces (MeshesArguents): Mesh faces.
            textures (Textures | None, optional): Mesh textures. Defaults to None.
            verts_normals (MeshesArguents | None, optional): Mesh vertex normals. Defaults to None.
            joints (MeshesArguents | None, optional): Joint locations. Defaults to None.
            poses (Pose | None, optional): Pose parameters. Defaults to None.
            shape_components (ShapeComponents | None, optional): Shape parameters. Defaults to None.
            root_positions (MeshesArguents | None, optional): Root positions. Defaults to None.
            root_orientation (MeshesArguents | None, optional): Root orientation. Defaults to None.
            vertices_template (MeshesArguents | None, optional): Template vertices. Defaults to None.

        """
        super().__init__(verts=verts, faces=faces, textures=textures, verts_normals=verts_normals)
        self.joints = joints
        self.poses = poses
        self.shape_components = shape_components
        self.root_positions = root_positions
        self.root_orientation = root_orientation
        self.vertices_template = vertices_template

import logging
import os
import platform
from typing import Iterable

import numpy as np
import pyrender
from PIL.Image import Image
from pyrender import Viewer
from pyrender.constants import RenderFlags
from pytorch3d.renderer import TexturesAtlas, TexturesVertex
from pytorch3d.structures.meshes import join_meshes_as_batch
from rich.logging import RichHandler
from trimesh import Trimesh

from src.tinyhumans.mesh import Meshes
from src.tinyhumans.tools import get_jet_colormap, img_from_array

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Initialize a logger
logger = logging.getLogger(__name__)

# Fix for np.infty used by pyrender
np.infty = np.inf


def set_pyopengl_platform():
    """
    Sets the appropriate value for the PYOPENGL_PLATFORM environment variable based on the operating system.
    """
    os_name = platform.system()

    if os_name == "Linux":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    elif os_name == "Darwin":
        if platform.processor() != "arm":
            os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif os_name == "Windows":
        os.environ["PYOPENGL_PLATFORM"] = "egl"


set_pyopengl_platform()


class Renderer:
    def __init__(
        self,
        image_size: tuple[int, int] = (800, 800),
        bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ambient_light: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_raymond_lighting: bool = True,
        use_direct_lighting: bool = True,
        camera_params: dict = {},
    ) -> None:
        self.image_size = image_size
        self.set_scene(bg_color, ambient_light)
        self.set_camera(**camera_params)

        if use_raymond_lighting:
            self.use_raymond_lighting()
        if use_direct_lighting:
            self.use_direct_lighting()

    def set_scene(self, *args, **kwargs) -> None:
        return None

    def set_camera(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def use_raymond_lighting(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def use_direct_lighting(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __call__(self, meshes: Iterable, render_params: dict) -> tuple[Image, np.ndarray]:
        raise NotImplementedError


def pytorch3d_to_trimesh(meshes: Meshes | list[Meshes]) -> list[Trimesh]:
    """
    Converts PyTorch3D Meshes objects to Trimesh objects.

    This function handles different texture types (vertex colors, atlas colors)
    and correctly extracts vertex and face information for creating Trimesh objects.

    Args:
        meshes(Meshes | list[Meshes]): A Meshes object or a list of Meshes objects.

    Returns:
        list[Trimesh]: A list of Trimesh objects corresponding to the input Meshes.

    Raises:
        NotImplementedError: If the texture type is TexturesUV.
    """

    meshes = join_meshes_as_batch(meshes) if isinstance(meshes, list) else meshes
    trimeshes = []
    for mesh in meshes:
        verts_colors = face_colors = None
        if mesh.textures is not None:
            if isinstance(mesh.textures, TexturesVertex):
                verts_colors = mesh.textures.verts_features_list()[0].detach().cpu().numpy()
            elif isinstance(mesh.textures, TexturesAtlas):
                atlas = mesh.textures.atlas_list()[0]
                if atlas.shape[-3:] == (1, 1, 3):
                    face_colors = atlas.view(-1, 3).detach().cpu().numpy()
            else:
                logger.warning("TexturesUV is not yet supported. Texture sampling will be used instead.")

            if verts_colors is None and face_colors is None:
                # source: https://github.com/facebookresearch/pytorch3d/issues/854#issuecomment-925737629
                verts_colors_packed = mesh.verts_packed().clone().detach()
                verts_colors_packed[mesh.faces_packed()] = mesh.textures.faces_verts_textures_packed()
                verts_colors = verts_colors_packed.cpu().numpy()

        trimeshes.append(
            Trimesh(
                vertices=mesh.vertices.detach().cpu().numpy(),
                faces=mesh.faces.detach().cpu().numpy(),
                vertex_colors=verts_colors,
                face_colors=face_colors,
                process=False,
            )
        )

    return trimeshes


class PyRenderer(Renderer):
    def __init__(
        self,
        image_size: tuple[int, int] = (800, 800),
        bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ambient_light: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_raymond_lighting: bool = True,
        use_direct_lighting: bool = True,
        camera_params: dict | None = None,
    ):
        camera_params = {
            "translation": [0.0, 0.0, 3.0],
            "rotation": np.eye(3),
            "yfov": np.pi / 3.0,
        } | (camera_params or {})
        super().__init__(image_size, bg_color, ambient_light, use_raymond_lighting, use_direct_lighting, camera_params)
        self.renderer = pyrender.OffscreenRenderer(*self.image_size)

    def set_scene(
        self,
        bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ambient_light: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)

    def set_camera(
        self,
        translation: Iterable[float] = [0.0, 0.0, 3.0],
        rotation: np.ndarray = np.eye(3),
        yfov: float = np.pi / 3.0,
    ):
        if not hasattr(self, "scene"):
            raise AttributeError("Scene must be set before setting the camera")

        camera = pyrender.PerspectiveCamera(yfov=yfov)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation
        camera_pose[:3, 3] = translation
        self._camera_node = self.scene.add(camera, pose=camera_pose)

    def set_image_size(self, image_size: tuple[int, int]) -> None:
        self.image_size = image_size
        self.renderer.viewport_width, self.renderer.viewport_height = image_size

    def set_background_color(self, color: Iterable[float] = [1.0, 1.0, 1.0]) -> None:
        self.scene.bg_color = color

    def set_camera_translation(self, translation: Iterable[float] = [0.0, 0.0, 3.0]) -> None:
        if (
            isinstance(translation, Iterable)
            and not isinstance(translation, str)
            and not isinstance(translation, np.ndarray)
        ):
            translation = np.array(translation)
        self._camera_node.translation = translation

    def set_camera_rotation(self, rotation: np.ndarray = np.eye(3)) -> None:
        self._camera_node.rotation = rotation

    def set_camera_pose(self, camera_pose: np.ndarray) -> None:
        self.scene.set_pose(self._camera_node, pose=camera_pose)

    def use_raymond_lighting(self, intensity: float = 1.0) -> None:
        for n in Viewer._create_raymond_lights(Viewer):
            n.light.intensity = intensity
            if not self.scene.has_node(n):
                self.scene.add_node(n, parent_node=self._camera_node)

    def use_direct_lighting(self, intensity: float = 1.0) -> None:
        direct_light = Viewer._create_direct_light(Viewer)
        direct_light.light.intensity = intensity
        if not self.scene.has_node(direct_light):
            self.scene.add_node(direct_light, parent_node=self._camera_node)

    def __call__(
        self,
        meshes: Meshes | list[Meshes],
        render_params: dict[str, bool] | None = None,
    ) -> tuple[Image, np.ndarray]:
        render_params = {
            "render_face_normals": False,
            "render_in_RGBA": False,
            "render_segmentation": False,
            "render_shadows": True,
            "render_vertex_normals": False,
            "render_wireframe": False,
            "skip_cull_faces": False,
        } | (render_params or {})

        self.scene.mesh_nodes.clear()
        for mesh in meshes.to_trimesh():
            self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False if mesh.visual.kind == "face" else True))

        flags = RenderFlags.NONE
        if render_params["render_in_RGBA"]:
            flags |= RenderFlags.RGBA
        if render_params["render_wireframe"]:
            flags |= RenderFlags.ALL_WIREFRAME
        if render_params["render_shadows"]:
            flags |= RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT
        if render_params["render_vertex_normals"]:
            flags |= RenderFlags.VERTEX_NORMALS
        if render_params["render_face_normals"]:
            flags |= RenderFlags.FACE_NORMALS
        if render_params["skip_cull_faces"]:
            flags |= RenderFlags.SKIP_CULL_FACES
        seg_node_map = None
        if render_params["render_segmentation"]:
            flags |= RenderFlags.SEG
            seg_node_map = {
                node: color for node, color in zip(self.scene.mesh_nodes, get_jet_colormap(len(meshes), dtype=np.uint8))
            }

        color, depth = self.renderer.render(self.scene, flags=flags, seg_node_map=seg_node_map)

        return img_from_array(color), depth

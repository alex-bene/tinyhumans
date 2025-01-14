"""PyRenderer for TinyHumans."""

from __future__ import annotations

import os
import platform
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pyrender
from pyrender import Viewer
from pyrender.constants import RenderFlags

from src.tinyhumans.tools import get_jet_colormap, get_logger, img_from_array

if TYPE_CHECKING:
    from PIL.Image import Image

    from src.tinyhumans.mesh import Meshes


# Initialize a logger
logger = get_logger(__name__)

# Fix for np.infty used by pyrender
np.infty = np.inf  # noqa: NPY201


def set_pyopengl_platform() -> None:
    """Set the appropriate value for the PYOPENGL_PLATFORM environment variable based on the operating system."""
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
        camera_params: dict = None,
    ) -> None:
        if camera_params is None:
            camera_params = {}
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


class PyRenderer(Renderer):
    def __init__(
        self,
        image_size: tuple[int, int] = (800, 800),
        bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ambient_light: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_raymond_lighting: bool = True,
        use_direct_lighting: bool = True,
        camera_params: dict | None = None,
    ) -> None:
        camera_params = {"translation": [0.0, 0.0, 3.0], "rotation": np.eye(3), "yfov": np.pi / 3.0} | (
            camera_params or {}
        )
        super().__init__(image_size, bg_color, ambient_light, use_raymond_lighting, use_direct_lighting, camera_params)
        self.renderer = pyrender.OffscreenRenderer(*self.image_size)

    def set_scene(
        self,
        bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ambient_light: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)

    def set_camera(
        self,
        translation: Iterable[float] = [0.0, 0.0, 3.0],
        rotation: np.ndarray = np.eye(3),
        yfov: float = np.pi / 3.0,
    ) -> None:
        if not hasattr(self, "scene"):
            msg = "Scene must be set before setting the camera"
            raise AttributeError(msg)

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
        for n in Viewer._create_raymond_lights(Viewer):  # noqa: SLF001
            n.light.intensity = intensity
            if not self.scene.has_node(n):
                self.scene.add_node(n, parent_node=self._camera_node)

    def use_direct_lighting(self, intensity: float = 1.0) -> None:
        direct_light = Viewer._create_direct_light(Viewer)  # noqa: SLF001
        direct_light.light.intensity = intensity
        if not self.scene.has_node(direct_light):
            self.scene.add_node(direct_light, parent_node=self._camera_node)

    def __call__(
        self, meshes: Meshes | list[Meshes], render_params: dict[str, bool] | None = None
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
            self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=mesh.visual.kind != "face"))

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
            seg_node_map = dict(zip(self.scene.mesh_nodes, get_jet_colormap(len(meshes), dtype=np.uint8)))

        color, depth = self.renderer.render(self.scene, flags=flags, seg_node_map=seg_node_map)

        return img_from_array(color), depth

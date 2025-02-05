"""PyRenderer for TinyHumans.

This module defines the Renderer and PyRenderer classes for rendering 3D meshes using pyrender. It includes
functionality for setting up scenes, cameras, lighting, and rendering meshes with various options.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pyrender
from pyrender import Viewer
from pyrender.constants import RenderFlags

from tinyhumans.tools import get_jet_colormap, get_logger, img_from_array

if TYPE_CHECKING:
    from PIL.Image import Image

    from tinyhumans.mesh import Meshes


# Initialize a logger
logger = get_logger(__name__)

# Fix for np.infty used by pyrender
np.infty = np.inf  # noqa: NPY201


def set_pyopengl_platform() -> None:
    """Set the appropriate value for the PYOPENGL_PLATFORM environment variable based on the operating system.

    This function sets the PYOPENGL_PLATFORM environment variable to "osmesa" for Linux, "egl" for macOS (if not arm64)
    and Windows, which is required for pyrender to work correctly.
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
    """Abstract base class for renderers."""

    def __init__(
        self,
        image_size: tuple[int, int] = (800, 800),
        bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ambient_light: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_raymond_lighting: bool = True,
        use_direct_lighting: bool = True,
        camera_params: dict | None = None,
    ) -> None:
        """Initialize the renderer.

        Args:
            image_size (tuple[int, int], optional): The size of the rendered image (width, height).
                Defaults to (800, 800).
            bg_color (tuple[float, float, float], optional): The background color of the scene (RGB).
                Defaults to (1.0, 1.0, 1.0).
            ambient_light (tuple[float, float, float], optional): The ambient light color (RGB).
                Defaults to (1.0, 1.0, 1.0).
            use_raymond_lighting (bool, optional): Whether to use raymond lighting. Defaults to True.
            use_direct_lighting (bool, optional): Whether to use direct lighting. Defaults to True.
            camera_params (dict, optional): Additional camera parameters. Defaults to None.

        """
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
        """Set up the scene for rendering. This method should be implemented by subclasses."""
        raise NotImplementedError

    def set_camera(self, *args, **kwargs) -> None:
        """Set up the camera for rendering. This method should be implemented by subclasses."""
        raise NotImplementedError

    def use_raymond_lighting(self, *args, **kwargs) -> None:
        """Enable raymond lighting. This method should be implemented by subclasses."""
        raise NotImplementedError

    def use_direct_lighting(self, *args, **kwargs) -> None:
        """Enable direct lighting. This method should be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, meshes: Iterable, render_params: dict) -> tuple[Image, np.ndarray]:
        """Render the meshes. This method should be implemented by subclasses."""
        raise NotImplementedError


class PyRenderer(Renderer):
    """PyRenderer class for rendering 3D meshes using pyrender.

    This class extends the Renderer class and provides functionality for setting up scenes, cameras, lighting, and
    rendering meshes with various options using pyrender.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (800, 800),
        bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ambient_light: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_raymond_lighting: bool = True,
        use_direct_lighting: bool = False,
        camera_params: dict | None = None,
    ) -> None:
        """Initialize the PyRenderer.

        Args:
            image_size (tuple[int, int], optional): The size of the rendered image (width, height).
                Defaults to (800, 800).
            bg_color (tuple[float, float, float], optional): The background color of the scene (RGB).
                Defaults to (1.0, 1.0, 1.0).
            ambient_light (tuple[float, float, float], optional): The ambient light color (RGB).
                Defaults to (1.0, 1.0, 1.0).
            use_raymond_lighting (bool, optional): Whether to use raymond lighting. Defaults to True.
            use_direct_lighting (bool, optional): Whether to use direct lighting. Defaults to False.
            camera_params (dict | None, optional): Additional camera parameters. Defaults to None.

        """
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
        """Set up the scene for rendering.

        Args:
            bg_color (tuple[float, float, float], optional): The background color of the scene (RGB).
                Defaults to (1.0, 1.0, 1.0).
            ambient_light (tuple[float, float, float], optional): The ambient light color (RGB).
                Defaults to (1.0, 1.0, 1.0).

        """
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)

    def set_camera(
        self,
        translation: Iterable[float] = [0.0, 0.0, 3.0],
        rotation: np.ndarray | None = None,
        yfov: float = np.pi / 3.0,
    ) -> None:
        """Set up the camera for rendering.

        Args:
            translation (Iterable[float], optional): The camera translation. Defaults to [0.0, 0.0, 3.0].
            rotation (np.ndarray, optional): The camera rotation matrix. If None, defaults to np.eye(3).
                Defaults to None.
            yfov (float, optional): The camera field of view. Defaults to np.pi / 3.0.

        Raises:
            AttributeError: If the scene has not been set up before setting the camera.

        """
        if not hasattr(self, "scene"):
            msg = "Scene must be set before setting the camera"
            raise AttributeError(msg)

        camera = pyrender.PerspectiveCamera(yfov=yfov)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation if rotation is not None else np.eye(3)
        camera_pose[:3, 3] = translation
        self._camera_node = self.scene.add(camera, pose=camera_pose)

    def set_image_size(self, image_size: tuple[int, int]) -> None:
        """Set the image size for rendering.

        Args:
            image_size (tuple[int, int]): The size of the rendered image (width, height).

        """
        self.image_size = image_size
        self.renderer.viewport_width, self.renderer.viewport_height = image_size

    def set_background_color(self, color: Iterable[float] = [1.0, 1.0, 1.0]) -> None:
        """Set the background color of the scene.

        Args:
            color (Iterable[float], optional): The background color (RGB). Defaults to [1.0, 1.0, 1.0].

        """
        self.scene.bg_color = color

    def set_camera_translation(self, translation: Iterable[float] = [0.0, 0.0, 3.0]) -> None:
        """Set the camera translation.

        Args:
            translation (Iterable[float], optional): The camera translation. Defaults to [0.0, 0.0, 3.0].

        """
        if (
            isinstance(translation, Iterable)
            and not isinstance(translation, str)
            and not isinstance(translation, np.ndarray)
        ):
            translation = np.array(translation)
        self._camera_node.translation = translation

    def set_camera_rotation(self, rotation: np.ndarray | None = None) -> None:
        """Set the camera rotation.

        Args:
            rotation (np.ndarray, optional): The camera rotation matrix. If None, defaults to np.eye(3).
                Defaults to None.

        """
        self._camera_node.rotation = rotation if rotation is not None else np.eye(3)

    def set_camera_pose(self, camera_pose: np.ndarray) -> None:
        """Set the camera pose.

        Args:
            camera_pose (np.ndarray): The camera pose matrix.

        """
        self.scene.set_pose(self._camera_node, pose=camera_pose)

    def use_raymond_lighting(self, intensity: float = 1.0) -> None:
        """Enable raymond lighting.

        Args:
            intensity (float, optional): The intensity of the raymond lighting. Defaults to 1.0.

        """
        for n in Viewer._create_raymond_lights(Viewer):
            n.light.intensity = intensity
            if not self.scene.has_node(n):
                self.scene.add_node(n, parent_node=self._camera_node)

    def use_direct_lighting(self, intensity: float = 1.0) -> None:
        """Enable direct lighting.

        Args:
            intensity (float, optional): The intensity of the direct lighting. Defaults to 1.0.

        """
        direct_light = Viewer._create_direct_light(Viewer)
        direct_light.light.intensity = intensity
        if not self.scene.has_node(direct_light):
            self.scene.add_node(direct_light, parent_node=self._camera_node)

    def __call__(
        self, meshes: Meshes | list[Meshes], render_params: dict[str, bool] | None = None
    ) -> tuple[Image, np.ndarray]:
        """Render the meshes.

        Args:
            meshes (Meshes | list[Meshes]): The meshes to render.
            render_params (dict[str, bool] | None, optional): Rendering parameters. Defaults to None.

        Returns:
            tuple[Image, np.ndarray]: A tuple containing the rendered image and the depth map.

        """
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

"""PyRenderer for TinyHumans.

This module defines the Renderer and PyRenderer classes for rendering 3D meshes using pyrender. It includes
functionality for setting up scenes, cameras, lighting, and rendering meshes with various options.
"""

from __future__ import annotations

import os
import platform
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pyrender
from pyrender import Viewer
from pyrender.constants import RenderFlags
from trimesh import Trimesh, transformations

from tinyhumans.tools import get_logger, img_from_array
from tinyhumans.visualize import get_jet_colormap

if TYPE_CHECKING:
    from PIL.Image import Image


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


class PyRenderer:
    """PyRenderer class for rendering 3D meshes using pyrender.

    This class extends the Renderer class and provides functionality for setting up scenes, cameras, lighting, and
    rendering meshes with various options using pyrender.
    """

    render_flags_map: ClassVar[dict[str, RenderFlags]] = {
        "RGBA": RenderFlags.RGBA,
        "all_wireframe": RenderFlags.ALL_WIREFRAME,
        "shadows": RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT,
        "vertex_normals": RenderFlags.VERTEX_NORMALS,
        "face_normals": RenderFlags.FACE_NORMALS,
        "skip_cull_faces": RenderFlags.SKIP_CULL_FACES,
        "segmentation": RenderFlags.SEG,
        "flat": RenderFlags.FLAT,
    }

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 512),
        bg_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        ambient_light: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_raymond_lighting: bool = True,
        use_direct_lighting: bool = False,
        camera_params: dict | None = None,
    ) -> None:
        """Initialize the PyRenderer.

        Args:
            image_size (tuple[int, int], optional): The size of the rendered image (width, height).
                Defaults to (512, 512).
            bg_color (tuple[float, float, float, float], optional): The background color of the scene (RGBA).
                Defaults to (1.0, 1.0, 1.0, 1.0).
            ambient_light (tuple[float, float, float], optional): The ambient light color (RGB).
                Defaults to (1.0, 1.0, 1.0).
            use_raymond_lighting (bool, optional): Whether to use raymond lighting. Defaults to True.
            use_direct_lighting (bool, optional): Whether to use direct lighting. Defaults to False.
            camera_params (dict | None, optional): Additional camera parameters. Defaults to None.

        """
        self.renderer = pyrender.OffscreenRenderer(*image_size)

        self.image_size = image_size
        self.set_scene(bg_color, ambient_light)
        camera_params = {"translation": [0.0, 0.0, 1.0], "rotation": np.eye(3), "focal_length": 600} | (
            camera_params or {}
        )
        self.set_camera(**camera_params)

        if use_raymond_lighting:
            self.use_raymond_lighting()
        if use_direct_lighting:
            self.use_direct_lighting()

    def set_scene(
        self,
        bg_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        ambient_light: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Set up the scene for rendering.

        Args:
            bg_color (tuple[float, float, float, float], optional): The background color of the scene (RGB).
                Defaults to (1.0, 1.0, 1.0, 1.0).
            ambient_light (tuple[float, float, float], optional): The ambient light color (RGB).
                Defaults to (1.0, 1.0, 1.0).

        """
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)

    def set_camera(
        self,
        translation: tuple[float, float, float] | None = None,
        rotation: np.ndarray | None = None,
        focal_length: int = 600,
    ) -> None:
        """Set up the camera for rendering.

        Args:
            translation (tuple[float, float, float] | None, optional): The camera translation. If None, defaults to
                (0, 0, 1). Defaults to None.
            rotation (np.ndarray | None, optional): The camera rotation matrix. If None, defaults to np.eye(3).
                Defaults to None.
            focal_length (int, optional): The camera focal length in pixels. Defaults to 600.

        Raises:
            AttributeError: If the scene has not been set up before setting the camera.

        """
        if not hasattr(self, "scene"):
            msg = "Scene must be set before setting the camera"
            raise AttributeError(msg)

        camera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length, cx=self.camera_center[0], cy=self.camera_center[1]
        )
        camera_pose = np.eye(4)
        if rotation is not None:
            camera_pose[:3, :3] = rotation
        camera_pose[:3, 3] = translation if translation is not None else (0.0, 0.0, 1.0)
        self._camera_node = self.scene.add(camera, pose=camera_pose)

    @property
    def image_size(self) -> tuple[int, int]:
        """Get the image size for rendering.

        Returns:
            tuple[int, int]: The size of the rendered image (width, height).

        """
        return self.renderer.viewport_width, self.renderer.viewport_height

    @image_size.setter
    def image_size(self, image_size: tuple[int, int]) -> None:
        """Set the image size for rendering.

        Args:
            image_size (tuple[int, int]): The size of the rendered image (width, height).

        """
        if hasattr(self, "renderer"):
            self.renderer.viewport_width, self.renderer.viewport_height = image_size
        self.camera_center = [image_size[0] // 2, image_size[1] // 2]
        # if _camera_node is set
        if hasattr(self, "_camera_node"):
            self._camera_node.camera.cx = self.camera_center[0]
            self._camera_node.camera.cy = self.camera_center[1]

    @property
    def background_color(self) -> tuple[float, float, float, float]:
        """Get the background color of the scene.

        Returns:
            tuple[float, float, float, float]: The background color (RGBA).

        """
        return self.scene.bg_color

    @background_color.setter
    def background_color(self, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> None:
        """Set the background color of the scene.

        Args:
            color (tuple[float, float, float, float], optional): The background color (RGBA).
                Defaults to (1.0, 1.0, 1.0, 1.0).

        """
        self.scene.bg_color = color

    @property
    def camera_translation(self) -> tuple[float, float, float]:
        """Get the camera translation.

        Returns:
            tuple[float, float, float]: The camera translation.

        """
        return self._camera_node.translation

    @camera_translation.setter
    def camera_translation(self, translation: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> None:
        """Set the camera translation.

        Args:
            translation (tuple[float, float, float], optional): The camera translation. Defaults to (0.0, 0.0, 1.0).

        """
        self._camera_node.translation = translation

    @property
    def camera_rotation(self) -> np.ndarray:
        """Get the camera rotation.

        Returns:
            np.ndarray: The camera rotation matrix.

        """
        return self._camera_node.rotation

    @camera_rotation.setter
    def camera_rotation(self, rotation: np.ndarray | None = None) -> None:
        """Set the camera rotation.

        Args:
            rotation (np.ndarray, optional): The camera rotation matrix. If None, defaults to np.eye(3).
                Defaults to None.

        """
        self._camera_node.rotation = rotation if rotation is not None else np.eye(3)

    @property
    def camera_pose(self) -> np.ndarray:
        """Get the camera pose.

        Returns:
            np.ndarray: The camera pose matrix.

        """
        return self.scene.get_pose(self._camera_node)

    @camera_pose.setter
    def camera_pose(self, pose: np.ndarray) -> None:
        """Set the camera pose.

        Args:
            pose (np.ndarray): The camera pose matrix.

        """
        self.scene.set_pose(self._camera_node, pose=pose)

    def use_raymond_lighting(self, intensity: float = 1.0) -> None:
        """Enable raymond lighting.

        Args:
            intensity (float, optional): The intensity of the raymond lighting. Defaults to 1.0.

        """
        self._raymond_lights = Viewer._create_raymond_lights(Viewer)
        for n in self._raymond_lights:
            n.light.intensity = intensity
            if not self.scene.has_node(n):
                self.scene.add_node(n, parent_node=self._camera_node)

    def disable_raymond_lighting(self) -> None:
        """Disable raymond lighting."""
        if hasattr(self, "_raymond_lights"):
            for n in self._raymond_lights:
                self.scene.remove_node(n)
            self._raymond_lights = []

    def use_direct_lighting(self, intensity: float = 1.0) -> None:
        """Enable direct lighting.

        Args:
            intensity (float, optional): The intensity of the direct lighting. Defaults to 1.0.

        """
        self._direct_light = Viewer._create_direct_light(Viewer)
        self._direct_light.light.intensity = intensity
        if not self.scene.has_node(self._direct_light):
            self.scene.add_node(self._direct_light, parent_node=self._camera_node)

    def disable_direct_lighting(self) -> None:
        """Disable direct lighting."""
        if hasattr(self, "_direct_light") and self._direct_light is not None:
            self.scene.remove_node(self._direct_light)
            self._direct_light = None

    def __call__(
        self,
        meshes: Trimesh | list[Trimesh],
        render_params: dict[str, bool] | None = None,
        meshes_colors: list | None = None,
        bg_color: tuple[int, int, int] | None = None,
        view: str = "front",
        material: pyrender.MetallicRoughnessMaterial | None = None,
    ) -> tuple[Image, np.ndarray]:
        """Render the meshes.

        Args:
            meshes (Trimesh | list[Trimesh]): The meshes to render.
            render_params (dict[str, bool], optional): Rendering parameters dictionary. Defaults to {}.
            meshes_colors (list | None, optional): A list of colors for each mesh. Applies only when
                `render_segmentation` is True. Defaults to None.
            bg_color (tuple[int, int, int] | None, optional): The background color of the rendered image. If None,
                defaults to the current background color. Defaults to None.
            view (str, optional): The view direction. Must be one of "front", "back", "left", "right", "top", "bottom".
                Defaults to "front".
            material (pyrender.MetallicRoughnessMaterial | None, optional): The material to for the meshes.
                Defaults to None.

        Returns:
            tuple[Image, np.ndarray]: A tuple containing the rendered image and the depth map.

        Raises:
            ValueError: If the view is not one of the valid options.

        """
        prev_bg_color = self.background_color
        self.background_color = bg_color

        # Set rotation angle and direction based on view
        if view not in ["front", "back", "left", "right", "top", "bottom"]:
            msg = f"Invalid view: {view}, must be one of 'front', 'back', 'left', 'right', 'top', 'bottom'"
            raise ValueError(msg)
        if view in ["front", "back", "left", "right"]:
            rotation_angle = {"front": 0.0, "left": 90.0, "back": 180.0, "right": -90.0}[view]
            rotation_direction = [0.0, 1.0, 0.0]
        else:  # if view in ["top", "bottom"]:
            rotation_angle = {"top": 90.0, "bottom": -90.0}[view]
            rotation_direction = [1.0, 0.0, 0.0]

        self.scene.mesh_nodes.clear()
        seg_node_map = {}
        if isinstance(meshes, Trimesh):
            meshes = [meshes]
        for i, mesh in enumerate(meshes):
            rtd_mesh = mesh
            if view != "front":
                rot = transformations.rotation_matrix(np.radians(rotation_angle), rotation_direction)
                rtd_mesh = mesh.copy().apply_transform(rot)
            node_id = self.scene.add(
                pyrender.Mesh.from_trimesh(rtd_mesh, smooth=mesh.visual.kind != "face", material=material), f"mesh_{i}"
            )
            if meshes_colors:
                seg_node_map[node_id] = meshes_colors[i]

        flags = RenderFlags.NONE
        if render_params:
            for param_name, flag in PyRenderer.render_flags_map.items():
                if render_params.get(param_name, False):
                    flags |= flag

            if render_params.get("segmentation", False):
                seg_node_map = dict(
                    zip(self.scene.mesh_nodes, get_jet_colormap(len(meshes), dtype=np.uint8), strict=False)
                )

        color, depth = self.renderer.render(self.scene, flags=flags, seg_node_map=seg_node_map)

        # Restore original scene
        self.background_color = prev_bg_color

        # Return
        return img_from_array(color), depth

    def __del__(self) -> None:
        """Need to delete before creating the renderer next time."""
        self.renderer.delete()

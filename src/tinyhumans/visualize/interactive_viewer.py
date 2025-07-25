"""Interactive viewer for .mcs files, which contain glTF data with custom extensions for SMPL and camera animations."""

import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import scipy.spatial.transform as sp_transform
import torch
import tyro
import viser
from scipy.spatial.transform import Rotation

from tinyhumans.models import SMPLX
from tinyhumans.tools import McsParser, Scene4D, get_logger

logger = get_logger(__name__, "info")


class ClientSession:
    """A class to manage the state and interaction for a single connected client."""

    def __init__(self, client: viser.ClientHandle, mcs_files: list[Path], smplx_model: SMPLX) -> None:
        """Initialize the client session.

        Args:
            client: The Viser client handle.
            mcs_files: A list of paths to available .mcs files.
            smplx_model: The loaded SMPLX model.

        """
        self.client = client
        self.mcs_files = mcs_files
        self.smplx_model = smplx_model
        self.smplx_faces = smplx_model.faces.cpu().numpy()

        self.animation_state = {"playing": False, "framerate": 10.0, "current_frame": 0}
        self.scene_loaded = False
        self.scene_data: Scene4D | None = None
        self.smpl_outputs: list[dict] = []

        self.gui_elements: list[viser.GuiHandle] = []
        self.frame_nodes: dict[str, viser.MeshHandle] = {}
        self.client_camera: viser.CameraFrustumHandle | None = None
        self.video_display: viser.ImageHandle | None = None

        self._setup_file_selection_gui()
        self._start_animation_loop()

    def _setup_file_selection_gui(self) -> None:
        """Create the GUI for file searching and loading."""
        search_box = self.client.gui.add_text("Search", "")
        file_dropdown = self.client.gui.add_dropdown("Select File", [f.parent.name for f in self.mcs_files])
        button_group = self.client.gui.add_button_group("Load Actions", ("Load Selected", "Load Random"))

        @search_box.on_update
        def _(_: viser.GuiInputHandle) -> None:
            query = search_box.value.lower()
            filtered = [f for f in self.mcs_files if query in f.parent.name.lower()]
            file_dropdown.disabled = not filtered
            file_dropdown.options = [f.parent.name for f in filtered]

        @button_group.on_click
        def _(_: viser.GuiButtonGroupHandle) -> None:
            if button_group.value == "Load Selected":
                path = next((f for f in self.mcs_files if f.parent.name == file_dropdown.value), None)
                if path:
                    self.load_scene(path)
            elif button_group.value == "Load Random" and self.mcs_files:
                self.load_scene(random.choice(self.mcs_files))  # noqa: S311

    def clear_scene(self) -> None:
        """Remove all scene-specific GUI elements and meshes."""
        for handle in self.gui_elements:
            handle.remove()
        self.gui_elements.clear()
        for handle in self.frame_nodes.values():
            handle.remove()
        self.frame_nodes.clear()
        if self.client_camera:
            self.client_camera.remove()
        if self.video_display:
            self.video_display.remove()
        self.scene_loaded = False
        self.scene_data = None
        self.smpl_outputs = []

    def _handle_load_error(self, mcs_path: Path, error: Exception) -> None:
        """Log and display an error message when a scene fails to load."""
        logger.exception("Failed to load %s", mcs_path, exc_info=error)
        self.client.gui.add_text("Status", f"Error loading {mcs_path.name}", disabled=True)

    def _validate_scene(self, mcs_path: Path) -> None:
        """Validate the loaded scene data.

        Args:
            mcs_path: The path to the .mcs file.

        Raises:
            ValueError: If the scene has no frames.

        """
        if self.scene_data and self.scene_data.num_frames == 0:
            msg = f"No frames found in {mcs_path.name}"
            raise ValueError(msg)

    def load_scene(self, mcs_path: Path) -> None:
        """Load a .mcs file and set up the scene.

        Args:
            mcs_path: The path to the .mcs file to load.

        """
        self.clear_scene()
        self.video_display = None
        status = self.client.gui.add_text("Status", "Loading...", disabled=True)
        try:
            self.scene_data = McsParser(mcs_path).parse()
            self._validate_scene(mcs_path)
            self._prepare_smpl_outputs()
            self._setup_scene_gui()
            self._setup_scene_elements()
            self.update_scene(0)
            self.scene_loaded = True
        except (OSError, ValueError, KeyError) as e:
            self._handle_load_error(mcs_path, e)
        finally:
            status.remove()

    @torch.inference_mode()
    def _prepare_smpl_outputs(self) -> None:
        """Pre-compute SMPL model outputs for the loaded scene."""
        if not self.scene_data:
            return
        self.smpl_outputs = []
        for body_data in self.scene_data.smpl_data:
            codec = body_data.codec
            device = self.smplx_model.device
            smpl_output = self.smplx_model(
                poses={"body": torch.from_numpy(codec.body_pose).to(device).flatten(1, 2)[:, 3:]},
                shape_components={
                    "betas": torch.from_numpy(codec.shape_parameters).to(device).expand(codec.frame_count, -1)
                },
                root_orientations=torch.from_numpy(codec.body_pose).to(device).flatten(1, 2)[:, :3],
                root_positions=torch.from_numpy(codec.body_translation).to(device),
            )
            smpl_output_np = {}
            for key, value in smpl_output.items():
                if isinstance(value, torch.Tensor):
                    smpl_output_np[key] = value.cpu().numpy()
            self.smpl_outputs.append(smpl_output_np)

    def _setup_scene_gui(self) -> None:
        """Set up the GUI for controlling the loaded scene."""
        if not self.scene_data:
            return

        native_framerate = self._get_native_framerate()
        self.animation_state["framerate"] = float(native_framerate)

        time_slider = self.client.gui.add_slider("Frame", 0, self.scene_data.num_frames - 1, 1, 0)
        follow_cam = self.client.gui.add_checkbox("Follow Camera", initial_value=False)
        play_button = self.client.gui.add_button("Play/Pause")
        fps_slider = self.client.gui.add_slider(
            "FPS",
            1,
            native_framerate,
            1,
            min(5, native_framerate),
            marks=[(i, str(i)) for i in range(0, int(native_framerate) + 1, 5)][1:],
        )
        native_fps_btn = self.client.gui.add_button(f"Set to Native FPS ({native_framerate:.2f})")
        self.gui_elements.extend([time_slider, follow_cam, play_button, fps_slider, native_fps_btn])

        if self.scene_data.video_frames:
            show_video = self.client.gui.add_checkbox("Show Video", initial_value=True)
            self.gui_elements.append(show_video)

            @show_video.on_update
            def _(_: viser.GuiInputHandle) -> None:
                if self.video_display:
                    self.video_display.visible = show_video.value

        @time_slider.on_update
        def _(_: viser.GuiSliderHandle) -> None:
            self.animation_state["current_frame"] = time_slider.value
            self.update_scene(time_slider.value)

        @play_button.on_click
        def _(_: viser.GuiButtonHandle) -> None:
            self.animation_state["playing"] = not self.animation_state["playing"]

        @fps_slider.on_update
        def _(_: viser.GuiSliderHandle) -> None:
            self.animation_state["framerate"] = float(fps_slider.value)

        @native_fps_btn.on_click
        def _(_: viser.GuiButtonHandle) -> None:
            fps_slider.value = native_framerate
            self.animation_state["framerate"] = float(native_framerate)

    def _get_native_framerate(self) -> float:
        """Determine the native framerate from the scene data."""
        if self.scene_data and self.scene_data.camera_data and len(self.scene_data.camera_data.times) > 1:
            return 1.0 / np.mean(np.diff(self.scene_data.camera_data.times))
        if self.scene_data and self.scene_data.smpl_data:
            return self.scene_data.smpl_data[0].codec.frame_rate
        return 30.0

    def _setup_scene_elements(self) -> None:
        """Create the visual elements (camera frustum, meshes) for the scene."""
        if not self.scene_data or not self.scene_data.camera_intrinsics:
            return
        intrinsics = self.scene_data.camera_intrinsics
        # Add camera
        self.client_camera = self.client.scene.add_camera_frustum(
            name="/camera", fov=intrinsics.yfov, aspect=intrinsics.aspect_ratio, scale=0.15, color=(200, 200, 200)
        )
        # Add video frames
        if self.scene_data.video_frames:
            self.video_display = self.client.scene.add_image(
                "/video_frame", self.scene_data.video_frames[0], render_width=0, render_height=0
            )
        # Add SMPL meshes
        for i, smpl_output in enumerate(self.smpl_outputs):
            self.frame_nodes[f"/human_{i}"] = self.client.scene.add_mesh_simple(
                f"/human_{i}", smpl_output["verts"][0], self.smplx_faces, color=(200, 200, 200)
            )

    @torch.inference_mode()
    def update_scene(self, frame_idx: int) -> None:
        """Update all elements in the scene to a specific frame.

        Args:
            frame_idx: The frame index to update the scene to.

        """
        if not self.scene_data or not self.client_camera:
            return

        # Update camera pose
        if self.scene_data.camera_data:
            cam_data = self.scene_data.camera_data
            R_cw = cam_data.R_cw[frame_idx]
            T_cw = cam_data.T_cw[frame_idx]

            # Viser expects the camera pose in the world frame (world-from-camera),
            # so we need to invert the camera-from-world transform.
            R_wc = R_cw.T
            T_wc = -R_wc @ T_cw

            # The camera data is stored in a CV-style coordinate system (Y-down, Z-forward),
            # but Viser uses a glTF-style coordinate system (Y-up, Z-backward).
            # To correct this, we apply a 180-degree rotation around the X-axis.
            fix_rot = sp_transform.Rotation.from_euler("x", 180, degrees=True).as_matrix()
            R_wc = R_wc @ fix_rot

            self.client_camera.position = T_wc
            quat_xyzw = Rotation.from_matrix(R_wc).as_quat()
            self.client_camera.wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])

            if cast("viser.GuiInputHandle", self.gui_elements[1]).value:
                self.client.camera.position = self.client_camera.position
                self.client.camera.wxyz = self.client_camera.wxyz

        # Update video frame
        if self.scene_data.video_frames and self.scene_data.camera_intrinsics and self.client_camera:
            intrinsics = self.scene_data.camera_intrinsics
            distance = 0.15 * 2.5  # Scale of the frustum * a factor
            height = 2 * distance * np.tan(intrinsics.yfov / 2) * 0.8
            width = height * intrinsics.aspect_ratio
            self.video_display.render_width = width
            self.video_display.render_height = height
            self.video_display.image = self.scene_data.video_frames[frame_idx]

            R_wc = Rotation.from_quat(np.roll(self.client_camera.wxyz, -1)).as_matrix()
            forward_dir = -R_wc[:, 2]
            up_dir = R_wc[:, 1]

            # Position it on the frustum plane, and then move it up.
            position = self.client_camera.position - forward_dir * distance - up_dir * height
            self.video_display.position = position
            self.video_display.wxyz = self.client_camera.wxyz

        # Update human meshes
        for i, smpl_output in enumerate(self.smpl_outputs):
            name = f"/human_{i}"
            visible = frame_idx < smpl_output["verts"].shape[0]
            self.frame_nodes[name].visible = visible
            if visible:
                self.frame_nodes[name].vertices = smpl_output["verts"][frame_idx]

    def _start_animation_loop(self) -> None:
        """Start the main animation loop for this client."""
        while True:
            if self.scene_loaded and self.animation_state["playing"]:
                time_slider = cast("viser.GuiSliderHandle", self.gui_elements[0])
                num_frames = time_slider.max + 1
                current_frame = (self.animation_state["current_frame"] + 1) % num_frames
                self.animation_state["current_frame"] = current_frame
                time_slider.value = current_frame
            time.sleep(1.0 / self.animation_state["framerate"])


class InteractiveViewer:
    """A class for the main application of the interactive viewer."""

    def __init__(self, data_path: str | Path, device: torch.device | str | None = None) -> None:
        """Initialize the interactive viewer.

        Args:
            data_path: The path to the directory containing .mcs files.

        """
        self.data_path = Path(data_path)
        self.device = (
            torch.device(
                "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
            if device is None
            else device
        )
        self.server = viser.ViserServer(label="TinyHumans Interactive Viewer", port=8043)
        self.smplx_model = self._load_smplx_model()
        self.mcs_files = sorted(self.data_path.rglob("*.mcs"))

    def _load_smplx_model(self) -> SMPLX:
        """Load the pretrained SMPLX model."""
        support_dir = Path("downloads")
        model_path = support_dir / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
        return SMPLX.from_pretrained(pretrained_model_path=model_path, num_betas=10, device_map=self.device)

    def _setup_scene(self) -> None:
        """Set up the initial scene environment."""
        self.server.scene.world_axes.visible = False
        self.server.scene.set_up_direction("+y")
        self.server.scene.add_grid(name="/ground", plane="xz", cell_size=1.0, cell_color=(0.2, 0.5, 0.1))

    def run(self) -> None:
        """Start the viewer application."""
        if not self.mcs_files:
            logger.error("No .mcs files found in %s", self.data_path)
            return

        logger.info("Found %d .mcs files.", len(self.mcs_files))
        self._setup_scene()

        self.server.on_client_connect(lambda client: ClientSession(client, self.mcs_files, self.smplx_model))

        logger.info("Viewer running... Press Ctrl+C to exit.")
        while True:
            time.sleep(1.0)


def main(
    data_path: str | Path = Path("/Users/abenetatos/GitRepos/tinyhumans/assets/"), device: str | None = None
) -> None:
    """Run the entry point for the interactive viewer.

    Args:
        data_path: The path to the directory containing .mcs files.
        device: The device to use for the viewer. If None, a device will be selected automatically with priority for
            CUDA, then MPS and finally CPU.

    """
    viewer = InteractiveViewer(data_path, device)
    viewer.run()


if __name__ == "__main__":
    tyro.cli(main)

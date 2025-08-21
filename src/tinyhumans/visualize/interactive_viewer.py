"""Interactive viewer for .mcs files, which contain glTF data with custom extensions for SMPL and camera animations."""

import random
import time
from enum import Enum
from pathlib import Path

import numpy as np
import scipy.spatial.transform as sp_transform
import torch
import tyro
import viser
from scipy.spatial.transform import Rotation

from tinyhumans.datatypes import Scene4D
from tinyhumans.models import SMPLX
from tinyhumans.tools import get_logger

logger = get_logger(__name__, "info")

DEFAULTS = {
    "FPS": 10,
    "SUBSAMPLING": 5,
    "SHOW_VIDEO": True,
    "VIDEO_IN_BG": False,
    "FOLLOW_CAMERA": True,
    "VIDEO_WITH_CAMERA": True,
    "VIDEO_SCALE_CAMERA": 0.6,
    "VIDEO_SCALE_VIEWPORT": 0.15,
}


class VideoFramePosition(Enum):
    """The position of the video frame in the scene."""

    RELATIVE_TO_VIEWPORT = 0
    RELATIVE_TO_CAMERA = 1
    RELATIVE_TO_CAMERA_BG = 2


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

        self.animation_state = {"playing": False, "framerate": 10.0, "current_frame": 0, "subsampling": 1}
        self.scene_loaded = False
        self.scene_data: Scene4D | None = None
        self.smpl_outputs: list[dict] = []

        self.gui_elements: dict[str, viser.GuiInputHandle] = {}
        self.gui_folders: dict[str, viser.GuiFolderHandle] = {}
        self.frame_nodes: dict[str, viser.MeshHandle] = {}
        self.scene_camera: viser.CameraFrustumHandle | None = None
        self.video_display: viser.ImageHandle | None = None
        self.status: viser.GuiInputHandle | None = None
        self.video_frame_position = VideoFramePosition.RELATIVE_TO_VIEWPORT

        self._setup_file_selection_gui()
        self._start_animation_loop()

    def _setup_file_selection_gui(self) -> None:
        """Create the GUI for file searching and loading."""
        with self.client.gui.add_folder("File Loading"):
            search_box = self.client.gui.add_text("Search", "")
            file_dropdown = self.client.gui.add_dropdown("Select File", [f.parent.name for f in self.mcs_files])
            button_group = self.client.gui.add_button_group("Load File", ("🔍 Selected", "🎲 Random"))
            self.status = self.client.gui.add_text("Status", "", disabled=True, visible=False)

        @search_box.on_update
        def _(_: viser.GuiInputHandle) -> None:
            query = search_box.value.lower()
            filtered = [f for f in self.mcs_files if query in f.parent.name.lower()]
            file_dropdown.disabled = not filtered
            file_dropdown.options = [f.parent.name for f in filtered]

        @button_group.on_click
        def _(_: viser.GuiButtonGroupHandle) -> None:
            if button_group.value == "🔍 Selected":
                path = next((f for f in self.mcs_files if f.parent.name == file_dropdown.value), None)
            elif button_group.value == "🎲 Random" and self.mcs_files:
                path = random.choice(self.mcs_files)  # noqa: S311
            if path:
                self.load_scene(path)
                search_box.value = path.parent.name

    def _setup_playback_controls_gui(self) -> None:
        """Set up the GUI for playback control of the scene."""
        # Create GUI folder
        self.gui_folders["Playback Controls"] = self.client.gui.add_folder("Playback Controls")

        # Create GUI elements
        with self.gui_folders["Playback Controls"]:
            play_button = self.client.gui.add_button("Play/Pause")
            prev_next_buttons = self.client.gui.add_button_group("Frame Navigation", ("◀ Previous", "Next ▶"))
            time_slider = self.client.gui.add_slider(
                "Frame", 0, self.scene_data.num_frames - 1, DEFAULTS["SUBSAMPLING"], 0
            )
            native_framerate = round(self._get_native_framerate())
            fps_slider = self.client.gui.add_slider(
                "FPS",
                1,
                native_framerate,
                1,
                min(DEFAULTS["FPS"], native_framerate),
                marks=[*[(i, str(i)) for i in range(0, native_framerate + 1, 5)][1:], (native_framerate, "▶️")],
            )
            subsampling_slider = self.client.gui.add_slider(
                "Subsampling", 1, 15, 1, DEFAULTS["SUBSAMPLING"], marks=[(5, 5), (10, 10)]
            )

        self.animation_state["framerate"] = fps_slider.value

        # Update GUI elements dict
        self.gui_elements = {
            "play_button": play_button,
            "prev_next_buttons": prev_next_buttons,
            "time_slider": time_slider,
            "fps_slider": fps_slider,
            "subsampling_slider": subsampling_slider,
        }

        @play_button.on_click
        def _(_: viser.GuiButtonHandle) -> None:
            self.animation_state["playing"] = not self.animation_state["playing"]

        @prev_next_buttons.on_click
        def _(_: viser.GuiButtonGroupHandle) -> None:
            num_frames = self.scene_data.num_frames
            step = self.animation_state["subsampling"]
            if prev_next_buttons.value == "◀ Previous":
                new_frame = (self.animation_state["current_frame"] - step + num_frames) % num_frames
            else:
                new_frame = (self.animation_state["current_frame"] + step) % num_frames
            time_slider.value = min(new_frame, time_slider.max)

        @time_slider.on_update
        def _(_: viser.GuiSliderHandle) -> None:
            clamped_frame_idx = min(len(self.smpl_outputs) - 1, time_slider.value)
            self.animation_state["current_frame"] = clamped_frame_idx
            self.update_scene(clamped_frame_idx)

        @fps_slider.on_update
        def _(_: viser.GuiSliderHandle) -> None:
            self.animation_state["framerate"] = float(fps_slider.value)

        @subsampling_slider.on_update
        def _(_: viser.GuiSliderHandle) -> None:
            self.animation_state["subsampling"] = subsampling_slider.value
            time_slider.step = subsampling_slider.value
            time_slider.value = round(time_slider.value / time_slider.step) * time_slider.step

    def _setup_display_options_gui(self) -> None:  # noqa: PLR0915
        """Set up the GUI for display options of the loaded scene."""
        if not self.scene_data or not (self.scene_data.video_frames or self.scene_data.camera_intrinsics):
            return

        # Create GUI folder
        self.gui_folders["Display Options"] = self.client.gui.add_folder("Display Options")

        # Create GUI elements
        with self.gui_folders["Display Options"]:
            if self.scene_data.camera_intrinsics:
                follow_cam = self.client.gui.add_checkbox("Follow Camera", initial_value=DEFAULTS["FOLLOW_CAMERA"])
                self.gui_elements["follow_cam"] = follow_cam
            if self.scene_data.video_frames:
                show_video = self.client.gui.add_checkbox("Show Video", initial_value=DEFAULTS["SHOW_VIDEO"])
                self.gui_folders["Video Options"] = self.client.gui.add_folder("Video Options")
                with self.gui_folders["Video Options"]:
                    self.gui_elements["show_video"] = show_video
                    if self.scene_data.camera_intrinsics:
                        video_with_camera = self.client.gui.add_checkbox(
                            "Place Video Relative to Camera", initial_value=DEFAULTS["VIDEO_WITH_CAMERA"]
                        )
                        video_in_background = self.client.gui.add_checkbox(
                            "Place Video in Background", initial_value=DEFAULTS["VIDEO_IN_BG"]
                        )
                        mesh_opacity = self.client.gui.add_slider(
                            "Mesh Opacity", min=0.0, max=1.0, step=0.01, initial_value=1.0, visible=False
                        )
                        self.gui_elements |= {
                            "video_in_background": video_in_background,
                            "video_with_camera": video_with_camera,
                            "mesh_opacity": mesh_opacity,
                        }

                    video_scale = self.client.gui.add_slider(
                        "Video Scale",
                        min=0.0,
                        max=1.0 if DEFAULTS["VIDEO_WITH_CAMERA"] and self.scene_data.camera_intrinsics else 0.5,
                        step=0.01,
                        initial_value=DEFAULTS["VIDEO_SCALE_CAMERA"]
                        if DEFAULTS["VIDEO_WITH_CAMERA"] and self.scene_data.camera_intrinsics
                        else DEFAULTS["VIDEO_SCALE_VIEWPORT"],
                    )
                    self.gui_elements["video_scale"] = video_scale

        # Update GUI elements dict
        if "follow_cam" in self.gui_elements:

            @follow_cam.on_update
            def _(_: viser.GuiInputHandle) -> None:
                if follow_cam.value and self.scene_camera:
                    self.client.camera.position = self.scene_camera.position
                    self.client.camera.wxyz = self.scene_camera.wxyz
                self.update_camera_pose(self.animation_state["current_frame"])

        if "show_video" in self.gui_elements:

            @show_video.on_update
            def _(_: viser.GuiInputHandle) -> None:
                self.video_display.visible = show_video.value
                self.gui_folders["Video Options"].visible = show_video.value
                self.update_video_frame(self.animation_state["current_frame"])

        if "video_in_background" in self.gui_elements:

            @video_in_background.on_update
            def _(_: viser.GuiInputHandle) -> None:
                video_with_camera.disabled = video_in_background.value
                mesh_opacity.visible = video_in_background.value
                self.update_video_frame(self.animation_state["current_frame"])
                self.gui_elements["video_scale"].visible = not video_in_background.value
                self.update_smpl_meshes(self.animation_state["current_frame"])

        if "video_with_camera" in self.gui_elements:

            @video_with_camera.on_update
            def _(_: viser.GuiInputHandle) -> None:
                if video_with_camera.value:
                    video_scale.max = 1.0
                    video_scale.value = DEFAULTS["VIDEO_SCALE_CAMERA"]
                else:
                    video_scale.max = 0.5
                    video_scale.value = DEFAULTS["VIDEO_SCALE_VIEWPORT"]
                self.update_video_frame(self.animation_state["current_frame"])

        if "mesh_opacity" in self.gui_elements:

            @mesh_opacity.on_update
            def _(_: viser.GuiInputHandle) -> None:
                for handle in self.frame_nodes.values():
                    handle.opacity = mesh_opacity.value

        if "video_scale" in self.gui_elements:

            @video_scale.on_update
            def _(_: viser.GuiInputHandle) -> None:
                self.update_video_frame(self.animation_state["current_frame"])

    def clear_scene(self) -> None:
        """Remove all scene-specific GUI elements and meshes."""
        for handle in self.gui_elements.values():
            handle.remove()
        self.gui_elements.clear()
        if "Video Options" in self.gui_folders:
            self.gui_folders["Video Options"].remove()
            del self.gui_folders["Video Options"]
        for handle in self.gui_folders.values():
            handle.remove()
        self.gui_folders.clear()
        for handle in self.frame_nodes.values():
            handle.remove()
        self.frame_nodes.clear()
        if self.scene_camera:
            self.scene_camera.remove()
        if self.video_display:
            self.video_display.remove()
        self.scene_loaded = False
        self.scene_data = None
        self.smpl_outputs = []
        self.video_display = None
        self.animation_state = {
            "playing": False,
            "framerate": DEFAULTS["FPS"],
            "current_frame": 0,
            "subsampling": DEFAULTS["SUBSAMPLING"],
        }

    def _handle_load_error(self, mcs_path: Path, error: Exception) -> None:
        """Log and display an error message when a scene fails to load."""
        logger.exception("Failed to load %s", mcs_path, exc_info=error)
        self.status.value = f"Error loading {mcs_path.name}"
        self.status.visible = True

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
        self.status.value = "Loading..."
        self.status.visible = True
        try:
            self.scene_data = Scene4D.from_msc_file(mcs_path)
            self._validate_scene(mcs_path)
            self._prepare_smpl_outputs()
            self._setup_playback_controls_gui()
            self._setup_display_options_gui()
            self._setup_scene_elements()
            self.update_scene(0)
            self.scene_loaded = True
        except (OSError, ValueError, KeyError) as e:
            self._handle_load_error(mcs_path, e)
        finally:
            self.status.visible = False

        @self.client.camera.on_update
        def _(_: viser.CameraFrustumHandle) -> None:
            if not self.scene_data:
                return
            if self.video_frame_position == VideoFramePosition.RELATIVE_TO_VIEWPORT:
                self.update_video_frame(
                    self.gui_elements["time_slider"].value if "time_slider" in self.gui_elements else 0
                )

    @torch.inference_mode()
    def _prepare_smpl_outputs(self) -> None:
        """Pre-compute SMPL model outputs for the loaded scene."""
        if not self.scene_data:
            return
        # for body_data in self.scene_data.smpl_data:
        self.smpl_outputs = (
            self.smplx_model(self.scene_data.smpl_data.to(self.smplx_model.device))[0].detach().cpu().numpy()
        )["verts"]
        self.smpl_frame_presence = self.scene_data.smpl_data.frame_presence[0].detach().cpu().numpy()

    def _get_native_framerate(self) -> float:
        """Determine the native framerate from the scene data."""
        if self.scene_data and self.scene_data.camera_data is not None and len(self.scene_data.camera_data.times) > 1:
            return 1.0 / np.mean(np.diff(self.scene_data.camera_data.times))
        if self.scene_data and self.scene_data.smpl_data is not None:
            return self.scene_data.smpl_data.frame_rate
        return 30.0

    def _setup_scene_elements(self) -> None:
        """Create the visual elements (camera frustum, meshes) for the scene."""
        if not self.scene_data:
            return

        # Add camera
        if self.scene_data.camera_intrinsics:
            intrinsics = self.scene_data.camera_intrinsics
            self.scene_camera = self.client.scene.add_camera_frustum(
                name="/camera", fov=intrinsics.yfov, aspect=intrinsics.aspect_ratio, scale=0.1, color=(65, 120, 100)
            )

        # Add video frames
        if self.scene_data.video_frames:
            self.video_display = self.client.scene.add_image(
                "/video_frame", self.scene_data.video_frames[0], render_width=0, render_height=0, cast_shadow=False
            )

        # Add SMPL meshes
        for i, smpl_output in enumerate(self.smpl_outputs[0, :]):
            self.frame_nodes[f"/human_{i}"] = self.client.scene.add_mesh_simple(
                f"/human_{i}", smpl_output, self.smplx_faces, color=(200, 200, 200)
            )

    def _get_video_frame_position(self, frame_idx: int, video_frame_position: VideoFramePosition) -> dict:
        camera = (
            self.scene_camera if video_frame_position != VideoFramePosition.RELATIVE_TO_VIEWPORT else self.client.camera
        )

        R_wc = Rotation.from_quat(np.roll(camera.wxyz, -1)).as_matrix()
        forward_dir = -R_wc[:, 2]
        up_dir = R_wc[:, 1]
        left_dir = R_wc[:, 0]

        frame_aspect = self.scene_data.video_frames[0].shape[1] / self.scene_data.video_frames[0].shape[0]
        if video_frame_position == VideoFramePosition.RELATIVE_TO_CAMERA_BG:
            T_wc = camera.position
            R_cw = R_wc.T
            T_cw = -R_cw @ T_wc
            z_axis_world = R_wc[:, 2]  # Viser camera's Z axis points "behind" the camera.
            max_depth = 0.0
            for h_idx in range(self.smpl_outputs.shape[1]):
                frame_presence = self.smpl_frame_presence[:, h_idx]
                clamped_idx = min(frame_idx, len(frame_presence) - 1)
                verts_w = self.smpl_outputs[clamped_idx, h_idx]
                verts_c = (R_cw @ verts_w.T).T + T_cw
                max_depth = max(max_depth, verts_c[:, 2].max()) if frame_presence[clamped_idx] else max_depth
            distance = max_depth + 0.1 if max_depth > 0.0 else 10.0
            height = 2 * distance * np.tan(camera.fov / 2)
            position = camera.position + z_axis_world * distance
        else:
            video_scale = self.gui_elements["video_scale"].value
            distance = 0.1 * 2.5  # Scale of the frustum * a factor
            height = 2 * distance * np.tan(camera.fov / 2) * video_scale

            position = camera.position - forward_dir * distance
            if video_frame_position == VideoFramePosition.RELATIVE_TO_CAMERA:
                position -= up_dir * (distance * np.tan(camera.fov / 2) + height / 2)
            else:  # if video_frame_position == VideoFramePosition.RELATIVE_TO_VIEWPORT
                viewport_aspect = camera.image_width / camera.image_height
                position -= up_dir * (distance * np.tan(camera.fov / 2) - height / 2) + left_dir * (
                    distance * np.tan(camera.fov / 2) * viewport_aspect - height * frame_aspect / 2
                )

        return {"position": position, "width": height * frame_aspect, "height": height, "wxyz": camera.wxyz}

    def update_video_frame(self, frame_idx: int) -> None:
        """Update all video frame related elements in the scene to a specific frame.

        Args:
            frame_idx: The frame index to update the scene to.

        """
        if not self.scene_data.video_frames or not self.gui_elements["show_video"].value:
            return

        if not (self.scene_data.camera_intrinsics and self.scene_camera):
            self.video_frame_position = VideoFramePosition.RELATIVE_TO_VIEWPORT
        elif self.gui_elements["video_in_background"].value:
            self.video_frame_position = VideoFramePosition.RELATIVE_TO_CAMERA_BG
        elif self.gui_elements["video_with_camera"].value:
            self.video_frame_position = VideoFramePosition.RELATIVE_TO_CAMERA
        else:
            self.video_frame_position = VideoFramePosition.RELATIVE_TO_VIEWPORT

        frame_position_dict = self._get_video_frame_position(frame_idx, self.video_frame_position)
        self.video_display.position = frame_position_dict["position"]
        self.video_display.render_width = frame_position_dict["width"]
        self.video_display.render_height = frame_position_dict["height"]
        self.video_display.image = self.scene_data.video_frames[frame_idx]
        self.video_display.wxyz = frame_position_dict["wxyz"]

    def update_camera_pose(self, frame_idx: int) -> None:
        """Update all video frame related elements in the scene to a specific frame.

        Args:
            frame_idx: The frame index to update the scene to.

        """
        if self.scene_data.camera_data is None:
            return

        cam_data = self.scene_data.camera_data
        R_cw = cam_data.R_cw[frame_idx].cpu().numpy()
        T_cw = cam_data.T_cw[frame_idx].cpu().numpy()

        # Viser expects the camera pose in the world frame (world-from-camera),
        # so we need to invert the camera-from-world transform.
        R_wc = R_cw.T
        T_wc = -R_wc @ T_cw

        # The camera data is stored in a CV-style coordinate system (Y-down, Z-forward),
        # but Viser uses a glTF-style coordinate system (Y-up, Z-backward).
        # To correct this, we apply a 180-degree rotation around the X-axis.
        fix_rot = sp_transform.Rotation.from_euler("x", 180, degrees=True).as_matrix()
        R_wc = R_wc @ fix_rot

        self.scene_camera.position = T_wc
        quat_xyzw = Rotation.from_matrix(R_wc).as_quat()
        self.scene_camera.wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])

        if self.gui_elements["follow_cam"].value:
            self.client.camera.position = self.scene_camera.position
            self.client.camera.wxyz = self.scene_camera.wxyz

    def update_smpl_meshes(self, frame_idx: int) -> None:
        """Update all smpl meshes in the scene to a specific frame.

        Args:
            frame_idx: The frame index to update the scene to.

        """
        for h_idx in range(self.smpl_outputs.shape[1]):
            name = f"/human_{h_idx}"
            visible = self.smpl_frame_presence[frame_idx, h_idx]
            self.frame_nodes[name].visible = visible
            if not visible:
                continue
            self.frame_nodes[name].vertices = self.smpl_outputs[frame_idx, h_idx]
            self.frame_nodes[name].opacity = (
                self.gui_elements["mesh_opacity"].value
                if self.video_frame_position == VideoFramePosition.RELATIVE_TO_CAMERA_BG
                else 1.0
            )

    def update_scene(self, frame_idx: int) -> None:
        """Update all elements in the scene to a specific frame.

        Args:
            frame_idx: The frame index to update the scene to.

        """
        if not self.scene_data:
            return

        self.update_camera_pose(frame_idx)
        self.update_video_frame(frame_idx)
        self.update_smpl_meshes(frame_idx)

    def _start_animation_loop(self) -> None:
        """Start the main animation loop for this client."""
        while True:
            if self.scene_loaded and self.animation_state["playing"]:
                time_slider = self.gui_elements["time_slider"]
                num_frames = time_slider.max + 1
                subsampling = self.animation_state["subsampling"]
                current_frame = (self.animation_state["current_frame"] + subsampling) % num_frames
                self.animation_state["current_frame"] = current_frame
                time_slider.value = current_frame
            time.sleep(1.0 / (self.animation_state["framerate"] / self.animation_state["subsampling"]))


class InteractiveViewer:
    """A class for the main application of the interactive viewer."""

    def __init__(self, data_path: str | Path, device: torch.device | str | None = None, port: int = 8042) -> None:
        """Initialize the interactive viewer.

        Args:
            data_path: The path to the directory containing .mcs files.
            device: The device to use for the viewer. If None, a device will be selected automatically with priority for
                CUDA, then MPS and finally CPU.
            port: The port to use for the web server.

        """
        self.data_path = Path(data_path)
        self.device = (
            torch.device(
                "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
            if device is None
            else device
        )
        self.server = viser.ViserServer(label="TinyHumans Interactive Viewer", port=port)
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
    data_path: str | Path = Path("/home/abenetatos/GitRepos/tinyhumans/hoigen_results_jz_mount"),
    device: str | None = None,
    port: int = 8047,
) -> None:
    """Run the entry point for the interactive viewer.

    Args:
        data_path: The path to the directory containing .mcs files.
        device: The device to use for the viewer. If None, a device will be selected automatically with priority for
            CUDA, then MPS and finally CPU.
        port: The port to use for the web server.

    """
    viewer = InteractiveViewer(data_path, device, port=port)
    viewer.run()


if __name__ == "__main__":
    tyro.cli(main)

"""A class to parse Meshcapade's .mcs files (.gltf style file) to extract scene data."""

import base64
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import scipy.spatial.transform as sp_transform
import torch
from smplcodec import SMPLCodec


class McsParser:
    """A class to parse a .mcs file (.gltf style file) to extract scene data."""

    # Mappings from glTF specifications
    COMPONENT_TYPE_MAP: ClassVar[dict[int, np.dtype]] = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}
    TYPE_COMPONENT_COUNT: ClassVar[dict[str, int]] = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def __init__(self, gltf_path: str | Path) -> None:
        """Initialize the parser with the path to the glTF file.

        Args:
            gltf_path: The path to the .gltf or .mcs file.

        """
        self.gltf_path = Path(gltf_path)
        self.gltf: dict[str, Any] = {}
        self.buffers: list[bytes] = []

    def _decode_buffers(self) -> None:
        """Decode base64 encoded buffers from the glTF file."""
        self.buffers = []
        for buffer_info in self.gltf["buffers"]:
            uri = buffer_info["uri"]
            _, encoded_data = uri.split(",")
            self.buffers.append(base64.b64decode(encoded_data))

    def _get_accessor_data(self, accessor_index: int) -> np.ndarray:
        """Decode data from a glTF accessor.

        Args:
            accessor_index: The index of the accessor to decode.

        Returns:
            The decoded data as a numpy array.

        """
        accessor = self.gltf["accessors"][accessor_index]
        buffer_view = self.gltf["bufferViews"][accessor["bufferView"]]
        buffer = self.buffers[buffer_view["buffer"]]

        component_type = McsParser.COMPONENT_TYPE_MAP[accessor["componentType"]]
        num_components = McsParser.TYPE_COMPONENT_COUNT[accessor["type"]]
        count = accessor["count"]

        byte_offset = buffer_view.get("byteOffset", 0)
        accessor_byte_offset = accessor.get("byteOffset", 0)
        total_offset = byte_offset + accessor_byte_offset

        data = np.frombuffer(buffer, dtype=component_type, count=count * num_components, offset=total_offset)
        return data.reshape(count, num_components) if num_components > 1 else data

    @staticmethod
    def list_of_smpl_data_to_smpldata_dict_args(list_of_smpl_data: list[dict]) -> dict:
        if not list_of_smpl_data:
            return {}
        keys = list_of_smpl_data[0].keys()
        args_dict: dict[str, np.ndarray] = {}
        for key in keys:
            args_dict[key] = np.stack([smpl_data[key] for smpl_data in list_of_smpl_data])
            if key != "shape_parameters" and args_dict[key].ndim >= 3:
                args_dict[key] = args_dict[key].swapaxes(0, 1)
            if all(v is None for v in args_dict[key]):
                args_dict[key] = None

        return args_dict

    def _load_smpl_data(self) -> dict[str, np.ndarray]:
        """Load SMPL data from the glTF extensions.

        Returns:
            A dict of data to initialize a SMPLData object

        """
        smpl_data = []
        frame_presence = []
        scene_desc = self.gltf["scenes"][0].get("extensions", {}).get("MC_scene_description", {})
        if "smpl_bodies" in scene_desc:
            for body_info in scene_desc["smpl_bodies"]:
                buffer_view = self.gltf["bufferViews"][body_info["bufferView"]]
                smpl_buffer_bytes = self.buffers[buffer_view["buffer"]]
                smpl_data.append(asdict(SMPLCodec.from_file(io.BytesIO(smpl_buffer_bytes))))
                frame_presence.append(body_info["frame_presence"])

        if not smpl_data:
            return {}
        keys = smpl_data[0].keys()
        smpl_data_dict: dict[str, np.ndarray] = {}
        frame_presence = np.array(frame_presence)
        max_frame_presence = frame_presence.max(0)[1].item()
        for key in keys:
            list_values = []
            for idx_i, smpl_data_i in enumerate(smpl_data):
                cur_v = smpl_data_i[key]
                if isinstance(cur_v, np.ndarray) and cur_v.ndim >= 2:
                    frame_presence_i = frame_presence[idx_i]
                    cur_v = np.pad(
                        cur_v,
                        (
                            (frame_presence_i[0].item(), max_frame_presence - frame_presence_i[1].item()),
                            *[(0, 0)] * (cur_v.ndim - 1),
                        ),
                    )
                list_values.append(cur_v)
            smpl_data_dict[key] = np.stack(list_values)
            if key != "shape_parameters" and smpl_data_dict[key].ndim >= 3:
                smpl_data_dict[key] = smpl_data_dict[key].swapaxes(0, 1)
            if all(v is None for v in smpl_data_dict[key]):
                smpl_data_dict[key] = None

        smpl_data_dict["frame_presence"] = np.stack(
            [
                [0] * frame_presence_i[0].item()
                + [1] * (frame_presence_i[1].item() - frame_presence_i[0].item())
                + [0] * (max_frame_presence - frame_presence_i[1].item())
                for frame_presence_i in frame_presence
            ]
        ).swapaxes(0, 1)
        smpl_data_dict.pop("frame_count", None)
        if "codec_version" in smpl_data_dict:
            smpl_data_dict["codec_version"] = smpl_data_dict["codec_version"][0].item()
        if "smpl_version" in smpl_data_dict:
            smpl_data_dict["smpl_version"] = smpl_data_dict["smpl_version"][0].item()
        if "gender" in smpl_data_dict:
            smpl_data_dict["gender"] = smpl_data_dict["gender"][0].item()
        if "frame_rate" in smpl_data_dict:
            smpl_data_dict["frame_rate"] = smpl_data_dict["frame_rate"][0].item()

        return smpl_data_dict

    def _load_camera_data(self) -> dict[str, torch.Tensor] | None:
        """Load camera animation data from the glTF animations."""
        if not self.gltf.get("animations"):
            return None

        animation = self.gltf["animations"][0]
        time_accessor_idx, trans_accessor_idx, rot_accessor_idx = -1, -1, -1

        for channel in animation["channels"]:
            sampler = animation["samplers"][channel["sampler"]]
            path = channel["target"]["path"]
            if path == "translation":
                trans_accessor_idx = sampler["output"]
                time_accessor_idx = sampler["input"]
            elif path == "rotation":
                rot_accessor_idx = sampler["output"]

        if any(idx == -1 for idx in [time_accessor_idx, trans_accessor_idx, rot_accessor_idx]):
            return None

        times = self._get_accessor_data(time_accessor_idx)
        translations = self._get_accessor_data(trans_accessor_idx)
        rotations = self._get_accessor_data(rot_accessor_idx)

        R_cw = sp_transform.Rotation.from_quat(rotations).as_matrix().transpose(0, 2, 1)
        T_cw = -np.einsum("nij,nj->ni", R_cw, translations)

        return {"times": torch.tensor(times), "R_cw": torch.tensor(R_cw), "T_cw": torch.tensor(T_cw)}

    def _load_camera_intrinsics(self) -> dict[str, int | float] | None:
        """Load camera intrinsics from the glTF cameras."""
        if not self.gltf.get("cameras"):
            return None
        perspective = self.gltf["cameras"][0]["perspective"]
        return {
            "yfov": perspective["yfov"],
            "aspect_ratio": perspective["aspectRatio"],
            "znear": perspective.get("znear"),
        }

    def parse(self) -> dict[str, list | dict | int | None]:
        """Load and parse the glTF file into a structured Scene4D object."""
        with self.gltf_path.open(encoding="utf-8") as f:
            self.gltf = json.load(f)

        self._decode_buffers()
        scene_desc = self.gltf["scenes"][0].get("extensions", {}).get("MC_scene_description", {})

        video_path = self.gltf_path.with_name("video.mp4")
        video_frames = None
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            video_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        return {
            "num_frames": scene_desc.get("num_frames", 0),
            "smpl_data": self._load_smpl_data(),
            "camera_data": self._load_camera_data(),
            "camera_intrinsics": self._load_camera_intrinsics(),
            "video_frames": video_frames,
        }

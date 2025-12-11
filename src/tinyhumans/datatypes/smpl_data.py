"""The main datatype to store and utilize SMPL data."""

import json
from collections.abc import Iterator
from contextlib import closing
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from smplcodec import SMPLGender, SMPLParamStructure, SMPLVersion
from smplcodec.utils import PathType, coerce_type, extract_item, to_camel, to_snake
from smplcodec.version import MAJOR
from tensordict import MetaData, TensorClass
from tensordict.utils import IndexType
from tinytools import get_logger
from torch import Tensor

from tinyhumans.tools import validate_tensor_shape

logger = get_logger(__name__)


def matching(thing: float | Tensor | None, other: float | Tensor | None) -> bool:
    """Check if two things are equal."""
    if thing is None or other is None:
        if not (torch.is_tensor(thing) or torch.is_tensor(other)):
            return thing is None and other is None
        # If one is a zero tensor and the other is None return True, else False
        return (torch.is_tensor(other) and other.abs().sum().allclose(torch.tensor(0.0))) or (
            torch.is_tensor(thing) and thing.abs().sum().allclose(torch.tensor(0.0))
        )
    if isinstance(thing, (int, float, str)):
        return thing == other
    if torch.is_tensor(thing) and torch.is_tensor(other):
        # All non-scalar fields contain float data
        return torch.allclose(thing, other)
    return False


class SMPLData(TensorClass):
    """SMPLData object.

    Ripped from smplcodec but translated into a pytorch-based tensorclass.

    This behaves as a tensor and a dataclass.
    All tensors are stored internally as [Batch Count (B) x Frame Count (T) x Human Count (H) x ...].
    If a tensor of less than the expected number of dimensions is passed in, it will be unsqueezed to the expected
    number of dimensions by adding dimensions in the front.
    For the "shape_parameters" tensor, read and write by the users happens as if it were a [B x H x ...] tensor
    (the shape does not change with frame count).
    Indexing a SMPLData object returns a SMPLData object and also keeps the number of dimensions the same with a size of
    one (1) for the indexed dimension.

    Attributes:
        codec_version (int): Codec version.
        smpl_version (SMPLVersion): SMPL version (SMPL, SMPLH, etc.).
        gender (SMPLGender): Gender.
        frame_rate (float | None): Frame rate.
        body_translation (Tensor | None): Body translation. Tensor of shape: [B x T x N x 3]
        body_pose (Tensor | None): Body pose. Tensor of shape: [B x T x N x J x 3]
        head_pose (Tensor | None): Head pose. Tensor of shape: [B x T x N x 3 x 3]
        left_hand_pose (Tensor | None): Left hand pose. Tensor of shape: [B x T x N x 15 x 3]
        right_hand_pose (Tensor | None): Right hand pose. Tensor of shape: [B x T x N x 15 x 3]
        expression_parameters (Tensor | None): FLAME expression parameters. Tensor of shape: [B x T x N x 10-100]
        dmpl_parameters (Tensor | None): DMPL soft-tissue dynamics parameters. Tensor of shape: [B x T x N x 8]
        shape_parameters (Tensor | None): Shape parameters. Tensor of shape: [B x T x N x 10-300] internally but
            interfaced as a tensor of shape [B x N x 10-300]
        shape_aggregation_method (Literal["mean", "first", "last"]): Shape aggregation method. Since shape parameters
            are the same for all frames, if we concatenate to SMPLData objects across the frame dimension, we could end
            up with a shape_parameters tensor that is not constant across frames. To handle this, we can either average
            the shape parameters across frames, take the first frame or the last one.

        batch_count (int | None): Number of batches.
        frame_count (int | None): Number of frames.
        human_count (int | None): Number of humans.
        batch_size (tuple[int, int, int]): Batch size (batch_count, frame_count, human_count).

    """

    codec_version: int = MAJOR
    smpl_version: SMPLVersion = SMPLVersion.SMPLX
    gender: SMPLGender = SMPLGender.NEUTRAL

    # motion metadata
    frame_rate: float | None = None

    # pose / motion data for frame_count=T frames, human_count=N
    body_translation: Tensor | None = None  # [B x T x N x 3] Global trans
    body_pose: Tensor | None = None  # [B x T x N x 22 x 3] pelvis..right_wrist
    head_pose: Tensor | None = None  # [B x T x N x 3 x 3] jaw, leftEye, rightEye
    left_hand_pose: Tensor | None = None  # [B x T x N x 15 x 3] left_index1..left_thumb3
    right_hand_pose: Tensor | None = None  # [B x T x N x 15 x 3] right_index1..right_thumb3
    expression_parameters: Tensor | None = None  # [B x T x N x 10-100] FLAME parameters
    dmpl_parameters: Tensor | None = None  # [B x T x N x 8] DMPL parameters
    frame_presence: Tensor | None = None  # [B x T x N] Frame presence tensor -> 0: missing, 1: present

    shape_parameters: Tensor | None = None  # [B x T x N x 10-300] betas
    shape_aggregation_method: Literal["mean", "first", "last"] = "first"

    def post_init(self) -> None:
        """Post-initialization method."""
        if self.shape_aggregation_method not in ["mean", "first", "last"]:
            msg = (
                f"Invalid shape aggregation method {self.shape_aggregation_method}. "
                "Should be 'mean', 'first' or 'last'."
            )
            raise ValueError(msg)

    @property
    def _shape_parameters(self) -> Tensor | None:
        """Access the internal representation of `shape_parameters` attribute."""
        return self._tensordict.get("shape_parameters", None)

    def _infer_batch_dimensions(self, dim: int = 0) -> int | None:
        if self.batch_size:
            return self.batch_size[dim]
        if dim < 0 or dim > 2:
            msg = f"Invalid dimension {dim}. Should be 0, 1 or 2."
            raise ValueError(msg)
        # If batch size is not yet set, we might have set a variable (tensor) that implies a frame count
        not_none_motion_fields = [
            field.name
            for field in fields(SMPLData)
            if field.name in self._tensordict and torch.is_tensor(getattr(self, field.name))
        ]
        if not not_none_motion_fields:
            return None

        if "shape_parameters" in not_none_motion_fields and len(not_none_motion_fields) > 1:
            not_none_motion_fields.remove("shape_parameters")

        if "shape_parameters" not in not_none_motion_fields:
            return getattr(self, not_none_motion_fields[0]).shape[dim]

        # "shape_parameters" is the only tensor set
        if dim != 1:
            return self._shape_parameters.shape[dim]

        # "shape_parameters" is the only tensor set and dim=1 return None because "shape_parameters" is constant
        # across time (dim=1)
        return None

    @property
    def batch_count(self) -> int | None:
        """Get batch count from batch size."""
        return self._infer_batch_dimensions(0)

    @property
    def frame_count(self) -> int | None:
        """Get frame count from batch size."""
        return self._infer_batch_dimensions(1)

    @property
    def human_count(self) -> int | None:
        """Get number of humnas from batch size."""
        return self._infer_batch_dimensions(2)

    def validate_tensor_shape_by_name(
        self, key: str, ndims: int, shape: list[int | str], value: Tensor | None = None
    ) -> None:
        """Validate field in SMPLCodec object by shape."""
        try:
            validate_tensor_shape(getattr(self, key) if value is None else value, ndims, shape, key)
        except ValueError as e:
            msg = f"Error validating tensor {key}: {e}"
            raise ValueError(msg) from e

    def __getitem__(self, index: IndexType) -> "SMPLData":
        """Get item in SMPLData.

        Overrides TensorClass.__getitem__ to unsqueeze batch dimension.

        Args:
            index (IndexType): Index to get.

        Returns:
            SMPLData: SMPLData object.

        """
        # Case 1: The index is a single integer.
        if isinstance(index, int):
            # Convert `5` to `slice(5, 6)`
            index = slice(index, index + 1)

        # Case 2: The index is a tuple of mixed types.
        if isinstance(index, tuple):
            # Process each part of the tuple index
            new_index_parts = []
            for idx_part in index:
                if isinstance(idx_part, int):
                    # Convert integer parts to slices
                    new_index_parts.append(slice(idx_part, idx_part + 1))
                else:
                    # Keep slices, Ellipsis, None, Tensors, etc. as they are
                    new_index_parts.append(idx_part)
            index = tuple(new_index_parts)

        return super().__getitem__(index)

    def __iter__(self: "SMPLData") -> Iterator["SMPLData"]:
        """Iterate over the batch dimension of SMPLData."""
        for i in range(self.batch_count):
            yield self[i]

    def iter_frames(self: "SMPLData") -> Iterator["SMPLData"]:
        """Iterate over the frame dimension of SMPLData."""
        for i in range(self.frame_count):
            yield self[:, i]

    def iter_humans(self: "SMPLData") -> Iterator["SMPLData"]:
        """Iterate over the humans dimension of SMPLData."""
        for i in range(self.human_count):
            yield self[:, :, i]

    def __getattribute__(self: "SMPLData", name: str) -> Any:
        """Get attribute in SMPLCodec.

        Overrides TensorClass.__getattribute__ to hide shape_parameters shape being (B x T x 10-300) internally.

        Args:
            name (str): Attribute to get.

        """
        if name == "shape_parameters" and "shape_parameters" in self._tensordict:
            selector = (
                0
                if self.shape_aggregation_method == "first"
                else (-1 if self.shape_aggregation_method == "last" else None)
            )
            return (
                self._tensordict["shape_parameters"].select(dim=1, index=selector)
                if selector is not None
                else self._tensordict["shape_parameters"].mean(dim=1)
            )
        return super().__getattribute__(name)

    def __setattr__(self: "SMPLData", name: str, value: Any) -> None:  # noqa: PLR0912, PLR0915
        """Set attribute in SMPLCodec.

        Overrides TensorClass.__setattr__ to automatically infer device and batch size.

        Args:
            name (str): Attribute set.
            value (Any): Value to set.

        """
        if name in ["codec_version", "smpl_version", "gender", "frame_rate", "shape_aggregation_method"]:
            if name == "frame_rate" and value is not None:
                value = float(value)
            if name == "codec" and value is not None:
                value = int(value)
            if name == "smpl_version" and value is not None:
                value = SMPLVersion(value)
            if name == "gender" and value is not None:
                value = SMPLGender(value)
            if name == "shape_aggregation_method" and value is not None:
                value = str(value)
                if value not in ["mean", "first", "last"]:
                    msg = f"shape_aggregation_method should be one of ['mean', 'first', 'last'], but got {value}"
                    raise ValueError(msg)
            value = MetaData(value)
        elif name == "frame_presence" and value is None and self.batch_size:
            value = torch.ones(*self.batch_size, device=self.device, dtype=self.dtype)
        elif value is not None and name in [field.name for field in self.fields()]:
            value = torch.tensor(coerce_type(value)) if not torch.is_tensor(value) else value

            if value.device != self.device and self.device is not None:
                msg = f"Device of {name} is different from expected object device: {self.device}."
                raise ValueError(msg)
            if value.dtype != self.dtype and self.dtype is not None:
                msg = f"Dtype of {name} is different from expected object dtype: {self.dtype}."
                raise ValueError(msg)

            B = self.batch_count
            T = self.frame_count
            H = self.human_count
            if name == "shape_parameters":
                # shape_parameters is a tensor of shape (B x T x N x 10-300) internally (for tensorclass to work nicely)
                # but should be used as a tensor of size (B x N x 10-300) or (10-300) by users
                if value.dim() == 4:  # B x T x N x 10-300
                    value = self._shape_parameters_aggregation(value)
                value: Tensor = self._unsqueeze_to_ndims(value, 3, is_shape=True)  # B x N x 10-300
                self.validate_tensor_shape_by_name(name, 3, [f"{B}, 1", f"{H}, 1", "10-300"], value=value)
                value = value.unsqueeze(1)  # B x 1 x N x 10-300
                shape = [-1]
            else:
                ndims = 5 if "pose" in name else (4 if name != "frame_presence" else 3)
                shape = (
                    getattr(SMPLParamStructure[self.smpl_version], name)
                    if "pose" in name
                    else (
                        [3]
                        if name == "body_translation"
                        else (
                            ["10-100"]
                            if name == "expression_parameters"
                            else ([8] if name == "dmpl_parameters" else [])  # frame_presence
                        )
                    )
                )
                if "pose" in name and value.size(-1) != 3:
                    value = value.reshape(*value.shape[:-1], -1, 3)
                value = self._unsqueeze_to_ndims(value, ndims)
                self.validate_tensor_shape_by_name(name, ndims, [f"{B}, 1", f"{T}, 1", f"{H}, 1", *shape], value=value)
                if name == "expression_parameters":
                    shape = [-1]
            value = value.expand(B, -1, -1, *shape) if B is not None else value
            value = value.expand(-1, T, -1, *shape) if T is not None else value
            value = value.expand(-1, -1, H, *shape) if H is not None else value

        super().__setattr__(name, value)
        if self.device is None:
            self.auto_device_()
        if not self.batch_size and torch.is_tensor(value) and name != "shape_parameters":
            if "shape_parameters" in self._tensordict:
                self.validate_tensor_shape_by_name(
                    "shape_parameters", 3, [f"{value.size(0)}, 1", f"{value.size(2)}, 1", "10-300"]
                )
                self.shape_parameters = self.shape_parameters.expand(value.shape[0], value.shape[2], -1)
            self.auto_batch_size_(3)
            try:
                if self.frame_presence is None:
                    self.frame_presence = torch.ones(*self.batch_size, device=self.device, dtype=self.dtype)
            except KeyError:
                pass

    def _unsqueeze_to_ndims(self, tensor: Tensor, ndims: int, is_shape: bool = False) -> Tensor:
        if tensor.ndim > ndims:
            msg = f"This tensor must be at most {ndims}-dimensional. [Batch Size x Frame Count x Human Count ...]"
            raise ValueError(msg)

        non_batch_dims = ndims - (3 if not is_shape else 2)
        while tensor.ndim < ndims:
            tensor = tensor.unsqueeze(0 if tensor.ndim <= non_batch_dims else 1)

        return tensor

    @property
    def body_orientation(self) -> Tensor:
        """Extract the body orientation from the body pose (first joint)."""
        return self.body_pose[..., 0, :] if self.body_pose is not None else None

    @body_orientation.setter
    def body_orientation(self, value: Tensor | None) -> None:
        """Set the body orientation from the body pose (first joint)."""
        if value is None:
            return
        if value.ndim != 4:
            msg = "body_orientation must be a 3-dimensional tensor (batch_count x frame_count x human_count x 3)."
            raise ValueError(msg)
        if self.body_pose is None:
            self.body_pose = torch.zeros(
                (
                    *(self.batch_size if self.batch_size else value.shape[:-2]),
                    *SMPLParamStructure[self.smpl_version].body_pose,
                ),
                device=self.device if self.device else value.device,
                dtype=self.dtype if self.dtype else value.dtype,
            )
        self.body_pose[..., 0, :] = value

    @property
    def full_pose(self) -> Tensor:
        """Create and return the full_pose tensor [batch_count x frame_count x human_count x num_joints x 3].

        If frame_count is 0 or None it is assumed to be 1 instead. This function will always return a full pose array,
        if any pose information is missing it will be filled with zeros automatic.
        """
        bs = self.batch_size if self.batch_size else (1, 1, 1)
        pose = torch.empty((*bs, 0, 3), device=self.device, dtype=self.dtype)
        for field in fields(SMPLParamStructure[self.smpl_version]):
            if getattr(SMPLParamStructure[self.smpl_version], field.name) is None:
                continue
            part_pose = getattr(self, field.name)
            if part_pose is None:
                part_pose = torch.zeros(
                    (*bs, *getattr(SMPLParamStructure[self.smpl_version], field.name)),
                    device=self.device,
                    dtype=self.dtype,
                )  # merge tuples for shape
            pose = torch.cat((pose, part_pose), dim=-2)
        return pose

    @full_pose.setter
    def full_pose(self, value: Tensor) -> None:
        """Set the individual pose components from the `full_pose` tensor."""
        # Check that the full pose tensor has a compatible shape
        if value.size(-1) != 3:
            value = value.reshape(*value.shape[:-1], -1, 3)  # [..., J, 3]

        # Check that the full pose tensor has the correct number of joints
        num_joints = 0
        for field in fields(SMPLParamStructure[self.smpl_version]):
            joints_shape = getattr(SMPLParamStructure[self.smpl_version], field.name)
            num_joints += joints_shape[0] if joints_shape is not None else 0
        if value.size(-2) != num_joints:
            msg = f"Expected full pose tensor to have {num_joints} joints, but got {value.size(-2)}."
            raise ValueError(msg)

        # Split the full tensor into individual pose components
        start_idx = 0
        for field in fields(SMPLParamStructure[self.smpl_version]):
            field_name = field.name
            field_shape = getattr(SMPLParamStructure[self.smpl_version], field_name)

            if field_shape is None:
                continue

            # Extract the component from full_pose
            setattr(self, field_name, value[..., start_idx : start_idx + field_shape[0], :])

            # Update start index
            start_idx += field_shape[0]

    def equals(self, other: "SMPLData") -> bool:
        """Check if two SMPLCodec objects are equal."""
        return all(matching(getattr(self, f.name), getattr(other, f.name)) for f in fields(self))

    @classmethod
    def from_full_pose(
        cls,
        full_pose: Tensor,
        smpl_version: SMPLVersion | str = SMPLVersion.SMPLX,
        gender: SMPLGender | str = SMPLGender.NEUTRAL,
        frame_rate: float | None = None,
        body_translation: Tensor | None = None,
        expression_parameters: Tensor | None = None,
        dmpl_parameters: Tensor | None = None,
        frame_presence: Tensor | None = None,
        shape_parameters: Tensor | None = None,
        shape_aggregation_method: Literal["mean", "first", "last"] = "first",
    ) -> "SMPLData":
        """Create SMPLData object from a full pose tensor.

        Args:
            full_pose (Tensor): Full pose tensor of shape [B x T x N x J x 3]
            codec_version (int | None): Codec version
            smpl_version (SMPLVersion | str | None): SMPL version
            gender (SMPLGender | str | None): Gender
            frame_rate (float | None): Frame rate
            body_translation (Tensor | None): Body translation tensor [B x T x N x 3]
            expression_parameters (Tensor | None): Expression parameters tensor [B x T x N x 10-100]
            dmpl_parameters (Tensor | None): DMPL parameters tensor [B x T x N x 8]
            frame_presence (Tensor | None): Frame presence tensor [B x T x N] (1 = present, 0 = not present)
            shape_parameters (Tensor | None): Shape parameters tensor [B x N x 10-300]
            shape_aggregation_method (Literal["mean", "first", "last"] | None): Shape aggregation method

        Returns:
            SMPLData: Initialized SMPLData object

        """
        # Create base SMPLData instance
        smpl_data = cls(
            smpl_version=SMPLVersion.from_string(smpl_version) if isinstance(smpl_version, str) else smpl_version,
            gender=SMPLGender.from_string(gender) if isinstance(gender, str) else gender,
            frame_rate=frame_rate,
            body_translation=body_translation,
            expression_parameters=expression_parameters,
            dmpl_parameters=dmpl_parameters,
            frame_presence=frame_presence,
            shape_parameters=shape_parameters,
            shape_aggregation_method=shape_aggregation_method,
        )
        smpl_data.full_pose = full_pose

        return smpl_data

    @classmethod
    def from_file(
        cls, filename: PathType, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float
    ) -> "SMPLData":
        """Load SMPLCodec object from file."""
        with filename if hasattr(filename, "read") else open(filename, "rb") as fp:  # noqa: PTH123
            try:
                data = {k: np.array(v) for k, v in json.load(fp).items()}
            except:  # noqa: E722
                fp.seek(0)
                data = dict(np.load(fp))

        data = {k: extract_item(v) for k, v in data.items()}
        data = {
            to_snake(k): np.expand_dims(v, axis=1 if v.ndim > 1 else 0)[None, ...] if isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }

        frame_count = data.pop("frame_count", None)
        if "codec_version" not in data:
            data["codec_version"] = 1

        obj = cls(**data).to(device=device).to(dtype=dtype)
        if frame_count is not None and frame_count != obj.frame_count:
            msg = (
                f"Frame count of {frame_count} from file does not match the inferred frame count of {obj.frame_count}."
            )
            raise ValueError(msg)
        return obj

    @classmethod
    def from_amass_npz(cls, filename: PathType, smpl_version_str: str = "smplx") -> "SMPLData":  # TODO: check
        """Load SMPLCodec object from AMASS .npz file."""
        with closing(np.load(filename, allow_pickle=True)) as infile:
            in_dict = dict(infile)

            mapped_dict = {
                "shape_parameters": in_dict.get("betas"),
                "body_translation": in_dict.get("trans"),
                "gender": SMPLGender.from_string(str(in_dict.get("gender", "neutral"))),
                "smpl_version": SMPLVersion.from_string(str(in_dict.get("surface_model_type", smpl_version_str))),
                "frame_count": int(
                    in_dict.get("frameCount", in_dict["trans"].shape[0])
                ),  # expects trans to be not None if frameCount does not exist
                # for frame rate, different names seem to exist in AMASS (SMPLH: mocap_framerate, SMPLX: mocap_frame_rate?!)
                "frame_rate": float(
                    in_dict.get("frameRate", in_dict.get("mocap_frame_rate", in_dict.get("mocap_framerate", 60.0)))
                ),  # default 60.0??
            }

            # check if pose parameters are stored separately
            if "body_pose" in in_dict:
                mapped_dict["body_pose"] = np.concatenate(
                    (in_dict["root_orient"], in_dict["pose_body"]), axis=-1
                ).reshape(mapped_dict["frame_count"], -1, 3)

                if "pose_jaw" in in_dict and "pose_eye" in in_dict:
                    mapped_dict["head_pose"] = np.concatenate(
                        (in_dict["pose_jaw"], in_dict["pose_eye"]), axis=-1
                    ).reshape(mapped_dict["frame_count"], -1, 3)

                if "pose_hand" in in_dict:
                    mapped_dict["left_hand_pose"] = in_dict["pose_hand"][
                        :, : int(in_dict["pose_hand"].shape[-1] / 2.0)
                    ].reshape(mapped_dict["frame_count"], -1, 3)
                    mapped_dict["right_hand_pose"] = in_dict["pose_hand"][
                        :, int(in_dict["pose_hand"].shape[-1] / 2.0) :
                    ].reshape(mapped_dict["frame_count"], -1, 3)
            elif "poses" in in_dict:
                # split "full pose" into separate parameters
                joint_info = SMPLParamStructure[mapped_dict["smpl_version"]]
                start_ind = 0

                for field in fields(joint_info):
                    num_params = np.prod(getattr(joint_info, field.name))

                    if num_params is not None:
                        mapped_dict[field.name] = in_dict["poses"][:, start_ind : start_ind + num_params].reshape(
                            (-1, *getattr(joint_info, field.name))
                        )
                        start_ind += num_params
            else:
                logger.warning("No pose parameters in file!!!")

            return cls(**{k: v for k, v in mapped_dict.items() if k != "frame_count"})

    def write(self, filename: str | Path, as_json: bool = True) -> None:
        """Write SMPLCodec object to file."""
        filename = Path(filename)
        if not filename.suffix:
            filename = filename.with_suffix(".smpl") if as_json else filename.with_suffix(".smplz")
        for idx_i, smpl_i in enumerate(self):
            smpl_i = cast("SMPLData", smpl_i)
            filename_i = filename
            if len(self) > 1:
                filename_i = filename.parent / f"{filename.stem}_{idx_i}{filename.suffix}"
            for idx_j, smpl_ij in enumerate(smpl_i.iter_humans()):
                filename_ij = filename_i
                if smpl_i.human_count > 1:
                    filename_ij = filename_i.parent / f"{filename_i.stem}_{idx_j}{filename.suffix}"
                data = {
                    to_camel(f): coerce_type(v).squeeze((2, 0) if isinstance(v, np.ndarray) and v.ndim >= 3 else 0)
                    for f, v in smpl_ij.detach().cpu().numpy().items()
                    if v is not None and f not in ("shape_aggregation_method", "dmpl_parameters", "frame_presence")
                }
                if smpl_ij.shape_parameters is not None:
                    data["shapeParameters"] = smpl_ij.shape_parameters[0, 0].detach().cpu().numpy()
                data[to_camel("frame_count")] = self.frame_count
                if as_json:
                    with filename_ij.open("w") as outfile:
                        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}, outfile)
                else:
                    with filename_ij.open("wb") as outfile:
                        np.savez_compressed(outfile, **data)

    def get_shape_tensor(
        self,
        shape_coeffs_size: int | None = None,
        expression_coeffs_size: int | None = None,
        dmpl_coeffs_size: int | None = None,
    ) -> Tensor:
        """Get shape tensor from a SMPLData object [batch_count x frame_count x human_count x shape_expr_dmpl_coeffs].

        This function will always return a full shape array, if any shape information is missing it will be filled with
        zeros automatic.

        Args:
            shape_coeffs_size (int | None, optional): Size of shape parameters. If `None`, the size is infered from the
                object data, if set. Defaults to None.
            expression_coeffs_size (int | None, optional): Size of expression parameters. If `None`, the size is infered
                from the object data, if set. Defaults to None.
            dmpl_coeffs_size (int | None, optional): Size of DMPL parameters. If `None`, the size is infered from the
                object data, if set. Defaults to None.

        Returns:
            Tensor: Tensor of concatenated shape and expression parameters.

        """
        # Make sure shape_coeffs_size, expression_coeffs_size and dmpl_coeffs_size are integers
        if shape_coeffs_size is None:
            shape_coeffs_size = self.shape_parameters.size(-1) if self.shape_parameters is not None else 0
        if expression_coeffs_size is None:
            expression_coeffs_size = (
                self.expression_parameters.size(-1) if self.expression_parameters is not None else 0
            )
        if dmpl_coeffs_size is None:
            dmpl_coeffs_size = self.dmpl_parameters.size(-1) if self.dmpl_parameters is not None else 0

        if not (
            isinstance(shape_coeffs_size, int)
            and isinstance(expression_coeffs_size, int)
            and isinstance(dmpl_coeffs_size, int)
        ):
            msg = "shape_coeffs_size, expression_coeffs_size and dmpl_coeffs_size must be ints. Set to `0` to disable."
            raise TypeError(msg)

        bs = self.batch_size if self.batch_size else (1, 1, 1)
        shape_parameters = [
            self.shape_parameters[..., :shape_coeffs_size].unsqueeze(1).expand(*bs, -1)
            if self.shape_parameters is not None
            else torch.zeros(*bs, shape_coeffs_size, device=self.device, dtype=self.dtype)
        ]
        dmpl_parameters = (
            [
                self.dmpl_parameters[..., :dmpl_coeffs_size]
                if self.dmpl_parameters is not None
                else torch.zeros(*bs, dmpl_coeffs_size, device=self.device, dtype=self.dtype)
            ]
            if dmpl_coeffs_size
            else []
        )
        expression_parameters = (
            [
                self.expression_parameters[..., :expression_coeffs_size]
                if self.expression_parameters is not None
                else torch.zeros(*bs, expression_coeffs_size, device=self.device, dtype=self.dtype)
            ]
            if expression_coeffs_size
            else []
        )
        return torch.cat(shape_parameters + dmpl_parameters + expression_parameters, dim=-1)

    def _shape_parameters_aggregation(self, shape_parameters: torch.Tensor) -> torch.Tensor:
        if self.shape_aggregation_method == "first":
            return shape_parameters[:, 0]
        if self.shape_aggregation_method == "last":
            return shape_parameters[:, -1]
        # Else if self.shape_aggregation_method == "mean"
        return shape_parameters.mean(dim=1)

    def set_shape_tensor(
        self,
        full_shape_tensor: Tensor,
        shape_coeffs_size: int | None = None,
        expression_coeffs_size: int | None = None,
        dmpl_coeffs_size: int | None = None,
    ) -> None:
        """Set shape tensor of a SMPLData object [batch_count x frame_count x human_count x shape_expr_dmpl_coeffs].

        Args:
            full_shape_tensor (Tensor): Tensor of concatenated shape, dmpl and expression parameters.
            shape_coeffs_size (int | None, optional): Size of shape parameters. If `None`, the size is infered from the
                object data, if set. If None and the corresponding object's field is also None, then throws an error.
                Defaults to None.
            expression_coeffs_size (int | None, optional): Size of expression parameters. If `None`, the size is infered
                from the object data, if set. If None and the corresponding object's field is also None, then throws an
                error. Defaults to None.
            dmpl_coeffs_size (int | None, optional): Size of DMPL parameters. If `None`, the size is infered
                from the object data, if set. If None and the corresponding object's field is also None, then throws an
                error. Defaults to None.

        """
        # Make sure shape_coeffs_size, expression_coeffs_size and dmpl_coeffs_size are integers. If None, check if the
        # corresponding object's field is also None and raise an error if so
        if shape_coeffs_size is None:
            if self.shape_parameters is None:
                msg = (
                    "shape_coeffs_size is None and self.shape_parameters is None so cannot infer the size. "
                    "Set to `0` to disable."
                )
                raise TypeError(msg)
            shape_coeffs_size = self.shape_parameters.size(-1)
        if expression_coeffs_size is None:
            if self.expression_parameters is None:
                msg = (
                    "expression_coeffs_size is None and self.expression_parameters is None so cannot infer the size. "
                    "Set to `0` to disable."
                )
                raise TypeError(msg)
            expression_coeffs_size = self.expression_parameters.size(-1)
        if dmpl_coeffs_size is None:
            if self.dmpl_parameters is None:
                msg = (
                    "dmpl_coeffs_size is None and self.dmpl_parameters is None so cannot infer the size. "
                    "Set to `0` to disable."
                )
                raise TypeError(msg)
            dmpl_coeffs_size = self.dmpl_parameters.size(-1)

        if not (
            isinstance(shape_coeffs_size, int)
            and isinstance(expression_coeffs_size, int)
            and isinstance(dmpl_coeffs_size, int)
        ):
            msg = "shape_coeffs_size, expression_coeffs_size and dmpl_coeffs_size must be ints. Set to `0` to disable."
            raise TypeError(msg)

        # Make sure the full_shape_tensor has the correct size
        if full_shape_tensor.ndim != 4:
            msg = f"Expected full_shape_tensor to be 4-dimensional, got {full_shape_tensor.ndim}"
            raise ValueError(msg)
        if full_shape_tensor.size(-1) != shape_coeffs_size + expression_coeffs_size + dmpl_coeffs_size:
            msg = (
                f"Expected shape tensor size to be {shape_coeffs_size + expression_coeffs_size + dmpl_coeffs_size}, got"
                f" {full_shape_tensor.size(-1)}"
            )
            raise ValueError(msg)

        ## for "shape_parameters" we aggregate based on the self.shape_aggregation_method
        if not shape_coeffs_size:
            self.shape_parameters = None
        else:
            self.shape_parameters = self._shape_parameters_aggregation(full_shape_tensor[..., :shape_coeffs_size])

        start_idx = shape_coeffs_size
        for attr_name, attr_size in (
            ("dmpl_parameters", dmpl_coeffs_size),
            ("expression_parameters", expression_coeffs_size),
        ):
            setattr(self, attr_name, full_shape_tensor[..., start_idx : start_idx + attr_size] if attr_size else None)
            start_idx += attr_size

    def presence_in_frame(self, frame_idx: int) -> Tensor | None:
        """Return a tensor of booleans indicating if each of the smpl bodies is present in the given frame.

        If frame_presence is not set (only when batch size is not set) this function will return None.
        """
        if self.frame_presence is None:
            return None
        if frame_idx >= self.frame_count:
            return torch.zeros(
                self.batch_count, self.frame_count, self.human_count, dtype=torch.bool, device=self.device
            )

        return self.frame_presence[:, frame_idx] > 0

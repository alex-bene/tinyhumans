"""Types for TinyHumans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING, Any

import torch
from tensordict.tensordict import TensorDict

if TYPE_CHECKING:
    from tensordict._nestedkey import NestedKey
    from tensordict.base import CompatibleType, T
    from tensordict.utils import DeviceType, IndexType
    from torch import Size, Tensor


MANO_POSE_SIZE = 15
SMPL_POSE_SIZE = 21
FLAME_POSE_SIZE = 4


class AutoTensorDict(TensorDict):
    def __init__(
        self,
        source: T | dict[NestedKey, CompatibleType] = None,
        batch_size: Sequence[int] | Size | int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool | None = None,
        lock: bool = False,
        **kwargs: dict[str, Any] | None,
    ) -> None:
        super().__init__(source, batch_size, device, names, non_blocking, lock, **kwargs)
        self.auto_batch_size_(1)
        self.auto_device_()

    def __setitem__(self, key: IndexType, value: Any) -> None:
        super().__setitem__(key, value)
        if self.device is None:
            self.auto_device_()
        if not self.batch_size:
            self.auto_batch_size_(1)


class LimitedAttrTensorDictWithDefaults(AutoTensorDict):
    valid_attr_keys: tuple[str, ...] = ()  # SHOULD BE IN ORDER FOR CONCATENATION
    valid_atrr_sizes: tuple[tuple[int, ...]] = ()  # SHOULD BE IN ORDER OF KEYS

    def __init__(
        self,
        source: T | dict[NestedKey, CompatibleType] = None,
        batch_size: Sequence[int] | Size | int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool | None = None,
        lock: bool = False,
        **kwargs: dict[str, Any] | None,
    ) -> None:
        if (source is not None) and kwargs:
            msg = "Either a dictionary or a sequence of kwargs must be provided, not both."
            raise ValueError(msg)
        source = kwargs if kwargs else source
        if source is not None:
            self.check_keys(source.keys())
        super().__init__(source, batch_size, device, names, non_blocking, lock, **kwargs)
        for attr_name, attr_size in zip(self.valid_attr_keys, self.valid_atrr_sizes):
            self[attr_name] = self.get(attr_name, default=None)
            if self[attr_name] is None:
                # Make a (batch_size, *attr_size) tensor of zeros using memory for just one element
                self[attr_name] = torch.zeros([1] * len(attr_size), dtype=self.dtype, device=self.device).expand(
                    self.batch_size[0], *attr_size
                )

    def __setitem__(self, key: IndexType, value: Any) -> None:
        self.check_keys(key)
        super().__setitem__(key, value)

    @classmethod
    def check_keys(cls, keys: str | Sequence[str]) -> None:
        msg = None
        if isinstance(keys, str) and (keys.lower() not in cls.valid_attr_keys):
            msg = f"Key {keys} is not a valid key for {cls.__name__}"
            keys = keys.lower()
        elif (
            not isinstance(keys, str)
            and isinstance(keys, Sequence)
            and not all(key in cls.valid_attr_keys for key in keys)
        ):
            msg = f"Sequence {keys} do not contain valid keys for {cls.__name__}"
            keys = [key.lower() for key in keys]

        if not msg:
            return

        msg += f"Valid keys for {cls.__name__} are {cls.valid_attr_keys!r}."
        raise KeyError(msg)

    def __getattr__(self, name: str) -> Any:
        if name in self.valid_attr_keys:
            return self[name]
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.valid_attr_keys:
            self[name] = value
            return
        super().__setattr__(name, value)


class Pose(LimitedAttrTensorDictWithDefaults):
    valid_model_types: tuple[str, ...] = ("smpl", "smplh", "smplx", "flame", "mano")

    @staticmethod
    def check_model_type(model_type: str | None) -> None:
        if model_type is None:
            return

        if model_type.lower() not in Pose.valid_model_types:
            msg = f"{model_type} is not a valid model type."
            msg += f"Valid types are: {', '.join([repr(m) for m in Pose.valid_model_types])}."
            raise ValueError(msg)

    @property
    def model_type(self) -> str:
        if self.__class__.__name__ == "Pose" or not self.__class__.__name__.endswith("Pose"):
            raise NotImplementedError

        return self.__class__.__name__[:-4].lower()

    def to_tensor(self) -> Tensor:
        return torch.cat([self[attr_name] for attr_name in self.valid_attr_keys], dim=-1)

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        raise NotImplementedError


class SMPLPose(Pose):
    valid_attr_keys: tuple[str, ...] = ("body", "hand")  # SHOULD BE IN ORDER FOR CONCATENATION
    valid_atrr_sizes: tuple[tuple[int, ...]] = ((63,), (6,))  # SHOULD BE IN ORDER OF KEYS
    full_pose_size: int = field(init=False, default=SMPL_POSE_SIZE * 3)

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        if model_type is not None:
            self.check_model_type(model_type)
            # TODO: this needs testing
            if model_type == "smplh":
                self.body, self.hand = pose_tensor.split([SMPL_POSE_SIZE * 3, MANO_POSE_SIZE * 3], dim=-1)
                self.hand = torch.cat([self.hand[:3], self.hand[MANO_POSE_SIZE * 3 : (MANO_POSE_SIZE + 1) * 3]], dim=-1)
                return
            if model_type == "mano":
                self.hand = torch.cat(
                    [pose_tensor[:3], pose_tensor[MANO_POSE_SIZE * 3 : (MANO_POSE_SIZE + 1) * 3]], dim=-1
                )
                return
            if model_type != "smpl":
                msg = f"{model_type.capitalize} pose loading to SMPL poses is not possible."
                raise ValueError(msg)

        self.body, self.hand = pose_tensor.split([63, 6], dim=-1)


class MANOPose(Pose):
    valid_attr_keys: tuple[str, ...] = ("hand",)  # SHOULD BE IN ORDER FOR CONCATENATION
    valid_atrr_sizes: tuple[tuple[int, ...]] = ((45,),)  # SHOULD BE IN ORDER OF KEYS
    full_pose_size: int = field(init=False, default=MANO_POSE_SIZE * 3)

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        if model_type is not None:
            self.check_model_type(model_type)
            # TODO: this needs testing
            if model_type == "smplh":
                _, self.hand = pose_tensor.split([SMPL_POSE_SIZE * 3, MANO_POSE_SIZE * 3], dim=-1)
                return
            if model_type != "mano":
                msg = f"{model_type.capitalize} pose loading to MANO poses is not possible."
                raise ValueError(msg)

        self.hand = pose_tensor


class SMPLHPose(SMPLPose):
    valid_atrr_sizes: tuple[tuple[int, ...]] = ((63,), (90,))  # SHOULD BE IN ORDER OF KEYS
    full_pose_size: int = field(init=False, default=(SMPL_POSE_SIZE + MANO_POSE_SIZE) * 3)

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        if model_type is not None:
            self.check_model_type(model_type)
            # TODO: this needs testing
            if model_type == "smpl":
                self.body, self.hand = pose_tensor.split([SMPL_POSE_SIZE * 3, 6], dim=-1)
                self.hand = torch.zeros([MANO_POSE_SIZE * 3, 3], dtype=pose_tensor.dtype, device=pose_tensor.device)
                self.hand[:3] = self.hand[:3]
                self.hand[MANO_POSE_SIZE * 3 : (MANO_POSE_SIZE + 1) * 3] = self.hand[3:]
                return
            if model_type == "mano":
                self.hand = pose_tensor
                return
            if model_type != "smplh":
                msg = f"{model_type.capitalize} pose loading to SMPL-H poses is not possible."
                raise ValueError(msg)

        self.body, self.hand = pose_tensor.split([SMPL_POSE_SIZE * 3, MANO_POSE_SIZE * 3], dim=-1)


class FLAMEPose(Pose):
    valid_attr_keys: tuple[str, ...] = ("body", "jaw", "eyes")  # SHOULD BE IN ORDER FOR CONCATENATION
    valid_atrr_sizes: tuple[tuple[int, ...]] = ((3,), (3,), (6,))  # SHOULD BE IN ORDER OF KEYS
    full_pose_size: int = field(init=False, default=FLAME_POSE_SIZE * 3)

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        if model_type is not None:
            self.check_model_type(model_type)
            # TODO: this needs testing
            if model_type == "smplx":
                body, self.jaw, self.eyes, _ = pose_tensor.split([SMPL_POSE_SIZE * 3, 3, 6, MANO_POSE_SIZE * 3], dim=-1)
                # TODO: no idea which part of the body flame is supposed to take
                self.body = body[:3]
                return
            if model_type != "flame":
                msg = f"{model_type.capitalize} pose loading to SMPL-X poses is not possible."
                raise ValueError(msg)

        self.body, self.jaw, self.eyes = pose_tensor.split([3, 3, 6], dim=-1)


class SMPLXPose(FLAMEPose):
    valid_attr_keys: tuple[str, ...] = (*FLAMEPose.valid_attr_keys, "hand")  # SHOULD BE IN ORDER FOR CONCATENATION
    valid_atrr_sizes: tuple[tuple[int, ...]] = ((63,), (3,), (6,), (90,))  # SHOULD BE IN ORDER OF KEYS
    full_pose_size: int = field(init=False, default=SMPL_POSE_SIZE * 3 + FLAME_POSE_SIZE * 3 + MANO_POSE_SIZE * 3)

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        if model_type is not None:
            self.check_model_type(model_type)
            # TODO: this needs testing
            if model_type == "flame":
                _, self.jaw, self.eyes = pose_tensor.split([3, 3, 6], dim=-1)
                return
            if model_type != "smplx":
                msg = f"{model_type.capitalize} pose loading to SMPL-X poses is not possible."
                raise ValueError(msg)

        self.body, self.jaw, self.eyes, self.hand = pose_tensor.split(
            [SMPL_POSE_SIZE * 3, 3, 6, MANO_POSE_SIZE * 3], dim=-1
        )


class ShapeComponents(LimitedAttrTensorDictWithDefaults):
    valid_attr_keys: tuple[str, ...] = ("betas", "expression", "dmpls")  # SHOULD BE IN ORDER FOR CONCATENATION

    def __init__(
        self,
        source: T | dict[NestedKey, CompatibleType] = None,
        use_expression: bool = False,
        use_dmpl: bool = False,
        batch_size: Sequence[int] | Size | int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool | None = None,
        lock: bool = False,
        **kwargs: dict[str, Any] | None,
    ) -> None:
        super().__init__(source, batch_size, device, names, non_blocking, lock, **kwargs)
        self.use_expression = use_expression
        self.use_dmpl = use_dmpl

    def to_tensor(self, default_pose: Pose | None = None) -> Tensor:
        return torch.cat(
            [
                self.get(attr_name, default=default_pose[attr_name].expand(self.batch_size[0], -1))
                for attr_name in self.valid_attr_keys
            ],
            dim=-1,
        )

    def to_tensor(self) -> Tensor:
        return torch.cat([self[attr_name] for attr_name in self.valid_attr_keys], dim=-1)

    def to_tensor(self) -> Tensor:
        components = [self["betas"]]
        if self.use_expression:
            components.append(self["expression"])
        if self.use_dmpl:
            components.append(self["dmpls"])

        return torch.cat(components, dim=-1)

    def from_tensor(self, shape_components_tensor: Tensor) -> None:
        # if self.use_expression:
        #     self.expression = shape_components_tensor[:, : self.expression_size]
        #     full_pose = shape_components_tensor[:, self.expression_size :]
        # if self.use_dmpl:
        #     self.dmpls = shape_components_tensor[:, : self.dmpls_size]
        #     full_pose = shape_components_tensor[:, self.dmpls_size :]

        # self.betas = shape_components_tensor
        raise NotImplementedError

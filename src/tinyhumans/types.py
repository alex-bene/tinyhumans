"""Types for TinyHumans."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum

import torch
from torch import Tensor

from src.tinyhumans.tensor_dict import TensorDict

MANO_POSE_SIZE = 15
SMPL_POSE_SIZE = 21
FLAME_POSE_SIZE = 4


class BetterStringEnum(str, Enum):
    """Case-insetitive string Enum with more descriptive error message for missing values."""

    @classmethod
    def _missing_(cls, value: str) -> BetterStringEnum:
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value.lower() == value:
                    return member

        msg = f"{value!r} is not a valid {cls.__name__}. Valid types are: {', '.join([repr(m.value) for m in cls])}."
        raise ValueError(msg)


class ModelType(BetterStringEnum):
    SMPL = "smpl"
    SMPLH = "smplh"
    SMPLX = "smplx"
    MANO = "mano"
    FLAME = "flame"
    Animal_Horse = "animal_horse"
    Animal_Dog = "animal_dog"
    Animal_Rat = "animal_rat"


@dataclass
class Pose(TensorDict):
    model_type: ModelType | None = None

    def to_tensor(self, default_pose: Pose | None = None) -> Tensor:
        attrs_to_concat = []
        all_attributes = [f.name for f in fields(self)]
        if "eyes" in all_attributes:  # SMPLX or FLAME
            attrs_to_concat = ["body", "jaw", "eyes"]
        elif "body" in all_attributes:  # SMPL, SMPLH, Animal_Horse, Animal_Dog, Animal_Rat
            attrs_to_concat = ["body"]
        if "hand" in all_attributes:  # SMPL, SMPLH, SMPLX, MANO
            attrs_to_concat.append("hand")

        return torch.cat(
            [
                (
                    getattr(self, attr_name)
                    if getattr(self, attr_name) is not None
                    else getattr(default_pose, attr_name).expand(self.batch_size, -1)
                )
                for attr_name in attrs_to_concat
            ],
            dim=-1,
        )

    def from_tensor(self, pose_tensor: Tensor, model_type: ModelType | str | None = None) -> None:
        raise NotImplementedError


@dataclass
class SMPLPose(Pose):
    body: Tensor | None = None
    hand: Tensor | None = None
    model_type: ModelType = field(init=False, default=ModelType.SMPL)
    full_pose_size: int = field(init=False, default=SMPL_POSE_SIZE * 3)

    def from_tensor(self, pose_tensor: Tensor, model_type: ModelType | str | None = None) -> None:
        if model_type is not None:
            model_type = ModelType(model_type)
            # TODO: this needs testing
            if model_type.value == "smplh":
                self.body, self.hand = pose_tensor.split([SMPL_POSE_SIZE * 3, MANO_POSE_SIZE * 3], dim=-1)
                self.hand = torch.cat([self.hand[:3], self.hand[MANO_POSE_SIZE * 3 : (MANO_POSE_SIZE + 1) * 3]], dim=-1)
                return
            if model_type.value == "mano":
                self.hand = torch.cat(
                    [pose_tensor[:3], pose_tensor[MANO_POSE_SIZE * 3 : (MANO_POSE_SIZE + 1) * 3]], dim=-1
                )
                return
            if model_type.value != "smpl":
                msg = f"{model_type.capitalize} pose loading to SMPL poses is not possible."
                raise ValueError(msg)

        self.body, self.hand = pose_tensor.split([63, 6], dim=-1)


@dataclass
class MANOPose(Pose):
    hand: Tensor | None = None
    model_type: ModelType = field(init=False, default=ModelType.MANO)
    full_pose_size: int = field(init=False, default=MANO_POSE_SIZE * 3)

    def from_tensor(self, pose_tensor: Tensor, model_type: ModelType | str | None = None) -> None:
        if model_type is not None:
            model_type = ModelType(model_type)
            # TODO: this needs testing
            if model_type.value == "smplh":
                _, self.hand = pose_tensor.split([SMPL_POSE_SIZE * 3, MANO_POSE_SIZE * 3], dim=-1)
                return
            if model_type.value != "mano":
                msg = f"{model_type.capitalize} pose loading to MANO poses is not possible."
                raise ValueError(msg)

        self.hand = pose_tensor


@dataclass
class SMPLHPose(SMPLPose):
    model_type: ModelType = field(init=False, default=ModelType.SMPLH)
    full_pose_size: int = field(init=False, default=(SMPL_POSE_SIZE + MANO_POSE_SIZE) * 3)

    def from_tensor(self, pose_tensor: Tensor, model_type: ModelType | str | None = None) -> None:
        if model_type is not None:
            model_type = ModelType(model_type)
            # TODO: this needs testing
            if model_type.value == "smpl":
                self.body, smpl_hand = pose_tensor.split([SMPL_POSE_SIZE * 3, 6], dim=-1)
                self.hand = torch.zeros([MANO_POSE_SIZE * 3, 3], dtype=pose_tensor.dtype, device=pose_tensor.device)
                self.hand[:3] = smpl_hand[:3]
                self.hand[MANO_POSE_SIZE * 3 : (MANO_POSE_SIZE + 1) * 3] = smpl_hand[3:]
                return
            if model_type.value == "mano":
                self.hand = pose_tensor
                return
            if model_type.value != "smplh":
                msg = f"{model_type.capitalize} pose loading to SMPL-H poses is not possible."
                raise ValueError(msg)

        self.body, self.hand = pose_tensor.split([SMPL_POSE_SIZE * 3, MANO_POSE_SIZE * 3], dim=-1)


@dataclass
class FLAMEPose(Pose):
    body: Tensor | None = None
    jaw: Tensor | None = None
    eyes: Tensor | None = None
    model_type: ModelType = field(init=False, default=ModelType.FLAME)
    full_pose_size: int = field(init=False, default=FLAME_POSE_SIZE * 3)

    def from_tensor(self, pose_tensor: Tensor, model_type: ModelType | str | None = None) -> None:
        if model_type is not None:
            model_type = ModelType(model_type)
            # TODO: this needs testing
            if model_type.value == "smplx":
                body, self.jaw, self.eyes, _ = pose_tensor.split([SMPL_POSE_SIZE * 3, 3, 6, MANO_POSE_SIZE * 3], dim=-1)
                # TODO: no idea which part of the body flame is supposed to take
                self.body = body[:3]
                return
            if model_type.value != "flame":
                msg = f"{model_type.capitalize} pose loading to SMPL-X poses is not possible."
                raise ValueError(msg)

        self.body, self.jaw, self.eyes = pose_tensor.split([3, 3, 6], dim=-1)


@dataclass
class SMPLXPose(FLAMEPose):
    hand: Tensor | None = None
    model_type: ModelType = field(init=False, default=ModelType.SMPLX)
    full_pose_size: int = field(init=False, default=SMPL_POSE_SIZE * 3 + FLAME_POSE_SIZE * 3 + MANO_POSE_SIZE * 3)

    def from_tensor(self, pose_tensor: Tensor, model_type: ModelType | str | None = None) -> None:
        if model_type is not None:
            model_type = ModelType(model_type)
            # TODO: this needs testing
            if model_type.value == "flame":
                _, self.jaw, self.eyes = pose_tensor.split([3, 3, 6], dim=-1)
                return
            if model_type.value != "smplx":
                msg = f"{model_type.capitalize} pose loading to SMPL-X poses is not possible."
                raise ValueError(msg)

        self.body, self.jaw, self.eyes, self.hand = pose_tensor.split(
            [SMPL_POSE_SIZE * 3, 3, 6, MANO_POSE_SIZE * 3], dim=-1
        )


@dataclass
class ShapeComponents(TensorDict):
    betas: torch.Tensor | None = None
    expression: torch.Tensor | None = None
    dmpls: torch.Tensor | None = None
    use_expression: bool = False
    use_dmpl: bool = False

    def to_tensor(self, default_shape_components: ShapeComponents | None = None) -> Tensor:
        dsc = default_shape_components
        components = [self.betas if self.betas is not None else dsc.betas]
        if self.use_expression:
            components.append(self.expression if self.expression is not None else dsc.expression)
        if self.use_dmpl:
            components.append(self.dmpls if self.dmpls is not None else dsc.dmpls)

        bs = max(len(comp) for comp in components if comp is not None)
        components = [comp.expand(bs, -1) if comp is not None else comp for comp in components]

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

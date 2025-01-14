"""TensorDict class for TinyHumans."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, overload

import torch
from torch import Tensor, is_tensor
from torch import is_floating_point as is_float_tensor

from src.tinyhumans.tools import get_logger

logger = get_logger(__name__)


@dataclass
class TensorDict:
    batch_size: int = field(init=False, default=None)
    device: torch.device = field(init=False, default=None)
    dtype: torch.dtype = field(init=False, default=None)
    _post_init: bool = field(init=False, default=False)

    @classmethod
    def create(
        cls: type[TensorDict],
        data: dict | TensorDict | torch.Tensor | None = None,
        use_expression: bool = False,
        use_dmpl: bool = False,
    ) -> TensorDict:
        kwargs = {"use_expression": use_expression, "use_dmpl": use_dmpl} if cls.__name__ == "ShapeComponents" else {}
        if isinstance(data, dict):
            res = cls(**data, **kwargs)
        elif is_tensor(data):
            res = cls(**kwargs)
            res.from_tensor(data)
        elif data is None:
            res = cls(**kwargs)
        elif isinstance(data, cls):
            res = data
        else:
            msg = f"`data` must be of type {cls}, dict, torch.Tensor or None."
            raise ValueError(msg)

        return res

    def __post_init__(self) -> None:
        all_attrs = [f.name for f in fields(self)]
        tensor_attrs = [attr_name for attr_name in all_attrs if is_tensor(getattr(self, attr_name))]

        if not tensor_attrs:
            return

        bszs = [len(getattr(self, attr_name)) for attr_name in tensor_attrs]
        dvcs = [getattr(self, attr_name).device for attr_name in tensor_attrs]
        dtps_fp = [
            getattr(self, attr_name).dtype for attr_name in tensor_attrs if is_float_tensor(getattr(self, attr_name))
        ]
        dtps_int = [
            getattr(self, attr_name).dtype
            for attr_name in tensor_attrs
            if not is_float_tensor(getattr(self, attr_name))
        ]

        if not all(bsz in [max(bszs), 1] for bsz in bszs):
            msg = "All tensors must have the same batch size or have a batch size of 1."
            raise ValueError(msg)
        if not all(dvc == dvcs[0] for dvc in dvcs):
            msg = "All tensors must have the same device."
            raise ValueError(msg)
        if not all(dtp == dtps_fp[0] for dtp in dtps_fp):
            msg = "All floating point tensors must have the same dtype."
            raise ValueError(msg)
        if not all(dtp == dtps_int[0] for dtp in dtps_int):
            msg = "All integer tensors must have the same dtype."
            raise ValueError(msg)

        self.batch_size = max(bszs)
        self.dtype = dtps_fp[0]
        self.device = dvcs[0]
        for attr_name in tensor_attrs:
            attr = getattr(self, attr_name)
            if attr is not None:
                setattr(self, attr_name, attr.expand(self.batch_size, -1))

        self._post_init = True

    def __setattr__(self, name: str, value: Any) -> None:
        if is_tensor(value):
            bs = value.shape[0]
            if bs == 1 and self.batch_size is not None:
                value = value.expand(self.batch_size, -1)
            elif self.batch_size not in (bs, None):
                msg = f"Tensor {name} must have the same batch size as the pose or have a batch size of 1."
                raise ValueError(msg)

            if self.device not in (value.device, None):
                value = value.to(self.device)
                logger.warning(f"Moving {name} to {self.device}.")
            if is_float_tensor(value) and self.dtype not in (value.dtype, None):
                value = value.to(self.dtype)
                logger.warning(f"Converting {name} to {self.dtype}.")

            if self._post_init:
                if self.device is None:
                    self.device = value.device
                if self.dtype is None:
                    self.dtype = value.dtype
                if self.batch_size is None:
                    self.batch_size = value.shape[0]

        super().__setattr__(name, value)

    def __len__(self) -> int:
        return self.batch_size

    @overload
    def to(self, dtype: torch.dtype, copy: bool = False) -> TensorDict:
        self.to(dtype=dtype, device=None, copy=copy)

    @overload
    def to(self, tensor: Tensor, copy: bool = False) -> TensorDict:
        self.to(dtype=tensor.dtype, device=tensor.device, copy=copy)

    def to(
        self, device: torch.device | str | int | None = ..., dtype: torch.dtype | None = ..., copy: bool = False
    ) -> TensorDict:
        if self.device == torch.device(device) and self.dtype == dtype and not copy:
            return self

        inputs = {}
        for attr_field in fields(self):
            attr_name = attr_field.name
            if not attr_field.init:
                continue
            attr = getattr(self, attr_name)
            inputs[attr_name] = attr
            if is_tensor(attr):
                inputs[attr_name] = attr.to(device=device, dtype=dtype, copy=copy)

        return self.__class__(**inputs)

    def to_tensor(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def from_tensor(self, data: Tensor, *args, **kwargs) -> None:
        raise NotImplementedError

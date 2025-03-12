"""Shape types for TinyHumans.

This module defines custom types and classes for working with human body models in TinyHumans. It includes classes for
shape components (ShapeComponents).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from tensordict import NonTensorData
from torch import Size

from tinyhumans.datatypes import AutoTensorDict, LimitedAttrTensorDictWithDefaults

if TYPE_CHECKING:
    from tensordict.base import CompatibleType, T
    from tensordict.utils import DeviceType, IndexType
    from torch import Tensor

    NestedKey = str | tuple["NestedKeyType", ...]  # type: ignore  # noqa: F821, PGH003


class ShapeComponents(LimitedAttrTensorDictWithDefaults):
    """Shape components for body models.

    This class extends LimitedAttrTensorDictWithDefaults and defines the shape components used in body models,
    including shape parameters (betas), expression parameters, and DMPL parameters.

    Attributes:
        valid_attr_keys (tuple[str, ...]): Valid attribute keys for ShapeComponents ("betas", "expression", "dmpls").

    """

    default_attr_sizes: ClassVar[dict[tuple[int, ...] | int]] = {
        "betas": 10,
        "expression": 5,
        "dmpls": 8,
        "use_expression": False,
        "use_dmpl": False,
    }

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
        """Initialize ShapeComponents.

        Args:
            source (T | dict[NestedKey, CompatibleType], optional): Source data. Defaults to None.
            use_expression (bool, optional): Whether to use expression parameters. Defaults to False.
            use_dmpl (bool, optional): Whether to use DMPL parameters. Defaults to False.
            batch_size (Sequence[int] | Size | int | None, optional): Batch size. Defaults to None.
            device (DeviceType | None, optional): Device. Defaults to None.
            names (Sequence[str] | None, optional): Names. Defaults to None.
            non_blocking (bool | None, optional): Non-blocking flag. Defaults to None.
            lock (bool, optional): Lock flag. Defaults to False.
            **kwargs (dict[str, Any] | None): Keyword arguments. Defaults to None.

        """
        if (source is not None) and kwargs:
            msg = "Either a dictionary or a sequence of kwargs must be provided, not both."
            raise ValueError(msg)
        source = kwargs if kwargs else source
        if source is not None:
            self.check_keys(list(source.keys()), use_expression, use_dmpl)
        if isinstance(source, dict):
            source = {key.lower(): value for key, value in source.items()}
        if source is None:
            source = {}
        source["use_expression"] = use_expression
        source["use_dmpl"] = use_dmpl
        AutoTensorDict.__init__(self, source, batch_size, device, names, non_blocking, lock)
        self.use_expression = self["use_expression"]
        self.use_dmpl = self["use_dmpl"]

    def check_keys(
        self, keys: str | Sequence[str], use_expression: bool | None = None, use_dmpl: bool | None = None
    ) -> None:
        """Check if keys are valid for the class.

        Args:
            keys (str | Sequence[str]): Key or sequence of keys to check.
            use_expression (bool | None, optional): If not None, use this value for use_expression instead of the
                value in the class. Defaults to None.
            use_dmpl (bool | None, optional): If not None, use this value for use_dmpl instead of the value in the
                class. Defaults to None.

        Raises:
            KeyError: If any key is not in valid_attr_keys.

        """
        if keys in ["use_expression", "use_dmpl"]:
            return

        if use_expression is None:
            use_expression = self.use_expression
        if use_dmpl is None:
            use_dmpl = self.use_dmpl

        valid_attr_keys = {"betas"}
        if use_expression:
            valid_attr_keys.add("expression")
        if use_dmpl:
            valid_attr_keys.add("dmpls")
        valid_attr_keys.update({"use_expression", "use_dmpl"})

        msg = None
        if isinstance(keys, str) and (keys.lower() not in valid_attr_keys):
            msg = f"Key {keys} is not a valid key for {self.__class__.__name__}"
        elif (
            not isinstance(keys, str)
            and isinstance(keys, Sequence)
            and not all(key.lower() in valid_attr_keys for key in keys)
        ):
            msg = f"Sequence {keys} do not contain valid keys for {self.__class__.__name__}"

        if not msg:
            return

        msg += f" with use_expression={use_expression} and use_dmpl={use_dmpl}."
        msg += f"Valid keys are {valid_attr_keys!r}."
        raise KeyError(msg)

    def to_tensor(self) -> Tensor:
        """Convert shape components to a tensor.

        Returns:
            Tensor: Concatenated tensor of shape components.

        """
        components = [self["betas"]]
        if self.use_expression:
            components.append(self["expression"])
        if self.use_dmpl:
            components.append(self["dmpls"])

        return torch.cat(components, dim=-1)

    def from_tensor(self, shape_components_tensor: Tensor) -> None:
        """Load shape components from a tensor.

        Args:
            shape_components_tensor (Tensor): Input shape components tensor.

        """
        betas_size = self.default_attr_sizes["betas"]
        self.betas = shape_components_tensor[:, :betas_size]
        if self.use_expression:
            self.expression = shape_components_tensor[
                :, betas_size : betas_size + self.default_attr_sizes["expression"]
            ]
        if self.use_dmpl:
            self.dmpls = shape_components_tensor[:, -self.default_attr_sizes["dmpls"] :]

    def __setattr__(self, key: IndexType, value: Any) -> None:
        """Set item in ShapeComponents.

        Overrides TensorDict.__setitem__ to automatically infer device and batch size.

        Args:
            key (IndexType): Key to set.
            value (Any): Value to set.

        """
        if key in ["use_expression", "use_dmpl"]:
            value = NonTensorData(value)
        super().__setattr__(key, value)

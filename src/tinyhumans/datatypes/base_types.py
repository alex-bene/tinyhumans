"""Types for TinyHumans.

This module defines custom types and classes for working with human body models in TinyHumans. It includes base utility
classes like AutoTensorDict and LimitedAttrTensorDictWithDefaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from tensordict.tensordict import TensorDict
from torch import Size

if TYPE_CHECKING:
    from tensordict.base import CompatibleType, T
    from tensordict.utils import DeviceType, IndexType

    NestedKey = str | tuple["NestedKeyType", ...]  # type: ignore  # noqa: F821, PGH003


MANO_POSE_SIZE = 15
SMPL_POSE_SIZE = 21
FLAME_POSE_SIZE = 4


class AutoTensorDict(TensorDict):
    """TensorDict with automatic device and batch size inference.

    This class extends TensorDict to automatically infer device and batch size from the input tensors. It initializes
    with a batch size of 1 and infers the device if not explicitly provided.
    """

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
        """Initialize AutoTensorDict.

        Args:
            source (T | dict[NestedKey, CompatibleType], optional): Source data to initialize the TensorDict.
                Defaults to None.
            batch_size (Sequence[int] | Size | int | None, optional): Batch size of the TensorDict. Defaults to None.
            device (DeviceType | None, optional): Device of the TensorDict. Defaults to None.
            names (Sequence[str] | None, optional): Names of the TensorDict. Defaults to None.
            non_blocking (bool | None, optional): Non-blocking flag. Defaults to None.
            lock (bool, optional): Lock flag. Defaults to False.
            **kwargs (dict[str, Any] | None): Additional keyword arguments. Defaults to None.

        """
        super().__init__(source, batch_size, device, names, non_blocking, lock, **kwargs)
        self.auto_batch_size_(1)
        if self.device is None:
            self.auto_device_()

    def __setitem__(self, key: IndexType, value: Any) -> None:
        """Set item in AutoTensorDict.

        Overrides TensorDict.__setitem__ to automatically infer device and batch size.

        Args:
            key (IndexType): Key to set.
            value (Any): Value to set.

        """
        super().__setitem__(key, value)
        if self.device is None:
            self.auto_device_()
        if not self.batch_size:
            self.auto_batch_size_(1)


class LimitedAttrTensorDictWithDefaults(AutoTensorDict):
    """TensorDict with limited attributes and default values.

    This class extends AutoTensorDict to provide a TensorDict with a predefined set of valid attribute keys and default
    values for missing attributes. It ensures that only valid keys are used and provides default zero tensors for
    missing attributes when accessed.

    Attributes:
        valid_attr_keys (tuple[str, ...]): Tuple of valid attribute keys.
        valid_attr_sizes (tuple[tuple[int, ...]]): Tuple of sizes for each valid attribute key.

    Raises:
        ValueError: If both source dictionary and kwargs are provided during initialization.
        ValueError: If valid_attr_sizes is not set with sizes for each valid_attr_keys.
        KeyError: If an invalid key is used to set or get an item.

    """

    default_attr_sizes: ClassVar[dict[tuple[int, ...] | int]] = {}

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
        """Initialize LimitedAttrTensorDictWithDefaults.

        Args:
            source (T | dict[NestedKey, CompatibleType], optional): Source data to initialize. Defaults to None.
            batch_size (Sequence[int] | Size | int | None, optional): Batch size. Defaults to None.
            device (DeviceType | None, optional): Device. Defaults to None.
            names (Sequence[str] | None, optional): Names. Defaults to None.
            non_blocking (bool | None, optional): Non-blocking flag. Defaults to None.
            lock (bool, optional): Lock flag. Defaults to False.
            **kwargs (dict[str, Any] | None): Keyword arguments for initialization. Defaults to None.

        Raises:
            ValueError: If both source dictionary and kwargs are provided.

        """
        if (source is not None) and kwargs:
            msg = "Either a dictionary or a sequence of kwargs must be provided, not both."
            raise ValueError(msg)
        source = kwargs if kwargs else source
        if source is not None:
            self.check_keys(list(source.keys()))
        if isinstance(source, dict):
            source = {key.lower(): value for key, value in source.items()}
        super().__init__(source, batch_size, device, names, non_blocking, lock)

    def __getitem__(self, key: IndexType) -> Any:
        """Get item with default value if missing.

        Overrides TensorDict.__getitem__ to return a default zero tensor if the key is in valid_attr_keys but not
        present in the TensorDict.

        Args:
            key (IndexType): Key to get.

        Returns:
            Any: Value associated with the key, or a default zero tensor if missing.

        Raises:
            ValueError: If valid_attr_sizes is not properly configured.

        """
        if key in self.default_attr_sizes and self.get(key, None) is None:
            # Make a (batch_size, *attr_size) tensor of zeros using memory for just one element
            attr_size = self.default_attr_sizes.get(key, None)
            if attr_size is None:
                msg = f"Attribute {key} not set and not found in `default_attr_sizes` to auto-initialize."
                raise ValueError(msg)
            if isinstance(attr_size, int):
                attr_size = (attr_size,)
            if self.batch_size is None or len(self.batch_size) == 0:
                self.batch_size = Size([1])
            self[key] = torch.zeros([1] * len(attr_size), dtype=self.dtype, device=self.device).expand(
                self.batch_size[0], *attr_size
            )
        return super().__getitem__(key)

    def __setitem__(self, key: IndexType, value: Any) -> None:
        """Set item, checking for valid keys.

        Overrides TensorDict.__setitem__ to check if the key is valid before setting the item.

        Args:
            key (IndexType): Key to set.
            value (Any): Value to set.

        Raises:
            KeyError: If the key is not valid.

        """
        self.check_keys(key)
        super().__setitem__(key, value)

    @classmethod
    def check_keys(cls, keys: str | Sequence[str]) -> None:
        """Check if keys are valid for the class.

        Args:
            keys (str | Sequence[str]): Key or sequence of keys to check.

        Raises:
            KeyError: If any key is not in valid_attr_keys.

        """
        msg = None
        if isinstance(keys, str) and (keys.lower() not in cls.default_attr_sizes):
            msg = f"Key {keys} is not a valid key for {cls.__name__}"
        elif (
            not isinstance(keys, str)
            and isinstance(keys, Sequence)
            and not all(key.lower() in cls.default_attr_sizes for key in keys)
        ):
            msg = f"Sequence {keys} do not contain valid keys for {cls.__name__}"

        if not msg:
            return

        msg += f"Valid keys for {cls.__name__} are {cls.default_attr_sizes!r}."
        raise KeyError(msg)

    def __getattr__(self, name: str) -> Any:
        """Get attribute, accessing TensorDict items as attributes.

        Overrides object.__getattr__ to allow accessing TensorDict items as attributes if the attribute name is in
        valid_attr_keys.

        Args:
            name (str): Attribute name.

        Returns:
            Any: Attribute value or TensorDict item.

        """
        if name in self.default_attr_sizes:
            return self[name]
        # I would expect this to be super().__getattr__(name) there is not __getattr__ in super().
        # Shouldn't this create an infinite loop? weird...
        return self.__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, setting TensorDict items as attributes.

        Overrides object.__setattr__ to allow setting TensorDict items as attributes if the attribute name is in
        valid_attr_keys.

        Args:
            name (str): Attribute name.
            value (Any): Attribute value or TensorDict item value.

        """
        if name in self.default_attr_sizes:
            self[name] = value
            return
        super().__setattr__(name, value)

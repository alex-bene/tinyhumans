"""Types for TinyHumans.

This module defines custom types and classes for working with human body models in TinyHumans. It includes classes for
pose parameters (Pose, SMPLPose, SMPLHPose, SMPLXPose, FLAMEPose, MANOPose) and shape components (ShapeComponents), as
well as utility classes like AutoTensorDict and LimitedAttrTensorDictWithDefaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from tensordict import NonTensorData
from tensordict.tensordict import TensorDict
from torch import Size

if TYPE_CHECKING:
    from tensordict.base import CompatibleType, T
    from tensordict.utils import DeviceType, IndexType
    from torch import Tensor

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

    valid_attr_keys: tuple[str, ...] = ()  # SHOULD BE IN ORDER FOR CONCATENATION
    valid_attr_sizes: tuple[tuple[int, ...]] = ()  # SHOULD BE IN ORDER OF KEYS

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
        if key in self.valid_attr_keys and self.get(key, default=None) is None:
            # Make a (batch_size, *attr_size) tensor of zeros using memory for just one element
            if not self.valid_attr_sizes:
                msg = "valid_attr_sizes tuple must be set with sized for each of the valid_attr_keys: "
                msg += f"{self.valid_attr_keys}."
                raise ValueError(msg)
            attr_size = self.valid_attr_sizes[self.valid_attr_keys.index(key)]
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
        if isinstance(keys, str) and (keys.lower() not in cls.valid_attr_keys):
            msg = f"Key {keys} is not a valid key for {cls.__name__}"
        elif (
            not isinstance(keys, str)
            and isinstance(keys, Sequence)
            and not all(key.lower() in cls.valid_attr_keys for key in keys)
        ):
            msg = f"Sequence {keys} do not contain valid keys for {cls.__name__}"

        if not msg:
            return

        msg += f"Valid keys for {cls.__name__} are {cls.valid_attr_keys!r}."
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
        if name in self.valid_attr_keys:
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
        if name in self.valid_attr_keys:
            self[name] = value
            return
        super().__setattr__(name, value)


class Pose(LimitedAttrTensorDictWithDefaults):
    """Base class for pose parameters.

    This class extends LimitedAttrTensorDictWithDefaults and serves as a base class for different pose representations
    (e.g., SMPLPose, SMPLHPose). It defines valid model types and provides methods for checking model types and
    converting pose parameters to tensors.

    Attributes:
        valid_model_types (tuple[str, ...]): Tuple of valid model types (e.g., "smpl", "smplh").

    Raises:
        NotImplementedError: If model_type property is accessed directly from Pose class.
        ValueError: If an invalid model type is provided.

    """

    valid_model_types: tuple[str, ...] = ("smpl", "smplh", "smplx", "flame", "mano")

    @staticmethod
    def check_model_type(model_type: str | None) -> None:
        """Check if model_type is valid.

        Args:
            model_type (str | None): Model type to check.

        Raises:
            ValueError: if model_type is not valid.

        """
        if model_type is None:
            return

        if model_type.lower() not in Pose.valid_model_types:
            msg = f"{model_type} is not a valid model type."
            msg += f"Valid types are: {', '.join([repr(m) for m in Pose.valid_model_types])}."
            raise ValueError(msg)

    @property
    def model_type(self) -> str:
        """str: Model type inferred from class name."""
        if self.__class__.__name__ == "Pose" or not self.__class__.__name__.endswith("Pose"):
            raise NotImplementedError

        return self.__class__.__name__[:-4].lower()

    def to_tensor(self) -> Tensor:
        """Convert pose parameters to a tensor.

        Returns:
            Tensor: Concatenated tensor of pose parameters.

        """
        return torch.cat([self[attr_name] for attr_name in self.valid_attr_keys], dim=-1)

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.

        """
        raise NotImplementedError


class SMPLPose(Pose):
    """Pose parameters for SMPL model.

    This class extends Pose and defines the pose parameters specific to the SMPL body model, including body pose and
    hand pose.

    Attributes:
        valid_attr_keys (tuple[str, ...]): Valid attribute keys for SMPLPose ("body", "hand").
        valid_attr_sizes (tuple[tuple[int, ...] | int, ...]): Sizes for each attribute key (body: 63, hand: 6).

    """

    valid_attr_keys: tuple[str, ...] = ("body", "hand")  # SHOULD BE IN ORDER FOR CONCATENATION
    valid_attr_sizes: tuple[tuple[int, ...] | int, ...] = (63, 6)  # SHOULD BE IN ORDER OF KEYS

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load SMPL pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            ValueError: If model_type is not compatible or loading to SMPL pose is not possible.

        """
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
    """Pose parameters for MANO model.

    This class extends Pose and defines the pose parameters specific to the MANO hand model, including hand pose.

    Attributes:
        valid_attr_keys (tuple[str, ...]): Valid attribute keys for MANOPose ("hand").
        valid_attr_sizes (tuple[tuple[int, ...] | int, ...]): Sizes for each attribute key (hand: 45).

    """

    valid_attr_keys: tuple[str, ...] = ("hand",)  # SHOULD BE IN ORDER FOR CONCATENATION
    valid_attr_sizes: tuple[tuple[int, ...] | int, ...] = (45,)  # SHOULD BE IN ORDER OF KEYS

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load MANO pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            ValueError: If model_type is not compatible or loading to MANO pose is not possible.

        """
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
    """Pose parameters for SMPL+H model.

    This class extends SMPLPose and defines the pose parameters specific to the SMPL+H body model, including body pose
    and hand pose with more DoF for hands.

    Attributes:
        valid_attr_sizes (tuple[tuple[int, ...] | int, ...]): Sizes for each attribute key (body: 63, hand: 90).

    """

    valid_attr_sizes: tuple[tuple[int, ...] | int, ...] = (63, 90)  # SHOULD BE IN ORDER OF KEYS

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load SMPL+H pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            ValueError: If model_type is not compatible or loading to SMPL+H pose is not possible.

        """
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
    """Pose parameters for FLAME model.

    This class extends Pose and defines the pose parameters specific to the FLAME face model, including body pose,
    jaw pose, and eye pose.

    Attributes:
        valid_attr_keys (tuple[str, ...]): Valid attribute keys for FLAMEPose ("body", "jaw", "eyes").
        valid_attr_sizes (tuple[tuple[int, ...] | int, ...]): Sizes for each attribute key (body: 3, jaw: 3, eyes: 6).

    """

    valid_attr_keys: tuple[str, ...] = ("body", "jaw", "eyes")  # SHOULD BE IN ORDER FOR CONCATENATION
    valid_attr_sizes: tuple[tuple[int, ...] | int, ...] = (3, 3, 6)  # SHOULD BE IN ORDER OF KEYS

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load FLAME pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            ValueError: If model_type is not compatible or loading to FLAME pose is not possible.

        """
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
    """Pose parameters for SMPL-X model.

    This class extends FLAMEPose and defines the pose parameters specific to the SMPL-X body and face model, including
    body pose, jaw pose, eye pose, and hand pose.

    Attributes:
        valid_attr_keys (tuple[str, ...]): Valid attribute keys for SMPLXPose (extended FLAMEPose keys + "hand").
        valid_attr_sizes (tuple[tuple[int, ...] | int, ...]): Sizes for each attribute key (extended FLAMEPose
        sizes + hand: 90).

    """

    valid_attr_keys: tuple[str, ...] = (*FLAMEPose.valid_attr_keys, "hand")  # SHOULD BE IN ORDER FOR CONCATENATION
    valid_attr_sizes: tuple[tuple[int, ...] | int, ...] = (63, 3, 6, 90)  # SHOULD BE IN ORDER OF KEYS

    def from_tensor(self, pose_tensor: Tensor, model_type: str | None = None) -> None:
        """Load SMPL-X pose parameters from a tensor.

        Args:
            pose_tensor (Tensor): Input pose tensor.
            model_type (str | None, optional): Model type. Defaults to None.

        Raises:
            ValueError: If model_type is not compatible or loading to SMPL-X pose is not possible.

        """
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
    """Shape components for body models.

    This class extends LimitedAttrTensorDictWithDefaults and defines the shape components used in body models,
    including shape parameters (betas), expression parameters, and DMPL parameters.

    Attributes:
        valid_attr_keys (tuple[str, ...]): Valid attribute keys for ShapeComponents ("betas", "expression", "dmpls").

    """

    # SHOULD BE IN ORDER FOR CONCATENATION
    valid_attr_keys: tuple[str, ...] = ("betas", "expression", "dmpls", "use_expression", "use_dmpl")

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
        super().__init__(source, batch_size, device, names, non_blocking, lock, **kwargs)
        self["use_expression"] = NonTensorData(use_expression)
        self["use_dmpl"] = NonTensorData(use_dmpl)

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

        Raises:
            NotImplementedError: This method is not implemented.

        """
        # if self.use_expression:
        #     self.expression = shape_components_tensor[:, : self.expression_size]
        #     full_pose = shape_components_tensor[:, self.expression_size :]
        # if self.use_dmpl:
        #     self.dmpls = shape_components_tensor[:, : self.dmpls_size]
        #     full_pose = shape_components_tensor[:, self.dmpls_size :]

        # self.betas = shape_components_tensor
        raise NotImplementedError

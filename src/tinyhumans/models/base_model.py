"""Base model for the tinyhumans library.

This module defines the BaseModel class, which serves as an abstract base class for models in this library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import torch
from tinytools import freeze_model, get_logger
from torch import nn

logger = get_logger(__name__)


class BaseModel(nn.Module):
    """Base class for models in the tinyhumans library."""

    def __init__(self) -> None:
        super().__init__()
        self.training: bool = True

    @property
    def device(self) -> torch.device:
        """torch.device: Device on which the model is loaded."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """torch.dtype: Data type of the model parameters."""
        return next(self.parameters()).dtype

    def train(self, mode: bool = True) -> Self:
        """Set the module in training mode.

        Args:
            mode (bool): If True, set the module in training mode. Otherwise, set it in evaluation mode.

        Returns:
            Self: The module in the specified mode.

        """
        super().train(mode)
        self.training = mode
        return self

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | Path,
        *model_args,
        device_map: str | torch.device = "auto",
        torch_dtype: torch.dtype = torch.float32,
        remove_key_prefix: str | None = None,
        replace_key_prefixes: dict[str, str] | None = None,
        **model_kwargs,
    ) -> Self:
        """Load a pre-trained model.

        Args:
            pretrained_model_path (str | Path): Path to the pretrained model.
            *model_args: Positional arguments to pass to the model constructor.
            device_map (str | torch.device): Device to map the model weights to.
            torch_dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.
            remove_key_prefix (str | None): Key prefix to remove in model weights.
            replace_key_prefixes (dict[str, str] | None): Key prefixes to replace in model weights.
            **model_kwargs: Keyword arguments to pass to the model constructor.

        Returns:
            Self: A pre-trained model instance.

        """
        if cls == BaseModel:
            msg = "`BaseModel` should only be instantiated through a subclass"
            raise NotImplementedError(msg)

        # Check pretrained model path
        pretrained_model_path: Path = Path(pretrained_model_path)
        if not pretrained_model_path.exists():
            msg = f"Could not find the pretrained model path: {pretrained_model_path}"
            raise ValueError(msg)

        # Infer device type
        if device_map == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        elif isinstance(device_map, str):
            device = torch.device(device_map)
        elif isinstance(device_map, torch.device):
            device = device_map
        else:
            msg = f"Invalid device_map: {device_map}"
            raise ValueError(msg)

        state_dict: dict[str, torch.Tensor] = torch.load(pretrained_model_path, map_location=device, weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Remove key prefix
        if remove_key_prefix is not None:
            state_dict = {
                k.replace(remove_key_prefix, "") if k.startswith(remove_key_prefix) else k: v
                for k, v in state_dict.items()
            }

        # Sequentially replace key prefixes
        if replace_key_prefixes is not None:
            for old_prefix, new_prefix in replace_key_prefixes.items():
                state_dict = {
                    k.replace(old_prefix, new_prefix) if k.startswith(old_prefix) else k: v
                    for k, v in state_dict.items()
                }

        # Convert to the specified data type
        if torch_dtype is not None:
            if torch_dtype == torch.float64 and device == "mps":
                logger.warning("Float64 precision is not supported on MPS. Converting to float32 instead.")
                torch_dtype = torch.float32
            state_dict = {k: v.to(torch_dtype) if torch.is_floating_point(v) else v for k, v in state_dict.items()}

        # Create model instance
        model_instance = cls(*model_args, **model_kwargs)

        # Load state dict
        instance_model_keys = list(model_instance.state_dict().keys())
        state_dict = {k: v for k, v in state_dict.items() if k in instance_model_keys}
        model_instance.load_state_dict(state_dict)

        # Freeze and set evaluation mode
        freeze_model(model_instance)
        model_instance.eval()

        logger.info("Loaded model in evaluation mode from pre-trained weights file: %s", pretrained_model_path)

        return model_instance.to(device, torch_dtype)

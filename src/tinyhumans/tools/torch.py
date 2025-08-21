"""Pytorch related tools."""

import torch
from torch import Tensor


def validate_tensor_shape(
    value: Tensor | None, ndims: int, shape: list[int | str], tensor_name: str | None = None
) -> None:
    """Validate Tensor object by shape."""
    if value is None:
        return
    if len(value.shape) != ndims:
        msg = f"shape argument should be a list of length {ndims}"
        raise ValueError(msg)
    for i, dim_size in enumerate(shape):
        if isinstance(dim_size, str):
            if "-" in dim_size:
                dim_range = dim_size.split("-")
                dim_range = range(int(dim_range[0]), int(dim_range[1]) + 1)
            elif "," in dim_size:
                dim_range = [int(d) if "None" not in d else None for d in dim_size.split(",")]
                if None in dim_range:
                    dim_range = [value.shape[i]]
            else:
                dim_range = [int(dim_size)]
        if (isinstance(dim_size, int) and value.shape[i] != dim_size) or (
            isinstance(dim_size, str) and value.shape[i] not in dim_range
        ):
            msg = f"{tensor_name} shape should be {shape}"
            raise ValueError(msg)


def freeze_model(model: torch.nn.Module) -> None:
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False

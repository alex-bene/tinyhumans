"""Colormaps for TinyHumans."""

from __future__ import annotations

import numpy as np


def get_jet_colormap(num_colors: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """Generate RGB values for a jet colormap equivalent.

    Args:
        num_colors(int): The number of colors to generate.
        dtype(np.dtype): The data type of the RGB values.

    Returns:
        np.ndarray: An array of RGB values for the jet colormap.

    """
    values = np.linspace(0, 1, num_colors)

    r = np.clip(1.5 - 4 * np.abs(values - 0.75), 0, 1)
    g = np.clip(1.5 - 4 * np.abs(values - 0.5), 0, 1)
    b = np.clip(1.5 - 4 * np.abs(values - 0.25), 0, 1)
    colors = np.stack([r, g, b], axis=-1)

    return (colors * 255).astype(dtype) if np.issubdtype(dtype, np.integer) else colors.astype(dtype)

"""Image tools for TinyHumans."""

from __future__ import annotations

import numpy as np
from PIL import Image


def img_from_array(img_array: np.ndarray, is_bgr: bool = False) -> Image.Image:
    """Convert a NumPy array representing image(s) to a list of PIL Image objects.

    Args:
        img_array (np.ndarray): A NumPy array representing the image(s). The array can have the following shapes:
            - (H, W, C): A color image, where H is height, W is width, and C is the number of channels.
            - (H, W): A grayscale image, where H is height and W is width.
        is_bgr (bool): A boolean indicating whether the input images are in BGR format.
            If True, the function will convert them to RGB. Defaults to False (assumes RGB or grayscale).

    Returns:
        Image.Image: A PIL Image.

    Raises:
        ValueError: If the number of dimensions in the input array is not 2D or 3D.

    """
    if is_bgr:
        img_array = np.flip(img_array, axis=-1)  # Convert BGR to RGB

    img_array = img_array.squeeze()  # Remove singleton dimensions

    if len(img_array.shape) < 2 or len(img_array.shape) > 3:
        msg = f"Invalid number of dimensions {len(img_array.shape)} for image array."
        raise ValueError(msg)

    # ensure correct type for all cases before doing anything
    img_array = img_array.astype(np.uint8)

    return Image.fromarray(img_array)


def imgs_from_array_batch(img_array_batch: np.ndarray, is_bgr: bool = False) -> list[Image.Image]:
    """Convert a NumPy array representing a batch of image(s) to a list of PIL Image objects.

    Args:
        img_array_batch (np.ndarray): An array representing the batch of images with one of the following shapes:
            - (N, H, W, C): A batch of color images, where N is the batch size, H is height, W is width, and C is the
                number of channels (e.g., 3 for RGB).
            - (N, H, W): A batch of grayscale images, where N is the batch size, H is height, and W is width.
        is_bgr (bool): A boolean indicating whether the input images are in BGR format.
            If True, the function will convert them to RGB. Defaults to False (assumes RGB or grayscale).

    Returns:
        list[Image.Image]: A list of PIL Image objects.

    Raises:
        ValueError: If the number of dimensions in the input array is not 3D or 4D.

    """
    if len(img_array_batch.shape) < 3 or len(img_array_batch.shape) > 4:
        msg = (
            f"Invalid number of dimensions {len(img_array_batch.shape)} for batch image array. "
            "Must be at least 3D (N, H, W) or 4D (N, H, W, C)."
        )
        raise ValueError(msg)

    return [img_from_array(img_array, is_bgr=is_bgr) for img_array in img_array_batch]


def image_grid(imgs: list[Image.Image], rows: int = 1, cols: int = 1) -> Image.Image:
    """Create a grid of images.

    Args:
        imgs (list[Image.Image]): A list of PIL Image objects representing the images to be arranged in a grid.
        rows (int): The number of rows in the grid.
        cols (int): The number of columns in the grid.

    Returns:
        Image.Image: A single PIL Image object representing the grid of images.

    Raises:
        ValueError: If the number of images does not match the number of rows and columns.

    """
    if len(imgs) != rows * cols:
        msg = "Number of images must match the number of rows and columns."
        raise ValueError(msg)

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene


def img_from_array(img_array: np.ndarray, is_bgr: bool = False) -> Image.Image:
    """
    Converts a NumPy array representing image(s) to a list of PIL Image objects.

    Args:
        img_array (np.ndarray): A NumPy array representing the image(s). The array can have the following shapes:
            - (H, W, C): A color image, where H is height, W is width, and C is the number of channels (e.g., 3 for RGB).
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
    """
    Converts a NumPy array representing a batch of image(s) to a list of PIL Image objects.

    Args:
        img_array_batch (np.ndarray): A NumPy array representing the batch of images. The array can have the following shapes:
            - (N, H, W, C): A batch of color images, where N is the batch size, H is height, W is width, and C is the number of channels (e.g., 3 for RGB).
            - (N, H, W): A batch of grayscale images, where N is the batch size, H is height, and W is width.
        is_bgr (bool): A boolean indicating whether the input images are in BGR format.
            If True, the function will convert them to RGB. Defaults to False (assumes RGB or grayscale).

    Returns:
        list[Image.Image]: A list of PIL Image objects.

    Raises:
        ValueError: If the number of dimensions in the input array is not 3D or 4D.
    """

    if len(img_array_batch.shape) < 3 or len(img_array_batch.shape) > 4:
        msg = f"Invalid number of dimensions {len(img_array_batch.shape)} for batch image array. Must be at least 3D (N, H, W) or 4D (N, H, W, C)."
        raise ValueError(msg)

    return [img_from_array(img_array, is_bgr=is_bgr) for img_array in img_array_batch]


def image_grid(imgs: list[Image.Image], rows: int = 1, cols: int = 1) -> Image.Image:
    """
    Create a grid of images.

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


def plot_meshes(
    meshes: Meshes,
    join: bool = True,
    join_direction: str = "right",
    colors_override=None,
    show: bool = True,
    num_columns: int = -1,
    subplot_size: int = 1000,
    subplot_titles: list | str | None = None,
    viewpoint_cameras: CamerasBase | None = None,
):
    """
    Plots a batch of meshes, optionally joining them in a single plot or displaying them individually in a grid.

    Args:
        meshes (pytorch3d.Meshes): A Meshes object containing the meshes to plot.
        join (bool, optional): If True, joins all meshes into a single plot. If False, plots each mesh individually in
            subplots. Defaults to True.
        join_direction (str, optional): The direction in which to join meshes when `join` is True.
            Options are "right", "left", "backward", "forward", and "diagonal". Defaults to "right".
        colors_override (array-like, optional):  An array-like object representing the colors to override mesh colors,
            broadcasted to all meshes. Defaults to None.
        show (bool, optional): If True, displays the plot. Defaults to True.
        num_columns (int, optional): The number of columns to use when plotting meshes individually.
            If negative, defaults to a maximum of 3 or the number of meshes if less than 3. Defaults to -1.
        subplot_size (int, optional): The size (width and height) of each subplot in pixels. Defaults to 1000.
        subplot_titles (list, optional): A list of titles to use for each subplot when plotting meshes individually.
            Defaults to None.
        viewpoint_cameras (CamerasBase, optional): A CamerasBase object to set the viewpoint of the plot.
            Defaults to None.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object.

    Raises:
        ValueError: If `join_direction` is not one of the valid options when `join` is True.
    """
    num_meshes = len(meshes)

    if join_direction == "right":
        shift = torch.tensor([1.0, 0.0, 0.0])
    elif join_direction == "left":
        shift = torch.tensor([-1.0, 0.0, 0.0])
    elif join_direction == "backward":
        shift = torch.tensor([0.0, 0.0, 1.0])
    elif join_direction == "forward":
        shift = torch.tensor([0.0, 0.0, -1.0])
    elif join_direction == "diagonal":
        shift = torch.tensor([1.0, 0.0, -1.0])
    elif join:
        raise ValueError('`join_direction` must be one of "right", "left", "backward", "forward", "diagonal"')

    if colors_override is not None:
        if not isinstance(colors_override, torch.Tensor):
            colors = torch.tensor(colors_override)
        colors = colors.expand(len(meshes), 3)
        meshes.textures = TexturesVertex(verts_features=colors)

    if join:
        meshes.offset_verts_(
            torch.arange(num_meshes)  # (num_meshes,)
            .view(num_meshes, 1)  # (num_meshes, 1)
            .repeat(1, meshes._V)  # (num_meshes, num_verts)
            .view(-1, 1)  # (num_meshes * num_verts, 1)
            * shift  # (num_meshes * num_verts, 3)
            * 0.7  # scale
        )

        fig = plot_scene(
            {
                subplot_titles if isinstance(subplot_titles, str) else "Joined meshes": {"mesh": meshes},
            },
            axis_args=AxisArgs(backgroundcolor="rgb(200,230,200)", showgrid=True, showticklabels=True),
            ncols=1,
            viewpoint_cameras=viewpoint_cameras,
        )
        cols = rows = 1
    else:
        cols = num_columns if num_columns > 0 else min(num_meshes, 3)
        rows = (num_meshes + cols - 1) // cols  # math.ceil(num_meshes / cols)

        fig = plot_batch_individually(
            meshes,
            axis_args=AxisArgs(backgroundcolor="rgb(200,230,200)", showgrid=True, showticklabels=True),
            ncols=num_columns if num_columns > 0 else min(num_meshes, 3),
            subplot_titles=subplot_titles,
            viewpoint_cameras=viewpoint_cameras,
        )

    fig.update_layout(autosize=False, width=cols * subplot_size, height=rows * subplot_size)

    if show:
        fig.show()

    return fig


def get_jet_colormap(num_colors: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Generator that yeilds RGB values for a jet colormap equivalent.

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

    if np.issubdtype(dtype, np.integer):
        colors = (colors * 255).astype(dtype)
    else:
        colors = colors.astype(dtype)

    return colors

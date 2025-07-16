"""Mesh plotting tools for TinyHumans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

if TYPE_CHECKING:
    from numpy import ndarray
    from plotly.graph_objs import Figure
    from pytorch3d.renderer.cameras import CamerasBase
    from pytorch3d.structures import Meshes


def plot_meshes(
    meshes: Meshes,
    join: bool = True,
    join_direction: str = "right",
    colors_override: torch.Tensor | ndarray | list | None = None,
    show: bool = True,
    num_columns: int = -1,
    subplot_size: int = 1000,
    subplot_titles: list | str | None = None,
    viewpoint_cameras: CamerasBase | None = None,
) -> Figure:
    """Plot a batch of meshes, optionally joining them in a single plot or displaying them individually in a grid.

    Args:
        meshes (pytorch3d.Meshes): A Meshes object containing the meshes to plot.
        join (bool, optional): If True, joins all meshes into a single plot. If False, plots each mesh individually in
            subplots. Defaults to True.
        join_direction (str, optional): The direction in which to join meshes when `join` is True.
            Options are "right", "left", "backward", "forward", and "diagonal". Defaults to "right".
        colors_override (array-like, optional): An array-like object representing the colors to override mesh colors,
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
        msg = '`join_direction` must be one of "right", "left", "backward", "forward", "diagonal"'
        raise ValueError(msg)

    if colors_override is not None:
        if not isinstance(colors_override, torch.Tensor):
            colors = torch.tensor(colors_override)
        colors = colors.expand(len(meshes), 3)
        meshes = meshes.clone()
        meshes.textures = TexturesVertex(verts_features=colors)

    if join:
        meshes = meshes.offset_verts(
            torch.arange(num_meshes)  # (num_meshes,)
            .view(num_meshes, 1)  # (num_meshes, 1)
            .repeat(1, meshes.num_verts_per_mesh().max())  # (num_meshes, num_verts)
            .view(-1, 1)  # (num_meshes * num_verts, 1)
            * shift  # (num_meshes * num_verts, 3)
            * 0.7  # scale
        )

        fig = plot_scene(
            {subplot_titles if isinstance(subplot_titles, str) else "Joined meshes": {"mesh": meshes}},
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

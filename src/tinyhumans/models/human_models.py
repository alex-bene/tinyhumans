"""Human body models based on BodyBaseParametricModel.

This module defines specific human body model classes such as SMPL, SMPLH, and SMPLX, which inherit from the
BodyBaseParametricModel class and provide model-specific initialization and shape component handling.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from tinyhumans.datasets.prepare.prepare_base import prepare as prepare_base

from .base_body_parametric_model import BodyBaseParametricModel

if TYPE_CHECKING:
    from tinyhumans.datatypes import SMPLData


class SMPL(BodyBaseParametricModel):
    """SMPL body model class.

    This class implements the SMPL body model, inheriting from BodyBaseParametricModel. It initializes the SMPL model with
    specific parameters and configurations, including handling of DMPL shape components.

    Attributes:
        dmpls_size (int): Size of DMPL shape parameters, if used.
        use_dmpl (bool): Flag indicating whether DMPL shape components are used.
        betas_size (int): Size of the betas shape parameters.

    """

    def __init__(
        self,
        vertices_template: Tensor,
        faces: Tensor,
        kinematic_tree_vector: Tensor,
        blending_weights: Tensor,
        num_joints: int,
        num_betas: int | None = None,
        num_dmpls: int | None = None,
        body_type: str | None = None,
        gender: str = "neutral",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize SMPL model.

        Args:
            vertices_template (torch.Tensor): Mean template vertices of body at rest.
            faces (torch.Tensor): Faces of the mesh topology.
            kinematic_tree_vector (torch.Tensor): Kinematic tree vector for the body model.
            blending_weights (torch.Tensor): Linear blend skinning weights.
            num_joints (int): Number of joints in the body model.
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to 10.
            num_dmpls (int | None, optional): Number of DMPL shape parameters. Defaults to None (no DMPL).
            body_type (str, optional): Type of the body model (e.g., "smpl", "smplh", "smplx"). Defaults to None.
            gender (str, optional): Gender of the model ("neutral", "male", "female"). Defaults to "neutral".
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        """
        # Setup shapes of default parameters
        self.use_dmpl = num_dmpls is not None and num_dmpls > 0
        self.dmpls_size = num_dmpls if self.use_dmpl else 0
        self.betas_size = num_betas if num_betas is not None and num_betas > 0 else 0

        super().__init__(
            body_type="smpl" if body_type is None else body_type,
            gender=gender,
            num_shape_coeffs=num_betas + self.dmpls_size,
            num_joints=num_joints,
            vertices_template=vertices_template,
            faces=faces,
            kinematic_tree_vector=kinematic_tree_vector,
            blending_weights=blending_weights,
            dtype=dtype,
        )

    @classmethod
    def _load_shape_components(
        cls,
        model_params_dict: dict[str, Tensor],
        num_betas: int | None = None,
        num_dmpls: int | None = None,
        dmpl_filename: str | Path | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Load shape components from model parameters.

        Args:
            model_params_dict (dict[str, Tensor]): Dictionary of model parameters.
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to None.
            num_dmpls (int | None, optional): Number of DMPL shape parameters. Defaults to None.
            dmpl_filename (str | Path | None, optional): Path to the DMPL eigenvectors file (.npz).
                Required if `num_dmpls` > 0. Defaults to None.
            device (torch.device | None, optional): Device to put the shape components on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        Returns:
            Tensor: Shape components tensor.

        Raises:
            ValueError: If `dmpl_filename` is not provided when `num_dmpls` > 0.

        """
        shape_blend_components = super()._load_shape_components(model_params_dict, num_betas, device, dtype)

        if num_dmpls is None:
            num_dmpls = 300 - model_params_dict["shapedirs"].shape[-1]
        elif num_dmpls < 0:
            num_dmpls = 0

        if num_dmpls > 0:
            if dmpl_filename is None:
                msg = "`dmpl_filename` should be provided when using dmpls!"
                raise ValueError(msg)
            dmpl_blend_components = torch.from_numpy(np.load(dmpl_filename)["eigvec"][:, :, :num_dmpls]).to(
                device, dtype
            )
            # Shape + DMPL components -> Shape components (num_vertices x 3 x num_shape_coeffs)
            shape_blend_components = torch.cat([shape_blend_components, dmpl_blend_components], dim=-1)

        return shape_blend_components

    def get_shape_tensor(self, smpl_data: SMPLData) -> Tensor:
        """Get shape tensor from a SMPLData object.

        Args:
            smpl_data (SMPLData): SMPL data object (poses, shape parameters).

        Returns:
            Tensor: Tensor of concatenated shape and dmpl parameters.

        """
        return smpl_data.get_shape_tensor(
            shape_coeffs_size=self.betas_size, dmpl_coeffs_size=self.dmpls_size, expression_coeffs_size=0
        )


"""SMPLH body model class."""
SMPLH = SMPL


class SMPLX(BodyBaseParametricModel):
    """SMPLX body model class.

    This class implements the SMPLX body model, inheriting from SMPLH. It further specializes SMPLH to include
    expression and jaw/eye pose parameters.

    Attributes:
        expression_coeffs_size (int): Size of expression shape parameters, if used.
        use_expression (bool): Flag indicating whether expression shape components are used.
        betas_size (int): Size of the betas shape parameters.

    """

    def __init__(
        self,
        vertices_template: Tensor,
        faces: Tensor,
        kinematic_tree_vector: Tensor,
        blending_weights: Tensor,
        num_joints: int,
        num_betas: int | None = None,
        num_expression_coeffs: int | None = None,
        body_type: str | None = None,
        gender: str = "neutral",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize SMPLX model.

        Args:
            vertices_template (torch.Tensor): Mean template vertices of body at rest.
            faces (torch.Tensor): Faces of the mesh topology.
            kinematic_tree_vector (torch.Tensor): Kinematic tree vector for the body model.
            blending_weights (torch.Tensor): Linear blend skinning weights.
            num_joints (int): Number of joints in the body model.
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to 10.
            num_expression_coeffs (int | None, optional): Number of expression shape parameters.
                Defaults to None (no expression).
            body_type (str, optional): Type of the body model (e.g., "smpl", "smplh", "smplx"). Defaults to None.
            gender (str, optional): Gender of the model ("neutral", "male", "female"). Defaults to "neutral".
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        """
        # Setup shapes of default parameters
        self.betas_size = num_betas if num_betas is not None and num_betas > 0 else 0
        self.use_expression = num_expression_coeffs is not None and num_expression_coeffs > 0
        self.expression_coeffs_size = num_expression_coeffs if self.use_expression else 0

        super().__init__(
            body_type="smplx" if body_type is None else body_type,
            gender=gender,
            num_shape_coeffs=num_betas + self.expression_coeffs_size,
            num_joints=num_joints,
            vertices_template=vertices_template,
            faces=faces,
            kinematic_tree_vector=kinematic_tree_vector,
            blending_weights=blending_weights,
            dtype=dtype,
        )

    @classmethod
    def _load_shape_components(
        cls,
        model_params_dict: dict[str, Tensor],
        num_betas: int | None = None,
        num_expression_coeffs: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Load shape components from model parameters.

        Args:
            model_params_dict (dict[str, Tensor]): Dictionary of model parameters.
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to None.
            num_expression_coeffs (int | None, optional): Number of expression shape parameters. Defaults to None.
            device (torch.device | None, optional): Device to put the shape components on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.

        Returns:
            Tensor: Shape components tensor.

        """
        shape_blend_components = super()._load_shape_components(model_params_dict, num_betas, device, dtype)

        if num_expression_coeffs is None:
            num_expression_coeffs = 300 - model_params_dict["shapedirs"].shape[-1]
        elif num_expression_coeffs < 0:
            num_expression_coeffs = 0

        # The orthonormal principal components for expression conditioned displacements
        # (num_vertices x 3 x num_expression_coeffs)
        if num_expression_coeffs > 0:
            expression_blend_components = torch.from_numpy(
                model_params_dict["shapedirs"][:, :, 300 : (300 + num_expression_coeffs)]
            ).to(device, dtype)
            # Shape + Expression components -> Shape components (num_vertices x 3 x num_shape_coeffs)
            shape_blend_components = torch.cat([shape_blend_components, expression_blend_components], dim=-1)

        return shape_blend_components

    def get_shape_tensor(self, smpl_data: SMPLData) -> Tensor:
        """Get shape tensor from a SMPLData object.

        Args:
            smpl_data (SMPLData): SMPL data object (poses, shape parameters).

        Returns:
            Tensor: Tensor of concatenated shape and expression parameters.

        """
        return smpl_data.get_shape_tensor(
            shape_coeffs_size=self.betas_size, dmpl_coeffs_size=0, expression_coeffs_size=self.expression_coeffs_size
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        num_betas: int | None = None,
        device_map: str | torch.device | None = "auto",
        torch_dtype: torch.dtype = torch.float32,
        username: str | None = None,
        password: str | None = None,
        cache_folder: Path | None = None,
        **kwargs,
    ) -> SMPLX:
        """Load a pre-trained model.

        Args:
            pretrained_model_name_or_path (str | Path): Path to the pretrained model parameters (.npz file) or model
                type to auto-download (e.g. "neutra", "lockedhead_female").
            num_betas (int | None, optional): Number of shape parameters (betas). Defaults to None.
            device_map (str | torch.device): Device to map the model weights to.
            torch_dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.
            username (str | None, optional): Username for downloading the model. Required if
                `pretrained_model_name_or_path` is not an existing file but a model type to download. Defaults to None.
            password (str | None, optional): Password for downloading the model. Required if
                `pretrained_model_name_or_path` is not an existing file but a model type to download. Defaults to None
            cache_folder (Path | None, optional): Cache folder for downloading the model. If None, the cache folder
                will be created in the user's home directory (~/.cache/tinyhumans/models). Defaults to None.
            **kwargs: Additional keyword arguments passed to the model constructor.

        Returns:
            SMPLX: A pre-trained SMPLX instance.

        """
        # Check pretrained model path
        pretrained_model_path = Path(pretrained_model_name_or_path)
        auto_download_options = (
            "male",
            "female",
            "neutral",
            "lockedhead_male",
            "lockedhead_female",
            "lockedhead_neutral",
        )
        if not pretrained_model_path.exists() and pretrained_model_name_or_path not in auto_download_options:
            msg = f"`pretrained_model_name_or_path` should be an existing file or one of {auto_download_options}"
            raise ValueError(msg)

        if cache_folder is None:
            cache_folder = Path.home() / ".cache" / "tinyhumans" / "models"

        output_dir = Path(cache_folder) / cls.__name__.lower()
        if not pretrained_model_path.exists():
            kwargs["gender"] = pretrained_model_name_or_path.replace("lockedhead_", "")
            pretrained_model_path = output_dir / f"{pretrained_model_name_or_path}.npz"

        if not pretrained_model_path.exists():
            if password is None or username is None:
                msg = (
                    "`username` and `password` must be provided if `pretrained_model_name_or_path` is one of "
                    f"{auto_download_options} and it's the first time the model is downloaded."
                )
                raise ValueError(msg)

            # --- Base URL for Downloads ---
            base_url = "https://download.is.tue.mpg.de/download.php"

            # --- Download files ---
            if "lockedhead" in pretrained_model_name_or_path:
                files_to_download = [
                    {
                        "id": f"{base_url}?domain=smplx&resume=1&sfile={'smplx_lockedhead_20230207.zip'}",
                        "name": "smplx_lockedhead_20230207.zip",
                        "output_dir": output_dir,
                        "gdrive": False,
                        "post_data": {"username": username, "password": password},
                        "check_exists": output_dir / f"{pretrained_model_name_or_path}.npz",
                    }
                ]
            else:
                files_to_download = [
                    {
                        "id": f"{base_url}?domain=smplx&resume=1&sfile={'models_smplx_v1_1.zip'}",
                        "name": "models_smplx_v1_1.zip",
                        "output_dir": output_dir,
                        "gdrive": False,
                        "post_data": {"username": username, "password": password},
                        "check_exists": output_dir / f"{pretrained_model_name_or_path}.npz",
                    }
                ]
            prepare_base(output_dir=output_dir, files_to_download=files_to_download, quiet=False)
            # move files around from smplx.text to smplx/dsta/text.tt
            for gender in ["FEMALE", "MALE", "NEUTRAL"]:
                if (output_dir / "models" / "smplx").exists():
                    src = output_dir / "models" / "smplx" / f"SMPLX_{gender}.npz"
                    dst = output_dir / f"{gender.lower()}.npz"
                    shutil.move(src, dst)
                if (output_dir / "models_lockedhead" / "smplx").exists():
                    src = output_dir / "models_lockedhead" / "smplx" / f"SMPLX_{gender}.npz"
                    dst = output_dir / f"lockedhead_{gender.lower()}.npz"
                    shutil.move(src, dst)
            if (output_dir / "models").exists():
                shutil.rmtree(output_dir / "models")
            if (output_dir / "models_lockedhead").exists():
                shutil.rmtree(output_dir / "models_lockedhead")

        return super().from_pretrained(pretrained_model_path, num_betas, device_map, torch_dtype, **kwargs)

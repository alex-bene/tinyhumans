"""Variational pose prior model.

This module defines the VPoser class, which is a variational pose prior model.
It also includes logic to load a pre-trained VPoser.

This code is adapted from the original VPoser implementation by Nima Ghorbani.
See: https://nghorbani.github.io/

License information for the original VPoser implementation can be found at:
https://github.com/nghorbani/human_body_prior/blob/master/LICENSE
"""

from __future__ import annotations

import torch
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from tinytools import get_logger
from torch import nn
from torch.nn import functional as F

from tinyhumans.models.base_model import BaseModel

logger = get_logger(__name__)


class VPoser(BaseModel):
    """Variational Pose Prior.

    This module implements a Variational Autoencoder (VAE) for 3D human pose.
    It learns a latent space representation of human poses, allowing for sampling and decoding of realistic poses.

    Attributes:
        num_joints (int): Number of joints.
        latent_dim (int): Dimension of the latent space.
        encoder_net (nn.Sequential): Encoder network.
        decoder_net (nn.Sequential): Decoder network.
        mu_layer (nn.Linear): Linear layer to predict the mean of the latent distribution.
        logvar_layer (nn.Linear): Linear layer to predict the log variance of the latent distribution.

    """

    def __init__(self) -> None:
        super().__init__()

        num_neurons, self.latent_dim = 512, 32

        self.num_joints = 21
        n_features = self.num_joints * 3

        self.encoder_net = nn.Sequential(
            nn.Flatten(1, -1),
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
        )

        self.mu_layer = nn.Linear(num_neurons, self.latent_dim)
        self.logvar_layer = nn.Linear(num_neurons, self.latent_dim)

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latent_dim, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
        )

    def encode(self, pose_body: torch.Tensor) -> torch.distributions.normal.Normal:
        """Encode the pose body into a normal distribution in latent space.

        Args:
            pose_body (torch.Tensor): The pose body tensor.

        Returns:
            torch.distributions.normal.Normal: A normal distribution in latent space.

        """
        x = self.encoder_net(pose_body)
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        scale = F.softplus(logvar)
        return torch.distributions.normal.Normal(mu, scale)

    def decode(self, latent_vector: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode a latent vector into pose parameters.

        Args:
            latent_vector (torch.Tensor): The input latent vector.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the decoded pose body and pose body matrix rotations.

        """
        # Pytorch3D as a convention drops the last row instead of the last column of the rotation matrix to get
        # the 6D rotation representation. This is why we need the various transpose operations here.
        rotation_matrix_parameters = rotation_6d_to_matrix(
            self.decoder_net(latent_vector).view(-1, self.num_joints, 3, 2).mT.flatten(-2)
        ).mT

        return {
            "pose_axis_angles": matrix_to_axis_angle(rotation_matrix_parameters),
            "pose_rotation_matrices": rotation_matrix_parameters.flatten(-2),
        }

    def forward(self, pose_body: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pose_body (torch.Tensor): The input pose body tensor.

        Returns:
            dict[str, torch.Tensor]: A dict containing the decoded pose results and the latent distribution parameters.

        """
        latent_distribution = self.encode(pose_body)
        latent_sample = latent_distribution.rsample()
        decode_results = self.decode(latent_sample)

        return decode_results | {
            "latent_pose_mean": latent_distribution.mean,
            "latent_pose_std": latent_distribution.scale,
            "latent_distribution": latent_distribution,
        }

    def sample(self, num_poses: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        """Sample poses from the VPoser.

        Args:
            num_poses (int): The number of poses to sample.
            seed (int | None): The random seed to use.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the sampled poses.

        """
        if seed is not None:
            torch.manual_seed(seed)

        set_train_mode = False
        if self.training:
            logger.info("Model is in training mode. Setting temporarily to evaluation mode.")
            self.eval()
            set_train_mode = True

        with torch.no_grad():
            latent_vector = torch.normal(
                0.0, 1.0, size=(num_poses, self.latent_dim), dtype=self.dtype, device=self.device
            )

        pose = self.decode(latent_vector)

        if set_train_mode:
            self.train()

        return pose

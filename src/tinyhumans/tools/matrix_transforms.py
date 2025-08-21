"""Matrix transformation tools for TinyHumans."""

from __future__ import annotations

import torch
from torch.nn.functional import pad


def get_homogeneous_transform_matrix(rotation_matrix: torch.Tensor, translations: torch.Tensor) -> torch.Tensor:
    """Create a homogeneous transformation matrix.

    Works for both batched and unbatched inputs.

    +---+---+
    | R | T |
    +---+---+
    | 0 | 1 |
    +---+---+

    Args:
        rotation_matrix (torch.Tensor): A batch of rotation matrices of shape (..., 3, 3)
        translations (torch.Tensor): A batch of translation vectors of shape (..., 3, 1)

    Returns:
        torch.Tensor: A batch of homogeneous transformation matrices of shape (..., 4, 4)

    """
    return torch.cat([pad(rotation_matrix, [0, 0, 0, 1]), pad(translations, [0, 0, 0, 1], value=1)], dim=-1)


def apply_rigid_transform(
    rot_mats: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a batch of rigid transformations to the joints.

    Args:
        rot_mats (torch.tensor): Tensor of rotation matrices with shape (..., num_joints, 3, 3)
        joints (torch.tensor): Locations of joints with shape (..., num_joints, 3)
        parents (torch.tensor): The kinematic tree of each object with shape (num_joints)

    Returns:
        torch.tensor : The locations of the joints after applying the pose rotations with shape (..., num_joints, 3)
        torch.tensor : The relative (with respect to the root joint) rigid transformations for all the joints in
            homogeneous coordinates with shape (..., num_joints, 4, 4)

    """
    joints = torch.unsqueeze(joints, dim=-1)  # (... x J x 3 x 1)

    rel_joints = joints.clone()
    rel_joints[..., 1:, :, :] -= joints[..., parents[1:], :, :]

    # Relative translation of each join based on its parent in the kinematic tree
    joint_rel_hom_transform_mats = get_homogeneous_transform_matrix(rot_mats, rel_joints)  # (... x J x 4 x 4)

    # Propagete the realtive transformations down the kinematic chain to get absolute transformations
    transform_chain = [joint_rel_hom_transform_mats[..., 0, :, :]]  # (... x 4 x 4)
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], joint_rel_hom_transform_mats[..., i, :, :])  # (... x 4x4)
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=-3)  # (... x J x 4 x 4)

    # The last column of the homogeneous transformations contains the location of the posed joints
    posed_joints = transforms[..., :3, 3]  # (... x J x 3)

    # NOTE: understand this
    joints_homogen = pad(joints, (0, 0, 0, 1))  # (... x J x 4 x 1)
    # # (... x J x 4 x 4) - (Pad((... x J x 4 x 4) * (... x J x 4 x 1) -> (... x J x 4 x 1)) -> (... x J x 4 x 4))
    rel_transforms = transforms - pad(torch.matmul(transforms, joints_homogen), (3, 0))

    return posed_joints, rel_transforms

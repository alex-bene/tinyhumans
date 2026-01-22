from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from smplcodec import SMPLGender, SMPLVersion

from src.tinyhumans.datatypes import SMPLData


@pytest.fixture
def smpl_data_path() -> Path:
    return Path(__file__).parent / "fixtures" / "subject-2.smpl"


def test_file_rw(tmp_path: Path, smpl_data_path: Path) -> None:
    smpl = SMPLData.from_file(smpl_data_path)
    smpl.write(tmp_path / "test", as_json=True)
    assert (tmp_path / "test.smpl").is_file()
    smpl.write(tmp_path / "test", as_json=False)
    assert (tmp_path / "test.smplz").is_file()
    smpl.write(tmp_path / "test.smpl2", as_json=True)
    assert (tmp_path / "test.smpl2").is_file()
    smpl.write(tmp_path / "test.smpl3", as_json=False)
    assert (tmp_path / "test.smpl3").is_file()
    assert smpl.equals(SMPLData.from_file(tmp_path / "test.smpl"))
    assert smpl.equals(SMPLData.from_file(tmp_path / "test.smplz"))
    assert smpl.equals(SMPLData.from_file(tmp_path / "test.smpl2"))
    assert smpl.equals(SMPLData.from_file(tmp_path / "test.smpl3"))

    smpl = SMPLData.from_file(smpl_data_path).expand(2, smpl.frame_count, 3)
    smpl.write(tmp_path / "test", as_json=True)
    for b in range(2):
        for h in range(3):
            assert (tmp_path / f"test_{b}_{h}.smpl").is_file()
            assert smpl.equals(SMPLData.from_file(tmp_path / f"test_{b}_{h}.smpl"))


def test_full_pose_rw(smpl_data_path: Path) -> None:
    smpl = SMPLData.from_file(smpl_data_path)
    assert smpl.full_pose.shape == (1, 295, 1, 55, 3)
    assert torch.allclose(smpl.full_pose[0, 0, 0, 3], torch.tensor([0.5279, 0.0309, -0.1366]), atol=1e-4)

    assert smpl.equals(
        SMPLData.from_full_pose(
            smpl.full_pose,
            shape_parameters=smpl.shape_parameters,
            body_translation=smpl.body_translation,
            frame_rate=smpl.frame_rate,
            smpl_version=smpl.smpl_version,
            gender=smpl.gender,
            shape_aggregation_method=smpl.shape_aggregation_method,
            expression_parameters=smpl.expression_parameters,
        )
    )
    smpl2 = SMPLData()
    smpl2.full_pose = smpl.full_pose
    assert torch.allclose(smpl2.full_pose, smpl.full_pose)
    assert torch.allclose(smpl2.body_pose, smpl.body_pose)
    assert torch.allclose(smpl2.head_pose.abs().sum(), torch.tensor(0.0))  # smpl.head_pose is None
    assert torch.allclose(smpl2.left_hand_pose.abs().sum(), torch.tensor(0.0))  # smpl.left_hand_pose is None
    assert torch.allclose(smpl2.right_hand_pose.abs().sum(), torch.tensor(0.0))  # smpl.right_hand_pose is None


def test_casting_and_dim_unsqueeze() -> None:
    smpl = SMPLData()
    smpl.shape_parameters = np.zeros(100)
    assert smpl.shape_parameters.shape == (1, 1, 100)
    assert smpl._shape_parameters.shape == (1, 1, 1, 100)
    smpl = SMPLData()
    smpl.body_translation = torch.ones(2, 3)
    assert smpl.body_translation.shape == (2, 1, 1, 3)
    smpl = SMPLData()
    smpl.body_pose = torch.ones(12, 2, 22, 3)
    assert smpl.body_pose.shape == (12, 1, 2, 22, 3)
    smpl = SMPLData()
    smpl.head_pose = torch.ones(14, 12, 2, 3, 3)
    assert smpl.head_pose.shape == (14, 12, 2, 3, 3)
    smpl.frame_rate = "3"
    assert smpl.frame_rate == 3.0
    smpl.smpl_version = 2
    assert smpl.smpl_version == SMPLVersion.SMPLX
    smpl.gender = 2
    assert smpl.gender == SMPLGender.FEMALE


def test_concatenation() -> None:
    smpl = SMPLData()
    smpl.shape_parameters = torch.zeros(12, 2, 100)
    # We can not cat SMPLData when batch_size is not yet set
    pytest.raises(RuntimeError, lambda: torch.cat([smpl, smpl]))
    smpl.body_pose = torch.rand(12, 50, 2, 22, 3)
    assert torch.cat([smpl, smpl]).shape == (24, 50, 2)
    assert torch.cat([smpl, smpl], dim=1).shape == (12, 100, 2)
    assert torch.cat([smpl, smpl], dim=2).shape == (12, 50, 4)


def test_shape_parameters() -> None:
    smpl = SMPLData(shape_parameters=torch.zeros(100))
    # Setting shape_parameters does not set frame_count but does set batch_count and human_count
    assert smpl.batch_count == 1
    assert smpl.frame_count is None
    assert smpl.human_count == 1
    assert not smpl.batch_size  # batch_size is never set when the only tensor ser it `shape_parameters`
    smpl = SMPLData(shape_parameters=torch.zeros(5, 100))
    assert smpl.batch_count == 5
    assert smpl.frame_count is None
    assert smpl.human_count == 1
    assert not smpl.batch_size
    smpl = SMPLData(shape_parameters=torch.zeros(12, 5, 100))
    assert smpl.batch_count == 12
    assert smpl.frame_count is None
    assert smpl.human_count == 5
    assert not smpl.batch_size
    # You are not allowed to set a time dimension for shape_parameters
    pytest.raises(ValueError, lambda: SMPLData(shape_parameters=torch.zeros(1, 1, 1, 100, 1)))

    # `shape_parameters` internally is saved as `_shape_parameters` and has an extra dimension for time
    # This is done because tensorclass and tensordict do not accept having a tensor that does not follow the
    # batch_size of the overall object.
    assert smpl._shape_parameters.shape == (12, 1, 5, 100)

    # Setting tensors that have a time dimension expands the internal representation of `shape_parameters`
    smpl.body_pose = torch.rand(12, 123, 5, 22, 3)
    assert smpl.batch_count == 12
    ## sets frame_count
    assert smpl.frame_count == 123
    assert smpl.human_count == 5
    ## sets batch_size
    assert smpl.batch_size == (12, 123, 5)
    ## does not change the user-facing shape_parameters
    assert smpl.shape_parameters.shape == (12, 5, 100)
    ## expands the internal representation of shape_parameters (_shape_parameters) to follow the object's batch_size
    assert smpl._shape_parameters.shape == (12, 123, 5, 100)


def test_shape_aggregation_method() -> None:
    smpl = SMPLData(shape_parameters=torch.zeros(12, 5, 100), body_pose=torch.rand(12, 60, 5, 22, 3))
    smpl2 = SMPLData(shape_parameters=torch.ones(12, 5, 100), body_pose=torch.rand(12, 120, 5, 22, 3))
    # When applying operations to SMPLData, we can end up with the internal representation of `shape_parameters`
    # not being constant across time (e.g. concatenation across the time dimension). In this case, this "inconsistent"
    # tensor will be the internal representation of `shape_parameters` (`_shape_parameters`) but the user-facing
    # `shape_parameters` will depend on the "shape_aggregation_method". If the shape_aggregation_method is "first",
    # (the default) then `shape_parameters = _shape_parameters[:, 0]`. If the shape_aggregation_method is "last", then
    # `shape_parameters = _shape_parameters[:, -1]`. If the shape_aggregation_method is "mean", then
    # `shape_parameters = _shape_parameters.mean(dim=1)`.
    cat_smpl: SMPLData = torch.cat([smpl, smpl2], dim=1)
    assert torch.allclose(cat_smpl.shape_parameters, smpl.shape_parameters)
    cat_smpl.shape_aggregation_method = "last"
    assert torch.allclose(cat_smpl.shape_parameters, smpl2.shape_parameters)
    cat_smpl.shape_aggregation_method = "mean"
    assert torch.allclose(
        cat_smpl.shape_parameters, torch.cat([smpl._shape_parameters, smpl2._shape_parameters], dim=1).mean(dim=1)
    )
    assert torch.allclose(
        cat_smpl.shape_parameters,
        (smpl.frame_count * smpl.shape_parameters + smpl2.frame_count * smpl2.shape_parameters)
        / (smpl.frame_count + smpl2.frame_count),
    )


def test_batch_related() -> None:
    smpl = SMPLData(shape_parameters=torch.zeros(100))
    # Setting shape_parameters does not set frame_count or batch_size
    assert smpl.batch_count == 1
    assert smpl.frame_count is None
    assert smpl.human_count == 1
    assert not smpl.batch_size  # batch_size is never set when the only tensor ser it `shape_parameters`
    smpl.body_translation = torch.ones(1, 50, 1, 3)
    assert smpl.batch_count == 1
    assert smpl.frame_count == 50
    assert smpl.human_count == 1
    assert smpl.batch_size == (1, 50, 1)
    smpl = SMPLData(body_pose=torch.ones(60, 14, 24, 22, 3))
    assert smpl.batch_count == 60
    assert smpl.frame_count == 14
    assert smpl.human_count == 24
    assert smpl.batch_size == (60, 14, 24)


def test_indexing_and_iter(smpl_data_path: Path) -> None:
    smpl = SMPLData.from_file(smpl_data_path)
    smpl = smpl.expand(12, 295, 2)
    assert smpl[0].shape == (1, 295, 2)
    assert smpl[:, 2].shape == (12, 1, 2)
    assert smpl[:, :, 1].shape == (12, 295, 1)

    for smpl_i in smpl:
        assert smpl_i.shape == (1, 295, 2)
    for smpl_i in smpl.iter_frames():
        assert smpl_i.shape == (12, 1, 2)
    for smpl_i in smpl.iter_humans():
        assert smpl_i.shape == (12, 295, 1)


def test_body_orientation_get_set() -> None:
    random_body_pose = torch.rand(60, 14, 24, 22, 3)
    smpl = SMPLData(body_pose=random_body_pose)
    assert torch.allclose(smpl.body_orientation, random_body_pose[..., 0, :])

    smpl.body_orientation = torch.rand(60, 14, 24, 3)
    assert torch.allclose(smpl.body_pose[..., 0, :], smpl.body_orientation)


def test_frame_presence() -> None:
    smpl = SMPLData(body_pose=torch.rand(60, 14, 24, 22, 3))
    assert torch.allclose(smpl.frame_presence, torch.ones(60, 14, 24))
    for fidx in range(smpl.frame_count):
        assert torch.allclose(smpl.presence_in_frame(fidx), torch.ones(60, 24, dtype=bool))
        assert smpl.presence_in_frame(fidx).dtype == torch.bool

    smpl = SMPLData(shape_parameters=torch.zeros(60, 24, 100))
    # shape_parameters does not set frame_count and batch size, thus frame_presence can not be set
    assert smpl.frame_presence is None
    smpl.left_hand_pose = torch.rand(60, 14, 24, 15, 3)
    assert torch.allclose(smpl.frame_presence, torch.ones(60, 14, 24))
    for fidx in range(smpl.frame_count):
        assert torch.allclose(smpl.presence_in_frame(fidx), torch.ones(60, 24, dtype=bool))
        assert smpl.presence_in_frame(fidx).dtype == torch.bool

    smpl.frame_presence = (torch.rand(60, 14, 24) > 0.5).to(smpl.dtype)
    assert torch.allclose(
        smpl.frame_presence.to(bool),
        torch.stack([smpl.presence_in_frame(fidx) for fidx in range(smpl.frame_count)], dim=1),
    )


def test_get_shape_tensor() -> None:
    # Test shape_parameters
    random_shape_parameters = torch.rand(60, 24, 100)
    smpl = SMPLData(body_pose=torch.rand(60, 14, 24, 22, 3), shape_parameters=random_shape_parameters)
    assert torch.allclose(smpl.get_shape_tensor(), random_shape_parameters.unsqueeze(1).expand(-1, 14, -1, -1))
    ## equivalent (when shape_coeffs_size is None, which is the default, the size is inferred from the shape_parameters,
    ## expression_parameters and dmpl_parameters if not None)
    assert torch.allclose(
        smpl.get_shape_tensor(shape_coeffs_size=None), random_shape_parameters.unsqueeze(1).expand(-1, 14, -1, -1)
    )
    ## equivalent
    assert torch.allclose(
        smpl.get_shape_tensor(shape_coeffs_size=100), random_shape_parameters.unsqueeze(1).expand(-1, 14, -1, -1)
    )
    # Test expression_parameters
    random_expression_parameters = torch.rand(60, 14, 24, 10)
    smpl.expression_parameters = random_expression_parameters
    assert torch.allclose(
        smpl.get_shape_tensor(),
        torch.cat([random_shape_parameters.unsqueeze(1).expand(-1, 14, -1, -1), random_expression_parameters], dim=-1),
    )
    assert torch.allclose(
        smpl.get_shape_tensor(shape_coeffs_size=10),
        torch.cat(
            [random_shape_parameters.unsqueeze(1).expand(-1, 14, -1, -1)[..., :10], random_expression_parameters],
            dim=-1,
        ),
    )
    # Test dmpl_parameters
    smpl.expression_parameters = None
    random_dmpl_parameters = torch.rand(60, 14, 24, 8)
    smpl.dmpl_parameters = random_dmpl_parameters
    assert torch.allclose(
        smpl.get_shape_tensor(),
        torch.cat([random_shape_parameters.unsqueeze(1).expand(-1, 14, -1, -1), random_dmpl_parameters], dim=-1),
    )
    assert torch.allclose(
        smpl.get_shape_tensor(shape_coeffs_size=10),
        torch.cat(
            [random_shape_parameters.unsqueeze(1).expand(-1, 14, -1, -1)[..., :10], random_dmpl_parameters], dim=-1
        ),
    )


def test_set_shape_tensor() -> None:
    # Test shape_parameters
    random_shape_parameters = torch.rand(60, 24, 100)
    smpl = SMPLData(body_pose=torch.rand(60, 14, 24, 22, 3), shape_parameters=random_shape_parameters)
    smpl.set_shape_tensor(smpl.get_shape_tensor(), expression_coeffs_size=0, dmpl_coeffs_size=0)
    assert torch.allclose(smpl.shape_parameters, random_shape_parameters)
    assert smpl.expression_parameters is None
    assert smpl.dmpl_parameters is None
    smpl2 = SMPLData(body_pose=torch.rand(60, 14, 24, 22, 3))
    smpl2.set_shape_tensor(smpl.get_shape_tensor(), shape_coeffs_size=100, expression_coeffs_size=0, dmpl_coeffs_size=0)
    assert torch.allclose(smpl2.shape_parameters, random_shape_parameters)
    assert smpl2.expression_parameters is None
    assert smpl2.dmpl_parameters is None

    # Test expression_parameters
    random_expression_parameters = torch.rand(60, 14, 24, 10)
    smpl.expression_parameters = random_expression_parameters
    smpl2.set_shape_tensor(
        smpl.get_shape_tensor(), shape_coeffs_size=100, expression_coeffs_size=10, dmpl_coeffs_size=0
    )
    assert torch.allclose(smpl.shape_parameters, smpl2.shape_parameters)
    assert torch.allclose(smpl.expression_parameters, smpl2.expression_parameters)
    assert smpl2.dmpl_parameters is None

    # Test dmpl_parameters
    smpl.expression_parameters = None
    random_dmpl_parameters = torch.rand(60, 14, 24, 8)
    smpl.dmpl_parameters = random_dmpl_parameters
    smpl2.set_shape_tensor(smpl.get_shape_tensor(), shape_coeffs_size=100, expression_coeffs_size=0, dmpl_coeffs_size=8)
    assert torch.allclose(smpl.shape_parameters, smpl2.shape_parameters)
    assert smpl2.expression_parameters is None
    assert torch.allclose(smpl.dmpl_parameters, smpl2.dmpl_parameters)

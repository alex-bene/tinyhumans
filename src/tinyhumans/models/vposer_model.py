#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2020.12.12

import glob
import os
import os.path as osp

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
from torch import nn
from torch.nn import functional as F

from tinyhumans.tools import get_logger

logger = get_logger(__name__)


def matrot2aa(pose_matrot):
    """:param pose_matrot: Nx3x3
    :return: Nx3
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(pose_matrot))


class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = "batch_flatten"

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))


class VPoser(nn.Module):
    def __init__(self, model_ps):
        super(VPoser, self).__init__()

        num_neurons, self.latentD = model_ps.model_params.num_neurons, model_ps.model_params.latentD

        self.num_joints = 21
        n_features = self.num_joints * 3

        self.encoder_net = nn.Sequential(
            BatchFlatten(),
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, self.latentD),
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )

    def encode(self, pose_body):
        """:param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        """
        return self.encoder_net(pose_body)

    def decode(self, Zin):
        bs = Zin.shape[0]

        prec = self.decoder_net(Zin)

        return {"pose_body": matrot2aa(prec.view(-1, 3, 3)).view(bs, -1, 3), "pose_body_matrot": prec.view(bs, -1, 9)}

    def forward(self, pose_body):
        """:param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        """
        q_z = self.encode(pose_body)
        q_z_sample = q_z.rsample()
        decode_results = self.decode(q_z_sample)
        decode_results.update({"poZ_body_mean": q_z.mean, "poZ_body_std": q_z.scale, "q_z": q_z})
        return decode_results

    def sample_poses(self, num_poses, seed=None):
        np.random.seed(seed)

        some_weight = [a for a in self.parameters()][0]
        dtype = some_weight.dtype
        device = some_weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0.0, 1.0, size=(num_poses, self.latentD)), dtype=dtype, device=device)

        return self.decode(Zgen)


def exprdir2model(expr_dir, model_cfg_override: dict = None):
    if not os.path.exists(expr_dir):
        raise ValueError(f"Could not find the experiment directory: {expr_dir}")

    model_snapshots_dir = osp.join(expr_dir, "snapshots")
    available_ckpts = sorted(glob.glob(osp.join(model_snapshots_dir, "*.ckpt")), key=osp.getmtime)
    assert len(available_ckpts) > 0, ValueError(f"No checkpoint found at {model_snapshots_dir}")
    trained_weights_fname = available_ckpts[-1]

    model_cfg_fname = glob.glob(osp.join("/", "/".join(trained_weights_fname.split("/")[:-2]), "*.yaml"))
    if len(model_cfg_fname) == 0:
        model_cfg_fname = glob.glob(osp.join("/".join(trained_weights_fname.split("/")[:-2]), "*.yaml"))

    model_cfg_fname = model_cfg_fname[0]
    model_cfg = OmegaConf.load(model_cfg_fname)
    if model_cfg_override:
        override_cfg_dotlist = [f"{k}={v}" for k, v in model_cfg_override.items()]
        override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)
        model_cfg = OmegaConf.merge(model_cfg, override_cfg)

    return model_cfg, trained_weights_fname


def load_model(
    expr_dir,
    model_code=None,
    remove_words_in_model_weights: str = None,
    load_only_cfg: bool = False,
    disable_grad: bool = True,
    model_cfg_override: dict = None,
    comp_device="gpu",
):
    """:param expr_dir:
    :param model_code: an imported module
    from supercap.train.supercap_smpl import SuperCap, then pass SuperCap to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    """
    import torch

    model_cfg, trained_weights_fname = exprdir2model(expr_dir, model_cfg_override=model_cfg_override)

    if load_only_cfg:
        return model_cfg

    assert model_code is not None, ValueError("model_code should be provided")
    model_instance = model_code(model_cfg)
    if disable_grad:  # i had to do this. torch.no_grad() couldnt achieve what i was looking for
        for param in model_instance.parameters():
            param.requires_grad = False

    if comp_device == "cpu" or not torch.cuda.is_available():
        logger.info("No GPU detected. Loading on CPU!")
        state_dict = torch.load(trained_weights_fname, map_location=torch.device("cpu"))["state_dict"]
    else:
        state_dict = torch.load(trained_weights_fname)["state_dict"]
    if remove_words_in_model_weights is not None:
        words = f"{remove_words_in_model_weights}"
        state_dict = {k.replace(words, "") if k.startswith(words) else k: v for k, v in state_dict.items()}

    ## keys that were in the model trained file and not in the current model
    instance_model_keys = list(model_instance.state_dict().keys())
    # trained_model_keys = list(state_dict.keys())
    # wts_in_model_not_in_file = set(instance_model_keys).difference(set(trained_model_keys))
    ## keys that are in the current model not in the training weights
    # wts_in_file_not_in_model = set(trained_model_keys).difference(set(instance_model_keys))
    # assert len(wts_in_model_not_in_file) == 0, ValueError('Some model weights are not present in the pretrained file. {}'.format(wts_in_model_not_in_file))

    state_dict = {k: v for k, v in state_dict.items() if k in instance_model_keys}
    model_instance.load_state_dict(state_dict, strict=False)
    # TODO fix the issues so that we can set the strict to true. The body model uses unnecessary registered buffers
    model_instance.eval()
    logger.info(f"Loaded model in eval mode with trained weights: {trained_weights_fname}")
    return model_instance, model_cfg

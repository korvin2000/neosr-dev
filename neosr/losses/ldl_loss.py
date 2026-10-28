from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.losses.base import Loss
from neosr.losses.basic_loss import chc_loss

if TYPE_CHECKING:
    from collections.abc import Callable


class ldl_loss(Loss):
    """LDL loss. Adapted from 'Details or Artifacts: A Locally Discriminative
    Learning Approach to Realistic Image Super-Resolution':
    https://arxiv.org/abs/2203.09195.

    Args:
    ----
        criterion (str): loss type. Default: 'huber'
        ksize (int): size of the local window. Default: 7

    """

    def __init__(self, criterion: str = "l1", ksize: int = 7) -> None:
        super().__init__()
        self.ksize = ksize
        self.criterion_type = criterion
        self.criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss | Callable

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "chc":
            self.criterion = chc_loss(loss_lambda=0, clip_min=0, clip_max=1)  # type: ignore[reportCallIssue]
        else:
            msg = f"{criterion} criterion has not been supported."
            raise NotImplementedError(msg)

    def get_local_weights(self, residual: Tensor) -> Tensor:
        """Get local weights for generating the artifact map of LDL.

        It is only called by the `get_refined_artifact_map` function.

        Args:
        ----
            residual (Tensor): Residual between predicted and ground truth images.

        Returns:
        -------
            Tensor: weight for each pixel to be discriminated as an artifact pixel

        """
        # unbiased variance of each reflect-padded k x k window from box-filter moments: the same
        # value as unfold + var, without the k^2-times unfolded copy in the backward. fp64: m2 - m1^2
        # cancels when the residual is biased and locally flat (one channel, so cheap even on GPU)
        k = self.ksize
        pad = (k - 1) // 2
        r = F.pad(residual.double(), pad=[pad, pad, pad, pad], mode="reflect")
        m1 = F.avg_pool2d(r, k, stride=1)
        m2 = F.avg_pool2d(r * r, k, stride=1)
        n = k * k
        return ((m2 - m1 * m1) * (n / (n - 1))).clamp_min(0).to(residual.dtype)

    def get_refined_artifact_map(self, img_gt: Tensor, img_output: Tensor) -> Tensor:
        """Calculate the artifact map of LDL
        (Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution. In CVPR 2022).

        Args:
        ----
            img_gt (Tensor): ground truth images.
            img_output (Tensor): output images given by the optimizing model.

        Returns:
        -------
            overall_weight: weight for each pixel to be discriminated as an artifact pixel
            (calculated based on both local and global observations).

        """
        residual_sr = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

        # fp32 + clamp: d(var**0.2)/dvar is inf at zero variance (constant residual)
        patch_level_weight = torch.var(
            residual_sr.float(), dim=(-1, -2, -3), keepdim=True
        ).clamp_min(1e-12) ** (1 / 5)
        pixel_level_weight = self.get_local_weights(residual_sr)
        return patch_level_weight * pixel_level_weight

    def forward(self, net_output: Tensor, gt: Tensor) -> Tensor:
        overall_weight = self.get_refined_artifact_map(gt, net_output)
        return self.criterion(overall_weight * net_output, overall_weight * gt)

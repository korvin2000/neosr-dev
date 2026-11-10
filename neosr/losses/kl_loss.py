import torch
from torch import Tensor
from torch.nn import functional as F

from neosr.losses.base import Loss
from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class kl_loss(Loss):
    """KL divergence between the channel softmaxes of the output and the target, per pixel.

    What it does not do (stage-2 L-12, measured): it is invariant to a per-pixel offset shared by
    the channels (an offset of 0.3 costs 6e-6), so it does not constrain brightness; `batchmean`
    sums over C, H and W, so the value grows with the patch area (x3.8 from 128 to 256 px) and a
    weight does not transfer across `patch_size`; on one-channel input it is identically 0.
    """

    def forward(self, net_output: Tensor, gt: Tensor):
        # Convert net_output and gt to probability distributions
        net_output_prob = F.softmax(net_output, dim=1)
        gt_prob = F.softmax(gt, dim=1)
        # Compute log probabilities
        net_output_log_prob = torch.log(net_output_prob + 1e-8)
        # Compute KL divergence
        loss = F.kl_div(net_output_log_prob, gt_prob, reduction="batchmean")
        # balance loss to avoid issues
        return loss * 0.03

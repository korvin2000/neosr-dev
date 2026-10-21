import torch
from torch import Tensor
from torch.nn import functional as F

from neosr.losses.base import Loss


class kl_loss(Loss):
    """KL-Divergence loss."""

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

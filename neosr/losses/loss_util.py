"""Utility helpers for loss-related operations."""

from torch import Tensor
from torch.nn import functional as F
import torch


def get_refined_artifact_map(img_gt: Tensor, img_output: Tensor, ksize: int = 7) -> Tensor:
    """Calculate the LDL artifact map for predicted images.

    This helper mirrors the logic used by :class:`neosr.losses.ldl_loss`
    to derive a pixel-wise artifact weighting that combines both
    patch-level and local variance observations.

    Args:
    ----
        img_gt: Ground truth images shaped ``(N, C, H, W)``.
        img_output: Predicted images shaped ``(N, C, H, W)``.
        ksize: Size of the local window used to compute local variance.

    Returns:
    -------
        Tensor: Artifact weights with the same spatial dimensions as the
        inputs, suitable for re-weighting losses.
    """

    residual_sr = torch.sum(torch.abs(img_gt - img_output), dim=1, keepdim=True)

    patch_level_weight = torch.var(
        residual_sr.clone(), dim=(-1, -2, -3), keepdim=True
    ) ** (1 / 5)

    pad = (ksize - 1) // 2
    residual_pad = F.pad(residual_sr, pad=[pad, pad, pad, pad], mode="reflect")
    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    pixel_level_weight = (
        torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True)
        .squeeze(-1)
        .squeeze(-1)
    )

    return patch_level_weight * pixel_level_weight

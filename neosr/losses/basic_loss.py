import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.losses.base import Loss

_reduction_modes: list[str] = ["none", "mean", "sum"]


def l1_loss(pred: Tensor, target: Tensor, reduction: str) -> Tensor:
    return F.l1_loss(pred, target, reduction=reduction)


def mse_loss(pred: Tensor, target: Tensor, reduction: str) -> Tensor:
    return F.mse_loss(pred, target, reduction=reduction)


def huber_loss(
    pred: torch.Tensor, target: torch.Tensor, reduction: str, delta: float = 1.0
) -> Tensor:
    return F.huber_loss(pred, target, reduction=reduction, delta=delta)


class psnr_loss(Loss):
    """-PSNR (peak 1) as a loss: 10 * log10(MSE_i + eps) per image in dB, then `reduction` over the
    images ('mean' | 'sum': a training term is a scalar); fp32 (the log of small MSEs)."""

    fp32 = True

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            msg = f"psnr_loss: reduction must be 'mean' or 'sum', got {reduction!r}"
            raise ValueError(msg)
        if (
            isinstance(eps, bool)
            or not isinstance(eps, int | float)
            or not 0 < eps < math.inf
        ):
            msg = f"psnr_loss: eps must be a finite number > 0, got {eps!r}"
            raise ValueError(msg)
        self.reduction, self.eps = reduction, float(eps)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        db = 10 * torch.log10((pred - target).square().flatten(1).mean(1) + self.eps)
        return getattr(db, self.reduction)()


class L1Loss(Loss):
    """L1 (mean absolute error, MAE) loss.

    Args:
    ----
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.

    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            msg = f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            raise ValueError(msg)

        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Args:
        ----
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.

        """
        return l1_loss(pred, target, reduction=self.reduction)


class MSELoss(Loss):
    """MSE (L2) loss.

    Args:
    ----
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.

    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            msg = f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            raise ValueError(msg)

        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Args:
        ----
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.

        """
        return mse_loss(pred, target, reduction=self.reduction)


class HuberLoss(Loss):
    """HuberLoss

    Args:
    ----
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        delta (float): Specifies the threshold at which to change between
            delta-scaled L1 and L2 loss. The value must be positive. Default: 1.0

    """

    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            msg = f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            raise ValueError(msg)

        self.reduction = reduction
        self.delta = delta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Args:
        ----
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.

        """
        return huber_loss(pred, target, delta=self.delta, reduction=self.reduction)


class chc_loss(Loss):
    """Clipped pseudo-Huber with Cosine Similarity Loss.

       For reference on research, see:
       https://github.com/HolmesShuan/AIM2020-Real-Super-Resolution
       https://github.com/dmarnerides/hdr-expandnet

    Args:
    ----
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        criterion (str): Specifies the loss to apply.
            Supported choices are 'l1' and 'huber'. Default: 'huber'.
        loss_lambda (float):  constant factor that adjusts the contribution of the cosine similarity term
        clip_min (float): threshold that sets the gradients of well-trained pixels to zeros
        clip_max (float): max clip limit, can act as a noise filter

    """

    def __init__(
        self,
        reduction: str = "mean",
        criterion: str = "huber",
        loss_lambda: float = 0,
        clip_min: float = 0.003921,
        clip_max: float = 0.996078,
    ) -> None:
        super().__init__()

        if reduction not in {"none", "mean", "sum"}:
            msg = f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            raise ValueError(msg)

        if criterion not in {"l1", "huber"}:
            msg = f"{criterion} not implemented."
            raise NotImplementedError(msg)

        self.reduction = reduction
        self.criterion = criterion
        self.similarity = nn.CosineSimilarity(dim=1)
        self.loss_lambda = loss_lambda  # 5/255 = 0.019607
        self.clip_min = clip_min  # 1/255 = 0.003921
        self.clip_max = clip_max

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Args:
        ----
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.

        """
        # at least fp32: the 1e-12 charbonnier eps and the cosine eps underflow in
        # fp16/bf16 -> NaN grads; fp64 inputs keep their precision
        dtype = torch.promote_types(pred.dtype, torch.float32)
        pred, target = pred.to(dtype), target.to(dtype)
        diff = pred - target
        # absolute or pseudo-huber (charbonnier)
        loss = diff.abs() if self.criterion == "l1" else torch.sqrt(diff**2 + 1e-12)
        if self.loss_lambda:
            loss = loss + self.loss_lambda * (1 - self.similarity(pred, target)).mean()
        loss = torch.clamp(loss, self.clip_min, self.clip_max)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

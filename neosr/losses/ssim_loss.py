import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.losses.base import Loss


class GaussianFilter2D(nn.Module):
    """Depthwise 2D Gaussian as a (1, k) then a (k, 1) zero-padded conv: the separable form of the
    k x k window (fp32 rounding apart), 2k instead of k^2 MACs per pixel. Takes up to
    3 * in_channels channels, so several moments can share one call."""

    def _get_gaussian_window1d(self) -> Tensor:
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        return w / w.sum()

    def __init__(
        self,
        window_size: int = 11,
        in_channels: int = 3,
        sigma: float = 1.5,
        padding: int | None = None,
    ) -> None:
        """2D Gaussian Filer.

        Args:
        ----
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
            padding (int, optional): The padding of the gaussian filter. Defaults to None.
                If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.

        """
        super().__init__()
        self.window_size = window_size
        if window_size % 2 != 1:
            msg = "Window size must be odd."
            raise ValueError(msg)
        self.padding = padding if padding is not None else window_size // 2
        self.sigma = sigma

        k = self._get_gaussian_window1d().repeat(3 * in_channels, 1)
        self.register_buffer("row", k.view(-1, 1, 1, window_size), persistent=False)
        self.register_buffer("col", k.view(-1, 1, window_size, 1), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        c, p = x.shape[1], self.padding
        x = F.conv2d(x, self.row[:c], padding=(0, p), groups=c)
        return F.conv2d(x, self.col[:c], padding=(p, 0), groups=c)


class mssim_loss(Loss):
    fp32 = True

    def __init__(
        self,
        window_size: int = 11,
        in_channels: int = 3,
        sigma: float = 1.5,
        K1: float = 0.01,
        K2: float = 0.03,
        L: int = 1,
        padding: int | None = None,
    ) -> None:
        """Adapted from 'A better pytorch-based implementation for the mean structural
            similarity. Differentiable simpler SSIM and MS-SSIM.':
                https://github.com/lartpang/mssim.pytorch.

            Calculate the mean SSIM (MSSIM) between two 4D tensors.

        Args:
        ----
            window_size (int): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float): The sigma of the gaussian filter. Defaults to 1.5.
            K1 (float): K1 of MSSIM. Defaults to 0.01.
            K2 (float): K2 of MSSIM. Defaults to 0.03.
            L (int): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
            padding (int, optional): The padding of the gaussian filter. Defaults to None. If it is set to None,
                the filter will use window_size//2 as the padding. Another common setting is 0.

        """
        super().__init__()

        self.window_size = window_size
        self.C1 = (K1 * L) ** 2  # equ 7 in ref1
        self.C2 = (K2 * L) ** 2  # equ 7 in ref1

        self.gaussian_filter = GaussianFilter2D(
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            padding=padding,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """x, y (Tensor): tensors of shape (N,C,H,W)
        Returns: Tensor.
        """
        assert x.shape == y.shape, f"x: {x.shape} and y: {y.shape} must be the same"
        assert x.ndim == y.ndim == 4, f"x: {x.ndim} and y: {y.ndim} must be 4"

        if x.type() != self.gaussian_filter.row.type():
            x = x.type_as(self.gaussian_filter.row)
        if y.type() != self.gaussian_filter.row.type():
            y = y.type_as(self.gaussian_filter.row)

        return 1.0 - self.msssim(x, y)

    def _ssim(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        # one filter call per side: the gt-only moments stay out of the graph (and its saved memory)
        mu_x, xx, xy = self.gaussian_filter(torch.cat([x, x * x, x * y], 1)).chunk(3, 1)
        mu_y, yy = self.gaussian_filter(torch.cat([y, y * y], 1)).chunk(2, 1)
        sigma2_x = xx - mu_x * mu_x  # equ 15
        sigma2_y = yy - mu_y * mu_y  # equ 15
        sigma_xy = xy - mu_x * mu_y  # equ 16

        A1 = 2 * mu_x * mu_y + self.C1
        A2 = 2 * sigma_xy + self.C2
        B1 = mu_x.pow(2) + mu_y.pow(2) + self.C1
        B2 = sigma2_x + sigma2_y + self.C2

        # equ 12, 13 in ref1
        l1 = A1 / B1
        cs = A2 / B2
        ssim = l1 * cs

        return ssim, cs

    def msssim(self, x: Tensor, y: Tensor) -> Tensor:
        ms_components = []
        for i, w in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
            ssim, cs = self._ssim(x, y)
            # reduce per (N, C) and clamp: negative (anti-correlated) bases give NaN in **w
            ssim = ssim.mean((2, 3)).clamp_min(1e-6)
            cs = cs.mean((2, 3)).clamp_min(1e-6)

            if i == 4:
                ms_components.append(ssim**w)
            else:
                ms_components.append(cs**w)
                padding = [s % 2 for s in x.shape[2:]]  # spatial padding
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)

        # equ 7 in ref2; a mul chain, not prod(): prod's backward syncs to look for zeros
        return math.prod(ms_components).mean()

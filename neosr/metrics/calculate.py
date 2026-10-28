from typing import cast

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.metrics.metric_util import reorder_image, to_y_channel
from neosr.utils.img_util import img2tensor


def _crop(img: np.ndarray, crop_border: int) -> np.ndarray:
    if crop_border < 0:
        msg = f"crop_border must be >= 0, got {crop_border}."
        raise ValueError(msg)
    if crop_border == 0:
        return img
    return img[crop_border:-crop_border, crop_border:-crop_border, ...]


def calculate_psnr(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 4,
    input_order: str = "HWC",
    test_y_channel: bool = False,
) -> float:
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
    ----
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
    -------
        float: PSNR result.

    """
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )
    if input_order not in {"HWC", "CHW"}:
        msg = f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        raise ValueError(msg)
    img = _crop(reorder_image(img, input_order=input_order), crop_border)
    img2 = _crop(reorder_image(img2, input_order=input_order), crop_border)
    if img.size == 0:
        msg = f"PSNR input is empty after cropping {crop_border} px."
        raise ValueError(msg)

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2) ** 2)
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(255.0 * 255.0 / mse)


@staticmethod
def _ssim(img: np.ndarray | MatLike, img2: np.ndarray | MatLike) -> float:
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
    ----
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
    -------
        float: SSIM result.

    """
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # valid mode for window size 11
    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = (
        cv2.filter2D(cast("MatLike", (img * img2)), -1, window)[5:-5, 5:-5] - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def calculate_ssim(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 4,
    input_order: str = "HWC",
    test_y_channel: bool = False,
) -> float:
    """Calculate SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
    ----
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
    -------
        float: SSIM result.

    """
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )
    if input_order not in {"HWC", "CHW"}:
        msg = f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        raise ValueError(msg)
    img = _crop(reorder_image(img, input_order=input_order), crop_border)
    img2 = _crop(reorder_image(img2, input_order=input_order), crop_border)
    if min(img.shape[:2]) < 11:
        msg = f"SSIM needs at least 11x11 px after cropping {crop_border} px, got {img.shape[:2]}."
        raise ValueError(msg)

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    ssims.extend([_ssim(img[..., i], img2[..., i]) for i in range(img.shape[2])])
    return float(np.array(ssims).mean())


# ------------------------------------------------------------------------------------------------
# On-device twins (`*_pt`) used by validation. They take what `img_util.quantize` returns for sr and
# gt: float32 RGB NCHW holding the uint8 values 0..255 that tensor2img would write. They reproduce
# the numpy references above (crop, Y conversion, float64 statistics) to rounding level and return a
# 0-d tensor on the input device, so validation syncs once per event instead of once per image.


def _prep_pt(
    img: Tensor, img2: Tensor, crop_border: int, input_order: str
) -> tuple[Tensor, Tensor]:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )
    if input_order not in {"HWC", "CHW"}:  # numpy layouts: tensors are always NCHW
        msg = f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        raise ValueError(msg)
    if crop_border < 0:
        msg = f"crop_border must be >= 0, got {crop_border}."
        raise ValueError(msg)
    if crop_border:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    return img, img2


def _div255(x: Tensor) -> Tensor:
    # CUDA divides by a Python scalar as a multiply by its reciprocal (not correctly rounded, unlike
    # numpy and the CPU); a 0-d device divisor gives the true quotient (a fill kernel, no copy)
    return x / torch.full((), 255.0, dtype=x.dtype, device=x.device)


def _y_pt(img: Tensor) -> Tensor:
    """to_y_channel's dtype sequence: fp32 / 255, fp64 BT.601 dot (BGR weights) + 16, / 255, fp32, x 255."""
    img = _div255(img.float())
    if img.shape[1] == 3:
        r, g, b = img.double().unbind(1)
        y = b * 24.966 + g * 128.553 + r * 65.481 + 16.0
        img = _div255(y).float().unsqueeze(1)
    return img * 255.0


def calculate_psnr_pt(
    img: Tensor,
    img2: Tensor,
    crop_border: int = 4,
    input_order: str = "HWC",
    test_y_channel: bool = False,
) -> Tensor:
    """calculate_psnr on quantised tensors (see above); integer squared errors sum exactly in fp64."""
    img, img2 = _prep_pt(img, img2, crop_border, input_order)
    if img.numel() == 0:
        msg = f"PSNR input is empty after cropping {crop_border} px."
        raise ValueError(msg)
    if test_y_channel:
        img, img2 = _y_pt(img), _y_pt(img2)
    mse = (img.double() - img2.double()).square().mean()
    return 10.0 * torch.log10(255.0 * 255.0 / mse)  # mse 0 -> inf, as the reference


# float64 (11, 1); the reference filters with its outer product. One copy per device: an H2D copy
# per call would block the host on every image
_SSIM_KERNEL = cv2.getGaussianKernel(11, 1.5)
_ssim_kernels: dict[torch.device, Tensor] = {}


def _ssim_kernel(device: torch.device) -> Tensor:
    k = _ssim_kernels.get(device)
    if k is None:
        k = _ssim_kernels[device] = (
            torch.from_numpy(_SSIM_KERNEL).to(device).view(1, 1, 11, 1)
        )
    return k


def calculate_ssim_pt(
    img: Tensor,
    img2: Tensor,
    crop_border: int = 4,
    input_order: str = "HWC",
    test_y_channel: bool = False,
) -> Tensor:
    """calculate_ssim on quantised tensors (see above): the same Gaussian window applied separably in
    fp64 to the five statistics at once, valid region only (what the reference keeps of filter2D)."""
    img, img2 = _prep_pt(img, img2, crop_border, input_order)
    if min(img.shape[-2:]) < 11:
        msg = f"SSIM needs at least 11x11 px after cropping {crop_border} px, got {tuple(img.shape[-2:])}."
        raise ValueError(msg)
    if test_y_channel:
        img, img2 = _y_pt(img), _y_pt(img2)
    n, c, h, w = img.shape
    k = _ssim_kernel(img.device)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssims = []
    for i in range(c):  # one channel at a time: the fp64 statistics are 5x the channel
        x, y = img[:, i].double(), img2[:, i].double()
        stats = torch.stack((x, y, x * x, y * y, x * y), 1).view(n * 5, 1, h, w)
        stats = F.conv2d(F.conv2d(stats, k), k.view(1, 1, 1, 11))
        mu1, mu2, xx, yy, xy = stats.view(n, 5, h - 10, w - 10).unbind(1)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * (xy - mu1_mu2) + c2)) / (
            (mu1_sq + mu2_sq + c1) * ((xx - mu1_sq) + (yy - mu2_sq) + c2)
        )
        ssims.append(ssim_map.mean((-2, -1)))
    return torch.stack(ssims, -1).mean()  # per channel, then over channels


# Building DISTS/TOPIQ costs far more than moving it (0.3-0.6 s and a 0.3-1 GiB host spike per
# build), so each network is built once per process and kept here by name. release_metric_nets(),
# called by the model's validation() after each event, parks the nets on the CPU: the VRAM is freed
# between events and the next event moves them back (~200 MB) instead of rebuilding them.
_metric_nets: dict[str, nn.Module] = {}


# the nets pull in torchvision models and weights: imported on first use
def _dists() -> nn.Module:
    return _streaming_dists()


def _topiq() -> nn.Module:
    from neosr.metrics.topiq import topiq  # noqa: PLC0415

    return topiq()


class _streaming_dists(nn.Module):
    """dists_loss (as_loss=False) evaluated one VGG stage at a time for both images: the same ops in
    the same order as dists_loss.forward (bit-identical value), but the two six-level pyramids are
    never held at once (MT-13: -28 % peak at 1080p). The weights are dists_loss's own (`net.`)."""

    def __init__(self) -> None:
        super().__init__()
        from neosr.losses.dists_loss import dists_loss  # noqa: PLC0415

        self.net = dists_loss(as_loss=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        net, c1, c2 = self.net, 1e-6, 1e-6
        w_sum = net.alpha.sum() + net.beta.sum()
        alpha = torch.split(net.alpha / w_sum, net.chns, dim=1)
        beta = torch.split(net.beta / w_sum, net.chns, dim=1)
        stages = (net.stage1, net.stage2, net.stage3, net.stage4, net.stage5)
        hx, hy = (x - net.mean) / net.std, (y - net.mean) / net.std
        fx, fy = x, y  # level 0 is the raw input
        dist1 = dist2 = 0
        for k in range(len(net.chns)):
            if k:
                hx, hy = stages[k - 1](hx), stages[k - 1](hy)
                fx, fy = hx, hy
            x_mean = fx.mean([2, 3], keepdim=True)
            y_mean = fy.mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)
            x_var = ((fx - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((fy - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (fx * fy).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)
        return 1 - (dist1 + dist2).squeeze()  # type: ignore[operator]


_metric_builders = {"dists": _dists, "topiq": _topiq}


def release_metric_nets() -> None:
    for net in _metric_nets.values():
        net.to("cpu")


def _metric_net(name: str, device: torch.device) -> nn.Module:
    net = _metric_nets.get(name)
    if net is None:
        net = _metric_builders[name]().eval().requires_grad_(False)
        _metric_nets[name] = net
    if next(net.parameters()).device != device:
        net.to(device)
    return net


def _metric_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        # "cuda" and "cuda:N" are the same placement: follow the current device
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _net_metric_pt(
    name: str, img: Tensor, img2: Tensor, device: torch.device | str | None
) -> Tensor:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )
    device = _metric_device(device)
    net = _metric_net(name, device)
    if img.shape[1] == 1:  # grayscale: both networks expect 3 channels
        img, img2 = img.expand(-1, 3, -1, -1), img2.expand(-1, 3, -1, -1)
    # quantised / 255: exactly the uint8 / 255 input of the reference
    x, y = _div255(img.to(device)), _div255(img2.to(device))
    with torch.inference_mode():
        return net(x, y)


def calculate_dists_pt(
    img: Tensor, img2: Tensor, device: torch.device | str | None = None
) -> Tensor:
    """DISTS (lower is better) on quantised tensors (see above)."""
    return _net_metric_pt("dists", img, img2, device)


def calculate_topiq_pt(
    img: Tensor, img2: Tensor, device: torch.device | str | None = None
) -> Tensor:
    """Full-reference TOPIQ (higher is better) on quantised tensors (see above)."""
    return _net_metric_pt("topiq", img, img2, device)


def _calculate_net_metric(
    name: str, img: np.ndarray, img2: np.ndarray, device: torch.device | str | None
) -> float:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )
    if img.ndim == 2:
        img, img2 = img[..., None], img2[..., None]
    # BGR uint8 HWC -> the quantised RGB NCHW tensors of the _pt functions
    x, y = (cast("Tensor", t).unsqueeze(0) for t in img2tensor([img, img2]))
    return _net_metric_pt(name, x, y, device).item()


def calculate_dists(
    img: np.ndarray, img2: np.ndarray, device: torch.device | str | None = None
) -> float:
    """Calculate the DISTS metric (lower is better).

    Args:
    ----
        img (np.ndarray): Images with range [0, 255] with order 'HWC' (BGR).
        img2 (np.ndarray): Images with range [0, 255] with order 'HWC' (BGR).
        device: Device to run on. Default: cuda if available, else cpu.

    Returns:
    -------
        float: DISTS result.

    """
    return _calculate_net_metric("dists", img, img2, device)


def calculate_topiq(
    img: np.ndarray, img2: np.ndarray, device: torch.device | str | None = None
) -> float:
    """Calculate the full-reference TOPIQ metric (higher is better).

    Args:
    ----
        img (np.ndarray): Images with range [0, 255] with order 'HWC' (BGR).
        img2 (np.ndarray): Images with range [0, 255] with order 'HWC' (BGR).
        device: Device to run on. Default: cuda if available, else cpu.

    Returns:
    -------
        float: TOPIQ result.

    """
    return _calculate_net_metric("topiq", img, img2, device)

from typing import cast

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from torch import Tensor, nn

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
    ssim_result = np.array(ssims).mean()
    if ssim_result <= 0:
        return float("inf")
    return ssim_result


# Building DISTS/TOPIQ costs far more than one forward pass, so each network is
# built once per (name, device) and reused. Call release_metric_nets() after
# validation to free the weights during training.
_metric_nets: dict[tuple[str, torch.device], nn.Module] = {}


# the nets pull in torchvision models and weights: imported on first use
def _dists() -> nn.Module:
    from neosr.losses.dists_loss import dists_loss  # noqa: PLC0415

    return dists_loss(as_loss=False)


def _topiq() -> nn.Module:
    from neosr.metrics.topiq import topiq  # noqa: PLC0415

    return topiq()


_metric_builders = {"dists": _dists, "topiq": _topiq}


def release_metric_nets() -> None:
    _metric_nets.clear()


def _metric_net(name: str, device: torch.device) -> nn.Module:
    net = _metric_nets.get((name, device))
    if net is None:
        net = _metric_builders[name]().to(device).eval().requires_grad_(False)
        _metric_nets[name, device] = net
    return net


def _calculate_net_metric(
    name: str, img: np.ndarray, img2: np.ndarray, device: torch.device | str | None
) -> float:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        # "cuda" and "cuda:N" must share one cache entry and follow the current device
        device = torch.device("cuda", torch.cuda.current_device())
    if img.ndim == 2:  # grayscale: both networks expect 3 channels
        img, img2 = (np.repeat(i[..., None], 3, axis=2) for i in (img, img2))
    net = _metric_net(name, device)
    # BGR uint8 HWC -> RGB [0, 1] NCHW
    x, y = (
        cast("Tensor", t).div(255).unsqueeze(0).to(device)
        for t in img2tensor([img, img2])
    )
    with torch.inference_mode():
        return net(x, y).item()


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

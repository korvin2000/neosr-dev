import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from neosr.utils.rng import rng

rng = rng()

# All augmentations work at native resolutions: boxes are drawn in LQ pixels and
# multiplied by `scale` for GT, so LQ is never resampled to GT size and back.


def _check_sizes(img_gt: Tensor, img_lq: Tensor, scale: int) -> None:
    if img_gt.shape[-2:] != (img_lq.shape[-2] * scale, img_lq.shape[-1] * scale):
        msg = f"img_gt {tuple(img_gt.shape)} must be {scale}x img_lq {tuple(img_lq.shape)}."
        raise ValueError(msg)


def _rand_box(h: int, w: int, cut_h: int, cut_w: int) -> tuple[int, int, int, int]:
    """Random box (top, bottom, left, right) with a uniform center, clipped to the
    image. Cut sizes are at least 1, so the box is never empty.
    """
    cut_h, cut_w = max(cut_h, 1), max(cut_w, 1)
    top = int(rng.integers(h)) - cut_h // 2  # type: ignore[attr-defined]
    left = int(rng.integers(w)) - cut_w // 2  # type: ignore[attr-defined]
    return (max(top, 0), min(top + cut_h, h), max(left, 0), min(left + cut_w, w))


def _perm(img: Tensor) -> Tensor:
    return torch.randperm(img.shape[0], device=img.device)


@torch.no_grad()
def mixup(
    img_gt: Tensor,
    img_lq: Tensor,
    alpha_min: float = 0.4,
    alpha_max: float = 0.6,
    scale: int = 1,
) -> tuple[Tensor, Tensor]:
    r"""MixUp augmentation.

    "Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)".
    In ICLR, 2018.
        https://github.com/facebookresearch/mixup-cifar10

    Args:
    ----
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W),
            GT `scale` times the size of LQ.
        alpha_min/max (float): The given min/max mixing ratio.
        scale (int): Scale ratio between GT and LQ.

    """
    _check_sizes(img_gt, img_lq, scale)
    lam = rng.uniform(alpha_min, alpha_max)  # type: ignore[attr-defined]

    # each image is mixed with the same partner in GT and LQ
    idx = _perm(img_gt)
    img_gt = lam * img_gt + (1 - lam) * img_gt[idx]
    img_lq = lam * img_lq + (1 - lam) * img_lq[idx]

    return img_gt, img_lq


@torch.no_grad()
def cutmix(
    img_gt: Tensor, img_lq: Tensor, alpha: float = 0.9, scale: int = 1
) -> tuple[Tensor, Tensor]:
    r"""CutMix augmentation.

    "CutMix: Regularization Strategy to Train Strong Classifiers with
    Localizable Features (https://arxiv.org/abs/1905.04899)". In ICCV, 2019.
        https://github.com/clovaai/CutMix-PyTorch

    Args:
    ----
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W),
            GT `scale` times the size of LQ. Modified in place.
        alpha (float): The given maximum mixing ratio.
        scale (int): Scale ratio between GT and LQ.

    """
    _check_sizes(img_gt, img_lq, scale)
    h, w = img_lq.shape[-2:]
    cut_rat = np.sqrt(1.0 - rng.uniform(0, alpha))  # type: ignore[attr-defined]
    y1, y2, x1, x2 = _rand_box(h, w, int(h * cut_rat), int(w * cut_rat))
    s = scale

    idx = _perm(img_gt)
    img_gt[..., y1 * s : y2 * s, x1 * s : x2 * s] = img_gt[idx][
        ..., y1 * s : y2 * s, x1 * s : x2 * s
    ]
    img_lq[..., y1:y2, x1:x2] = img_lq[idx][..., y1:y2, x1:x2]

    return img_gt, img_lq


@torch.no_grad()
def resizemix(
    img_gt: Tensor,
    img_lq: Tensor,
    scope: tuple[float, float] = (0.2, 0.9),
    scale: int = 1,
) -> tuple[Tensor, Tensor]:
    r"""ResizeMix augmentation.

    "ResizeMix: Mixing Data with Preserved Object Information and True Labels
    (https://arxiv.org/abs/2012.11101)".

    Args:
    ----
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W),
            GT `scale` times the size of LQ. Modified in place.
        scope (float): The given min/max box size ratio.
        scale (int): Scale ratio between GT and LQ.

    """
    _check_sizes(img_gt, img_lq, scale)
    h, w = img_lq.shape[-2:]
    tao = rng.uniform(scope[0], scope[1])  # type: ignore[attr-defined]
    y1, y2, x1, x2 = _rand_box(h, w, int(h * tao), int(w * tao))
    s = scale

    idx = _perm(img_gt)
    gt_resized = F.interpolate(
        img_gt[idx], ((y2 - y1) * s, (x2 - x1) * s), mode="bicubic"
    ).clamp(0, 1)
    lq_resized = F.interpolate(img_lq[idx], (y2 - y1, x2 - x1), mode="bicubic").clamp(
        0, 1
    )
    img_gt[..., y1 * s : y2 * s, x1 * s : x2 * s] = gt_resized
    img_lq[..., y1:y2, x1:x2] = lq_resized

    return img_gt, img_lq


@torch.no_grad()
def cutblur(
    img_gt: Tensor, img_lq: Tensor, alpha: float = 0.7, scale: int = 1
) -> tuple[Tensor, Tensor]:
    r"""CutBlur Augmentation.

    "Rethinking Data Augmentation for Image Super-resolution:
        A Comprehensive Analysis and a New Strategy"
        (https://arxiv.org/abs/2004.00448)

    Args:
    ----
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W),
            GT `scale` times the size of LQ. img_lq is modified in place.
        alpha (float): The given max mixing ratio.
        scale (int): Scale ratio between GT and LQ.

    """
    _check_sizes(img_gt, img_lq, scale)
    h, w = img_lq.shape[-2:]
    lam = rng.uniform(0.2, alpha)  # type: ignore[attr-defined]
    y1, y2, x1, x2 = _rand_box(h, w, int(h * lam), int(w * lam))
    s = scale

    # paste the GT region into LQ, downscaled to LQ resolution
    patch = img_gt[..., y1 * s : y2 * s, x1 * s : x2 * s]
    if s > 1:
        patch = F.interpolate(
            patch, (y2 - y1, x2 - x1), mode="bicubic", antialias=True
        ).clamp(0, 1)
    img_lq[..., y1:y2, x1:x2] = patch

    return img_gt, img_lq


_AUGMENTS = {
    "mixup": mixup,
    "cutmix": cutmix,
    "resizemix": resizemix,
    "cutblur": cutblur,
}
# application order when several are drawn
_ORDER = ("cutmix", "mixup", "resizemix", "cutblur")


@torch.no_grad()
def apply_augment(
    img_gt: Tensor,
    img_lq: Tensor,
    scale: int = 1,
    augs: tuple[str, ...] = ("none", "mixup", "cutmix", "resizemix", "cutblur"),
    prob: tuple[float, ...] | None = (0.1, 0.3, 0.2, 0.7, 0.8),
    multi_prob: float = 0.3,
) -> tuple[Tensor, Tensor]:
    r"""Applies Augmentations. The inputs are never modified.

    Args:
    ----
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
        scale (int): Scale ratio between GT and LQ.
        augs (list): List of possible augmentations to apply. Supported
            values are: "none", "mixup", "cutmix", "resizemix" and "cutblur"
        prob (list | None): Relative weight of each augmentation (zero
            disables it). Default None: uniform.
        multi_prob (float): Probability that to apply >1 augmentations from the list.
            Default: 0.3
    Return:
        img_gt, img_lq (Tensor)

    """
    if prob is None:
        prob = (1.0,) * len(augs)
    if len(augs) != len(prob):
        msg = "Length of 'augmentation' and aug_prob don't match!"
        raise ValueError(msg)
    unknown = set(augs) - {"none", *_AUGMENTS}
    if unknown:
        msg = f"Unknown augmentations: {sorted(unknown)}."
        raise ValueError(msg)
    if any(p < 0 for p in prob) or sum(prob) <= 0:
        msg = f"aug_prob must be non-negative with a positive sum, got {prob}."
        raise ValueError(msg)

    active = [(a, p) for a, p in zip(augs, prob, strict=True) if p > 0]
    names = [a for a, _ in active]
    weights = np.array([p for _, p in active], dtype=np.float64)
    weights /= weights.sum()

    if len(active) > 1 and rng.random() < multi_prob:  # type: ignore[attr-defined]
        num_augs = rng.integers(2, len(active)) if len(active) > 2 else len(active)  # type: ignore[attr-defined]
        chosen = set(rng.choice(names, size=num_augs, replace=False, p=weights))  # type: ignore[attr-defined]
    else:
        chosen = {rng.choice(names, p=weights)}  # type: ignore[attr-defined]

    chosen = [a for a in _ORDER if a in chosen]
    if not chosen:  # only "none"
        return img_gt, img_lq
    if img_gt.shape[0] == 1 and chosen != ["cutblur"]:
        msg = "Augmentations need batch >1 to work."
        raise ValueError(msg)

    # the mixing functions write in place: never touch the caller's tensors
    img_gt, img_lq = img_gt.clone(), img_lq.clone()
    for aug in chosen:
        img_gt, img_lq = _AUGMENTS[aug](img_gt, img_lq, scale=scale)

    return img_gt, img_lq

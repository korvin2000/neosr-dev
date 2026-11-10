"""Separable antialiased resampling with a kernel table (stage 3 OTF design, D1).

The PIL algorithm, vectorised: per axis one dense (out, in) weight matrix whose kernel is stretched by
the scale only when downscaling; rows are truncated at the border and renormalised. `aa_bicubic` and
`aa_bilinear` equal `F.interpolate(mode, antialias=True)` (to 1e-15 in float64).
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["KERNELS", "resample", "scaled_size"]


def _triangle(x: Tensor) -> Tensor:
    return (1 - x.abs()).clamp(min=0)


def _cubic(b: float, c: float) -> Callable[[Tensor], Tensor]:
    """Mitchell-Netravali family: (0, 0.5) is the Keys cubic a = -0.5, (1/3, 1/3) Mitchell."""

    def kernel(x: Tensor) -> Tensor:
        x = x.abs()
        near = (12 - 9 * b - 6 * c) * x**3 + (-18 + 12 * b + 6 * c) * x**2 + (6 - 2 * b)
        far = (
            (-b - 6 * c) * x**3
            + (6 * b + 30 * c) * x**2
            + (-12 * b - 48 * c) * x
            + (8 * b + 24 * c)
        )
        return torch.where(x < 1, near, torch.where(x < 2, far, 0.0)) / 6

    return kernel


def _lanczos(a: int) -> Callable[[Tensor], Tensor]:
    def kernel(x: Tensor) -> Tensor:
        return torch.where(x.abs() < a, torch.sinc(x) * torch.sinc(x / a), 0.0)

    return kernel


# name -> kernel in input-pixel units, zero outside its support (None: the box, see _box_window)
KERNELS: dict[str, Callable[[Tensor], Tensor] | None] = {
    "aa_box": None,
    "aa_bilinear": _triangle,
    "aa_bicubic": _cubic(0, 0.5),
    "aa_mitchell": _cubic(1 / 3, 1 / 3),
    "aa_lanczos2": _lanczos(2),
    "aa_lanczos3": _lanczos(3),
}


def _box_window(n: int, m: int, device: torch.device) -> Tensor:
    """The half-open box [-0.5, 0.5) around each output centre, decided in integers: a float offset
    puts a sample that sits exactly on a window boundary into the wrong window."""
    i = torch.arange(m, device=device)[:, None]
    j = torch.arange(n, device=device)
    if n >= m:  # the input centre (j + 0.5) / n lies in [i / m, (i + 1) / m)
        inside = ((2 * j + 1) * m >= 2 * i * n) & ((2 * j + 1) * m < 2 * (i + 1) * n)
    else:  # nearest sample: the output centre lies in (j - 0.5, j + 0.5]
        inside = (2 * j * m < (2 * i + 1) * n) & ((2 * i + 1) * n <= 2 * (j + 1) * m)
    return inside.double()


def _weights(n: int, m: int, kernel: str, like: Tensor) -> Tensor:
    """(m, n) matrix that resamples an axis of n samples to m; built in float64 on `like`'s device."""
    fn = KERNELS[kernel]
    if fn is None:
        w = _box_window(n, m, like.device)
    else:
        scale = n / m
        centre = (
            torch.arange(m, dtype=torch.float64, device=like.device) + 0.5
        ) * scale - 0.5
        pos = torch.arange(n, dtype=torch.float64, device=like.device)
        w = fn((pos - centre[:, None]) / max(scale, 1.0))
    return (w / w.sum(1, keepdim=True)).to(like.dtype)


def resample(x: Tensor, size: tuple[int, int], kernel: str) -> Tensor:
    """`x` (..., h, w) resampled to `size` = (height, width) with a `KERNELS` name, in `x`'s dtype.

    An axis whose length does not change is returned as is (also for kernels that do not interpolate,
    such as Mitchell); coordinates follow the sizes (in / out), not a scale factor.
    """
    h, w = x.shape[-2:]
    if min(size) < 1:
        msg = f"resample: output size must be positive, got {tuple(size)} from {(h, w)}"
        raise ValueError(msg)
    # an ambient autocast would run the matmuls, and return the image, in bf16/fp16
    with torch.autocast(x.device.type, enabled=False):
        if size[0] != h:
            x = _weights(h, size[0], kernel, x) @ x
        if size[1] != w:
            x = x @ _weights(w, size[1], kernel, x).T
    return x


def scaled_size(h: int, w: int, factor: float) -> tuple[int, int]:
    """The output size of `F.interpolate(scale_factor=factor)`: floor(n * factor)."""
    return math.floor(h * factor), math.floor(w * factor)

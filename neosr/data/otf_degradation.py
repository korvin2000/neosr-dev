"""On-the-fly (Real-ESRGAN) degradation of GT batches into LQ batches, plus the training pair pool.

Python `random` draws the resize kind and the interpolation modes, `neosr.utils.rng.rng` the scalar
probabilities and scales, torch's RNG the per-sample noise and JPEG qualities.
"""

import logging
import random
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from neosr.config import ConfigError
from neosr.data.degradations import (  # type: ignore[attr-defined]
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from neosr.utils.diffjpeg import DiffJPEG, filter2D  # type: ignore[attr-defined]
from neosr.utils.rng import rng

# key -> required list length (0: scalar); every key exists per stage, suffixed "" and "2"
_STAGE_KEYS = {
    "resize_prob": 3,
    "resize_range": 2,
    "gaussian_noise_prob": 0,
    "noise_range": 2,
    "poisson_scale_range": 2,
    "gray_noise_prob": 0,
}
_MODES = ("area", "bilinear", "bicubic")
# BT.601 luma, the weights of cv2's BGR2GRAY that the paired datasets use for color='y'
_LUMA = (0.299, 0.587, 0.114)


def _quantize(x: Tensor) -> Tensor:
    return torch.clamp((x * 255.0).round(), 0, 255) / 255.0


@contextmanager
def _no_cudnn_autotune() -> Generator[None]:
    """cudnn.benchmark off inside the Degrader: the second blur sees a new image size almost every
    batch (275 distinct conv shapes in 3000 batches at canvas 288, 413 at 512). At batch 1 filter2D
    takes the cuDNN path, where benchmark_limit=0 autotuned each new shape (~300 ms); batches >= 2
    run the native depthwise kernel and are unaffected."""
    cudnn = torch.backends.cudnn
    prev, cudnn.benchmark = cudnn.benchmark, False
    try:
        yield
    finally:
        cudnn.benchmark = prev


class Degrader:
    """Two-stage degradation (blur, resize, noise, JPEG; then the final resize + sinc filter and
    JPEG in random order). `cfg` is the merged `datasets.train` section; its keys are checked once.
    """

    def __init__(
        self, cfg: dict[str, Any], *, scale: int, device: torch.device | str
    ) -> None:
        self.jpeg = bool(cfg.get("jpeg_compress", True))
        keys = {f"{k}{s}": n for s in ("", "2") for k, n in _STAGE_KEYS.items()}
        keys["second_blur_prob"] = 0
        if self.jpeg:
            keys |= {"jpeg_range": 2, "jpeg_range2": 2}
        missing = sorted(k for k in keys if cfg.get(k) is None)
        if missing:
            msg = f"model_type 'otf' needs these [degradations] keys: {', '.join(missing)}"
            raise ConfigError(msg)
        for k, n in keys.items():
            if n and (not isinstance(cfg[k], list | tuple) or len(cfg[k]) != n):
                msg = (
                    f"[degradations] {k} must be a list of {n} numbers, got {cfg[k]!r}"
                )
                raise ConfigError(msg, key=k)
        self.cfg = {k: cfg[k] for k in keys}
        self.scale = scale
        self.gray = cfg.get("color") == "y"
        self.jpeger = DiffJPEG(differentiable=False).to(device) if self.jpeg else None

    @torch.no_grad()
    def __call__(
        self, gt: Tensor, kernel1: Tensor, kernel2: Tensor, sinc_kernel: Tensor
    ) -> Tensor:
        """`gt` (b,c,h,w) in [0, 1] -> LQ (b,c,h/scale,w/scale), quantised to 8 bit."""
        with _no_cudnn_autotune():
            return self._degrade(gt, kernel1, kernel2, sinc_kernel)

    def _degrade(
        self, gt: Tensor, kernel1: Tensor, kernel2: Tensor, sinc_kernel: Tensor
    ) -> Tensor:
        k1, k2, sinc = (
            k.to(gt.device, non_blocking=True) for k in (kernel1, kernel2, sinc_kernel)
        )
        h, w = gt.shape[2:]
        # first stage
        out = filter2D(gt, k1)
        out = F.interpolate(
            out, scale_factor=self._resize(""), mode=random.choice(_MODES)
        )
        out = self._jpeg(self._noise(out, ""), "")
        # second stage
        if rng.uniform() < self.cfg["second_blur_prob"]:
            out = filter2D(out, k2)
        s = self._resize("2")
        size = (int(h / self.scale * s), int(w / self.scale * s))
        out = self._noise(
            F.interpolate(out, size=size, mode=random.choice(_MODES)), "2"
        )
        # [resize back + sinc] and JPEG in either order; other orders give twisted lines
        size = (h // self.scale, w // self.scale)
        if rng.uniform() < 0.5:
            out = filter2D(
                F.interpolate(out, size=size, mode=random.choice(_MODES)), sinc
            )
            out = self._jpeg(out, "2")
        else:
            out = self._jpeg(out, "2")
            out = filter2D(
                F.interpolate(out, size=size, mode=random.choice(_MODES)), sinc
            )
        return _quantize(out)

    def degrade(
        self, gt: Tensor, kernel1: Tensor, kernel2: Tensor, sinc_kernel: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The training pair (gt, lq). color='y' degrades the RGB gt (real chroma JPEG artifacts),
        then converts gt and lq to luma; lq is re-quantised to 8 bit."""
        lq = self(gt, kernel1, kernel2, sinc_kernel)
        if not self.gray:
            return gt, lq
        w = gt.new_tensor(_LUMA).view(1, 3, 1, 1)
        return (gt * w).sum(1, keepdim=True), _quantize((lq * w).sum(1, keepdim=True))

    def _resize(self, s: str) -> float:
        kind = random.choices(("up", "down", "keep"), self.cfg[f"resize_prob{s}"])[0]
        lo, hi = self.cfg[f"resize_range{s}"]
        return (
            rng.uniform(1, hi)
            if kind == "up"
            else rng.uniform(lo, 1)
            if kind == "down"
            else 1
        )

    def _noise(self, out: Tensor, s: str) -> Tensor:
        c, gray = self.cfg, self.cfg[f"gray_noise_prob{s}"]
        if rng.uniform() < c[f"gaussian_noise_prob{s}"]:
            return random_add_gaussian_noise_pt(
                out,
                sigma_range=c[f"noise_range{s}"],
                clip=True,
                rounds=False,
                gray_prob=gray,
            )
        return random_add_poisson_noise_pt(
            out,
            scale_range=c[f"poisson_scale_range{s}"],
            gray_prob=gray,
            clip=True,
            rounds=False,
        )

    def _jpeg(self, out: Tensor, s: str) -> Tensor:
        if self.jpeger is None:
            return out
        quality = out.new_zeros(out.size(0)).uniform_(*self.cfg[f"jpeg_range{s}"])
        # clamp to [0, 1], otherwise DiffJPEG gives unpleasant artifacts; its output is a strided
        # crop, which filter2D cannot view once a CPU reflect pad keeps that layout
        return self.jpeger(torch.clamp(out, 0, 1), quality=quality).contiguous()


class PairPool:
    """Training pair pool: mixes degradations across batches, since one batch shares its resize
    factors. Pass-through while filling; once full, each batch replaces `b` random slots and the
    pairs held there are returned (indexed swap: no full-pool shuffle or copy).

    A member whose first batch round-trips exactly through 8 bit (k/255 in [0, 1]: the Degrader's
    lq, the 8-bit decoded RGB gt) is stored as uint8, 4x smaller and bit-exact; others (the luma gt
    of color='y') keep their dtype.
    """

    def __init__(self, size: int, batch: int, *, device: torch.device | str) -> None:
        if batch < 1 or size < batch:
            msg = f"queue_size ({size}) must be >= batch_size ({batch})"
            raise ConfigError(msg, key="queue_size")
        self.batch = batch
        self.size = (
            size // batch * batch
        )  # as before: rounded down to a multiple of the batch
        self.device = torch.device(device)
        self.ptr = 0
        self.lq: Tensor | None = None
        self.gt: Tensor | None = None
        self.dtypes = (torch.float32, torch.float32)  # lq, gt as passed in

    def _alloc(self, x: Tensor) -> Tensor:
        exact = torch.equal(_quantize(x), x)  # one host sync, on the first batch only
        dtype = torch.uint8 if exact else x.dtype
        return x.new_empty((self.size, *x.shape[1:]), dtype=dtype, device=self.device)

    @staticmethod
    def _enc(store: Tensor, x: Tensor) -> Tensor:
        return (x * 255.0).round().to(torch.uint8) if store.dtype == torch.uint8 else x

    @staticmethod
    def _dec(x: Tensor, dtype: torch.dtype) -> Tensor:
        return x.to(dtype) / 255.0 if x.dtype == torch.uint8 else x

    @torch.no_grad()
    def __call__(self, lq: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        b = self.batch
        if lq.size(0) != b or gt.size(0) != b:
            msg = f"pair pool needs lq/gt batches of batch_size {b}, got {lq.size(0)}/{gt.size(0)}"
            raise ValueError(msg)
        if self.lq is None or self.gt is None:
            self.dtypes = (lq.dtype, gt.dtype)
            self.lq, self.gt = self._alloc(lq), self._alloc(gt)
            mib = (self.lq.nbytes + self.gt.nbytes) / 2**20
            logging.getLogger("neosr").info(
                f"Pair pool: {self.size} pairs, {mib:.0f} MiB "
                f"(lq {str(self.lq.dtype)[6:]}, gt {str(self.gt.dtype)[6:]})"
            )
        if self.ptr < self.size:  # filling: store a copy, train on the batch itself
            self.lq[self.ptr : self.ptr + b] = self._enc(self.lq, lq)
            self.gt[self.ptr : self.ptr + b] = self._enc(self.gt, gt)
            self.ptr += b
            return lq, gt
        idx = torch.randperm(self.size, device=self.device)[:b]
        # advanced indexing copies
        out_lq, out_gt = self._dec(self.lq[idx], self.dtypes[0]), self._dec(self.gt[idx], self.dtypes[1])
        self.lq[idx], self.gt[idx] = self._enc(self.lq, lq), self._enc(self.gt, gt)
        return out_lq, out_gt

import inspect
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from functools import partial
from typing import Any

import torch

from neosr.config import ConfigError
from neosr.utils.factory import resolve

__all__ = ["METRICS", "build_metric"]

METRICS = {
    "calculate_psnr": ("calculate", None),
    "calculate_ssim": ("calculate", None),
    "calculate_dists": ("calculate", None),
    "calculate_topiq": ("calculate", None),
}


def build_metric(
    cfg: Mapping[str, Any], device: torch.device, *, section: str = "val.metrics"
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Resolve and bind a `val.metrics.<name>` table once per run.

    `type` names the numpy reference (calculate.py); the callable is its on-device twin
    `<type>_pt`: it takes the quantised sr, gt tensors of `img_util.quantize` (float32 RGB NCHW,
    0..255) and returns a 0-d tensor. DISTS/TOPIQ nets are built at the first call and reused.
    """
    kwargs = deepcopy(dict(cfg))
    name = kwargs.pop("type", None)
    kwargs.pop("better", None)  # read by the best-metric bookkeeping, not by the metric
    ref = resolve(METRICS, "neosr.metrics", section, name)
    fn = getattr(sys.modules[ref.__module__], f"{name}_pt")
    sig = inspect.signature(fn)
    extra = {"device": device} if "device" in sig.parameters else {}
    try:  # a typo, or a `device` key that contradicts the run device
        sig.bind(None, None, **kwargs, **extra)
    except TypeError as err:
        msg = f"[{section}] {name}: {err}"
        raise ConfigError(msg) from err
    return partial(fn, **kwargs, **extra)

import inspect
from collections.abc import Callable, Mapping
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import torch

from neosr.config import ConfigError
from neosr.utils.factory import resolve

__all__ = ["METRICS", "build_metric", "calculate_metric"]

METRICS = {
    "calculate_psnr": ("calculate", None),
    "calculate_ssim": ("calculate", None),
    "calculate_dists": ("calculate", None),
    "calculate_topiq": ("calculate", None),
}


def build_metric(
    cfg: Mapping[str, Any], device: torch.device, *, section: str = "val.metrics"
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Resolve and bind a `val.metrics.<name>` table once per run.

    The callable takes the uint8 HWC BGR images from `tensor2img` (sr, gt). DISTS/TOPIQ
    nets are imported and built on `device` at the first call, then reused (eval, no grad).
    """
    kwargs = deepcopy(dict(cfg))
    name = kwargs.pop("type", None)
    kwargs.pop("better", None)  # read by the best-metric bookkeeping, not by the metric
    fn = resolve(METRICS, "neosr.metrics", section, name)
    sig = inspect.signature(fn)
    extra = {"device": device} if "device" in sig.parameters else {}
    try:  # a typo, or a `device` key that contradicts the run device
        sig.bind(None, None, **kwargs, **extra)
    except TypeError as err:
        msg = f"[{section}] {name}: {err}"
        raise ConfigError(msg) from err
    return partial(fn, **kwargs, **extra)


def calculate_metric(data: Mapping[str, np.ndarray], opt: Mapping[str, Any]) -> float:
    """Transitional: the old `models/image.py` call shape, until wave W3 builds each metric once."""
    opt = dict(opt)  # a legacy `device` key still picks the device here
    device = torch.device(
        opt.pop("device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return build_metric(opt, device)(data["img"], data["img2"])

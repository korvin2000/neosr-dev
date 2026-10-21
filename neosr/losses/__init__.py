"""Loss table, single-loss builder and the training-objective builder (design T1 §4.2, T3 L-D)."""

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from neosr.config import ConfigError
from neosr.losses.base import Loss
from neosr.utils.factory import construct, inject_scale, resolve

__all__ = [
    "LOSSES",
    "RESERVED",
    "RESERVED_PREFIX",
    "SLOT_KEY",
    "TARGETS",
    "build_loss",
    "build_losses",
]

# type -> (module under neosr.losses, scale kwarg | None)
LOSSES: dict[str, tuple[str, str | None]] = {
    "L1Loss": ("basic_loss", None),
    "MSELoss": ("basic_loss", None),
    "HuberLoss": ("basic_loss", None),
    "chc_loss": ("basic_loss", None),
    "consistency_loss": ("consistency_loss", None),
    "dists_loss": ("dists_loss", None),
    "fdl_loss": ("fdl_loss", None),
    "ff_loss": ("ff_loss", None),
    "gan_loss": ("gan_loss", None),
    "kl_loss": ("kl_loss", None),
    "ldl_loss": ("ldl_loss", None),
    "msswd_loss": ("msswd_loss", None),
    "ncc_loss": ("ncc_loss", None),
    "mssim_loss": ("ssim_loss", None),
    "vgg_perceptual_loss": ("vgg_perceptual_loss", None),
}

# legacy log names; any other slot logs as l_g_<slot>
SLOT_KEY = {"pixel": "pix", "perceptual": "percep"}
TARGETS = {"gt": "gt", "lq": "lq_up"}  # config value -> Context attribute
RESERVED = {
    "l_g_total",
    "l_d_total",
    "l_d_real",
    "l_d_fake",
    "out_d_real",
    "out_d_fake",
}
RESERVED_PREFIX = ("grad_norm_", "skipped_")


def build_loss(
    cfg: Mapping[str, Any], *, scale: int | None = None, section: str = "train.loss"
) -> nn.Module:
    kwargs = deepcopy(dict(cfg))
    name = kwargs.pop("type", None)
    obj = resolve(LOSSES, "neosr.losses", section, name)
    inject_scale(LOSSES, section, name, kwargs, scale)
    return construct(obj, section, name, kwargs)


def build_losses(
    train: Mapping[str, Any], *, scale: int, device: torch.device, has_d: bool
) -> tuple[list[Loss], list[Loss]]:
    """Every non-empty `train.<slot>_opt` table becomes a term; returns (generator terms, discriminator terms)."""
    terms: list[Loss] = []
    for key, cfg in train.items():
        if not key.endswith("_opt") or not cfg:
            continue
        name, spec, section = key[:-4], dict(cfg), f"train.{key}"
        if name in {"consistency", "msswd"} and train.get("match_lq_colors"):
            spec.setdefault("target", "lq")  # legacy global -> per-loss key
        if name == "gan" and train.get("wavelet_guided"):
            spec.setdefault("wavelet_init", train.get("wavelet_init", 0))
        if "loss_weight" in spec and "perceptual_weight" in spec:
            msg = f"[{section}] set loss_weight or perceptual_weight, not both"
            raise ConfigError(msg)
        weight = spec.pop("loss_weight", spec.pop("perceptual_weight", None))
        if weight is not None and (
            isinstance(weight, bool)
            or not isinstance(weight, int | float)
            or not math.isfinite(weight)
        ):  # builder-owned keys are opaque to load_config
            msg = f"[{section}] loss_weight must be a finite number, got {weight!r}"
            raise ConfigError(msg)
        target = spec.pop("target", "gt")
        if not isinstance(target, str) or target not in TARGETS:
            msg = f"[{section}] target must be one of {sorted(TARGETS)}, got {target!r}"
            raise ConfigError(msg)
        loss = build_loss(spec, scale=scale, section=section)
        loss.loss_weight = float(type(loss).loss_weight if weight is None else weight)
        loss.target, loss.log_key = TARGETS[target], f"l_g_{SLOT_KEY.get(name, name)}"
        terms += [t for t in loss.expand() if t.phase == "d" or t.loss_weight]

    g = [t for t in terms if t.phase == "g"]
    d = [t for t in terms if t.phase == "d"]
    keys = [t.log_key for t in terms if t.log_key]
    if not g:
        msg = "[train] at least one generator loss with a non-zero weight is required"
        raise ConfigError(msg)
    if has_d != bool(d):
        msg = "[train] network_d requires gan_opt and gan_opt requires network_d"
        raise ConfigError(msg)
    if train.get("wavelet_guided") and not d:
        msg = "[train] wavelet_guided requires gan_opt"
        raise ConfigError(msg)
    if len(set(keys)) != len(keys) or any(
        k in RESERVED or k.startswith(RESERVED_PREFIX) for k in keys
    ):
        msg = f"[train] loss log keys must be unique and not reserved: {keys}"
        raise ConfigError(msg)
    # frozen after construction, so ctor-made parameters (DISTS alpha/beta) are covered too
    nn.ModuleList(terms).to(device).requires_grad_(requires_grad=False).eval()
    return g, d

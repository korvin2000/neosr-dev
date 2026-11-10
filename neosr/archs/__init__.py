from collections.abc import Mapping
from typing import Any

from torch import nn

from neosr.utils.factory import build
from neosr.utils.registry import ARCH_REGISTRY

__all__ = ["ARCH_REGISTRY", "build_network"]


def build_network(
    cfg: Mapping[str, Any], *, scale: int | None = None, section: str = "network_g"
) -> nn.Module:
    return build(ARCH_REGISTRY, section, cfg, scale=scale)

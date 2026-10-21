"""Trainer table (design T3 §3.2): `model_type` -> class in `neosr.models.<module>`, imported on use."""

from typing import TYPE_CHECKING, Any

from neosr.utils import get_root_logger
from neosr.utils.factory import resolve

if TYPE_CHECKING:
    from neosr.models.base import base
    from neosr.runtime import Runtime

__all__ = ["MODELS", "build_model"]

MODELS: dict[str, tuple[str, str | None]] = {
    "image": ("image", None),
    "otf": ("otf", None),
}


def build_model(opt: dict[str, Any], runtime: "Runtime") -> "base":
    """Build the trainer; with `path.resume_state` set (training) it also resumes (T2 §3 step 7)."""
    model = resolve(MODELS, "neosr.models", "model_type", opt["model_type"])(
        opt, runtime
    )
    get_root_logger().info(f"Using model [{type(model).__name__}].")
    return model

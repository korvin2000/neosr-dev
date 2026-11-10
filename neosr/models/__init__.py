"""Trainer builder (design T3 §3.2, T7): `model_type` is a `MODEL_REGISTRY` type, imported on use."""

from typing import TYPE_CHECKING, Any

from neosr.utils import get_root_logger
from neosr.utils.factory import resolve
from neosr.utils.registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from neosr.models.base import base
    from neosr.runtime import Runtime

__all__ = ["MODEL_REGISTRY", "build_model"]


def build_model(opt: dict[str, Any], runtime: "Runtime") -> "base":
    """Build the trainer; with `path.resume_state` set (training) it also resumes (T2 §3 step 7)."""
    model = resolve(MODEL_REGISTRY, "model_type", opt["model_type"])(opt, runtime)
    get_root_logger().info(f"Using model [{type(model).__name__}].")
    return model

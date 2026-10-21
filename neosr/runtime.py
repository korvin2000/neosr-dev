"""Process runtime: distributed init, seeding, backend flags and the device. Called once per entry point."""

import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from neosr.config import ConfigError
from neosr.utils.dist_util import get_dist_info, init_dist

logger = logging.getLogger("neosr")


@dataclass(frozen=True, slots=True)
class Runtime:
    device: torch.device
    rank: int
    world_size: int
    seed: int
    distributed: bool


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002
    torch.manual_seed(seed)  # also seeds every CUDA device
    # seam (lane L-C): seed the process-wide numpy Generator here with neosr.utils.rng.seed_rng(seed)


def _check_gpu(cfg: dict[str, Any]) -> None:
    major, _ = torch.cuda.get_device_capability()
    amp, bf16 = cfg.get("use_amp"), cfg.get("bfloat16")
    if bf16 and major <= 6:
        msg = "bfloat16 is not supported on Pascal or older GPUs (compute capability <= 6.x)"
        raise ConfigError(msg)
    if bf16 and major == 7:
        logger.warning(
            "Turing GPU detected: bfloat16 support may be limited or slow, consider disabling it."
        )
    if amp and major <= 6:
        logger.warning(
            "Pascal or older GPU has no Tensor Cores: AMP may not provide benefits."
        )
    if not amp and major >= 8:
        logger.warning(
            "Ampere or newer GPU detected: consider use_amp with bfloat16 for a speedup."
        )


def setup_runtime(cfg: dict[str, Any], *, launcher: str = "none") -> Runtime:
    """Initialise the process group, seeds and backend flags; record them in `cfg` and return a Runtime."""
    if cfg.get("compile") and sys.platform.startswith("win"):
        msg = "compile is not supported on Windows (no Triton backend); disable it"
        raise ConfigError(msg)
    num_gpu = cfg.get("num_gpu", "auto")
    if launcher == "none" and isinstance(num_gpu, int) and num_gpu > 1:
        msg = f"num_gpu={num_gpu} needs a launcher (torchrun --nproc_per_node={num_gpu} ... --launcher pytorch); DataParallel was removed"
        raise ConfigError(msg)
    seed = cfg.get("manual_seed")
    deterministic = seed is not None
    if deterministic:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
        )  # before the first cuBLAS handle
    else:
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    distributed = launcher != "none"
    if distributed:
        params = cfg.get("dist_params", {})
        init_dist(
            launcher, **{k: params[k] for k in ("backend", "port") if k in params}
        )
    rank, world_size = get_dist_info()
    if seed is None:  # rank 0 picks, every rank uses it
        box = [random.SystemRandom().randint(1024, 10000) if rank == 0 else None]
        if world_size > 1:
            dist.broadcast_object_list(box, src=0)
        seed = box[0]
    seed_all(seed + rank)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if not deterministic:
        torch.backends.cudnn.benchmark_limit = 0
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = deterministic  # type: ignore[attr-defined]
    if cfg.get("fast_matmul"):
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_accumulation = True

    if torch.cuda.is_available():
        _check_gpu(cfg)
        index = (
            int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count()
            if distributed
            else 0
        )
        device = torch.device("cuda", index)
    else:
        device = torch.device("cpu")
    cfg.update(
        dist=distributed,
        rank=rank,
        world_size=world_size,
        manual_seed=seed,
        deterministic=deterministic,
        num_gpu=torch.cuda.device_count() if num_gpu == "auto" else num_gpu,
    )
    return Runtime(device, rank, world_size, seed, distributed)

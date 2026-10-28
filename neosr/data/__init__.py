import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from neosr.utils.factory import build
from neosr.utils.rng import seed_rng

if TYPE_CHECKING:
    from neosr.runtime import Runtime

__all__ = ["DATASETS", "build_dataloader", "build_dataset"]

DATASETS: dict[str, tuple[str, str | None]] = {
    "otf": ("otf_dataset", None),
    "paired": ("paired_dataset", None),
    "single": ("single_dataset", None),
}


def build_dataset(cfg: Mapping[str, Any]) -> Dataset:
    section = f"datasets.{cfg.get('phase')}"
    return build(DATASETS, "neosr.data", section, cfg, whole=True)


def worker_init_fn(_: int) -> None:
    # torch already seeds random, torch and legacy numpy from this per-worker seed
    seed_rng(get_worker_info().seed)  # type: ignore[union-attr]
    # one thread per worker: the workers are the parallelism (cv2 would spawn a pool per worker)
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def auto_workers(world_size: int) -> int:
    """`num_worker_per_gpu = "auto"`: the CPUs this process may run on (affinity-aware), split over
    the processes, minus one core for each training process itself."""
    cpus = os.process_cpu_count() or 1
    return max(cpus // max(world_size, 1) - 1, 1)


def build_dataloader(
    dataset: Dataset,
    cfg: Mapping[str, Any],
    *,
    runtime: "Runtime",
    sampler: Sampler | None = None,
) -> DataLoader:
    """Train: `batch_size` and `num_worker_per_gpu` (default 4, "auto": `auto_workers`) per process,
    shuffled unless `sampler` is given. Val/test: batch 1, in order; `num_worker_per_gpu` (default 0:
    the main process) lets test.py decode the next images while the GPU runs.

    The loader generator, seeded `runtime.seed + runtime.rank`, drives shuffling and the per-epoch
    worker seeds.
    """
    phase = cfg["phase"]
    if phase not in {"train", "val", "test"}:
        msg = f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'."
        raise ValueError(msg)
    train = phase == "train"
    workers, batch_size = cfg.get("num_worker_per_gpu", 4 if train else 0), 1
    if workers == "auto":
        workers = auto_workers(runtime.world_size)
        logging.getLogger("neosr").info(f"num_worker_per_gpu auto: {workers}")
    if train:
        batch_size = cfg["batch_size"]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=workers,
        drop_last=train,
        generator=torch.Generator().manual_seed(runtime.seed + runtime.rank),
        worker_init_fn=worker_init_fn,
        pin_memory=cfg.get("pin_memory", torch.cuda.is_available()),
        persistent_workers=cfg.get("persistent_workers", workers > 0) and workers > 0,
        prefetch_factor=2 if workers else None,
    )

import os
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from neosr.utils.factory import resolve
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
    return resolve(DATASETS, "neosr.data", section, cfg.get("type"))(
        deepcopy(dict(cfg))
    )


def worker_init_fn(_: int) -> None:
    # torch already seeds random, torch and legacy numpy from this per-worker seed
    seed_rng(get_worker_info().seed)  # type: ignore[union-attr]


def build_dataloader(
    dataset: Dataset,
    cfg: Mapping[str, Any],
    *,
    runtime: "Runtime | None" = None,
    sampler: Sampler | None = None,
    num_gpu: int = 1,
    dist: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """Train: `batch_size` and `num_worker_per_gpu` (default 4, "auto": CPUs per process) per process,
    shuffled unless `sampler` is given. Val/test: batch 1 in the main process.

    The loader generator, seeded `runtime.seed + runtime.rank`, drives shuffling and the per-epoch
    worker seeds. `num_gpu`/`dist`/`seed` serve the pre-Runtime call shape of train.py/test.py
    (DataParallel multiplies batch and workers; torch's global RNG seeds the workers; a train
    loader seeds the main-process `rng` with `seed`, as the old import-time `rng()` did) and are
    ignored when `runtime` is given.
    """
    phase = cfg["phase"]
    if phase not in {"train", "val", "test"}:
        msg = f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'."
        raise ValueError(msg)
    if runtime is not None:
        num_gpu, dist = runtime.world_size, runtime.distributed
    train = phase == "train"
    if runtime is None and train and seed is not None:
        # legacy bridge until W4; with a runtime, runtime.seed_all seeds it
        seed_rng(seed)
    multiplier = max(num_gpu, 1)
    workers, batch_size = 0, 1
    if train:
        workers = cfg.get("num_worker_per_gpu", 4)
        if workers == "auto":
            workers = max((os.cpu_count() or 1) // multiplier, 1)
        batch_size = cfg["batch_size"]
        if not dist:  # DataParallel splits one batch over all GPUs
            batch_size, workers = batch_size * multiplier, workers * multiplier
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=workers,
        drop_last=train,
        generator=None
        if runtime is None
        else torch.Generator().manual_seed(runtime.seed + runtime.rank),
        worker_init_fn=worker_init_fn,
        pin_memory=cfg.get("pin_memory", torch.cuda.is_available()),
        persistent_workers=cfg.get("persistent_workers", workers > 0) and workers > 0,
        prefetch_factor=2 if workers else None,
    )

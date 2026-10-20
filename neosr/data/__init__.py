import importlib
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import Sampler

from neosr.data import augmentations, degradations, otf_dataset
from neosr.utils import get_root_logger, scandir
from neosr.utils.registry import DATASET_REGISTRY

__all__ = ["build_dataloader", "build_dataset"]

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = Path(Path(__file__).resolve()).parent
dataset_filenames = [
    Path(Path(v).name).stem
    for v in scandir(str(data_folder))
    if v.endswith("_dataset.py")
]

# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f"neosr.data.{file_name}")
    for file_name in dataset_filenames
]


def build_dataset(dataset_opt: dict[str, Any]):
    """Build dataset from options.

    Args:
    ----
        dataset_opt (dict): Configuration for dataset. It must contain:
            type (str): Dataset type.

    """
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt["type"])(dataset_opt)  # type: ignore[operator]
    logger = get_root_logger()
    logger.info(f"Dataset [{dataset.__class__.__name__}] is built.")
    return dataset


def seed_data_rngs(seed: int) -> None:
    """Replace the module-level numpy Generators of the data pipeline.

    Workers inherit (fork) or re-create (spawn, from the same manual_seed) identical
    Generators, so every worker would draw the same degradations unless reseeded.
    """
    modules = (augmentations, degradations, otf_dataset)
    for module, child in zip(
        modules, np.random.SeedSequence(seed).spawn(len(modules)), strict=True
    ):
        module.rng = np.random.default_rng(child)  # type: ignore[attr-defined]


def worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    # torch seeds each worker with base_seed + worker_id, where base_seed is drawn
    # from the main process RNG (seeded per rank) for every new iterator: distinct
    # per worker, rank and epoch, and reproducible under manual_seed
    seed = torch.initial_seed()
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002
    seed_data_rngs(seed)


def build_dataloader(
    dataset: data.Dataset,
    dataset_opt: dict[str, Any],
    num_gpu: int = 1,
    dist: bool = False,
    sampler: Sampler | None = None,
    seed: int | None = None,  # noqa: ARG001
) -> data.DataLoader:
    """Build dataloader.

    Args:
    ----
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int | 'auto'): Number of workers for each GPU.
            batch_size (int): Training batch size for each GPU.
            pin_memory (bool): Default: True for training with CUDA.
            persistent_workers (bool): Default: True when there are workers.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Unused, kept for compatibility: worker seeds derive
            from torch's per-worker seed (see worker_init_fn).

    """
    phase = dataset_opt["phase"]

    if phase == "train":
        multiplier = max(num_gpu, 1)
        num_workers = dataset_opt.get("num_worker_per_gpu")
        if num_workers is None:
            num_workers = 4
        elif num_workers == "auto":
            num_workers = max((os.cpu_count() or 1) // multiplier, 1)
        batch_size = dataset_opt["batch_size"]
        if not dist:  # DataParallel splits one batch across all GPUs
            batch_size *= multiplier
            num_workers *= multiplier
        dataloader_args = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": sampler is None,
            "sampler": sampler,
            "num_workers": num_workers,
            "prefetch_factor": 2 if num_workers > 0 else None,
            "drop_last": True,
            "worker_init_fn": worker_init_fn,
            "pin_memory": dataset_opt.get("pin_memory", torch.cuda.is_available()),
            # torch rejects persistent workers without workers
            "persistent_workers": dataset_opt.get("persistent_workers", True)
            and num_workers > 0,
        }
    elif phase in {"val", "test"}:
        dataloader_args = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": dataset_opt.get("pin_memory", False),
        }
    else:
        msg = f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'."
        raise ValueError(msg)

    return data.DataLoader(**dataloader_args)

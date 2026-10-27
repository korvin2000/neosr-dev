import logging
import os
import random
import time
from collections.abc import Iterator
from os import path as osp
from pathlib import Path
from typing import Any

import torch

from neosr.utils.dist_util import master_only


class tc:
    # ansi color codes
    std = "\x1b[38;5;195m"
    black = "\033[0;30m"
    red = "\033[1;31m"
    strong_red = "\033[0;31m"
    light_green = "\033[1;32m"
    light_yellow = "\033[1;33m"
    light_blue = "\033[1;34m"
    light_cyan = "\033[1;36m"
    light_white = "\033[1;37m"
    bold = "\033[1m"
    italic = "\033[3m"
    underline = "\033[4m"
    end = "\033[0m"


def set_random_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)


def get_time_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def mkdir_and_rename(path: str) -> None:
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
    ----
        path (str): Folder path.

    """
    if Path(path).exists():
        new_name = str(path) + "_archived_" + get_time_str()
        logging.getLogger("neosr").info(f"{path} exists: archived as {new_name}")
        Path(path).rename(new_name)
    Path(path).mkdir(parents=True, exist_ok=True)


@master_only
def make_exp_dirs(opt: dict[str, Any], archive: bool = True) -> None:
    """Create the run folders. `archive` (fresh runs) first renames an existing run folder; a
    resume passes False and keeps whatever exists."""
    p = opt["path"]
    root = p["experiments_root" if opt["is_train"] else "results_root"]
    if archive:
        mkdir_and_rename(root)
    Path(root).mkdir(parents=True, exist_ok=True)
    for key in ("models", "training_states", "log", "visualization"):
        if key in p:
            Path(p[key]).mkdir(parents=True, exist_ok=True)


def scandir(
    dir_path: str,
    suffix: str | None = None,
    recursive: bool = False,
    full_path: bool = False,
) -> Iterator[Any]:
    """Scan a directory to find the interested files.

    Args:
    ----
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
    -------
        A generator for all the interested files with relative paths.

    """
    if (suffix is not None) and not isinstance(suffix, str | tuple):
        msg = '"suffix" must be a string or tuple of strings'
        raise TypeError(msg)
    if suffix is not None:  # matched case-insensitively: 'png' also finds 'X.PNG'
        suffix = tuple(s.lower() for s in ((suffix,) if isinstance(suffix, str) else suffix))

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if entry.name.startswith("."):  # hidden files and folders (.DS_Store, .thumbs)
                continue
            if entry.is_file():
                return_path = entry.path if full_path else osp.relpath(entry.path, root)
                if suffix is None or return_path.lower().endswith(suffix):
                    yield return_path
            elif recursive and entry.is_dir(follow_symlinks=False):
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


@staticmethod
def sizeof_fmt(size: int, suffix: str = "B") -> str:
    """Get human readable file size.

    Args:
    ----
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
    ------
        str: Formatted file size.

    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}{suffix}"
        size //= 1024
    return f"{size:3.1f} Y{suffix}"

import logging
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from neosr.data.file_client import FileClient
from neosr.utils import imfrombytes, scandir

# files the datasets and the lmdb tool pick up from folders, matched case-insensitively
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")
# the optional ' (h,w,c)' and ' level' a meta_info line appends to the file name (lmdb_util writes both)
_META_TAIL = re.compile(r" \(\d+,\d+,\d+\)( \d+)?$")


def image_paths(folder: str, *, recursive: bool = False) -> list[str]:
    """Sorted full paths of the images in `folder` (hidden entries skipped)."""
    return sorted(scandir(folder, suffix=IMG_EXTENSIONS, recursive=recursive, full_path=True))


def meta_info_names(meta_info: str) -> list[str]:
    """File names listed in a meta_info file, one per line; names may contain spaces."""
    with Path(meta_info).open(encoding="utf-8") as fin:
        return [_META_TAIL.sub("", line.strip()) for line in fin if line.strip()]


class PathList(Sequence[str]):
    """Read-only list of paths kept in one bytes buffer plus offsets. Indexing touches no per-item
    object, so the pages forked DataLoader workers share stay shared (a list of str costs each worker
    about 360 B per entry once read: copy-on-access through the refcounts)."""

    def __init__(self, paths: Iterable[str]) -> None:
        enc = [os.fsencode(p) for p in paths]
        self._buf = np.frombuffer(b"".join(enc), np.uint8)
        self._off = np.cumsum([0, *map(len, enc)], dtype=np.int64)

    def __len__(self) -> int:
        return len(self._off) - 1

    def __getitem__(self, i: int) -> str:  # type: ignore[override]
        if not -len(self) <= i < len(self):
            raise IndexError(i)
        i %= len(self)
        return os.fsdecode(self._buf[self._off[i] : self._off[i + 1]].tobytes())


def read_img(
    file_client: FileClient,
    path: str,
    client_key: str,
    retries: int = 3,
    *,
    float32: bool = True,
) -> np.ndarray:
    """Read and decode an image as HWC BGR: float32 in [0, 1], or the decoded uint8 with
    `float32=False` (crop first, then `to_float32`: identical values, less work).

    Transient read errors (OSError, e.g. a congested network share) are retried
    ``retries`` times; missing files, empty data and decode failures raise at once
    with the path.
    """
    for attempt in range(retries):
        try:
            img_bytes = file_client.get(path, client_key)  # type: ignore[arg-type]
            break
        except FileNotFoundError:
            raise
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(1)
    if not img_bytes:
        msg = f"No data returned from path: {path}"
        raise OSError(msg)
    try:
        img = imfrombytes(img_bytes, float32=float32)
    except (AttributeError, ValueError, cv2.error) as e:  # e.g. imdecode gave None
        msg = f"Failed to decode image: {path}"
        raise OSError(msg) from e
    if img is None:  # imdecode gives None on undecodable bytes
        msg = f"Failed to decode image: {path}"
        raise OSError(msg)
    return img  # type: ignore[return-value]


def train_retry[T](load: Callable[[int], T], index: int, size: int, phase: str) -> T:
    """`load(index)`. In the train phase a data defect (OSError: unreadable file; ValueError:
    mis-sized pair) is logged with its path and a random other index is loaded instead, at most
    10 in a row, so one bad file cannot end a run but a wrong dataroot still does. Val/test raise."""
    if phase != "train":
        return load(index)
    for _ in range(10):
        try:
            return load(index)
        except (OSError, ValueError) as e:
            err = e
            logging.getLogger("neosr").warning(f"Skipping a training sample: {e}")
            index = random.randrange(size)  # python random: seeded per worker
    msg = "10 consecutive unusable training samples; check the dataset"
    raise RuntimeError(msg) from err


def to_float32(img: np.ndarray) -> np.ndarray:
    """uint8 image -> float32 in [0, 1], as `read_img(float32=True)` computes it."""
    return img.astype(np.float32) / 255.0


def lmdb_keys(folder: str) -> list[str]:
    """Keys of an lmdb folder: the names of its meta_info.txt (`{key}.png (h,w,c) level`) without
    their final extension, so dots and spaces inside a key are kept."""
    return [n.rsplit(".", 1)[0] for n in meta_info_names(str(Path(folder) / "meta_info.txt"))]


def paired_paths_from_lmdb(folders: list[str], keys: list[str]) -> list[str]:
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    ::

        lq.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
    ----
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
    -------
        list[str]: Returned path list.

    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    )
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith(".lmdb") and gt_folder.endswith(".lmdb")):
        msg = (
            f"{input_key} folder and {gt_key} folder should both in lmdb "
            f"formats. But received {input_key}: {input_folder}; "
            f"{gt_key}: {gt_folder}"
        )
        raise ValueError(msg)
    # ensure that the two meta_info files are the same
    input_lmdb_keys, gt_lmdb_keys = lmdb_keys(input_folder), lmdb_keys(gt_folder)
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        msg = f"Keys in {input_key}_folder and {gt_key}_folder are different."
        raise ValueError(msg)
    paths: list[Any] = []
    paths.extend(
        {f"{input_key}_path": lmdb_key, f"{gt_key}_path": lmdb_key}
        for lmdb_key in sorted(input_lmdb_keys)
    )
    return paths


def paired_paths_from_meta_info_file(
    folders: list[str], keys: list[str], meta_info_file: str
) -> list[dict[str, str]]:
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
    ----
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.

    Returns:
    -------
        list[str]: Returned path list.

    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    )
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    gt_names = meta_info_names(meta_info_file)
    paths: list[dict[str, str]] = []
    for gt_name in gt_names:
        input_path = str(Path(input_folder) / gt_name)
        gt_path = str(Path(gt_folder) / gt_name)
        paths.append({f"{input_key}_path": input_path, f"{gt_key}_path": gt_path})
    return paths


def paired_paths_from_folder(
    folders: list[str], keys: list[str]
) -> list[dict[str, str]]:
    """Generate paired paths from folders.

    Args:
    ----
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].

    Returns:
    -------
        list[str]: Returned path list.

    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    )

    input_folder, gt_folder = folders
    input_key, gt_key = keys

    # normalised as Path joins them below, e.g. './LQ/x.png' -> 'LQ/x.png'
    input_paths = {str(Path(p)) for p in image_paths(input_folder, recursive=True)}
    gt_paths = image_paths(gt_folder, recursive=True)  # sorted: the same order on every rank

    assert len(input_paths) == len(gt_paths), (
        f"{input_key} and {gt_key} datasets have different number of images: "
        f"{len(input_paths)}, {len(gt_paths)}."
    )
    paths: list[dict[str, str]] = []
    for gt_path in gt_paths:
        input_path = str(Path(input_folder) / Path(gt_path).relative_to(gt_folder))
        assert input_path in input_paths, f"{input_path} is not in {input_key}_paths."
        paths.append({f"{input_key}_path": input_path, f"{gt_key}_path": gt_path})
    return paths


def paths_from_lmdb(folder: str) -> list[str]:
    """Generate paths from lmdb.

    Args:
    ----
        folder (str): Folder path.

    Returns:
    -------
        list[str]: Returned path list.

    """
    if not folder.endswith(".lmdb"):
        msg = f"Folder {folder}folder should in lmdb format."
        raise ValueError(msg)
    return lmdb_keys(folder)

import os
import shutil
from itertools import batched
from multiprocessing import Pool
from pathlib import Path

import cv2
import lmdb
from tqdm import tqdm


def read_img_worker(path, key, compress_level):
    """Read image worker.

    Args:
    ----
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.

    Returns:
    -------
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.

    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        msg = f"Cannot read image: {path}"
        raise OSError(msg)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode(
        ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
    )
    return (key, img_byte, (h, w, c))


def _encode(item):
    return read_img_worker(*item)


def make_lmdb_from_imgs(
    data_path,
    lmdb_path,
    img_path_list,
    keys,
    batch=256,
    compress_level=1,
    multiprocessing_read=True,
    n_thread=None,
    map_size=None,
) -> None:
    """Make lmdb from images.

    Contents of lmdb. The file structure is:

    ::

        example.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt (UTF-8) records per image 1) the image name (the key plus
    '.png': values are stored PNG-encoded), 2) the image shape and 3) the compression
    level, separated by a space, e.g. `000_00000000.png (720,1280,3) 1`. Keys are
    UTF-8 and may contain dots and spaces.

    Images are encoded in a pool (`n_thread` processes, default: the usable CPUs)
    and written `batch` at a time, one commit each: memory stays bounded. The database is built as `<lmdb_path>.tmp` and renamed when complete;
    any error (with the failing path) removes it, so no partial output is left.

    Args:
    ----
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys; must be unique.
        batch (int): Images per commit, also the most held in memory. Default: 256.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Encode in a process pool. Default: True.
        n_thread (int | None): Pool size.
        map_size (int | None): Initial map size; it grows when full. Default: twice
            the size of the source files.

    """
    assert len(img_path_list) == len(keys), (
        "img_path_list and keys should have the same length, "
        f"but got {len(img_path_list)} and {len(keys)}"
    )
    if not lmdb_path.endswith(".lmdb"):
        msg = "lmdb_path must end with '.lmdb'."
        raise ValueError(msg)
    if Path(lmdb_path).exists():
        msg = f"Folder {lmdb_path} already exists."
        raise FileExistsError(msg)
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})[:5]
        msg = f"Duplicate lmdb keys (two images map to one record): {dup}"
        raise ValueError(msg)
    print(f"Create lmdb for {data_path}, save to {lmdb_path}...")
    print(f"Total images: {len(img_path_list)}")
    paths = [Path(data_path) / p for p in img_path_list]
    if map_size is None:
        map_size = max(2 * sum(p.stat().st_size for p in paths), 1 << 24)
    tmp = Path(f"{lmdb_path}.tmp")
    if tmp.exists():  # left by a killed build
        shutil.rmtree(tmp)
    items = ((p, k, compress_level) for p, k in zip(paths, keys, strict=True))
    n_thread = n_thread or os.process_cpu_count() or 1
    env = lmdb.open(str(tmp), map_size=map_size)
    try:
        with (
            Pool(n_thread) if multiprocessing_read else _NoPool() as pool,
            (tmp / "meta_info.txt").open("w", encoding="utf-8") as meta,
            tqdm(total=len(paths), unit="image") as pbar,
        ):
            for window in batched(items, batch, strict=False):
                # ordered; a worker's error (with its path) is re-raised here
                chunk = list(pool.imap(_encode, window, chunksize=max(len(window) // (4 * n_thread), 1)))
                _put_all(env, chunk)
                meta.writelines(f"{k}.png ({h},{w},{c}) {compress_level}\n" for k, _, (h, w, c) in chunk)
                pbar.update(len(chunk))
        env.close()
        tmp.rename(lmdb_path)
    except BaseException:
        env.close()
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print("\nFinish writing lmdb.")


class _NoPool:
    def __enter__(self):
        return self

    def __exit__(self, *_) -> None:
        pass

    @staticmethod
    def imap(fn, items, **_):
        return map(fn, items)


def _put_all(env, chunk) -> None:
    """Write one chunk in one transaction; a full map is doubled and the chunk retried."""
    while True:
        try:
            with env.begin(write=True) as txn:
                for key, buf, _ in chunk:
                    if not txn.put(key.encode("utf-8"), buf, overwrite=False):
                        msg = f"Duplicate lmdb key: {key}"
                        raise ValueError(msg)
            return
        except lmdb.MapFullError:
            env.set_mapsize(env.info()["map_size"] * 2)


class LmdbMaker:
    """LMDB Maker.

    Args:
    ----
        lmdb_path (str): Lmdb save path.
        map_size (int): Map size for lmdb env. Default: 1024 ** 4, 1TB.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.

    """

    def __init__(
        self, lmdb_path, map_size=1024**4, batch=5000, compress_level=1
    ) -> None:
        if not lmdb_path.endswith(".lmdb"):
            msg = "lmdb_path must end with '.lmdb'."
            raise ValueError(msg)
        if Path(lmdb_path).exists():
            msg = f"Folder {lmdb_path} already exists."
            raise FileExistsError(msg)

        self.lmdb_path = lmdb_path
        self.batch = batch
        self.compress_level = compress_level
        self.env = lmdb.open(lmdb_path, map_size=map_size)
        self.txn = self.env.begin(write=True)
        self.txt_file = Path.open(
            Path(lmdb_path) / "meta_info.txt", "w", encoding="utf-8"
        )
        self.counter = 0

    def put(self, img_byte, key, img_shape) -> None:
        self.counter += 1
        key_byte = key.encode("utf-8")
        self.txn.put(key_byte, img_byte)
        # write meta information
        h, w, c = img_shape
        self.txt_file.write(f"{key}.png ({h},{w},{c}) {self.compress_level}\n")
        if self.counter % self.batch == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self) -> None:
        self.txn.commit()
        self.env.close()
        self.txt_file.close()

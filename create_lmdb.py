import argparse
from pathlib import Path

from neosr.data.data_util import IMG_EXTENSIONS
from neosr.utils import scandir
from neosr.utils.lmdb_util import make_lmdb_from_imgs


def prepare_keys(folder_path):
    """Prepare image path list and keys.

    Args:
    ----
        folder_path (str): Folder path.

    Returns:
    -------
        list[str]: Image path list (every IMG_EXTENSIONS image, any case).
        list[str]: Key list: each path without its final extension, so `a.b.png`
            and `a.png` stay distinct keys.

    """
    print("Reading image path list ...")
    img_path_list = sorted(scandir(folder_path, suffix=IMG_EXTENSIONS, recursive=False))
    keys = [Path(p).with_suffix("").as_posix() for p in img_path_list]
    return img_path_list, keys


def create_lmdb():
    """Create lmdb files.
    Large images are best tiled first: `python -m scripts.extract_subimages`.
    """
    folder_path = args.input
    lmdb_path = args.output
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(
        folder_path,
        lmdb_path,
        img_path_list,
        keys,
        multiprocessing_read=True,
        n_thread=args.workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, help=("Input Path"))
    parser.add_argument("--output", type=str, help=("Output Path"))
    parser.add_argument(
        "--workers", type=int, default=None, help="encoder processes (default: usable CPUs)"
    )
    args = parser.parse_args()
    create_lmdb()

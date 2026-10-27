"""Cut large images into overlapping tiles: decode cost scales with image area, so a loader reading
512 px tiles serves about 10x more samples per second than one reading 2K images.

    python -m scripts.extract_subimages -i GT -o GT_tiles --crop 512 --step 384
    python -m scripts.extract_subimages -i LQ -o LQ_tiles --crop 128 --step 96   # x4 LQ: crop and step / 4

Run from the repository root. Tiles are named `<stem>_s001.png`, ...; the same crop/step (divided by the scale for LQ) gives aligned pairs.
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2

from neosr.data.data_util import image_paths


def starts(n: int, crop: int, step: int) -> list[int]:
    """Tile origins along one axis; a last tile is added flush with the border."""
    s = list(range(0, max(n - crop, 0) + 1, step))
    return s if s[-1] + crop >= n else [*s, n - crop]


def tile(path: str, out: Path, crop: int, step: int, level: int) -> int:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        msg = f"Cannot read image: {path}"
        raise OSError(msg)
    h, w = img.shape[:2]
    idx = 0
    for y in starts(h, crop, step):
        for x in starts(w, crop, step):
            idx += 1
            dst = out / f"{Path(path).stem}_s{idx:03d}.png"
            if not cv2.imwrite(str(dst), img[y : y + crop, x : x + crop], [cv2.IMWRITE_PNG_COMPRESSION, level]):
                msg = f"Cannot write {dst}"
                raise OSError(msg)
    return idx


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--step", type=int, default=384)
    p.add_argument("--level", type=int, default=3, help="PNG compression level")
    p.add_argument("--workers", type=int, default=os.process_cpu_count())
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)  # never mix tiles of two runs
    paths = image_paths(a.input)
    with Pool(a.workers) as pool:
        n = sum(pool.imap_unordered(partial(tile, out=out, crop=a.crop, step=a.step, level=a.level), paths, 4))
    print(f"{len(paths)} images -> {n} tiles in {out}")

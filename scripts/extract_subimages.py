"""Cut large images into overlapping tiles: decode cost scales with image area, so a loader reading
512 px tiles serves about 10x more samples per second than one reading 2K images.

    python -m scripts.extract_subimages -i GT -o GT_tiles --crop 512 --step 384
    python -m scripts.extract_subimages -i LQ -o LQ_tiles --crop 128 --step 96   # x4 LQ: crop and step / 4

Run from the repository root. Tiles are named `<stem>_s001.png`, ...; the same crop/step (divided by the scale for LQ) gives aligned pairs.

`--bits8` stores 16-bit sources as their top 8 bits: exactly what the loaders decode (IMREAD_COLOR keeps the
high byte), at about 1.65x less decode time. `--crop 0` converts without tiling (names kept):

    python -m scripts.extract_subimages -i GT -o GT8 --crop 0 --bits8
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from neosr.data.data_util import image_paths


def starts(n: int, crop: int, step: int) -> list[int]:
    """Tile origins along one axis; a last tile is added flush with the border."""
    s = list(range(0, max(n - crop, 0) + 1, step))
    return s if s[-1] + crop >= n else [*s, n - crop]


def _write(dst: Path, img, level: int) -> None:
    if not cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, level]):
        msg = f"Cannot write {dst}"
        raise OSError(msg)


def tile(
    path: str, out: Path, crop: int, step: int, level: int, bits8: bool = False
) -> int:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        msg = f"Cannot read image: {path}"
        raise OSError(msg)
    if bits8 and img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    if crop == 0:
        _write(out / f"{Path(path).stem}.png", img, level)
        return 1
    h, w = img.shape[:2]
    idx = 0
    for y in starts(h, crop, step):
        for x in starts(w, crop, step):
            idx += 1
            _write(
                out / f"{Path(path).stem}_s{idx:03d}.png",
                img[y : y + crop, x : x + crop],
                level,
            )
    return idx


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--crop", type=int, default=512, help="tile size; 0: whole images")
    p.add_argument("--step", type=int, default=384)
    p.add_argument("--level", type=int, default=3, help="PNG compression level")
    p.add_argument(
        "--bits8",
        action="store_true",
        help="16-bit sources as 8 bit (what training reads)",
    )
    p.add_argument("--workers", type=int, default=os.process_cpu_count())
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)  # never mix tiles of two runs
    paths = image_paths(a.input)
    with Pool(a.workers) as pool:
        n = sum(
            pool.imap_unordered(
                partial(
                    tile,
                    out=out,
                    crop=a.crop,
                    step=a.step,
                    level=a.level,
                    bits8=a.bits8,
                ),
                paths,
                4,
            )
        )
    print(f"{len(paths)} images -> {n} tiles in {out}")

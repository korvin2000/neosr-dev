import torch
from torch import Tensor, nn


def tile_spans(n: int, tile: int, halo: int) -> list[tuple[int, int, int]]:
    """(window start, core start, core end) along one axis of length n. Windows are `tile` wide and snapped
    inside the image; cores tile [0, n) exactly once and keep >= halo context on every interior side."""
    if n <= tile:
        return [(0, 0, n)]
    starts = [*range(0, n - tile, tile - 2 * halo), n - tile]
    # the last window is snapped back, so its core starts where the previous one ends (no double writes)
    ends = [s + tile - halo for s in starts[:-1]] + [n]
    return [
        (s, e0, e1) for s, e0, e1 in zip(starts, [0, *ends[:-1]], ends, strict=True)
    ]


def infer_tiled(
    net: nn.Module, lq: Tensor, scale: int, tile: int, halo: int, out_device="cpu"
) -> Tensor:
    """Fixed tile x tile windows snapped inside the image (smaller only when the image is). Each writes its
    core, so context never leaves the image. Exact iff RF <= halo and the arch is translation-equivariant on
    the tile lattice (tile, tile-2*halo and the image size multiples of its stride/window). Output stitched on
    out_device (CPU: tiling exists for memory; 8K x4 fp32 is 5.9 GiB)."""
    if not 0 <= 2 * halo < tile:
        msg = f"tiled inference needs tile > 2 * tile_overlap >= 0, got tile={tile}, tile_overlap={halo}"
        raise ValueError(msg)
    _, _, h, w = lq.shape
    out = None
    for y, y0, y1 in tile_spans(h, tile, halo):
        for x, x0, x1 in tile_spans(w, tile, halo):
            o = net(lq[..., y : y + tile, x : x + tile])
            if out is None:
                size = (*o.shape[:2], h * scale, w * scale)
                out = torch.empty(size, dtype=o.dtype, device=out_device)
            out[..., y0 * scale : y1 * scale, x0 * scale : x1 * scale] = o[
                ...,
                (y0 - y) * scale : (y1 - y) * scale,
                (x0 - x) * scale : (x1 - x) * scale,
            ]
    assert out is not None
    return out

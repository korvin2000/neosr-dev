import pytest
import torch
from torch import nn

from neosr.models.tiling import infer_tiled, tile_spans

SHAPES = [(37, 53), (64, 64), (10, 70), (100, 33), (5, 5), (1, 1)]


def conv_net(scale=2, cin=3, cout=2, depth=6):
    # depth 3x3 convs: receptive radius = depth LR pixels; pixel shuffle adds none
    torch.manual_seed(0)
    layers = [nn.Conv2d(cin, 8, 3, padding=1)]
    for _ in range(depth - 2):
        layers += [nn.LeakyReLU(0.2), nn.Conv2d(8, 8, 3, padding=1)]
    layers += [
        nn.LeakyReLU(0.2),
        nn.Conv2d(8, cout * scale**2, 3, padding=1),
        nn.PixelShuffle(scale),
    ]
    return nn.Sequential(*layers).double().eval()


def strided_net(scale=2):
    # stride-4 lattice: exact only when tile starts land on multiples of 4
    torch.manual_seed(0)
    return (
        nn
        .Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.AvgPool2d(4),
            nn.Upsample(scale_factor=4 * scale, mode="nearest"),
            nn.Conv2d(8, 3, 3, padding=1),
        )
        .double()
        .eval()
    )


def max_err(net, lq, scale, tile, halo):
    with torch.no_grad():
        return (infer_tiled(net, lq, scale, tile, halo) - net(lq)).abs().max().item()


@pytest.mark.parametrize("halo", [6, 8])
@pytest.mark.parametrize(("h", "w"), SHAPES)
def test_exact_when_halo_covers_rf(h, w, halo):
    net = conv_net()
    lq = torch.rand(2, 3, h, w, dtype=torch.float64)
    with torch.no_grad():
        out = infer_tiled(net, lq, 2, 24, halo)
        ref = net(lq)
    assert out.shape == ref.shape == (2, 2, 2 * h, 2 * w)
    assert torch.allclose(out, ref, rtol=0, atol=1e-12)


def test_inexact_below_rf():
    lq = torch.rand(1, 3, 37, 53, dtype=torch.float64)
    assert max_err(conv_net(), lq, 2, 24, 3) > 1e-6


def test_strided_alignment():
    net, lq = strided_net(), torch.rand(1, 3, 64, 96, dtype=torch.float64)
    assert max_err(net, lq, 2, 32, 8) < 1e-12
    assert max_err(net, lq, 2, 32, 5) > 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_output_on_out_device():
    net, lq = (
        conv_net().cuda(),
        torch.rand(1, 3, 40, 40, dtype=torch.float64, device="cuda"),
    )
    with torch.no_grad():
        out, ref = infer_tiled(net, lq, 2, 24, 6), net(lq)
        assert out.device.type == "cpu"
        assert torch.allclose(out, ref.cpu(), rtol=0, atol=1e-12)
        assert infer_tiled(net, lq, 2, 24, 6, out_device=lq.device).device == lq.device


@pytest.mark.parametrize(
    ("tile", "halo"), [(8, 0), (8, 3), (16, 5), (24, 8), (32, 8), (64, 16), (7, 3)]
)
def test_coverage_exactly_once(tile, halo):
    for n in range(1, 300):
        hits = [0] * n
        for s, c0, c1 in tile_spans(n, tile, halo):
            end = s + min(tile, n)
            assert s >= 0
            assert end <= n
            assert c0 - s >= (halo if c0 > 0 else 0)
            assert end - c1 >= (halo if c1 < n else 0)
            for i in range(c0, c1):
                hits[i] += 1
        assert hits == [1] * n, (n, tile, halo)


class CountingBuffer:
    # stands in for the output buffer: counts destination writes, checks source/dest shapes agree
    def __init__(self, size, **_):
        self.hits = torch.zeros(size, dtype=torch.int64)

    def __setitem__(self, key, value):
        assert value.shape == self.hits[key].shape
        self.hits[key] += 1


@pytest.mark.parametrize(("scale", "tile", "halo"), [(1, 8, 3), (2, 24, 8), (3, 16, 5)])
def test_output_written_exactly_once(monkeypatch, scale, tile, halo):
    monkeypatch.setattr(torch, "empty", CountingBuffer)
    net = nn.Upsample(scale_factor=scale)
    for h, w in [*SHAPES, (tile, tile + 1), (3 * tile - 1, 2 * tile + 7)]:
        out = infer_tiled(net, torch.zeros(1, 1, h, w), scale, tile, halo)
        assert out.hits.shape == (1, 1, h * scale, w * scale)
        assert (out.hits == 1).all(), (h, w)


def test_rejects_bad_overlap():
    lq = torch.rand(1, 3, 8, 8, dtype=torch.float64)
    for tile, halo in [(16, 8), (16, -1)]:
        with pytest.raises(ValueError, match="tile_overlap"):
            infer_tiled(conv_net(), lq, 2, tile, halo)

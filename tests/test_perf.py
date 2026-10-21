"""C6 performance pass (docs/refactor/analysis/perf.md): each rewrite keeps the old numerics."""

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from neosr.data import degradations as dg
from neosr.losses import build_loss
from neosr.losses.base import Context
from tests.test_model import build, make_opt

CUDA = torch.cuda.is_available()
DINO = Path(torch.hub.get_dir(), "facebookresearch_dinov2_main").is_dir()


def _levels(x):  # pre-C6: one torch.unique (a host sync on CUDA) per sample
    vals = [2 ** np.ceil(np.log2(len(torch.unique(s)))) for s in x]
    return x.new_tensor(vals).view(-1, 1, 1, 1)


def _old_poisson(img, scale, gray_noise):
    """generate_poisson_noise_pt before C6, tensor arguments."""
    b, _, h, w = img.size()
    gray_noise = gray_noise.view(b, 1, 1, 1)
    gray = torch.sum(gray_noise) > 0
    if gray:
        g = rgb_to_grayscale(img, num_output_channels=1)
        g = torch.clamp((g * 255.0).round(), 0, 255) / 255.0
        vals = _levels(g)
        noise_gray = (torch.poisson(g * vals) / vals - g).expand(b, 3, h, w)
    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.0
    vals = _levels(img)
    noise = torch.poisson(img * vals) / vals - img
    if gray:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise * scale.view(b, 1, 1, 1)


def test_poisson_levels_match_unique():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        lo = int(torch.randint(0, 255, (1,), generator=g))
        hi = int(torch.randint(lo + 1, 257, (1,), generator=g))
        x = torch.randint(lo, hi, (3, 2, 9, 7), generator=g).float() / 255
        assert torch.equal(dg._poisson_levels(x), _levels(x))


@pytest.mark.parametrize("dev", ["cpu", *(["cuda"] if CUDA else [])])
@pytest.mark.parametrize("gray", [0.0, 0.5, 1.0])
def test_poisson_noise_is_bitwise_unchanged(dev, gray):
    g = torch.Generator().manual_seed(0)
    # per-sample contrast: different level counts per sample
    img = torch.rand(4, 3, 24, 20, generator=g) * torch.rand(4, 1, 1, 1, generator=g)
    scale, mask = (
        torch.rand(4, generator=g),
        (torch.rand(4, generator=g) < gray).float(),
    )
    img, scale, mask = img.to(dev), scale.to(dev), mask.to(dev)
    torch.manual_seed(1)
    ref = _old_poisson(img, scale, mask)
    torch.manual_seed(1)
    assert torch.equal(dg.generate_poisson_noise_pt(img, scale, mask), ref)


def _no_sync(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "sync debug mode is a prototype"
        torch.cuda.set_sync_debug_mode("error")
        try:
            return fn()
        finally:
            torch.cuda.set_sync_debug_mode(0)


@pytest.mark.skipif(not CUDA, reason="host syncs are a CUDA notion")
def test_mssim_backward_does_not_sync():
    loss = build_loss({"type": "mssim_loss"}).cuda()
    x = torch.rand(2, 3, 64, 64, device="cuda", requires_grad=True)
    value = loss(x, torch.rand(2, 3, 64, 64, device="cuda"))
    _no_sync(value.backward)  # prod() did (input == 0).sum().item()
    assert x.grad.abs().sum() > 0


@pytest.mark.skipif(not CUDA, reason="host syncs are a CUDA notion")
def test_poisson_noise_has_no_per_sample_sync():
    img = torch.rand(8, 3, 32, 32, device="cuda")
    scale = torch.rand(8, device="cuda")
    _no_sync(lambda: dg.generate_poisson_noise_pt(img, scale, 0))


def test_compile_wraps_the_generator_only(tmp_path):
    """A compiled spectral-norm D run on real and fake before one backward fails autograd's
    saved-tensor version check (torch 2.9.1); D stays eager."""
    model = build(make_opt(tmp_path, d=True) | {"compile": True})
    assert model.net_g_fwd._orig_mod is model.net_g
    assert model.net_d_fwd is model.net_d
    assert model.net_d_adv is model.net_d


def test_fdl_precision_policy():
    assert build_loss({
        "type": "fdl_loss",
        "model": "vgg",
    }).fp32  # bf16 VGG features: phase blow-up
    assert build_loss({"type": "fdl_loss", "model": "vgg", "fp32_backbone": False}).fp32


@pytest.mark.skipif(not DINO, reason="DINOv2 hub repository not cached")
def test_fdl_dinov2_drops_unused_blocks_bitwise():
    assert build_loss({"type": "fdl_loss", "model": "dinov2"}).fp32  # fp32 by default
    loss = build_loss({"type": "fdl_loss", "model": "dinov2", "fp32_backbone": False})
    assert not loss.fp32
    cut = loss.model
    assert len(cut.model.blocks) == max(cut.layers) + 1
    full = copy.deepcopy(cut)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full.model = (
            torch.hub
            .load(
                "facebookresearch/dinov2",
                "dinov2_vitb14",
                trust_repo="check",
                verbose=False,
            )
            .to(memory_format=torch.channels_last)
            .eval()
        )
    x = torch.rand(1, 3, 28, 28, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for a, b in zip(cut(x), full(x), strict=True):
            assert torch.equal(a, b)


def _pink(n, size, g):
    """1/f-spectrum colour images in [0, 1]: natural-image statistics without fixture files."""
    f = torch.fft.fftfreq(size).abs()
    r = (f.view(-1, 1) ** 2 + f.view(1, -1) ** 2).sqrt().clamp_min(1 / size)
    img = torch.fft.ifft2(
        torch.fft.fft2(torch.randn(n, 3, size, size, generator=g)) / r
    ).real
    lo, hi = img.amin((1, 2, 3), keepdim=True), img.amax((1, 2, 3), keepdim=True)
    return (img - lo) / (hi - lo)


@pytest.fixture
def _strict_fp32():
    """The reference precision: no TF32 matmuls or convolutions (runtime fast_matmul enables both)."""
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


@pytest.mark.skipif(
    not (CUDA and DINO), reason="needs CUDA autocast and the DINOv2 hub cache"
)
@pytest.mark.usefixtures("_strict_fp32")
@pytest.mark.parametrize("size", [64, 128, 256])
def test_fdl_dinov2_amp_policy_within_tolerance(size):
    """perf.md §3.1 tolerance of the default policy against the all-fp32 loss, through a fixed G with
    bf16 output (as in training) and fp32 gt: loss rel. error < 1e-2, G-gradient cos >= 0.97,
    sr-gradient rel. L2 < 1. Below amp_min_size the backbone must stay fp32 (bf16 there: G cos
    0.72-0.91 on this data at 64 px, 0.22-0.97 on real pairs at 64-192 px)."""
    loss = build_loss({"type": "fdl_loss", "model": "dinov2"}).cuda()
    g = torch.Generator().manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 48, 3, padding=1),
        nn.PixelShuffle(4),
    )
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    net.cuda()
    gt = _pink(4, size, g).cuda()
    lq = F.interpolate(gt, scale_factor=0.25, mode="bicubic", antialias=True)
    up = F.interpolate(lq, scale_factor=4, mode="bicubic").bfloat16()

    def run(*, fp32):
        loss.fp32 = fp32  # True: the whole loss in fp32, as with fp32_backbone = true
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sr = net(lq) + up
            value = loss.compute(Context(sr=sr, gt=gt, lq=lq, it=1))
        assert sr.dtype == torch.bfloat16
        assert value.dtype == torch.float32
        *g_net, g_sr = torch.autograd.grad(value, [*net.parameters(), sr])
        return (
            value.item(),
            torch.cat([t.flatten() for t in g_net]).double(),
            g_sr.double(),
        )

    (ref, ref_net, ref_sr), (got, got_net, got_sr) = run(fp32=True), run(fp32=False)
    assert torch.isfinite(got_net).all()
    assert abs(got - ref) / abs(ref) < 1e-2
    assert F.cosine_similarity(got_net, ref_net, dim=0) >= 0.97
    assert (got_sr - ref_sr).norm() / ref_sr.norm() < 1

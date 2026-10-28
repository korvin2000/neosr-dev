"""Stage 2 losses lane (docs/refactor/stage2/analysis/losses.md, stability.md, gpu_profile.md): exact kernel
rewrites against in-test copies of the old kernels, and the opt-in GAN mechanisms."""

import warnings
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur

from neosr.losses import build_loss
from neosr.losses.base import Context
from tests.test_losses import CASES, make_pair

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CUDA = torch.cuda.is_available()
cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def grads_of(loss_fn, pred, target):
    x = pred.detach().clone().requires_grad_()
    value = loss_fn(x, target)
    value.backward()
    return value.detach(), x.grad


def rel_l2(a, b):
    return ((a - b).double().norm() / b.double().norm().clamp_min(1e-30)).item()


def near(sigma, dev=DEV):
    """A converging operating point: gt plus uniform noise of amplitude sigma (0.01 ~ 40 dB)."""
    pred, gt = make_pair("random", dev)
    return (gt + sigma * (pred - 0.5)).clamp(0, 1), gt


def check_rewrite(make_new, make_old, atol=1e-6):
    """Values match on every case; at real operating points the sr gradient of the rewrite is at
    least as close to the fp64 old kernel as the fp32 old kernel is (degenerate cases have ~0
    gradients, so a relative gradient check means nothing there)."""
    new, old, truth = (
        make_new().to(DEV),
        make_old().to(DEV),
        make_old().to(DEV).double(),
    )
    for case in CASES:
        pred, target = make_pair(case)
        torch.testing.assert_close(
            grads_of(new, pred, target)[0],
            grads_of(old, pred, target)[0],
            rtol=1e-5,
            atol=atol,
        )
    for pred, target in (make_pair("random"), near(0.03), near(0.01)):
        g64 = grads_of(truth, pred.double(), target.double())[1]
        err_new = rel_l2(grads_of(new, pred, target)[1], g64)
        err_old = rel_l2(grads_of(old, pred, target)[1], g64)
        assert err_new <= 1.5 * err_old + 1e-6, (err_new, err_old)
    return new, old


def _no_sync(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "sync debug mode is a prototype"
        torch.cuda.set_sync_debug_mode("error")
        try:
            return fn()
        finally:
            torch.cuda.set_sync_debug_mode(0)


# ---- L-02 / L-04 consistency: fp32 and a fixed separable blur ------------------------------------


def old_blur(x):
    """torchvision GaussianBlur(21, 3): a 21x21 kernel rebuilt per call, reflect pad 10."""
    return gaussian_blur(x, [21, 21], [3.0, 3.0])


def old_consistency():
    loss = build_loss({"type": "consistency_loss"})
    loss.blur = old_blur  # the pre-stage-2 kernel, everything else shared
    return loss


def test_consistency_separable_blur_matches_torchvision():
    # grad rel. L2 vs fp64 at 27-40 dB: 4-11e-4 new, 7-15e-4 old (measured)
    new, _ = check_rewrite(
        lambda: build_loss({"type": "consistency_loss"}), old_consistency
    )
    for case in CASES:
        x = make_pair(case)[0]
        torch.testing.assert_close(new.blur(x), old_blur(x), rtol=1e-5, atol=2e-6)
    assert not new.state_dict()


def test_consistency_draws_no_rng():
    loss = build_loss({"type": "consistency_loss"})
    pred, target = make_pair("random", "cpu")
    state = torch.get_rng_state()
    loss(pred, target)
    assert torch.equal(
        torch.get_rng_state(), state
    )  # GaussianBlur.get_params drew twice per call


@cuda
def test_consistency_is_fp32_under_autocast():
    """L-02: the blur conv and the luma matmul ran in bf16 (0.37-0.60 grad rel. L2 at 27-40 dB)."""
    loss = build_loss({"type": "consistency_loss"}).cuda()
    assert loss.fp32
    pred, gt = make_pair("random", "cuda")
    pred = gt + 0.01 * (
        pred - 0.5
    )  # ~40 dB: where bf16 rounding was the size of the residual
    out = []
    for amp in (False, True):
        sr = pred.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            value = loss.compute(Context(sr=sr, gt=gt, lq=gt, it=1))
        value.backward()
        out.append((value.detach(), sr.grad))
    (v32, g32), (vamp, gamp) = out
    assert vamp.dtype == torch.float32
    torch.testing.assert_close(vamp, v32, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(gamp, g32, rtol=1e-6, atol=1e-9)


# ---- L-05 LDL: box-filter moments instead of unfold + var ----------------------------------------


def old_local_weights(residual, ksize=7):
    pad = (ksize - 1) // 2
    r = F.pad(residual, pad=[pad, pad, pad, pad], mode="reflect")
    u = r.unfold(2, ksize, 1).unfold(3, ksize, 1)
    return (
        torch.var(u, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)
    )


def old_ldl():
    loss = build_loss({"type": "ldl_loss"})
    loss.get_local_weights = old_local_weights
    return loss


def test_ldl_box_variance_matches_unfold():
    new, _ = check_rewrite(
        lambda: build_loss({"type": "ldl_loss"}), old_ldl
    )  # grad rel. L2 ~5e-7
    for case in CASES:
        pred, target = make_pair(case)
        r = (target - pred).abs().sum(1, keepdim=True)
        torch.testing.assert_close(
            new.get_local_weights(r), old_local_weights(r), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            grads_of(new, pred, target)[1], grads_of(old_ldl().to(DEV), pred, target)[1],
            rtol=1e-5, atol=1e-9,
        )  # fmt: skip


def test_ldl_box_variance_is_never_negative():
    loss = build_loss({"type": "ldl_loss"})
    flat = torch.full((1, 1, 16, 16), 0.3) + 1e-7 * torch.randn(
        1, 1, 16, 16
    )  # m2 - m1^2 cancels
    assert (loss.get_local_weights(flat) >= 0).all()


# ---- L-06 MS-SSIM: separable window, one filter call per side ------------------------------------


class Gaussian2D(nn.Module):
    """The pre-stage-2 11x11 depthwise window (any channel count); `row` is what mssim casts to."""

    def __init__(self, window_size=11, sigma=1.5, padding=None):
        super().__init__()
        x = torch.arange(-(window_size // 2), window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / (sigma * sigma))
        w = (w / w.sum()).reshape(1, 1, 1, window_size)
        self.register_buffer("row", torch.matmul(w.transpose(-1, -2), w))
        self.padding = window_size // 2 if padding is None else padding

    def forward(self, x):
        w = self.row.expand(x.shape[1], -1, -1, -1)
        return F.conv2d(x, w, padding=self.padding, groups=x.shape[1])


def old_mssim(**kw):
    loss = build_loss({"type": "mssim_loss", **kw})
    loss.gaussian_filter = Gaussian2D(padding=kw.get("padding"))
    return loss


def test_mssim_separable_matches_2d_window():
    # grad rel. L2 vs fp64 at 27-40 dB: 3-8e-5 new, 6-18e-5 old (measured)
    new, old = check_rewrite(lambda: build_loss({"type": "mssim_loss"}), old_mssim)
    x = make_pair("random")[0]
    torch.testing.assert_close(
        new.gaussian_filter(x), old.gaussian_filter(x), rtol=1e-5, atol=1e-6
    )
    for p in (0, 3):  # a custom padding is still the 2D window's padding
        a, b = build_loss({"type": "mssim_loss", "padding": p}), old_mssim(padding=p)
        torch.testing.assert_close(
            a.gaussian_filter(x.cpu()), b.gaussian_filter(x.cpu()), rtol=1e-5, atol=1e-6
        )
    assert not new.state_dict()  # the window is a constant, not state


def test_mssim_gray_input_unchanged():
    """1-channel input: the old 3-row window made 3 identical channels; the mean is the same."""
    pred, target = (t[:, :1] for t in make_pair("random", "cpu"))
    new, old = build_loss({"type": "mssim_loss"}), old_mssim()
    torch.testing.assert_close(
        new(pred, target), old(pred, target), rtol=1e-4, atol=1e-6
    )


# ---- L-10 ff_loss: no host sync --------------------------------------------------------------------


@cuda
def test_ff_loss_does_not_sync():
    loss = build_loss({"type": "ff_loss"}).cuda()
    x = torch.rand(2, 3, 64, 64, device="cuda", requires_grad=True)
    y = torch.rand(2, 3, 64, 64, device="cuda")
    _no_sync(lambda: loss(x, y).backward())  # two .item() asserts synced per call
    assert x.grad.abs().sum() > 0


# ---- GP-03 / L-17 FDL(DINOv2): cached position embedding, opt-in recomputation ---------------------

DINO = Path(torch.hub.get_dir(), "facebookresearch_dinov2_main").is_dir()
dino = pytest.mark.skipif(not DINO, reason="DINOv2 hub repository not cached")


def saved_bytes(fn):
    """Bytes of the distinct storages autograd saves for backward while fn() runs."""
    seen = {}

    def pack(t):
        seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, sum(seen.values())


@dino
def test_fdl_pos_encoding_cache_is_exact_and_bounded():
    vit = build_loss({"type": "fdl_loss", "model": "dinov2"}).model.model
    g = torch.Generator().manual_seed(0)

    def tokens(h, w):
        return torch.rand(1, 1 + (h // 14) * (w // 14), 768, generator=g)

    for h, w in ((28, 42), (42, 56), (56, 28), (28, 42)):
        x = tokens(h, w)
        hit = vit.interpolate_pos_encoding(x, h, w)
        assert torch.equal(hit, type(vit).interpolate_pos_encoding(vit, x, h, w))
    assert len(vit.pos_cache) == 3  # the repeated size was a hit
    assert not any(t.requires_grad for t in vit.pos_cache.values())
    for h in range(14, 14 * 12, 14):
        vit.interpolate_pos_encoding(tokens(h, 14), h, 14)
    assert len(vit.pos_cache) == 8


@dino
@pytest.mark.parametrize(
    "amp", [False, pytest.param(True, marks=cuda)], ids=["fp32", "bf16"]
)
def test_fdl_pos_encoding_cache_keeps_the_loss_bitwise(amp):
    dev = "cuda" if amp else "cpu"
    loss = build_loss({
        "type": "fdl_loss",
        "model": "dinov2",
        "fp32_backbone": not amp,
    }).to(dev)
    loss.amp_min_size = 56  # let the small test patch take the autocast path
    pred, gt = (t[:1, :, :56, :70] for t in make_pair("random", dev))
    out = []
    for cached in (True, False):
        if not cached:
            del loss.model.model.interpolate_pos_encoding  # back to the hub method
        sr = pred.clone().requires_grad_()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
            value = loss.compute(Context(sr=sr, gt=gt, lq=gt, it=1))
        value.backward()
        out.append((value.detach(), sr.grad))
    assert torch.equal(out[0][0], out[1][0])
    assert torch.equal(out[0][1], out[1][1])


@dino
def test_fdl_checkpoint_is_bitwise_and_saves_memory():
    plain = build_loss({"type": "fdl_loss", "model": "dinov2"})
    ckpt = build_loss({"type": "fdl_loss", "model": "dinov2", "checkpoint": True})
    pred, gt = (t[:2, :, :56, :56] for t in make_pair("random", "cpu"))
    out = []
    for loss in (plain, ckpt):
        sr = pred.clone().requires_grad_()
        value, nbytes = saved_bytes(
            lambda loss=loss, sr=sr: loss.compute(Context(sr=sr, gt=gt, lq=gt, it=1))
        )
        value.backward()
        out.append((value.detach(), sr.grad, nbytes))
    (v0, g0, b0), (v1, g1, b1) = out
    assert torch.equal(v0, v1)
    assert torch.equal(g0, g1)
    assert b1 < 0.5 * b0, (b1, b0)


def test_fdl_checkpoint_needs_dinov2():
    with pytest.raises(ValueError, match="dinov2"):
        build_loss({"type": "fdl_loss", "model": "vgg", "checkpoint": True})

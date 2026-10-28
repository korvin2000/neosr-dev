"""Stage 2 losses lane (docs/refactor/stage2/analysis/losses.md, stability.md, gpu_profile.md): exact kernel
rewrites against in-test copies of the old kernels, and the opt-in GAN mechanisms."""

import math
import warnings
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur

from neosr.archs import build_network
from neosr.losses import build_loss
from neosr.losses.base import Context
from neosr.losses.gan_loss import batch_coupled, eager, gan_loss
from neosr.runtime import Runtime
from tests.test_gan import LABELS, Critic, Forbidden, Spy, images
from tests.test_gan import ctx as gan_ctx
from tests.test_losses import CASES, make_pair
from tests.test_model import batches as model_batches
from tests.test_model import build as build_model_seeded
from tests.test_model import make_opt as model_opt
from tests.test_model import run as run_model

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


# ---- GAN: one-forward D update, start/ramp, lazy R1/R2, rpgan ---------------------------------------


def _gan(**kw):
    g_term, d_term = gan_loss(**kw).to(DEV).expand()
    g_term.loss_weight = 0.1
    return g_term, d_term


def _critic(seed=0, bn=False):
    torch.manual_seed(seed)
    critic = Critic()
    if bn:
        critic.body.insert(1, nn.BatchNorm2d(8))
    return critic.to(DEV)


def test_d_update_is_one_forward_on_cat():
    """GP-01: real and fake go through D as one batch; same values and grads as two forwards
    (eval mode: no SN step, no dropout), same logs."""
    critic = _critic().eval()
    ref, spy = deepcopy(critic), Spy(critic)
    _, d_term = _gan(**LABELS)
    sr, gt = images(), images(seed=1)
    c = gan_ctx(sr, gt, net_d=spy, net_d_adv=Forbidden())
    d_term.compute(c).backward()
    assert len(spy.seen) == 1
    assert torch.equal(spy.seen[0], torch.cat([gt, sr]))
    real, fake = ref(gt), ref(sr)
    want = F.binary_cross_entropy_with_logits(
        real, torch.full_like(real, 0.9)
    ) + F.binary_cross_entropy_with_logits(fake, torch.full_like(fake, 0.2))
    want.backward()
    assert set(c.logs) == {"l_d_real", "l_d_fake", "out_d_real", "out_d_fake"}
    torch.testing.assert_close(c.logs["out_d_fake"], fake.mean().detach())
    for p, q in zip(critic.parameters(), ref.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


def test_batchnorm_d_keeps_two_forwards():
    """Batch statistics couple real and fake: such a D keeps its two forwards, bitwise as before."""
    critic = _critic(bn=True).train()
    ref, spy = deepcopy(critic), Spy(critic)
    assert batch_coupled(spy)
    assert not batch_coupled(_critic())
    _, d_term = _gan()
    sr, gt = images(), images(seed=1)
    torch.manual_seed(5)  # the same dropout draws in both runs
    value = d_term.compute(gan_ctx(sr, gt, net_d=spy, net_d_adv=Forbidden()))
    assert [x.shape[0] for x in spy.seen] == [2, 2]
    torch.manual_seed(5)
    real, fake = ref(gt), ref(sr)
    want = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
    want = want + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
    assert torch.equal(value, want)


def test_eager_unwraps_compile_only():
    net = Critic()
    assert eager(net) is net
    assert eager(torch.compile(net)) is net  # lazy: nothing is compiled here


def test_gan_start_and_ramp():
    """ST-05: G and D terms are off through start_iter; then G's weight ramps linearly to 1."""
    critic = _critic().eval()
    g_term, d_term = _gan(start_iter=3, ramp_iters=4)
    full_term, _ = _gan()
    sr, gt = images(), images(seed=1)
    full = full_term.compute(gan_ctx(sr, gt, net_d=Forbidden(), net_d_adv=critic, it=4))
    for it in range(1, 11):
        assert g_term.active(it) is d_term.active(it) is (it > 3)
        if it <= 3:
            continue
        value = g_term.compute(
            gan_ctx(sr, gt, net_d=Forbidden(), net_d_adv=critic, it=it)
        )
        if it >= 3 + 4:
            assert torch.equal(value, full)  # the full term is not scaled at all
        else:
            torch.testing.assert_close(value, full * (it - 3) / 4, rtol=1e-6, atol=0)
    g0, d0 = _gan()  # the defaults: always on, never scaled
    assert all(g0.active(it) and d0.active(it) for it in (1, 2, 10**6))


def test_gan_start_keeps_d_idle_under_phase_filtering():
    """The trainer drops inactive terms (Loss.active, PLAN interface 1): before start_iter neither
    D handle is ever called and the D objective has no term (an idle phase)."""
    g_term, d_term = _gan(start_iter=2)
    sr, gt = images(), images(seed=1)
    for it in (1, 2):
        c = gan_ctx(sr, gt, net_d=Forbidden(), net_d_adv=Forbidden(), it=it)
        assert [t for t in (g_term, d_term) if t.active(c.it)] == []
    critic = _critic().eval()
    c = gan_ctx(sr, gt, net_d=critic, net_d_adv=critic, it=3)
    assert all(torch.isfinite(t.compute(c)) for t in (g_term, d_term) if t.active(3))


@pytest.mark.parametrize(
    "kw",
    [
        {"start_iter": -1},
        {"ramp_iters": -2},
        {"r1_gamma": -1.0},
        {"reg_every": 0},
        {"gan_type": "rpgan", "real_label_val": 0.9},
        {"gan_type": "rpgan", "fake_label_val": 0.1},
    ],
)
def test_gan_rejects_bad_options(kw):
    with pytest.raises(ValueError, match=r"gan_loss|rpgan"):
        gan_loss(**kw)


def _penalty_reference(critic, real, fake, r1, r2, k):
    xr, xf = real.clone().requires_grad_(), fake.clone().requires_grad_()
    out_r, out_f = critic(xr), critic(xf)
    pen = 0
    for gm, out, x in ((r1, out_r, xr), (r2, out_f, xf)):
        (gx,) = torch.autograd.grad(out.flatten(1).mean(1).sum(), x, create_graph=True)
        pen = pen + gm * gx.square().flatten(1).sum(1).mean()
    return out_r, out_f, 0.5 * k * pen


def test_lazy_r1_r2_matches_reference():
    """ST-06: every reg_every-th update adds gamma * k / 2 * E||grad_x D(x)||^2 for real (R1) and
    fake (R2) inputs, through D's parameters (double backward); other updates are unchanged."""
    critic = _critic().eval()
    ref = deepcopy(critic)
    _, d_term = _gan(r1_gamma=2.0, r2_gamma=0.5, reg_every=4)
    _, plain = _gan()
    sr, gt = images(), images(seed=1)
    for it in (3, 5):  # not a regularised update: the plain objective, bitwise
        c, cp = (
            gan_ctx(sr, gt, net_d=critic, net_d_adv=Forbidden(), it=it) for _ in "ab"
        )
        assert torch.equal(d_term.compute(c), plain.compute(cp))
        assert "l_d_reg" not in c.logs
    c = gan_ctx(sr, gt, net_d=critic, net_d_adv=Forbidden(), it=8)
    value = d_term.compute(c)
    value.backward()
    out_r, out_f, reg = _penalty_reference(ref, gt, sr, 2.0, 0.5, 4)
    want = F.binary_cross_entropy_with_logits(out_r, torch.ones_like(out_r))
    want = (
        want + F.binary_cross_entropy_with_logits(out_f, torch.zeros_like(out_f)) + reg
    )
    want.backward()
    torch.testing.assert_close(value, want.detach())
    torch.testing.assert_close(c.logs["l_d_reg"], reg.detach())
    assert reg > 0
    for p, q in zip(critic.parameters(), ref.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


def test_lazy_r1_runs_in_fp32_under_autocast():
    """The regularised D forward leaves autocast: bf16 input gradients would be rounded away."""
    critic = _critic().eval()
    _, d_term = _gan(r1_gamma=1.0, reg_every=1)
    sr, gt = images(), images(seed=1)
    out = []
    for amp in (False, True):
        c = gan_ctx(sr, gt, net_d=critic, net_d_adv=Forbidden(), it=1)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=amp):
            out.append(d_term.compute(c).detach())
        assert c.logs["out_d_real"].dtype == torch.float32
    torch.testing.assert_close(out[1], out[0], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=cuda)])
@pytest.mark.parametrize("d", ["metagan", "unet", "patchgan"])
def test_lazy_r1_double_backward_through_discriminators(d, dev):
    """Train-mode SN, GroupNorm/RMSNorm, Mish, DropPath and SDPA attention (metagan) all take the
    double backward; metagan only under the math SDPA kernel (losses.md L-08)."""
    cfg = {"metagan": {}, "unet": {"num_feat": 8}, "patchgan": {"num_feat": 8}}[d]
    torch.manual_seed(0)
    net = build_network({"type": d, **cfg}, section="network_d").to(dev).train()
    _, d_term = gan_loss(r1_gamma=1.0, r2_gamma=1.0, reg_every=1).to(dev).expand()
    sr, gt = images((2, 3, 64, 64), dev=dev), images((2, 3, 64, 64), dev=dev, seed=1)
    c = gan_ctx(sr, gt, net_d=net, net_d_adv=Forbidden(), it=1)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        value = d_term.compute(c)
    value.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None and g.isfinite().all() for g in grads)
    assert torch.isfinite(c.logs["l_d_reg"])
    assert c.logs["l_d_reg"] > 0


def test_rpgan_g_term_matches_reference():
    """ST-07: G minimises softplus(D(real) - D(fake)); D(real) is a no-grad constant of the pair."""
    critic = _critic().eval()
    ref, spy = deepcopy(critic), Spy(critic)
    g_term, _ = _gan(gan_type="rpgan")
    sr, gt = images().requires_grad_(), images(seed=1)
    value = g_term.compute(gan_ctx(sr, gt, net_d=Forbidden(), net_d_adv=spy))
    gd = torch.autograd.grad(value, [sr, *critic.parameters()])
    assert [x.shape[0] for x in spy.seen] == [2, 2]
    assert torch.equal(spy.seen[0], gt)
    ref_sr = sr.detach().clone().requires_grad_()
    want = F.softplus(ref(gt).detach() - ref(ref_sr)).mean() * 0.1
    want_g = torch.autograd.grad(want, [ref_sr, *ref.parameters()])
    torch.testing.assert_close(value, want)
    for a, b in zip(gd, want_g, strict=True):  # D params see only the fake side
        torch.testing.assert_close(a, b)


def test_rpgan_d_term_matches_reference():
    critic = _critic().eval()
    ref = deepcopy(critic)
    _, d_term = _gan(gan_type="rpgan")
    sr, gt = images(), images(seed=1)
    c = gan_ctx(sr, gt, net_d=critic, net_d_adv=Forbidden())
    value = d_term.compute(c)
    value.backward()
    real, fake = ref(gt), ref(sr)
    want = F.softplus(fake - real).mean()  # falls as D(fake) - D(real) falls
    want.backward()
    torch.testing.assert_close(value, want.detach())
    assert set(c.logs) == {"out_d_margin", "out_d_real", "out_d_fake"}
    torch.testing.assert_close(c.logs["out_d_margin"], (real - fake).mean().detach())
    for p, q in zip(critic.parameters(), ref.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


# ---- GP-07: wrap() compiles a one-forward D --------------------------------------------------------


def test_compile_wraps_a_one_forward_d(tmp_path):
    opt = model_opt(tmp_path, d=True) | {"compile": True}
    m = build_model_seeded(opt)
    assert m.net_g_fwd._orig_mod is m.net_g
    assert m.net_d_fwd._orig_mod is m.net_d
    assert m.net_d_adv is m.net_d_fwd
    opt["network_d"] = {
        "type": "patchgan",
        "num_feat": 4,
        "norm_type": "batch",
        "use_sn": False,
    }
    m = build_model_seeded(opt)
    assert m.net_d_fwd is m.net_d_adv is m.net_d  # two forwards: eager


def test_compiled_spectral_norm_d_trains(tmp_path):
    """Two compiled SN-D forwards before one backward failed autograd's version check (perf.md);
    the one-forward update trains under inductor (CPU here: the GPU is shared)."""
    gan = {"type": "gan_loss", "loss_weight": 0.1, "r1_gamma": 1.0, "reg_every": 2}
    opt = model_opt(tmp_path, d=True) | {"compile": True}
    opt["train"]["gan_opt"] = gan
    m = build_model_seeded(opt)
    run_model(m, model_batches(3))
    log = m.get_current_log()
    assert m.update == 3
    assert {"l_d_reg", "l_d_real", "grad_norm_d"} <= set(log)
    assert all(math.isfinite(v) for v in log.values())


# ---- DDP: the regularised D update through the reducer (2-rank gloo, CPU) -------------------------


def _ddp_worker(rank: int, tmp: str, out: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp}/pg", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )  # fmt: skip
    try:
        opt = model_opt(Path(tmp) / "run", d=True, accumulate=2)
        opt["train"]["gan_opt"] = {
            "type": "gan_loss", "loss_weight": 0.1, "gan_type": "rpgan",
            "ramp_iters": 2, "r1_gamma": 1.0, "r2_gamma": 0.5, "reg_every": 2,
        }  # fmt: skip
        m = build_model_seeded(
            opt,
            rt=Runtime(torch.device("cpu"), rank, 2, 0, distributed=True),
            seed=rank,
        )
        run_model(m, model_batches(6, seed=rank))  # rank-specific data, 3 updates
        log = m.get_current_log()
        nets = {n: {k: v.detach().clone() for k, v in net.state_dict().items()}
                for n, net in m.nets.items()}  # fmt: skip
        torch.save({"log": log, "nets": nets, "update": m.update}, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_ddp_regularised_rpgan_keeps_ranks_in_sync(tmp_path):
    """autograd.grad on D's inputs under the DDP-wrapped D, then one backward: the reducer still
    averages D's (penalty-including) gradients, so both ranks hold the same weights."""
    out = str(tmp_path / "res")
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path), out), nprocs=2)
    r0, r1 = (torch.load(f"{out}.{r}", weights_only=False) for r in (0, 1))
    assert r0["update"] == r1["update"] == 3
    assert {"l_d_reg", "out_d_margin", "grad_norm_d"} <= set(r0["log"])
    assert all(math.isfinite(v) for v in r0["log"].values())
    assert r0["log"] == r1["log"]  # window means are all-reduced
    for n in r0["nets"]:
        for k, v in r0["nets"][n].items():
            assert torch.equal(v, r1["nets"][n][k]), (n, k)


# ---- ST-01: scripts/term_grads.py -------------------------------------------------------------------


def test_term_grads_reports_shares_and_leaves_the_model(tmp_path):
    from scripts.term_grads import term_grads  # ruff: ignore[import-outside-top-level]

    opt = model_opt(tmp_path, d=True)
    opt["train"]["mse_opt"] = {"type": "MSELoss", "loss_weight": 0.5}
    m = build_model_seeded(opt)
    before = deepcopy(m.net_g.state_dict())
    stats = term_grads(m, model_batches(3))
    assert set(stats) == {"l_g_pix", "l_g_mse", "l_g_gan"}
    assert sum(s["share"] for s in stats.values()) == pytest.approx(1.0)
    assert all(-1 - 1e-6 <= s["cos"] <= 1 + 1e-6 for s in stats.values())
    assert list(stats) == sorted(stats, key=lambda k: -stats[k]["share"])
    for k, v in m.net_g.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert (m.update, m.micro) == (0, 0)
    assert all(not s for s in m.g.opt.state.values())
    assert all(p.grad is None for p in m.net_g.parameters())

    # one batch, by hand: the L1 term's gradient norm over G's trainable parameters
    one = term_grads(m, model_batches(1))
    m.feed_data(model_batches(1)[0])
    loss = F.l1_loss(m.net_g(m.lq).clamp(1 / 255, 1), m.gt)
    gs = torch.autograd.grad(loss, m.g.params)
    norm = torch.cat([g.flatten() for g in gs]).norm()
    assert one["l_g_pix"]["norm"] == pytest.approx(float(norm), rel=1e-5)


def test_term_grads_skips_inactive_terms(tmp_path):
    from scripts.term_grads import term_grads  # ruff: ignore[import-outside-top-level]

    opt = model_opt(tmp_path, d=True)
    opt["train"]["gan_opt"]["start_iter"] = 5
    m = build_model_seeded(opt)
    only = term_grads(m, model_batches(1))  # update 1: the GAN term is off
    assert set(only) == {"l_g_pix"}
    assert only["l_g_pix"]["share"] == pytest.approx(1.0)
    assert only["l_g_pix"]["cos"] == pytest.approx(1.0)
    assert set(term_grads(m, model_batches(1), it=6)) == {"l_g_pix", "l_g_gan"}

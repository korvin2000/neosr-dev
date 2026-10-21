"""GAN terms and the adversarial view (design T1 §4.3, acceptance §6 row A2)."""

import subprocess
import types
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import neosr.losses as losses
from neosr.archs.esrgan_arch import esrgan
from neosr.archs.metagan_arch import metagan
from neosr.losses import build_losses
from neosr.losses.base import Context, Loss
from neosr.losses.gan_loss import gan_d_loss, gan_loss
from neosr.losses.wavelet_guided import AdvView

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LEGACY_REV = "c48252d"  # last revision with the @torch.no_grad wavelet_guided(output, gt) function
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def legacy_wavelet_guided():
    try:
        src = subprocess.run(
            ["git", "show", f"{LEGACY_REV}:neosr/losses/wavelet_guided.py"],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git history with {LEGACY_REV} is not available")
    mod = types.ModuleType("legacy_wavelet_guided")
    exec(compile(src, mod.__name__, "exec"), mod.__dict__)  # noqa: S102
    return mod.wavelet_guided


def images(shape=(2, 3, 32, 40), dev=DEV, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.rand(shape, device=dev, generator=g)


def rms(x):
    return x.double().pow(2).mean().sqrt().item()


class Critic(nn.Module):
    """A tiny train-mode D with an active dropout: enough to tell G from D gradients."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x):
        return self.body(x)


class Spy(nn.Module):
    """A distinct D handle over the same critic that records what it is called on."""

    def __init__(self, d: nn.Module) -> None:
        super().__init__()
        self.d, self.seen = d, []

    def forward(self, x):
        self.seen.append(x)
        return self.d(x)


class Forbidden(nn.Module):
    """The D handle a term must not use (T1 D5: G term -> net_d_adv, D term -> net_d)."""

    def forward(self, _x):
        msg = "wrong D handle"
        raise AssertionError(msg)


CRITS = {
    "bce": F.binary_cross_entropy_with_logits,
    "mse": F.mse_loss,
    "huber": F.huber_loss,
}
LABELS = {
    "real_label_val": 0.9,
    "fake_label_val": 0.2,
}  # nondefault: a swapped label shows


def ctx(sr, gt, *, net_d, net_d_adv, it=1):
    return Context(
        sr=sr, gt=gt, lq=gt[..., ::2, ::2], it=it, net_d=net_d, net_d_adv=net_d_adv
    )


# ---- AdvView ------------------------------------------------------------------------------------


def test_advview_identity_then_hf():
    view, x = AdvView(5).to(DEV), images()
    assert view(x, 4) is x
    for it in (5, 6):
        hf = view(x, it)
        assert hf.shape == (2, 3, 32, 40)
        assert hf.dtype == torch.float32
    assert not list(view.parameters())
    assert not view.state_dict()  # the filters are constants, not checkpoint state
    assert {n for n, _ in view.named_buffers()} == {
        "y_w",
        "band_w",
        "swt.h0_col",
        "swt.h1_col",
        "swt.h0_row",
        "swt.h1_row",
    }


def test_advview_equals_legacy_wavelet_guided():
    gt = images()
    _, hf_gt = legacy_wavelet_guided()(gt, gt)
    torch.testing.assert_close(AdvView().to(DEV)(gt, 0), hf_gt, rtol=1e-6, atol=1e-6)


def test_advview_gradcheck():
    view = AdvView().double()
    x = torch.rand(1, 3, 24, 24, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: view(t, 0), (x,), fast_mode=True)


@cuda
def test_advview_bf16_symmetry():
    """Fake (bf16 sr) and real (fp32 gt) HF now carry the same error; today's fake path carried ~2x."""
    x = images()
    q = x.bfloat16()  # what G emits under bf16 autocast
    view = AdvView().to(DEV)
    ref = view(x, 0)
    real_err = rms(
        view(q.float(), 0) - ref
    )  # the gt path: the same quantised values, all fp32
    with torch.autocast("cuda", dtype=torch.bfloat16):
        new_err = rms(view(q, 0) - ref)
        old_hf, _ = legacy_wavelet_guided()(q, x)
    old_err = rms(old_hf - ref)
    assert new_err <= 1.2 * real_err
    assert old_err > 1.2 * real_err  # the check can see the asymmetry it guards against


# ---- gan_loss / gan_d_loss --------------------------------------------------------------------


@pytest.mark.parametrize("wavelet_init", [None, 0])
def test_g_term_reaches_sr_and_leaves_d_grads(wavelet_init):
    torch.manual_seed(0)
    net_g, net_d = nn.Conv2d(3, 3, 3, padding=1).to(DEV), Critic().to(DEV).train()
    g_term, d_term = gan_loss(wavelet_init=wavelet_init).to(DEV).expand()
    gt, lq = images(), images(seed=1)

    sr = lq.clone().requires_grad_()  # G term reaches sr
    g_term.compute(ctx(sr, gt, net_d=Forbidden(), net_d_adv=net_d)).backward(
        inputs=[sr]
    )
    assert sr.grad is not None
    assert sr.grad.abs().sum() > 0

    d_term.compute(
        ctx(net_g(lq), gt, net_d=net_d, net_d_adv=Forbidden())
    ).backward()  # pre-existing D grads (accumulation)
    before = [p.grad.clone() for p in net_d.parameters()]
    params_g = list(net_g.parameters())
    g_term.compute(ctx(net_g(lq), gt, net_d=Forbidden(), net_d_adv=net_d)).backward(
        inputs=params_g
    )
    assert all(
        torch.equal(p.grad, b) for p, b in zip(net_d.parameters(), before, strict=True)
    )
    assert all(p.grad.abs().sum() > 0 for p in params_g)


@pytest.mark.parametrize("gan_type", sorted(CRITS))
@pytest.mark.parametrize("it", [4, 5, 6])  # before, at and after wavelet_init
def test_g_term_matches_independent_reference(gan_type, it):
    """G calls only net_d_adv, on AdvView(sr, it); value and sr.grad match a hand-built reference."""
    torch.manual_seed(0)
    critic = Critic().to(DEV).eval()  # deterministic, so the reference can re-run it
    adv, view = Spy(critic), AdvView(5).to(DEV)
    g_term = gan_loss(gan_type=gan_type, wavelet_init=5, **LABELS).to(DEV)
    sr, gt = images().requires_grad_(), images(seed=1)
    value = g_term.compute(ctx(sr, gt, net_d=Forbidden(), net_d_adv=adv, it=it))
    value.backward()
    assert len(adv.seen) == 1
    torch.testing.assert_close(adv.seen[0], view(sr, it), rtol=0, atol=0)

    ref_sr = sr.detach().clone().requires_grad_()
    logits = critic(view(ref_sr, it))
    want = CRITS[gan_type](logits, torch.full_like(logits, 0.9)) * 0.1
    want.backward()
    torch.testing.assert_close(value, want)
    torch.testing.assert_close(sr.grad, ref_sr.grad)
    assert sr.grad.abs().sum() > 0
    assert gan_loss.loss_weight == pytest.approx(0.1)
    assert not hasattr(g_term, "update_loss_weight")


@pytest.mark.parametrize("gan_type", sorted(CRITS))
@pytest.mark.parametrize("wavelet_init", [None, 0, 10])
def test_d_term_matches_independent_reference(gan_type, wavelet_init):
    """D calls only net_d with sr detached; its grads equal a cloned critic's real + fake
    reference, and each branch contributes."""
    torch.manual_seed(0)
    critic = Critic().to(DEV).eval()
    ref = deepcopy(critic)
    _, d_term = (
        gan_loss(gan_type=gan_type, wavelet_init=wavelet_init, **LABELS)
        .to(DEV)
        .expand()
    )
    assert isinstance(d_term, gan_d_loss)
    assert (d_term.phase, d_term.log_key, d_term.loss_weight) == ("d", "", 1.0)
    sr, gt = images().requires_grad_(), images(seed=1)
    c = ctx(sr, gt, net_d=critic, net_d_adv=Forbidden())
    value = d_term.compute(c)
    value.backward()
    assert sr.grad is None
    assert set(c.logs) == {"l_d_real", "l_d_fake", "out_d_real", "out_d_fake"}

    view = AdvView(10 if wavelet_init is None else wavelet_init).to(DEV)
    real, fake = (
        ref(view(gt, 1)),
        ref(view(sr.detach(), 1)),
    )  # it=1: identity for None and 10
    l_real = CRITS[gan_type](real, torch.full_like(real, 0.9))
    l_fake = CRITS[gan_type](fake, torch.full_like(fake, 0.2))
    params = list(ref.parameters())
    g_real = torch.autograd.grad(l_real, params)
    g_fake = torch.autograd.grad(l_fake, params)
    assert all(sum(g.abs().sum() for g in gs) > 0 for gs in (g_real, g_fake))
    for p, a, b in zip(critic.parameters(), g_real, g_fake, strict=True):
        torch.testing.assert_close(p.grad, a + b)
    for key, want in (
        ("l_d_real", l_real),
        ("l_d_fake", l_fake),
        ("out_d_real", real.mean()),
        ("out_d_fake", fake.mean()),
    ):
        torch.testing.assert_close(c.logs[key], want.detach())
    torch.testing.assert_close(value, (l_real + l_fake).detach())


def test_unknown_gan_type():
    with pytest.raises(ValueError, match="gan_type"):
        gan_loss(gan_type="wgan")


# ---- builder ------------------------------------------------------------------------------------


class Pix(Loss):
    def forward(self, x, y):
        return F.l1_loss(x, y)


def real_gan_build(cfg, *, scale, section):  # noqa: ARG001
    """The real gan_loss; a stub pixel term (B1 migrates the real kernels separately)."""
    kw = dict(cfg)
    return gan_loss(**kw) if kw.pop("type") == "gan_loss" else Pix()


@pytest.mark.parametrize(
    ("weight", "g_keys"), [(0, ["l_g_pix"]), (None, ["l_g_pix", "l_g_gan"])]
)
def test_builder(weight, g_keys, monkeypatch):
    monkeypatch.setattr(losses, "build_loss", real_gan_build)
    gan = {"type": "gan_loss"} | ({} if weight is None else {"loss_weight": weight})
    train = {
        "pixel_opt": {"type": "L1Loss"},
        "gan_opt": gan,
        "wavelet_guided": True,
        "wavelet_init": 7,
    }
    g, d = build_losses(train, scale=2, device=torch.device(DEV), has_d=True)
    assert [t.log_key for t in g] == g_keys
    assert [type(t) for t in d] == [gan_d_loss]
    assert d[0].g.view.start_iter == 7
    assert d[0].g.loss_weight == pytest.approx(0.1 if weight is None else 0.0)


# ---- train-mode MetaGAN: backward(inputs=) vs the freeze toggle ---------------------------------


@cuda
def test_metagan_inputs_equals_freeze_toggle():
    """The 5 % peak gate's workload: esrgan x4 G, train-mode MetaGAN, batch 2, 32 -> 128 px,
    bf16 autocast, AdvView on; measured from zero_grad to the end of backward. Not universal:
    D params that require grad make D save its inputs for weight grads, a fixed D-side cost
    (~5.5 MB per 256 px sample). esrgan b8 64 -> 256 px: 1.011; compact b8 64 -> 256 px: 1.123."""
    torch.manual_seed(0)
    net_g = esrgan(scale=4).cuda()
    net_d = metagan().cuda().train()  # SN power iteration, dropout and DropPath active
    g_term = gan_loss(wavelet_init=0).cuda()
    lq, gt = images((2, 3, 32, 32)), images((2, 3, 128, 128), seed=1)
    state_g, state_d = deepcopy(net_g.state_dict()), deepcopy(net_d.state_dict())
    params_g = list(net_g.parameters())

    def run(freeze):
        net_g.load_state_dict(state_g)
        net_d.load_state_dict(state_d)
        net_g.zero_grad(set_to_none=True)
        net_d.zero_grad(set_to_none=True)
        torch.manual_seed(1)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        if freeze:
            net_d.requires_grad_(requires_grad=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sr = net_g(lq)
            loss = g_term.compute(
                Context(sr=sr, gt=gt, lq=lq, it=1, net_d=net_d, net_d_adv=net_d)
            )
        loss.backward(**({} if freeze else {"inputs": params_g}))
        torch.cuda.synchronize()
        peak = (
            torch.cuda.max_memory_allocated() - base
        )  # before the test's own grad copies
        net_d.requires_grad_(requires_grad=True)
        assert all(p.grad is None for p in net_d.parameters())
        return [p.grad.clone() for p in params_g], peak

    run(freeze=True)  # warm-up: cudnn/allocator first-call effects
    grads_t, peak_t = run(freeze=True)
    grads_i, peak_i = run(freeze=False)
    for a, b in zip(grads_t, grads_i, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert peak_i <= 1.05 * peak_t, (peak_i, peak_t)

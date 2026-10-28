"""Stage 3 R3GAN track (docs/refactor/stage3/r3gan/DESIGN.md, neosr_fit.md section 7). CPU."""

import itertools
import math
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from neosr.archs import build_network
from neosr.archs.r3gan_arch import Down, r3gan
from neosr.config import ConfigError, validate_train
from neosr.losses import RESERVED
from neosr.losses import gan_loss as glm
from neosr.losses.gan_loss import gan_loss
from neosr.runtime import Runtime
from tests.test_gan import Forbidden, images
from tests.test_gan import ctx as gan_ctx
from tests.test_model import batches as model_batches
from tests.test_model import build as model_build
from tests.test_model import make_opt as model_make_opt
from tests.test_model import run as model_run
from tests.test_resume import assert_same, continuation
from tests.test_resume import make_opt as resume_make_opt

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
ROWS = {  # row -> (widths, head, GT whose final map is 4x4 | None, params in M)
    "r3gan": ((32, 64, 128, 256, 256), "patch", None, 0.87),
    "r3gan_scalar128": ((32, 64, 128, 256, 256, 256), "scalar", 128, 1.21),
    "r3gan_scalar256": ((32, 64, 128, 256, 256, 256, 256), "scalar", 256, 1.55),
}


def _d(name: str = "r3gan", **kw) -> torch.nn.Module:
    return build_network({"type": name, **kw}, section="network_d")


def _golden(widths, head, gw=16, e=2) -> dict[str, tuple]:
    """Key -> shape of the R3GAN D (neosr_fit.md 2.1/2.6), spelled out."""
    out = {"extraction.weight": (widths[0], 3, 1, 1)}
    for i, c in enumerate(widths):
        p = f"stages.{i}.blocks.0"
        out |= {
            f"{p}.lin1.weight": (e * c, c, 1, 1), f"{p}.act1.bias": (e * c,),
            f"{p}.lin2.weight": (e * c, gw, 3, 3), f"{p}.act2.bias": (e * c,),
            f"{p}.lin3.weight": (c, e * c, 1, 1),
        }  # fmt: skip
        if i < len(widths) - 1 and widths[i + 1] != c:
            out[f"stages.{i}.down.proj.weight"] = (widths[i + 1], c, 1, 1)
    if head == "scalar":
        out["basis.weight"] = (widths[-1], 1, 4, 4)
    return out | {"head.weight": (1, widths[-1], 1, 1)}


def _randomise(net: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """A trained-looking D: non-zero lin3 and biases (at init every block is the identity)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return net


# ---- R1: the discriminator (tests 1-5) ------------------------------------------------------------


@pytest.mark.parametrize("name", list(ROWS))
def test_keys_params_and_shapes(name):
    widths, head, gt, mparams = ROWS[name]
    net = _d(name)
    sd = net.state_dict()
    assert {k: tuple(v.shape) for k, v in sd.items()} == _golden(widths, head)
    # the FIR kernel is not persistent
    assert set(sd) == {n for n, _ in net.named_parameters()}
    n = sum(p.numel() for p in net.parameters())
    assert n == pytest.approx(mparams * 1e6, rel=5e-3)
    size = gt or 256
    out = net(torch.rand(2, 3, size, size))
    assert out.shape == ((2, 1, 1, 1) if gt else (2, 1, 16, 16))
    if gt:
        with pytest.raises(
            ValueError, match=rf"needs 4x4: train with GT {gt} or use r3gan"
        ):
            net(torch.rand(1, 3, gt // 2, gt // 2))
    else:
        assert net(torch.rand(1, 3, 128, 96)).shape == (1, 1, 8, 6)


@pytest.mark.parametrize("kw", [{"head": "pool"}, {"group_width": 48}])
def test_rejects_bad_options(kw):
    with pytest.raises(ValueError, match="r3gan"):
        r3gan(**kw)


@pytest.mark.parametrize("name", list(ROWS))
def test_identity_blocks_and_linear_at_init(name):
    """Fix-up init: every lin3 is zero, so every block is the identity and D is linear."""
    torch.manual_seed(0)
    net, gt = _d(name), ROWS[name][2] or 64
    assert all(
        not p.any() for n, p in net.named_parameters() if n.endswith("lin3.weight")
    )
    x, y = torch.rand(2, 3, gt, gt), torch.rand(2, 3, gt, gt)
    with torch.no_grad():
        mixed, lin = net(0.7 * x - 1.3 * y), 0.7 * net(x) - 1.3 * net(y)
    torch.testing.assert_close(mixed, lin, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(("blocks", "head"), [(1, "scalar"), (2, "patch")])
def test_init_is_the_reference_fixup(monkeypatch, blocks, head):
    """Every weight is drawn as normal(0, gain / sqrt(fan_in)) with fan_in of its (grouped)
    kernel: block lin1/lin2 gain sqrt(2/1.04) * (blocks * stages)^(-1/4), lin3 zero, every other
    layer gain 1; biases start at zero (recorded draw arguments, not sample statistics)."""
    drawn = {}
    normal = torch.nn.init.normal_

    def spy(t, mean, std):
        drawn[id(t)] = (mean, std)
        return normal(t, mean, std)

    monkeypatch.setattr(torch.nn.init, "normal_", spy)
    widths = (16, 32, 32)
    net = r3gan(widths=widths, blocks=blocks, head=head)
    block_gain = (2 / 1.04) ** 0.5 * (blocks * len(widths)) ** -0.25
    for name, w in net.named_parameters():
        if name.endswith("bias"):
            assert not w.any(), name
        elif name.endswith("lin3.weight"):
            assert id(w) not in drawn, name
            assert not w.any(), name
        else:
            gain = block_gain if name.endswith(("lin1.weight", "lin2.weight")) else 1
            fan_in = w.shape[1] * w.shape[2] * w.shape[3]
            assert drawn[id(w)] == (
                0.0,
                pytest.approx(gain / fan_in**0.5, rel=1e-12),
            ), name


def _reference(sd: dict, widths, gw: int, head: str, x: torch.Tensor) -> torch.Tensor:
    """The R3GAN D written out with functional ops (not in place): x + lin3(act(lin2(act(lin1 x)))),
    act = leaky_relu(. + bias, 0.2) with gain 1, FIR then projection, basis then head."""

    def act(t, b):
        return F.leaky_relu(t + b[:, None, None], 0.2)

    k = torch.tensor([1.0, 3.0, 3.0, 1.0], dtype=x.dtype)
    x = F.conv2d(x, sd["extraction.weight"])
    for i, c in enumerate(widths):
        b = f"stages.{i}.blocks.0"
        h = act(F.conv2d(x, sd[f"{b}.lin1.weight"]), sd[f"{b}.act1.bias"])
        h = F.conv2d(h, sd[f"{b}.lin2.weight"], padding=1, groups=2 * c // gw)
        x = x + F.conv2d(act(h, sd[f"{b}.act2.bias"]), sd[f"{b}.lin3.weight"])
        if i < len(widths) - 1:
            fir = (k[:, None] * k / 64).expand(c, 1, 4, 4)
            x = F.conv2d(x, fir, stride=2, padding=1, groups=c)
            if (w := sd.get(f"stages.{i}.down.proj.weight")) is not None:
                x = F.conv2d(x, w)
    if head == "scalar":
        x = F.conv2d(x, sd["basis.weight"], groups=widths[-1])
    return F.conv2d(x, sd["head.weight"])


@pytest.mark.parametrize(("head", "size"), [("patch", 24), ("scalar", 16)])
def test_trained_forward_equals_the_functional_reference(head, size):
    """Non-zero lin3 and biases (both activation branches active): the module computes the
    reference's block order, activation slope, gain 1 and FIR."""
    torch.manual_seed(0)
    widths = (8, 16, 16)
    net = _randomise(r3gan(widths=widths, group_width=4, head=head))
    x = torch.rand(2, 3, size, size)
    with torch.no_grad():
        want = _reference(net.state_dict(), widths, 4, head, x)
        torch.testing.assert_close(net(x), want, rtol=1e-5, atol=1e-5)
    h = net.stages[0].blocks[0].lin1(net.extraction(x))
    pre = h + net.stages[0].blocks[0].act1.bias[:, None, None]
    assert (pre > 0).any()
    assert (pre < 0).any()


def test_fir_is_the_reference_lowpass():
    """Grouped conv == the reference's reshape-to-(B*C,1,H,W) conv == the explicit stride-2 window
    sum of [1,3,3,1] x [1,3,3,1] / 64 with padding 1; channels_last gives the same values."""
    torch.manual_seed(0)
    down, x = Down(5, 5), torch.rand(2, 5, 12, 10)
    out = down(x)
    k = torch.tensor([1.0, 3.0, 3.0, 1.0])
    ref = F.conv2d(
        x.reshape(10, 1, 12, 10), (k[:, None] * k / 64)[None, None], stride=2, padding=1
    ).reshape(2, 5, 6, 5)
    xp = F.pad(x, (1, 1, 1, 1))
    explicit = sum(
        k[a] * k[b] / 64 * xp[..., a : a + 12 : 2, b : b + 10 : 2]
        for a in range(4)
        for b in range(4)
    )
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(out, explicit)
    cl = down.to(memory_format=torch.channels_last)
    torch.testing.assert_close(cl(x.to(memory_format=torch.channels_last)), out)


@pytest.mark.parametrize(("head", "size"), [("patch", 12), ("scalar", 8)])
def test_double_backward(head, size):
    """gradgradcheck (fp64) through grouped convs, the FIR and the in-place biased activation."""
    torch.manual_seed(0)
    net = _randomise(r3gan(widths=(4, 8), group_width=2, head=head)).double()
    x = torch.rand(1, 3, size, size, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(net, (x,), eps=1e-6, atol=1e-4)


def test_penalty_parameter_gradients_match_finite_differences():
    """The R1-style penalty ||grad_x sum_p eps_p D_p(x)||^2 differentiated w.r.t. D's weights and
    biases (the double backward a D update runs) agrees with finite differences (fp64)."""
    torch.manual_seed(0)
    net = _randomise(r3gan(widths=(4, 8), group_width=2)).double()
    x = torch.rand(2, 3, 8, 8, dtype=torch.float64)
    eps = torch.randint(0, 2, (2, 1, 4, 4), dtype=torch.float64) * 2 - 1
    names = [
        "extraction.weight",
        "stages.0.blocks.0.lin1.weight",
        "stages.0.blocks.0.act1.bias",
        "stages.0.blocks.0.lin2.weight",
        "stages.1.blocks.0.lin3.weight",
        "head.weight",
    ]
    params = dict(net.named_parameters())

    def penalty(*ws):
        xi = x.clone().requires_grad_()
        out = torch.func.functional_call(
            net, params | dict(zip(names, ws, strict=True)), (xi,)
        )
        (gx,) = torch.autograd.grad((out * eps).sum(), xi, create_graph=True)
        return gx.square().sum((1, 2, 3)).mean()

    ws = tuple(params[n].detach().clone().requires_grad_() for n in names)
    assert penalty(*ws) > 0
    assert torch.autograd.gradcheck(penalty, ws, eps=1e-6, atol=1e-5)


def test_activations_stay_bf16_under_autocast():
    """The bias follows the activation dtype: a fp32 bias would promote every block to fp32."""
    net, seen = _randomise(r3gan()), []
    net.stages[0].blocks[0].act1.register_forward_hook(
        lambda _m, _i, o: seen.append(o.dtype)
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(torch.rand(1, 3, 64, 64))
    assert seen == [torch.bfloat16]
    assert out.dtype == torch.bfloat16


def test_compiled_forward_equals_eager():
    torch.manual_seed(0)
    net = _randomise(r3gan(widths=(8, 16, 16))).eval()
    x = torch.rand(2, 3, 32, 32)
    torch.testing.assert_close(
        torch.compile(net, backend="aot_eager")(x), net(x), rtol=1e-5, atol=1e-6
    )


def test_regularised_update_uses_the_eager_d():
    """A compiled graph has no double backward: the R1/R2 update runs the module under compile."""
    torch.manual_seed(0)
    net = _randomise(r3gan(widths=(8, 16, 16)))
    _, d_term = gan_loss(
        gan_type="rpgan", r1_gamma=1.0, r2_gamma=1.0, reg_every=1
    ).expand()
    sr, gt = (
        images((2, 3, 32, 32), dev="cpu"),
        images((2, 3, 32, 32), dev="cpu", seed=1),
    )
    compiled = torch.compile(net, backend="aot_eager")
    d_term.compute(gan_ctx(sr, gt, net_d=compiled, net_d_adv=Forbidden())).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters())


# ---- R2: the loss path (tests 6-18) --------------------------------------------------------------


def _d_term(start_iter: int = 0, **kw):
    loss = gan_loss(gan_type="rpgan", **kw)
    loss.schedule(start_iter=start_iter)
    return loss.expand()[1]


def _ctx(net, it: int = 1, shape=(2, 3, 32, 32), sr=None):
    gt = images(shape, dev="cpu", seed=1)
    sr = images(shape, dev="cpu") if sr is None else sr
    return gan_ctx(sr, gt, net_d=net, net_d_adv=Forbidden(), it=it)


def _tiny(head: str = "patch") -> nn.Module:
    torch.manual_seed(0)
    return _randomise(r3gan(widths=(4, 8), group_width=2, head=head))


def _trained(net: nn.Module, seed: int = 0) -> nn.Module:
    """Init-scale weights with live residual branches: lin3 drawn like lin1, small biases."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in net.named_parameters():
            if n.endswith("lin3.weight"):
                p.copy_(torch.randn(p.shape, generator=g) / p[0].numel() ** 0.5)
            elif n.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    return net


PEN_KEYS = {"r1", "r2", "l_d_r1", "l_d_r2", "gamma_r1", "gamma_r2", "l_d_reg"}


def test_rpgan_pairs_each_fake_with_its_own_real_per_position():
    """softplus(D_p(fake_i) - D_p(real_i)) averaged over i and p: shuffling the fakes across the
    batch or across positions changes the loss (a relativistic average would not)."""
    torch.manual_seed(0)
    net, d_term = nn.Conv2d(3, 1, 1), _d_term()
    c = _ctx(net)
    value = d_term.compute(c).detach()
    with torch.no_grad():
        want = F.softplus(net(c.sr) - net(c.gt)).mean()
    torch.testing.assert_close(value, want)
    for sr in (c.sr.flip(0), c.sr.roll(1, dims=-1)):
        other = d_term.compute(_ctx(net, sr=sr)).detach()
        assert not torch.isclose(other, value, rtol=1e-4, atol=0)


@pytest.mark.parametrize("seed", range(3))
def test_disjoint_receptive_fields_give_the_exact_penalty_for_any_draw(seed):
    """Linear per-position critics on disjoint 4x4 patches: ||sum_p eps_p w_p||^2 / P = ||w||^2."""
    torch.manual_seed(seed)
    net = nn.Conv2d(3, 1, 4, stride=4, bias=False)
    c = _ctx(net)
    _d_term(r1_gamma=1.0, r2_gamma=1.0, reg_every=1).compute(c)
    for side in ("r1", "r2"):
        torch.testing.assert_close(c.logs[side], net.weight.detach().square().sum())


def _exact_penalty(net, real, fake, gammas, k=1):
    """0.5 k gamma mean_{i,p} ||grad_x D_p(x_i)||^2 per side: one backward per position."""
    pen = 0
    for gm, side in zip(gammas, (real, fake), strict=True):
        x = side.clone().requires_grad_()
        out = net(x).reshape(len(x), -1)
        per = [
            torch.autograd
            .grad(out[:, p].sum(), x, create_graph=True)[0]
            .square()
            .flatten(1)
            .sum(1)
            for p in range(out.shape[1])
        ]
        pen = pen + 0.5 * k * gm * torch.stack(per).mean()
    return pen


@pytest.mark.parametrize("positions", [2, 3])
def test_estimator_averages_to_the_per_position_penalty(monkeypatch, positions):
    """Overlapping receptive fields (a randomised r3gan with a 1 x P map): the mean over all 2^P
    sign patterns equals the exact per-position penalty in value and in its D-parameter gradients,
    while single patterns differ (cross terms are real)."""
    net, gammas = _tiny(), (1.0, 2.0)
    ref = deepcopy(net)
    d_term = _d_term(r1_gamma=gammas[0], r2_gamma=gammas[1], reg_every=1)
    d_term.adv_loss = lambda real, fake, _ctx: 0 * (real.sum() + fake.sum())
    shape = (2, 3, 2, 2 * positions)
    patterns = list(itertools.product((-1.0, 1.0), repeat=positions))
    vals = []
    for (
        pat
    ) in patterns:  # one pattern for every sample and side: each norm sees only its row
        monkeypatch.setattr(
            glm, "signs", lambda o, pat=pat: torch.tensor(pat).expand_as(o)
        )
        c = _ctx(net, shape=shape)
        d_term.compute(c).backward()
        assert c.logs["r1"].numel() == 1
        vals.append(c.logs["l_d_reg"])
    c = _ctx(ref, shape=shape)
    exact = _exact_penalty(ref, c.gt, c.sr, gammas)
    (exact * len(patterns)).backward()  # the patterns' penalty-gradient sum
    torch.testing.assert_close(
        torch.stack(vals).mean(), exact.detach(), rtol=1e-5, atol=0
    )
    assert max(abs(v - exact.detach()) for v in vals) > 1e-3 * exact
    for p, q in zip(net.parameters(), ref.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)


def _old_d_value(net, gt, sr, gammas, k):
    """gan_d_loss on a regularised update before stage 3 R2 (mean-logit score, fp32 eager), with
    the reshape fix: flatten(1) raised IndexError on a (B,) output."""
    xs = [
        x.detach().float().requires_grad_(gm > 0)
        for x, gm in zip((gt, sr), gammas, strict=True)
    ]
    real, fake = net(torch.cat(xs)).chunk(2)
    on = [
        (o, x, gm) for o, x, gm in zip((real, fake), xs, gammas, strict=True) if gm > 0
    ]
    score = sum(o.reshape(len(o), -1).mean(1).sum() for o, _, _ in on)
    grads = torch.autograd.grad(score, [x for _, x, _ in on], create_graph=True)
    pen = sum(
        gm * gx.square().flatten(1).sum(1).mean()
        for (_, _, gm), gx in zip(on, grads, strict=True)
    )
    return F.softplus(fake.float() - real.float()).mean() + 0.5 * k * pen


class Pooled(nn.Module):
    """A scalar critic with a 1-D (B,) output (the owner's traiNNer D returned that shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 3, padding=1)

    def forward(self, x):
        return F.mish(self.conv(x)).mean((1, 2, 3))


@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("make", [lambda: _tiny("scalar"), Pooled])
def test_one_logit_per_sample_is_bitwise_the_old_penalty_and_draws_nothing(make, k):
    """P = 1 (the scalar head; also a (B,) output, which crashed flatten(1)): today's penalty,
    bitwise in value and gradients, and the RNG is not touched."""
    torch.manual_seed(0)
    net = make()
    ref = deepcopy(net)
    c = _ctx(net, it=8, shape=(2, 3, 8, 8))
    rng = torch.get_rng_state()
    c.it = 2 * k
    value = _d_term(r1_gamma=2.0, r2_gamma=0.3, reg_every=k).compute(c)
    assert torch.equal(torch.get_rng_state(), rng)
    value.backward()
    want = _old_d_value(ref, c.gt, c.sr, (2.0, 0.3), k)
    want.backward()
    assert torch.equal(value, want)
    for p, q in zip(net.parameters(), ref.parameters(), strict=True):
        assert torch.equal(p.grad, q.grad)


def test_zero_effective_gamma_is_the_plain_update():
    """Gammas 0, or a cosine decayed to 0: the plain objective bitwise, no penalty logs, no draw."""
    net, plain = _tiny(), _d_term()
    decayed = _d_term(
        start_iter=2,
        r1_gamma=1.0,
        r2_gamma=1.0,
        reg_every=1,
        gamma_decay=0.0,
        gamma_iters=4,
    )
    assert decayed.g.gamma_scale(6) > 0
    for term, it in ((plain, 7), (decayed, 7), (decayed, 50)):
        c, rng = _ctx(net, it=it), torch.get_rng_state()
        value = term.compute(c)
        assert torch.equal(torch.get_rng_state(), rng)
        assert not PEN_KEYS & set(c.logs)
        assert torch.equal(value, plain.compute(_ctx(net, it=it)))


def test_lazy_penalty_is_k_times_the_every_step_penalty():
    net, plain = _tiny(), _d_term()
    logs = []
    for k in (1, 4):
        torch.manual_seed(0)  # the same signs
        c = _ctx(net, it=8)
        _d_term(r1_gamma=1.0, r2_gamma=0.5, reg_every=k).compute(c)
        logs.append(c.logs)
    for key in ("l_d_r1", "l_d_r2", "l_d_reg"):
        assert torch.equal(logs[1][key], 4 * logs[0][key]), key
    for key in ("r1", "r2", "gamma_r1", "gamma_r2"):
        assert torch.equal(logs[1][key], logs[0][key]), key
    lazy = _d_term(r1_gamma=1.0, r2_gamma=0.5, reg_every=4)
    for it in (5, 7):  # not regularised: the plain objective, bitwise
        c = _ctx(net, it=it)
        assert torch.equal(lazy.compute(c), plain.compute(_ctx(net, it=it)))
        assert not PEN_KEYS & set(c.logs)


def test_gamma_schedule_is_a_pure_cosine_of_the_update():
    """1 at the first GAN update (start_iter + 1), (1 + d)/2 halfway, d from N on; the D term
    logs gamma times that factor."""
    loss = gan_loss(
        gan_type="rpgan", r1_gamma=4.0, reg_every=1, gamma_decay=0.1, gamma_iters=100
    )
    loss.schedule(start_iter=10)
    m = loss.gamma_scale
    exact = pytest.approx  # with rel=0, abs=0 below: the endpoints are exact
    assert m(1) == m(11) == exact(1.0, rel=0, abs=0)
    assert m(61) == pytest.approx(0.55, rel=1e-12)
    assert m(111) == m(10**6) == exact(0.1, rel=0, abs=0)
    seq = [m(i) for i in range(1, 140)]
    assert all(a >= b for a, b in itertools.pairwise(seq))
    assert [m(i) for i in (500, 30, 61)] == [m(61), m(30), m(500)][::-1]
    const = gan_loss(r1_gamma=4.0).gamma_scale
    assert all(const(i) == exact(1.0, rel=0, abs=0) for i in (1, 50, 10**6))
    c = _ctx(_tiny(), it=61)
    loss.expand()[1].compute(c)
    assert c.logs["gamma_r1"].item() == pytest.approx(4.0 * 0.55, rel=1e-6)
    assert not {"r2", "l_d_r2", "gamma_r2"} & set(c.logs)


@pytest.mark.parametrize(
    "kw",
    [
        {"r1_gamma": float("nan")},
        {"r2_gamma": float("inf")},
        {"r1_gamma": True},
        {"r1_gamma": 1.0, "gamma_decay": 1.5},
        {"r1_gamma": 1.0, "gamma_decay": -0.1, "gamma_iters": 10},
        {"r1_gamma": 1.0, "gamma_iters": -1},
        {"r1_gamma": 1.0, "gamma_iters": 10.0},
        {"r1_gamma": 1.0, "gamma_iters": True},
        {"reg_every": 2.0},
        {"reg_every": True},
        {"gamma_iters": 100},  # a schedule without a penalty
        {"r1_gamma": 1.0, "gamma_decay": 0.1},  # a decay without a length
        {"penalty_fp32": 1},
    ],
)
def test_rejects_bad_penalty_options(kw):
    with pytest.raises(ValueError, match="gan_loss"):
        gan_loss(**kw)


def _penalty_run(net, autocast=None, **kw):
    """The D term's penalty alone (adversarial part zeroed) and its D-parameter gradients."""
    d_term = _d_term(r1_gamma=1.0, r2_gamma=1.0, reg_every=1, **kw)
    seen = []
    d_term.adv_loss = lambda real, fake, _ctx: (
        seen.append(real.dtype) or 0 * (real.float().sum() + fake.float().sum())
    )
    net.zero_grad()
    c = _ctx(net, shape=(2, 3, 64, 64))
    torch.manual_seed(0)  # identical signs in every precision
    with torch.autocast("cpu", dtype=autocast, enabled=autocast is not None):
        value = d_term.compute(c)
    value.backward()
    grad = torch.cat([p.grad.flatten() for p in net.parameters()])
    return c.logs, value.detach(), grad, seen[0]


def test_bf16_penalty_matches_fp32_with_identical_signs():
    """A bf16 phase runs the penalty in bf16 (fp32 leaves and norms): within 3 % of fp32 and the
    penalty's D-parameter gradient keeps its direction; penalty_fp32 and fp16 phases run fp32."""
    net = _trained(r3gan())
    logs32, v32, g32, _ = _penalty_run(net)
    logs16, _, g16, dtype16 = _penalty_run(net, torch.bfloat16)
    assert dtype16 == torch.bfloat16  # the pass really ran in bf16
    for key in ("r1", "r2"):
        assert logs16[key].dtype == torch.float32
        torch.testing.assert_close(logs16[key], logs32[key], rtol=0.03, atol=0)
    assert F.cosine_similarity(g16, g32, dim=0) > 0.999
    assert (g16.norm() / g32.norm()).item() == pytest.approx(1, abs=0.03)
    for kw, dtype in (({"penalty_fp32": True}, torch.bfloat16), ({}, torch.float16)):
        _, v, g, seen = _penalty_run(net, dtype, **kw)
        assert seen == torch.float32
        assert torch.equal(v, v32)
        assert torch.equal(g, g32)


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize(("name", "size"), [("r3gan", 64), ("r3gan_scalar128", 128)])
def test_r1_r2_double_backward_through_r3gan(name, size, dev):
    """Under the trainer's bf16 + channels_last on CUDA (fp32 on CPU), with the MATH SDPA kernel."""
    torch.manual_seed(0)
    mf = torch.channels_last if dev == "cuda" else torch.contiguous_format
    net = _trained(_d(name)).to(dev, memory_format=mf)
    d_term = gan_loss(gan_type="rpgan", r1_gamma=1.0, r2_gamma=1.0, reg_every=1)
    d_term = d_term.to(dev).expand()[1]
    shape = (2, 3, size, size)
    sr, gt = (
        images(shape, dev=dev, seed=s).contiguous(memory_format=mf) for s in (0, 1)
    )
    c = gan_ctx(sr, gt, net_d=net, net_d_adv=Forbidden())
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        value = d_term.compute(c)
    value.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters())
    assert all(c.logs[k].isfinite() and c.logs[k] > 0 for k in ("r1", "r2", "l_d_reg"))


def test_dunet_has_no_double_backward_and_is_refused_by_name():
    torch.manual_seed(0)
    net = build_network({"type": "dunet", "dim": 4}, section="network_d")
    c = _ctx(net)
    with pytest.raises(RuntimeError, match="grid_sampler_2d_backward"):
        _d_term(r1_gamma=1.0, reg_every=1).compute(c).backward()
    cfg = {
        "scale": 4,
        "datasets": {"train": {"patch_size": 16, "batch_size": 2}},
        "logger": {"save_checkpoint_freq": 100},
        "network_d": {"type": "dunet"},
        "train": {
            "optim_g": {},
            "optim_d": {},
            "gan_opt": {"type": "gan_loss", "r1_gamma": 1.0},
        },
    }
    with pytest.raises(ConfigError, match="dunet"):
        validate_train(cfg)
    other = deepcopy(cfg)  # any slot may hold the gan_loss
    other["train"]["gan_opt"]["r1_gamma"] = 0
    other["train"]["adv_opt"] = {"type": "gan_loss", "r2_gamma": 0.5}
    with pytest.raises(ConfigError, match=r"train\.adv_opt\.r1_gamma/r2_gamma.*dunet"):
        validate_train(other)
    cfg["network_d"]["type"] = "r3gan"
    validate_train(cfg)
    cfg["network_d"]["type"], cfg["train"]["gan_opt"]["r1_gamma"] = "dunet", 0
    validate_train(cfg)


def test_batch_coupled_d_with_a_penalty_is_refused_at_the_first_call():
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Conv2d(4, 1, 1)
    )
    with pytest.raises(ConfigError, match="batch statistics"):
        _d_term(r1_gamma=1.0).compute(_ctx(net))  # update 1 is not even regularised
    assert (
        _d_term().compute(_ctx(net)).isfinite()
    )  # no penalty: two forwards, as before


def test_penalty_logs():
    net = _tiny()
    d_term = _d_term(r1_gamma=1.0, r2_gamma=0.5, reg_every=2)
    c = _ctx(net, it=1)
    d_term.compute(c)
    assert not PEN_KEYS & set(c.logs)
    c = _ctx(net, it=2)
    d_term.compute(c)
    assert PEN_KEYS <= set(c.logs) <= RESERVED
    lg = c.logs
    torch.testing.assert_close(lg["l_d_r1"] + lg["l_d_r2"], lg["l_d_reg"])
    assert torch.equal(lg["l_d_r1"], 0.5 * 2 * 1.0 * lg["r1"])
    assert (lg["gamma_r1"].item(), lg["gamma_r2"].item()) == (1.0, 0.5)
    assert all(lg[k] > 0 for k in ("r1", "r2"))


def _r3gan_opt(tmp):
    """The resume harness's sebica_mini G (BatchNorm, accumulate 2) with the R3GAN recipe's D
    side: r3gan patch head, Adam (0, 0.99), rpgan, every-step R1/R2 on a cosine across the resume."""
    opt = resume_make_opt(tmp, "c-no-d")
    opt["network_d"] = {"type": "r3gan", "widths": [8, 16, 16], "group_width": 8}
    opt["train"] |= {
        "optim_d": {"type": "adam", "lr": 1e-3, "betas": [0.0, 0.99], "weight_decay": 0},
        "gan_opt": {
            "type": "gan_loss", "gan_type": "rpgan", "loss_weight": 0.3, "start_iter": 1,
            "r1_gamma": 1.0, "r2_gamma": 1.0, "reg_every": 1, "gamma_decay": 0.1,
            "gamma_iters": 4,
        },
    }  # fmt: skip
    return opt


def test_resume_is_exact_across_the_schedule_and_the_draws(tmp_path):
    """6 updates straight vs 3 + resume + 3 (accumulate 2: a draw per micro-batch, the cosine
    crosses the resume): bitwise weights, optimizer state and logs (the CPU RNG holds the signs)."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(mode=True)
    try:
        a, _, sa, sr = continuation(tmp_path, _r3gan_opt, acc=2)
    finally:
        torch.use_deterministic_algorithms(mode=prev)
    assert type(a.net_d).__name__ == "r3gan"
    assert tuple(a.d.opt.param_groups[0]["betas"]) == (0.0, 0.99)
    assert {"r1", "l_d_r2", "gamma_r1"} <= set(a.get_current_log())
    assert_same(sa, sr)


def _ddp_worker(rank: int, tmp: str, out: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp}/pg", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )  # fmt: skip
    try:
        opt = model_make_opt(Path(tmp) / "run", d=True, accumulate=2)
        opt["network_d"] = {"type": "r3gan", "widths": [8, 16, 16], "group_width": 8}
        opt["train"]["optim_d"] = {"type": "adam", "lr": 1e-3, "betas": [0.0, 0.99]}
        opt["train"]["gan_opt"] = {
            "type": "gan_loss", "loss_weight": 0.3, "gan_type": "rpgan", "r1_gamma": 1.0,
            "r2_gamma": 1.0, "reg_every": 1, "gamma_decay": 0.1, "gamma_iters": 2,
        }  # fmt: skip
        m = model_build(
            opt,
            rt=Runtime(torch.device("cpu"), rank, 2, 0, distributed=True),
            seed=rank,
        )
        model_run(
            m, model_batches(6, seed=rank)
        )  # rank-specific data and signs, 3 updates
        log = m.get_current_log()
        sd = {k: v.detach().clone() for k, v in m.net_d.state_dict().items()}
        torch.save({"log": log, "d": sd, "update": m.update}, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_ddp_regularised_r3gan_keeps_ranks_in_sync(tmp_path):
    """The per-position penalty through the DDP-wrapped r3gan: rank-specific signs, but the reducer
    averages D's penalty gradients, so both ranks hold the same D."""
    out = str(tmp_path / "res")
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path), out), nprocs=2)
    r0, r1 = (torch.load(f"{out}.{r}", weights_only=False) for r in (0, 1))
    assert r0["update"] == r1["update"] == 3
    assert {"r1", "r2", "l_d_reg", "out_d_margin"} <= set(r0["log"])
    assert all(math.isfinite(v) for v in r0["log"].values())
    assert r0["log"] == r1["log"]
    for k, v in r0["d"].items():
        assert torch.equal(v, r1["d"][k]), k

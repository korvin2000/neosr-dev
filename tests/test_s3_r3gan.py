"""Stage 3 R3GAN track (docs/refactor/stage3/r3gan/DESIGN.md, neosr_fit.md section 7). CPU."""

import pytest
import torch
from torch.nn import functional as F

from neosr.archs import build_network
from neosr.archs.r3gan_arch import Down, r3gan
from neosr.losses.gan_loss import gan_loss
from tests.test_gan import Forbidden, images
from tests.test_gan import ctx as gan_ctx

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

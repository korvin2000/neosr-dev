"""Stage 3 A6: RealPLKSR opt-in pre-LayerNorm and ICNR init; PLK eval + grad. CPU."""

from functools import partial

import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from neosr.archs import build_network, plksr_arch
from neosr.archs.arch_util import LayerNorm2d, icnr_


def _plk(scale=4, **kw):
    torch.manual_seed(0)
    return build_network({"type": "realplksr", "n_blocks": 4, **kw}, scale=scale).eval()


def test_layer_norm_keys_follow_spandrel_and_gn_stays_the_default():
    gn, ln = _plk().state_dict(), _plk(layer_norm=True).state_dict()
    assert gn.keys() == _plk(layer_norm=False).state_dict().keys()
    assert "feats.1.norm.weight" in gn
    assert not [k for k in gn if "layer_norm" in k]
    assert {"feats.1.layer_norm.weight", "feats.1.layer_norm.bias"} <= ln.keys()
    assert not [k for k in ln if ".norm." in k]  # the post-branch GroupNorm is gone
    assert ln.keys() - gn.keys() == {
        f"feats.{i}.layer_norm.{p}" for i in range(1, 5) for p in ("weight", "bias")
    }
    assert _plk(layer_norm=True).feats[1].layer_norm.eps == 1e-6


def test_gn_default_keys_are_the_old_ones():
    """12 tensors per block, as before the option existed: every old net_*.pth loads strictly."""
    per_block = {
        f"{m}.{p}"
        for m in (
            "channel_mixer.0",
            "channel_mixer.2",
            "lk.conv",
            "attn.f.0",
            "refine",
            "norm",
        )
        for p in ("weight", "bias")
    }
    keys = _plk().state_dict().keys()
    assert {k.split(".", 2)[2] for k in keys if k.startswith("feats.1.")} == per_block
    assert len(keys) == 4 * 12 + 4  # + head and tail convs


def test_pre_layer_norm_starts_with_a_small_output():
    x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    with torch.random.fork_rng():
        torch.manual_seed(0)
        net = build_network({"type": "realplksr", "layer_norm": True}, scale=4).eval()
    with torch.no_grad():
        ln = net(x)
    assert ln.std().item() < 0.5  # the 28-block GroupNorm default starts above 2 (A-07)


def test_layer_norm_block_trains_and_keeps_channels_last():
    net = (
        _plk(2, layer_norm=True, icnr=True)
        .train()
        .to(memory_format=torch.channels_last)
    )
    x = torch.rand(2, 3, 12, 10).contiguous(memory_format=torch.channels_last)
    block = net.feats[1]
    h = net.feats[0](x)
    assert block.layer_norm(h).is_contiguous(memory_format=torch.channels_last)
    y = net(x.requires_grad_())
    y.mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters())
    # residual from the raw input: a zero branch (zeroed refine) leaves the block an identity
    with torch.no_grad():
        block.refine.weight.zero_()
        block.refine.bias.zero_()
        torch.testing.assert_close(block.eval()(h), h)


def test_icnr_is_ignored_with_dysample():
    a = _plk(2, dysample=True, icnr=True).state_dict()
    b = _plk(2, dysample=True).state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)  # same seed, no ICNR applied


def test_layer_norm2d_is_fp16_safe_and_layout_free():
    ln = LayerNorm2d(8, eps=1e-6)
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.uniform_(-0.1, 0.1)
    g = torch.Generator().manual_seed(2)
    x = (
        torch.randn(2, 8, 5, 7, generator=g) * 1000
    )  # |x - mean| far above 256: a naive fp16 var overflows
    ref = F.layer_norm(x.permute(0, 2, 3, 1), (8,), ln.weight, ln.bias, 1e-6).permute(
        0, 3, 1, 2
    )
    cl = x.contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(ln(cl), ref)
    naive = x.half() - x.half().mean(1, keepdim=True)
    assert not naive.pow(2).isfinite().all()
    torch.testing.assert_close(ln.half()(x.half()).float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("scale", [2, 3])
def test_icnr_starts_every_sub_pixel_phase_equal(scale):
    net = _plk(scale, icnr=True)
    assert net.state_dict().keys() == _plk(scale).state_dict().keys()
    x = torch.rand(1, 3, 6, 5, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        y = net(x)
    phases = [y[..., i::scale, j::scale] for i in range(scale) for j in range(scale)]
    for p in phases[1:]:
        torch.testing.assert_close(p, phases[0], rtol=0, atol=1e-6)
    with torch.no_grad():
        z = _plk(scale)(x)
    assert (z[..., ::scale, ::scale] - z[..., 1::scale, 1::scale]).abs().max() > 1e-4


def test_icnr_helper_layout():
    conv = torch.nn.Conv2d(4, 3 * 4, 3)
    icnr_(conv, 2)
    w, b = conv.weight, conv.bias
    for c in range(3):
        assert all(torch.equal(w[c * 4 + k], w[c * 4]) for k in range(4))
        assert torch.all(b[c * 4 : c * 4 + 4] == b[c * 4])


# ---------------------------------------------------------------- PLK in-place fast path vs autograd
@pytest.mark.parametrize("layer_norm", [False, True])
def test_eval_with_grad_backprops_and_no_grad_stays_in_place(layer_norm):
    """An eval-mode net with grad enabled used to fail backward: the PLK fast path wrote into
    the conv's saved input. It now takes the split/cat path; no_grad keeps the in-place one."""
    net = _plk(2, layer_norm=layer_norm)
    x = torch.rand(1, 3, 12, 10, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        ref = net(x)
    y = net(x.clone().requires_grad_())
    y.mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters())
    torch.testing.assert_close(y.detach(), ref, rtol=0, atol=0)
    lk, inplace = net.feats[1].lk, []
    hook = lk.register_forward_hook(lambda _, i, o: inplace.append(o is i[0]))
    with torch.no_grad():
        net(x.clone().requires_grad_())  # a tracked image: the LK input is untracked
    with torch.inference_mode():
        net(x.clone())
    hook.remove()
    assert inplace == [True, True]  # inference keeps the in-place fast path
    torch.jit.trace(lk, torch.rand(1, 64, 6, 5))  # no_grad checker: same graph


def test_plksr_lk_layers_backprop_in_eval_with_grad():
    torch.manual_seed(0)
    kw = {"use_max_kernel": True, "with_idt": True}
    sparse = partial(plksr_arch.SparsePLKConv2d, 8, 9, (5, 3), (1, 2), **kw)
    layers = [
        plksr_arch.PLKConv2d(8, 7, with_idt=True),
        plksr_arch.PLKConv2d(8, 7, with_idt=False),
        plksr_arch.RectSparsePLKConv2d(8, 9),
        sparse(),
    ]
    for m in (plksr_arch.PLKConv2d(8, 7, with_idt=True), sparse()):
        m.convert()  # deploy form: identity folded into one conv
        layers.append(m)
    x = torch.rand(1, 12, 9, 11)

    def grads(m, fn):
        m.zero_grad()
        xg = x.clone().requires_grad_()
        fn(xg.clone()).square().mean().backward()  # non-leaf input, as inside a block
        return [xg.grad, *(p.grad.clone() for p in m.parameters())]

    for m in layers:
        m.eval()
        with torch.no_grad():
            ref = m(x.clone())
        eager = grads(m, m)
        assert all(g.isfinite().all() for g in eager)
        torch.testing.assert_close(m(x.clone()).detach(), ref, rtol=0, atol=0)
        # the reentrant checkpoint's no_grad pass must not overwrite the input it recomputes from
        ckpt = grads(m, lambda t, m=m: checkpoint(m, t, use_reentrant=True))
        assert all(map(torch.equal, eager, ckpt))

"""Group A1 arch checks: FlexNet, SPAN, SPAN+ (forward/backward, reparam lifecycle, attention parity).

The full-size forward/backward sweep needs CUDA; the small lifecycle/parity checks run on CPU.
"""

import pytest
import torch
from torch import nn

from neosr.archs import flexnet_arch, span_arch, spanplus_arch

pytestmark = pytest.mark.usefixtures("_fp32_exact")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VARIANTS = {
    "span": (span_arch.span, {"upscale": 4}),
    "span_fast": (span_arch.span_fast, {"upscale": 4}),
    "spanplus": (spanplus_arch.spanplus, {"upscale": 4}),
    "spanplus_sts": (spanplus_arch.spanplus_sts, {"upscale": 4}),
    "spanplus_s": (spanplus_arch.spanplus_s, {"upscale": 4}),
    "spanplus_st": (spanplus_arch.spanplus_st, {"upscale": 4}),
    "flexnet": (flexnet_arch.flexnet, {"scale": 4}),
    "metaflexnet": (flexnet_arch.metaflexnet, {"scale": 4}),
}
SMALL = {"span": {}, "spanplus": {}, "flexnet": {"num_blocks": (1, 1), "dim": 16}}
SHAPES = [(64, 64), (67, 53), (20, 12)]  # aligned, unaligned, smaller than metaflexnet's 64 pad


@pytest.fixture
def _fp32_exact():
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


def _build(name, device="cpu", **kw):
    fn, base = VARIANTS[name]
    torch.manual_seed(0)
    return fn(**base, **kw).to(device)


def _deploy_convs(net):
    return [c for m in net.modules() for c in (getattr(m, "eval_conv", None), getattr(m, "conv5x5_reparam", None)) if c]


def _assert_eval_matches_train(net, x, tol=1e-5):
    with torch.no_grad():
        ref = net.train()(x)
        out = net.eval()(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
    return out


@cuda
@pytest.mark.parametrize("name", VARIANTS)
def test_forward_backward_and_reparam(name):
    net = _build(name, "cuda")
    assert not any(p.requires_grad for c in _deploy_convs(net) for p in c.parameters())
    ids = {k: id(v) for k, v in net.named_parameters()}
    # backward on a small unaligned input (shared GPU: default flexnet at 64x64 needs >1 GB)
    net.train()
    net(torch.rand(1, 3, 27, 21, device="cuda")).square().mean().backward()
    for k, p in net.named_parameters():  # every trainable param is used (DDP-safe), grads finite
        assert (p.grad is not None) == p.requires_grad, k
        assert p.grad is None or torch.isfinite(p.grad).all(), k
    for h, w in SHAPES:
        out = _assert_eval_matches_train(net, torch.rand(1, 3, h, w, device="cuda"), 1e-4)
        assert out.shape == (1, 3, h * 4, w * 4)
    assert ids == {k: id(v) for k, v in net.named_parameters()}  # fusion never replaces params


@pytest.mark.parametrize("name", SMALL)
def test_lifecycle(name):
    net = _build(name, **SMALL[name])
    x = torch.rand(1, 3, 24, 16)
    # train() after eval(): a training step reaches the next eval output
    net.eval()
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1)
    net.train()
    net(x).mean().backward()
    opt.step()
    ref = _assert_eval_matches_train(net, x)
    # weights changed while already in eval mode (schedule-free optimizer.eval() after model.eval(),
    # which writes through p.data and bumps no version counter) are picked up
    net.eval()
    with torch.no_grad():
        for p in net.parameters():  # mild change: random weights blow activations up past fp32 tolerance
            if p.ndim > 1:
                p.data.mul_(1.1)
    with torch.inference_mode():
        out = net(x)
    with torch.no_grad():
        torch.testing.assert_close(out, net.train()(x), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(out, ref)
    # loading into a model that is already in eval mode is honoured
    other = _build(name, **SMALL[name]).eval()
    other.load_state_dict(net.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(other(x), out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", SMALL)
def test_autocast_and_channels_last(name):
    net = _build(name, **SMALL[name])
    x = torch.rand(1, 3, 24, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        net.eval()  # fusion inside autocast must not cache a bf16-rounded kernel
        net(x)
    _assert_eval_matches_train(net, x)
    for m in net.modules():  # eval() left the exact fp32 fused kernel in the deploy conv (export)
        if hasattr(m, "fused_kernel"):
            k = m.fused_kernel()
            assert torch.equal(getattr(m, "eval_conv", getattr(m, "conv5x5_reparam", None)).weight, k[0] if isinstance(k, tuple) else k)
    net = net.to(memory_format=torch.channels_last)
    _assert_eval_matches_train(net, x.to(memory_format=torch.channels_last))


@pytest.mark.parametrize("window_size", [8, 16])
def test_flexnet_sdpa_matches_manual(window_size):
    x = torch.rand(1, 3, 2 * window_size + 3, window_size + 5)
    outs, grads = [], []
    for flash in (True, False):
        net = _build("flexnet", num_blocks=(1, 1), dim=16, window_size=window_size, flash_attn=flash)
        y = net(x)
        y.square().mean().backward()
        outs.append(y)
        grads.append({k: p.grad for k, p in net.named_parameters() if p.grad is not None})
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-4, atol=1e-5)
    assert grads[0].keys() == grads[1].keys()
    for k in grads[0]:
        torch.testing.assert_close(grads[0][k], grads[1][k], rtol=1e-3, atol=1e-6, msg=k)


def test_flexnet_dropout_off_in_eval():
    x = torch.rand(1, 3, 19, 13)
    outs = []
    for flash in (True, False):
        net = _build("flexnet", num_blocks=(1, 1), dim=16, attn_drop=0.5, proj_drop=0.5, flash_attn=flash).eval()
        with torch.no_grad():
            outs.append(net(x))
            torch.testing.assert_close(net(x), outs[-1], rtol=0, atol=0)
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", SMALL)
def test_eval_compiles_without_graph_breaks(name):
    net = _build(name, **SMALL[name]).eval()
    x = torch.rand(1, 3, 19, 13)
    compiled = torch.compile(net, backend="eager", fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), net(x))


def test_spanplus_honours_num_out_ch():
    for up, scale in (("ps", 2), ("conv", 1), ("dys", 2)):
        net = spanplus_arch.spanplus(num_in_ch=3, num_out_ch=1, upscale=scale, upsampler=up)
        assert net(torch.rand(1, 3, 8, 8)).shape == (1, 1, 8 * scale, 8 * scale)


def test_span_sequential_flattens_nested():
    a, b, c = nn.Conv2d(1, 1, 1), nn.ReLU(), nn.Conv2d(1, 1, 1)
    seq = span_arch.sequential(nn.Sequential(a, b), c)
    assert list(seq) == [a, b, c]

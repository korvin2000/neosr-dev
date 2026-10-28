"""Stage-2 lane cnn (archs_cnn.md A-xx, metrics.md MT-01/03/09, archs_mixed.md MX-13/14): CPU checks.

Gradient census of every default ARCHS row of the lane, online re-parameterisation parity (outputs and
parameter gradients against the branch implementations kept here), eval-cache freshness (eval(), EMA
lerp, load_state_dict), DySample fp32 grid and dynamic ONNX export.
"""

import copy
import importlib.util
import io
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from neosr.archs import (
    ARCHS,
    aether_arch,
    build_network,
    cfsr_arch,
    eimn_arch,
    flexnet_arch,
    man_arch,
    metagan_arch,
    moesr_arch,
    plainusr_arch,
    plksr_arch,
    safmn_arch,
    span_arch,
    spanplus_arch,
)
from neosr.archs.arch_util import DySample
from neosr.models.phase import Ema

LANE = {
    *("esrgan", "compact", "span", "spanplus", "realplksr", "plksr", "mosrv2", "moesr"),
    *("aether", "cugan", "rcan", "krgn", "safmn", "eimn", "man", "msdan", "ninasr"),
    *("sebica", "cfsr", "hasn", "plainusr", "ea2fpn", "desrgan", "flexnet", "unet"),
    *("dunet", "patchgan", "metagan", "resgan"),
}
ROWS = sorted(n for n, (stem, _) in ARCHS.items() if stem.removesuffix("_arch") in LANE)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _net(name: str, scale: int = 4, **kw) -> nn.Module:
    cfg = {"type": name, **kw}
    if ARCHS[name][1] is None:
        return build_network(cfg, section="network_d")
    return build_network(cfg, scale=scale)


# ---------------------------------------- frozen parameters, small fixes (A-05, A-10/F12, A-14, A-15)
@pytest.mark.parametrize("scale", [2, 3, 4])
def test_msdan_freezes_only_the_unused_upsampler(scale: int) -> None:
    net = _net("msdan", scale)
    frozen = {n.split(".")[1] for n, p in net.named_parameters() if not p.requires_grad}
    assert frozen == (
        {"BSConv_2", "BSConv1_2"} if scale == 3 else {"BSConv_3", "BSConv1_3"}
    )
    net(torch.rand(1, 3, 8, 8)).mean().backward()
    assert all((p.grad is None) != p.requires_grad for p in net.parameters())


@pytest.mark.parametrize(("name", "kw"), [("krgn", {}), ("rcan", {"norm": True})])
def test_meanshift_is_frozen(name: str, kw: dict) -> None:
    """F12 / A-10: the fixed colour transform is not trained; its keys still load."""
    net = _net(name, **kw)
    shifts = [m for m in net.modules() if type(m).__name__ == "MeanShift"]
    assert len(shifts) == 2
    assert not any(p.requires_grad for m in shifts for p in m.parameters())
    sd = net.state_dict()
    _net(name, **kw).load_state_dict(sd, strict=True)


def test_metagan_attention_dropout_only_in_training() -> None:
    att = metagan_arch.Attention(32, head_dim=16, attn_drop=0.5).eval()
    x = torch.rand(2, 4, 4, 32)
    with torch.no_grad():
        torch.testing.assert_close(att(x), att(x), rtol=0, atol=0)
        assert not torch.equal(att.train()(x), att(x))


def test_plk_convert_keeps_device_and_dtype() -> None:
    """A-14: the deploy helpers build their identity and conv on the kernel's device and dtype."""
    x = torch.rand(1, 12, 9, 11, dtype=torch.float64)
    for conv in (
        plksr_arch.PLKConv2d(8, 7, with_idt=True),
        plksr_arch.SparsePLKConv2d(
            8, 9, (5, 3), (1, 2), use_max_kernel=True, with_idt=True
        ),
    ):
        m = conv.to(torch.float64).eval()
        with torch.no_grad():
            ref = m(x.clone())
            m.convert()
            torch.testing.assert_close(m(x.clone()), ref, rtol=1e-10, atol=1e-10)
        assert m.conv.weight.dtype == torch.float64


# ---------------------------------------------------------------- DySample (A-02 / MT-03 / MX-13)
def _zero_offsets(m: DySample) -> DySample:
    for t in (m.offset.weight, m.offset.bias, m.scope.weight, m.init_pos):
        t.data.zero_()  # offset 0: the grid reaches grid_sample unchanged
    return m


def test_dysample_grid_stays_fp32(monkeypatch) -> None:
    """Under autocast the grid is the fp32 grid (grid_sample promotes x to it); a bf16 net casts
    the finished fp32 grid once. The old code rounded `arange + 0.5` and the normaliser first."""
    H, W, s = 2, 512, 2
    seen = []
    grid_sample = F.grid_sample
    monkeypatch.setattr(
        F, "grid_sample", lambda x, g, **k: seen.append(g) or grid_sample(x, g, **k)
    )
    m = _zero_offsets(DySample(4, 3, scale=s, groups=4))
    x = torch.rand(1, 4, H, W)
    ref = 2 * (torch.arange(W, dtype=torch.float32) + 0.5) / W - 1
    ref = ref.repeat_interleave(s).expand(4, s * H, -1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        m(x)
    m.to(torch.bfloat16)(x.to(torch.bfloat16))
    assert seen[0].dtype == torch.float32
    assert torch.equal(seen[0][..., 0], ref)
    assert seen[1].dtype == torch.bfloat16
    assert torch.equal(seen[1][..., 0], ref.to(torch.bfloat16))


def _old_dysample_forward(m: DySample, x: torch.Tensor) -> torch.Tensor:
    """The pre-fix forward (host-built normaliser, grid cast to x.dtype first)."""
    offset = m.offset(x) * m.scope(x).sigmoid() * 0.5 + m.init_pos
    B, _, H, W = offset.shape
    offset = offset.view(B, 2, -1, H, W)
    coords_h = torch.arange(H, dtype=torch.float32) + 0.5
    coords_w = torch.arange(W, dtype=torch.float32) + 0.5
    coords = (
        torch
        .stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
        .transpose(1, 2)[None, :, None]
        .to(x.dtype)
    )
    normalizer = torch.tensor([W, H], dtype=x.dtype).view(1, 2, 1, 1, 1)
    coords = 2 * (coords + offset) / normalizer - 1
    coords = (
        F
        .pixel_shuffle(coords.reshape(B, -1, H, W), m.scale)
        .view(B, 2, -1, m.scale * H, m.scale * W)
        .permute(0, 2, 3, 4, 1)
        .contiguous()
        .flatten(0, 1)
    )
    out = F.grid_sample(
        x.reshape(B * m.groups, -1, H, W),
        coords,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    ).view(B, -1, m.scale * H, m.scale * W)
    return m.end_conv(out) if m.end_convolution else out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dysample_bit_identical_in_full_precision(dtype) -> None:
    m = DySample(16, 3, scale=2, groups=4).to(dtype)
    nn.init.normal_(m.scope.weight, std=0.5)  # non-trivial offsets
    x = torch.randn(2, 16, 13, 9, dtype=dtype, requires_grad=True)
    out = m(x)
    ref = _old_dysample_forward(m, x)
    assert torch.equal(out, ref)
    g = torch.randn_like(out)
    got = torch.autograd.grad(out, [x, *m.parameters()], g)
    exp = torch.autograd.grad(ref, [x, *m.parameters()], g)
    for a, b in zip(got, exp, strict=True):
        assert torch.equal(a, b)


def _ort_errors(net: nn.Module, trace: tuple, shapes: list) -> list[float]:
    ort = pytest.importorskip("onnxruntime")
    f = io.BytesIO()
    net = net.eval()
    with torch.inference_mode():
        torch.onnx.export(
            net,
            torch.rand(*trace),
            f,
            opset_version=17,
            dynamo=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "b", 2: "h", 3: "w"},
                "output": {0: "b", 2: "H", 3: "W"},
            },
        )
    sess = ort.InferenceSession(f.getvalue(), providers=["CPUExecutionProvider"])
    errs = []
    for shape in shapes:
        x = torch.rand(*shape)
        with torch.inference_mode():
            ref = net(x).numpy()
        errs.append(float(abs(sess.run(None, {"input": x.numpy()})[0] - ref).max()))
    return errs


@pytest.mark.filterwarnings("ignore")
def test_dysample_dynamic_onnx_export() -> None:
    """F07: the normaliser was a trace constant (max error 1.85 at 12x10)."""
    m = DySample(16, 3, scale=2, groups=4)
    nn.init.normal_(m.scope.weight, std=0.5)
    errs = _ort_errors(
        m, (1, 16, 8, 8), [(1, 16, 8, 8), (1, 16, 12, 10), (2, 16, 33, 17)]
    )
    assert max(errs) < 1e-4, errs


def test_dysample_has_no_host_tensor() -> None:
    """MT-03 / MX-13: no torch.tensor(list) per forward (an H2D copy and sync on CUDA)."""
    m = DySample(16, 3, scale=2, groups=4)
    calls = []
    real = torch.tensor
    try:
        torch.tensor = lambda *a, **k: calls.append(a) or real(*a, **k)
        m(torch.rand(1, 16, 5, 7))
    finally:
        torch.tensor = real
    assert not calls


# ---------------------------------------------------------------- channels-first LayerNorm (A-08)
def _ln_manual(m, x):
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    return (
        m.weight[:, None, None] * ((x - u) / torch.sqrt(s + m.eps))
        + m.bias[:, None, None]
    )


def _bf16_errors(m, x, g, fwd, ref=None, ref_only=False):
    """Relative L2 errors of output, x gradient and every parameter gradient of fwd under bf16
    autocast against the fp64 module (upstream gradient g, non-uniform)."""

    def run(mod, f, xx, *, ac):
        xx = xx.detach().requires_grad_()
        ps = [p for p in mod.parameters() if p.requires_grad]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=ac):
            out = f(xx)
        grads = torch.autograd.grad(out, [xx, *ps], g.to(out.dtype))
        return [t.detach().double() for t in (out, *grads)]

    if ref_only:
        md = copy.deepcopy(m).double()
        return run(md, md, x.double(), ac=False)
    got = run(m, fwd, x, ac=True)
    return [float((a - r).norm() / r.norm()) for a, r in zip(got, ref, strict=True)]


LAYER_NORMS = {
    "safmn": lambda c: safmn_arch.LayerNorm(c, data_format="channels_first"),
    "man": lambda c: man_arch.LayerNorm(c, data_format="channels_first"),
    "cfsr": lambda c: cfsr_arch.LayerNorm(c, data_format="channels_first"),
    "eimn": lambda c: eimn_arch.LayerNorm(c, data_format="channels_first"),
    "moesr": moesr_arch.LayerNorm,
}


@pytest.mark.parametrize("fmt", [torch.contiguous_format, torch.channels_last])
@pytest.mark.parametrize("name", LAYER_NORMS)
def test_channels_first_layer_norm(name: str, fmt) -> None:
    """F.layer_norm on the NHWC view: the hand-written formula at fp32 rounding (outputs and
    gradients), x's memory format kept; under bf16 autocast the mean is no longer rounded first."""
    m = LAYER_NORMS[name](16)
    with torch.no_grad():
        m.weight.normal_(1, 0.2)
        m.bias.normal_(0, 0.2)
    x = (torch.randn(2, 16, 7, 9) + 3).to(memory_format=fmt).requires_grad_()
    out, grads = _out_and_grads(m, x)
    ref, ref_grads = _out_and_grads(m, x, lambda t: _ln_manual(m, t))
    assert out.is_contiguous(memory_format=fmt)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    for g, r in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-5)
    # bf16 autocast against fp64: output, x and weight gradients gain an order of magnitude; the
    # bias gradient (exact before: a plain sum) is now at bf16 rounding
    xb = (torch.randn(2, 16, 8, 8) + 16 * torch.randn(2, 1, 8, 8)).to(torch.bfloat16)
    g = torch.randn(2, 16, 8, 8)
    exact = _bf16_errors(m, xb, g, m, ref_only=True)
    new = _bf16_errors(m, xb, g, m, exact)
    old = _bf16_errors(m, xb, g, lambda t: _ln_manual(m, t), exact)
    assert max(new) < 1e-2, new
    assert all(n < o / 2 for n, o in zip(new[:3], old[:3], strict=True)), (new, old)


# ---------------------------------------------------------------- gradient census (A-03, A-05, A-06, A-10)
@pytest.mark.parametrize("name", ROWS)
def test_gradient_census(name: str) -> None:
    """Every trainable parameter gets a finite gradient from one train step (DDP with
    find_unused_parameters=False needs that); frozen ones are deploy copies or unused branches."""
    net = _net(name).train()
    sr = ARCHS[name][1] is not None
    x = torch.rand(1, 3, *((24, 20) if sr else (64, 64)))
    out = net(x)
    out = out[0] if isinstance(out, tuple | list) else out
    out.float().square().mean().backward()
    missing = [
        n for n, p in net.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, missing
    stray = [
        n
        for n, p in net.named_parameters()
        if not p.requires_grad and p.grad is not None
    ]
    assert not stray, stray
    bad = [
        n
        for n, p in net.named_parameters()
        if p.grad is not None and not p.grad.isfinite().all()
    ]
    assert not bad, bad


@pytest.mark.parametrize("name", ["plainusr", "plainusr_ultra", "plainusr_large"])
def test_plainusr_trains_the_network_it_evaluates(name: str) -> None:
    """A-03: plainusr_ultra's training graph dropped blockm/block2; eval ran them untrained."""
    net = _net(name)
    x = torch.rand(1, 3, 13, 11)
    with torch.no_grad():
        torch.testing.assert_close(net.train()(x), net.eval()(x), rtol=1e-5, atol=1e-5)


def test_aether_prepare_qat_in_reduced_precision() -> None:
    """A-13: the priming forward of prepare_qat() runs in the net's dtype."""
    net = _net("aether_small").to(torch.bfloat16)
    net.prepare_qat()
    assert net.is_quantized


# ---------------------------------------------------------------- online re-parameterisation (A-01)
# The branch forms these blocks trained with before A-01, kept here as the reference.
def _conv3xc_branch(m, x):
    return m.conv(F.pad(x, (1, 1, 1, 1), "constant", 0)) + m.sk(x)


def _omnishift_branch(m, x):
    a = m.alpha
    return a[0] * x + a[1] * m.conv1x1(x) + a[2] * m.conv3x3(x) + a[3] * m.conv5x5(x)


def _mbconv_branch(m, x):
    out = m.expand_conv(x)
    t, b0 = (
        F.pad(out, (1, 1, 1, 1), "constant", 0),
        m.expand_conv.bias.view(1, -1, 1, 1),
    )
    t[:, :, :1, :] = b0  # old pad_tensor: the 1-px border holds the expand bias
    t[:, :, -1:, :] = b0
    t[:, :, :, :1] = b0
    t[:, :, :, -1:] = b0
    return m.reduce_conv(m.se(m.fea_conv(t)) + out) + x


def _lk_branch(m, x):
    lk, sk = m.lk_conv(x), m.sk_conv(x)
    return lk + m.lk_bias.view(1, -1, 1, 1) + sk + m.sk_bias.view(1, -1, 1, 1)


BRANCH = {
    span_arch.Conv3XC: _conv3xc_branch,  # SPAN+ Conv3XC subclasses it
    flexnet_arch.OmniShift: _omnishift_branch,
    plainusr_arch.MBConv: _mbconv_branch,
    aether_arch.ReparamLargeKernelConv: _lk_branch,
}
BLOCKS = {
    "conv3xc": lambda: span_arch.Conv3XC(6, 5, gain1=2),
    "conv3xc_nobias": lambda: span_arch.Conv3XC(6, 5, gain1=2, bias=False),
    "conv3xc_s2": lambda: span_arch.Conv3XC(6, 5, gain1=2, s=2),
    "spanplus_conv3xc": lambda: spanplus_arch.Conv3XC(6, 5, gain=2),
    "omnishift": lambda: flexnet_arch.OmniShift(6),
    "mbconv": lambda: plainusr_arch.MBConv(6),
    "reparam_lk": lambda: aether_arch.ReparamLargeKernelConv(6, 6, 11, 1, 6, 3),
}


def _branch_of(m: nn.Module):
    return next(f for cls, f in BRANCH.items() if isinstance(m, cls))


def _out_and_grads(net, x, fwd=None):
    out = (fwd or net)(x)
    params = [p for p in net.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        out, [x, *params], torch.ones_like(out) / out.numel() ** 0.5
    )
    return out, grads


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("name", BLOCKS)
def test_fused_training_block_equals_branch_form(name: str, dtype) -> None:
    """Same function of (x, params): outputs and every gradient match, borders included."""
    m = BLOCKS[name]().to(dtype).train()
    with torch.no_grad():  # non-trivial biases, alphas and an unsaturated ASR
        for p in m.parameters():
            if p.requires_grad:
                p.normal_(0, 0.3)
    x = torch.randn(2, 6, 17, 23, dtype=dtype, requires_grad=True)
    out, grads = _out_and_grads(m, x)
    ref, ref_grads = _out_and_grads(m, x, lambda t: _branch_of(m)(m, t))
    tol = (
        {"rtol": 1e-10, "atol": 1e-12}
        if dtype == torch.float64
        else {"rtol": 1e-4, "atol": 1e-5}
    )
    torch.testing.assert_close(out, ref, **tol)
    for g, r in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(g, r, **tol)


@pytest.mark.parametrize("name", BLOCKS)
def test_fused_training_block_under_bf16_autocast(name: str) -> None:
    """Production autocast against fp64: output, x gradient and every parameter gradient of the
    fused block stay at bf16 rounding and within 1.5x of the branch form's error (Conv3XC and
    MBConv come out lower, OmniShift and the large-kernel conv about equal)."""
    torch.manual_seed(0)
    m = BLOCKS[name]().train()
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.normal_(0, 0.3)
    x = torch.randn(2, 6, 17, 23)
    g = torch.randn_like(m(x))
    ref = _bf16_errors(m, x, g, m, ref_only=True)
    fused = _bf16_errors(m, x, g, m, ref)
    branch = _bf16_errors(m, x, g, lambda t: _branch_of(m)(m, t), ref)
    assert max(fused) < 2e-2, fused
    assert all(f < 1.5 * b + 1e-3 for f, b in zip(fused, branch, strict=True)), (
        fused,
        branch,
    )


REPARAM_NETS = {
    "span": {},
    "span_fast": {},
    "spanplus": {},
    "spanplus_sts": {},
    "flexnet": {"num_blocks": (1, 1), "dim": 16},
    "plainusr": {},
    "plainusr_ultra": {},
    "plainusr_large": {},
    "aether_small": {"drop_path_rate": 0.0},
}


@pytest.mark.parametrize("name", REPARAM_NETS)
def test_fused_training_network_equals_branch_form(monkeypatch, name: str) -> None:
    """fp64, whole network in train mode: output and every parameter gradient as with the branch
    forms (state_dict keys and init are unchanged, so old checkpoints train on identically)."""
    net = _net(name, **REPARAM_NETS[name]).double().train()
    x = torch.rand(1, 3, 19, 13, dtype=torch.float64, requires_grad=True)
    out, grads = _out_and_grads(net, x)
    for cls, f in BRANCH.items():
        monkeypatch.setattr(cls, "forward", f)
    ref, ref_grads = _out_and_grads(net, x)
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-11)
    for g, r in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(g, r, rtol=1e-8, atol=1e-11)


# ---------------------------------------------------------------- eval kernel cache (MT-09)
CACHED = {"span": {}, "spanplus": {}, "flexnet": {"num_blocks": (1, 1), "dim": 16}}


def _fresh(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """What the current weights give: a train-mode copy fuses on the fly. Also asserts that the
    deploy kernel each block just ran with is the fused kernel of its current weights."""
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, span_arch.Conv3XC | flexnet_arch.OmniShift):
                k = m.fused_kernel()
                k = k[0] if isinstance(k, tuple) else k
                deploy = m.eval_conv if hasattr(m, "eval_conv") else m.conv5x5_reparam
                assert torch.equal(deploy.weight, k)
        return copy.deepcopy(net).train()(x)


@pytest.mark.parametrize("name", CACHED)
def test_eval_forward_uses_the_cached_kernel(monkeypatch, name: str) -> None:
    net = _net(name, **CACHED[name]).eval()  # eval() fuses once
    calls = []
    for cls in (span_arch.Conv3XC, flexnet_arch.OmniShift):
        f = cls.fused_kernel
        monkeypatch.setattr(
            cls, "fused_kernel", lambda self, f=f: calls.append(1) or f(self)
        )
    x = torch.rand(1, 3, 20, 12)
    with torch.inference_mode():
        net(x)
        net(x)
    assert not calls


@pytest.mark.parametrize("name", CACHED)
def test_eval_kernel_follows_ema_optimizer_and_load(name: str) -> None:
    """The EMA copy stays in eval and is lerped in place; load_state_dict and optimizer steps write
    in place too: every in-place update reaches the next eval forward without an eval() call."""
    net = _net(name, **CACHED[name]).train()
    x = torch.rand(1, 3, 20, 12)
    ema = Ema(net, decay=0.5)
    ema.update()  # the first update copies
    with torch.inference_mode():
        before = ema.module(x)
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.5)
    for _ in range(2):
        net(x).square().mean().backward()
        opt.step()
        opt.zero_grad()
        ema.update()  # torch._foreach_lerp_
    with torch.inference_mode():
        out = ema.module(x)
    assert not torch.allclose(out, before)
    torch.testing.assert_close(out, _fresh(ema.module, x))
    # load_state_dict into a model that already cached its kernels in eval mode
    other = _net(name, **CACHED[name]).eval()
    with torch.inference_mode():
        other(x)
    other.load_state_dict(net.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(other(x), _fresh(net, x))
    # an optimizer step on a net that is in eval mode
    other(x).square().mean().backward()
    torch.optim.SGD(other.parameters(), lr=0.5).step()
    with torch.inference_mode():
        torch.testing.assert_close(other(x), _fresh(other, x))


@pytest.mark.parametrize("name", CACHED)
def test_eval_forward_under_autograd_keeps_every_gradient(name: str) -> None:
    """The cached deploy conv is frozen: an eval forward that autograd records runs the fused
    kernel instead, so every branch parameter gets the train-mode gradient (fp64)."""
    net = _net(name, **CACHED[name]).double().eval()
    x = torch.rand(1, 3, 19, 13, dtype=torch.float64)
    g = torch.randn_like(net(x))
    grads = {}
    for mode in (False, True):
        net.train(mode).zero_grad(set_to_none=True)
        out = net(x)
        out.backward(g)
        grads[mode] = (out.detach(), {n: p.grad for n, p in net.named_parameters()})
    (out, ev), (ref, tr) = grads[False], grads[True]
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-11)
    for n, p in net.named_parameters():
        if p.requires_grad:
            assert ev[n] is not None, n
            torch.testing.assert_close(ev[n], tr[n], rtol=1e-8, atol=1e-11)


@pytest.mark.parametrize("name", CACHED)
def test_built_under_inference_mode(name: str) -> None:
    """Inference tensors have no version counter: the blocks re-fuse on every eval forward."""
    x = torch.rand(1, 3, 20, 12)
    with torch.inference_mode():
        net = _net(name, **CACHED[name]).eval()
        out = net(x)
        torch.testing.assert_close(net(x), out)
        ref = net.train()(x)
        net.eval()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- ONNX export (MT-01)
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", ["span", "span_fast", "spanplus", "plainusr_ultra"])
def test_reparam_family_exports_dynamic_onnx(name: str) -> None:
    """The TorchScript exporter rejects a conv whose kernel is computed: SPAN's eval forward is the
    deploy conv again, PlainUSR (and float Aether, test_qat) export their branch form; spanplus
    also carries DySample's dynamic normaliser."""
    errs = _ort_errors(
        _net(name),
        (1, 3, 20, 20),
        [(1, 3, 20, 20), (1, 3, 32, 24), (1, 3, 27, 41), (2, 3, 24, 32)],
    )
    assert max(errs) < 1e-4, errs


def _convert_py():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "neosr_entry_convert_s2", root / "convert.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.mark.filterwarnings("ignore")
def test_convert_exports_span(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    torch.save({"params": _net("span", 2).state_dict()}, tmp_path / "w.pth")
    argv = ["--input", str(tmp_path / "w.pth"), "-net", "span", "-s", "2"]
    assert _convert_py().main(argv) == 0
    assert (tmp_path / "w_fp32.onnx").is_file()


# ---------------------------------------------------------------- Aether float graph (A-09)
def test_aether_float_training_is_one_dynamo_graph() -> None:
    """Idle QuantStubs were 68 skipped calls: 10 graphs / 9 breaks per forward."""
    torch._dynamo.reset()
    try:
        net = _net("aether_small").train()
        exp = torch._dynamo.explain(net)(torch.rand(1, 3, 16, 16))
    finally:
        torch._dynamo.reset()
    assert (exp.graph_count, exp.graph_break_count) == (1, 0), exp.break_reasons


# ---------------------------------------------------------------- PlainUSR deploy (A-11)
@pytest.mark.parametrize("name", ["plainusr", "plainusr_ultra", "plainusr_large"])
def test_plainusr_deploy_matches_eval(name: str) -> None:
    net = _net(name)
    with torch.no_grad():  # trained-like weights: unsaturated ASR, non-uniform PConv
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    net.eval()
    x = torch.rand(1, 3, 13, 11)
    with torch.no_grad():
        ref = net(x)
        deployed = copy.deepcopy(net)
        deployed.switch_to_deploy()
        out = deployed(x)
        net.switch_to_deploy(prune=True)
        pruned = net(x)
    blocks = (plainusr_arch.MBConv, plainusr_arch.PConv)
    assert not any(isinstance(m, blocks) for m in deployed.modules())
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    assert pruned.shape == ref.shape
    assert pruned.isfinite().all()


# ---------------------------------------------------------------- FlexNet SDPA (MX-14)
def test_flexnet_sdpa_gets_4d_inputs(monkeypatch) -> None:
    """3-D q/k/v always take the math kernel; (B', 1, N, C) is flash/efficient-eligible."""
    seen = []
    sdpa = F.scaled_dot_product_attention
    monkeypatch.setattr(
        F,
        "scaled_dot_product_attention",
        lambda q, k, v, **kw: seen.append(q.ndim) or sdpa(q, k, v, **kw),
    )
    net = _net("flexnet", num_blocks=(1, 1), dim=16)
    assert net(torch.rand(1, 3, 19, 13)).isfinite().all()
    assert seen
    assert set(seen) == {4}


# ---------------------------------------------------------------- span_fast init (A-06)
def test_span_fast_default_init_is_the_legacy_fan_out() -> None:
    """The A-06 init is opt-in (fan_in_init) until an A/B: by default every weight is the legacy
    kaiming fan_out draw. Same seed, same draws: the two inits differ by std ratios only."""
    torch.manual_seed(0)
    old = _net("span_fast")
    torch.manual_seed(0)
    new = _net("span_fast", fan_in_init=True)
    for (n, a), b in zip(old.named_parameters(), new.parameters(), strict=True):
        if a.ndim < 2 or not a.requires_grad or "eval_conv" in n:
            continue
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(a)
        scale = {"conv_9x9.weight": 8, "conv_cat.weight": 1 / 8}.get(n, 1)
        torch.testing.assert_close(b, a * scale * (fan_out / fan_in) ** 0.5, msg=n)


def test_span_fast_fan_in_init_keeps_every_block_alive() -> None:
    """fan_out init made the bias-free SPAB chain underflow to exact zeros by block 3: blocks 3-5
    and conv_2 got zero/subnormal gradients that Adam (eps 1e-8) cannot move."""
    net = _net("span_fast", fan_in_init=True).train()
    feats = []
    for b in (net.block_1, net.block_3, net.block_5):
        b.register_forward_hook(lambda _m, _i, o: feats.append(o[0].std().item()))
    net(torch.rand(2, 3, 32, 32)).square().mean().backward()
    assert min(feats) > 1e-2, feats
    weak = [
        n
        for n, p in net.named_parameters()
        if p.requires_grad and p.grad.abs().max() < 1e-6
    ]
    assert not weak, weak

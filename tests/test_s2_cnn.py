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


def test_dysample_fp32_bit_identical() -> None:
    m = DySample(16, 3, scale=2, groups=4)
    nn.init.normal_(m.scope.weight, std=0.5)  # non-trivial offsets
    x = torch.randn(2, 16, 13, 9, requires_grad=True)
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

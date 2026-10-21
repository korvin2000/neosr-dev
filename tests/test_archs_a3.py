"""Regression tests for group A3 archs: EIMN, GRFormer, KRGN, ESC, OmniSR, HiT-SRF."""

import hashlib

import pytest
import torch

from neosr.archs import (
    eimn_arch,
    esc_arch,
    grformer_arch,
    hitsrf_arch,
    krgn_arch,
    omnisr_arch,
)
from neosr.archs import ARCHS as TABLE
from neosr.utils.factory import resolve

_REGISTERED = (eimn_arch, grformer_arch, hitsrf_arch, krgn_arch)  # imported to register
pytestmark = pytest.mark.usefixtures("_env")

# CPU always runs; CUDA cases skip only when no GPU is present
DEVS = (
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
    ),
)
# heavy sweeps run once: on CUDA when present (catches CPU-built tensors), else on CPU
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 4
# name -> kwargs; scale kwarg name differs per arch
ARCHS = {
    "eimn": {"scale": SCALE},
    "eimn_l": {"scale": SCALE},
    "eimn_a": {"scale": SCALE},
    "grformer": {"upscale": SCALE},
    "grformer_medium": {"upscale": SCALE},
    "grformer_large": {"upscale": SCALE},
    "krgn": {"scale": SCALE},
    "esc": {"upscaling_factor": SCALE},
    "esc_light": {"upscaling_factor": SCALE},
    "esc_large": {"upscaling_factor": SCALE},
    "esc_fp": {"upscaling_factor": SCALE},
    "omnisr": {"upsampling": SCALE},
    "hit_srf": {"upscale": SCALE},
    # hit_srf_medium/large: windows up to 128/192 px need a dense (win_area^2 x heads)
    # position-bias gather (~6-10 GiB per block); too heavy for a smoke test.
}
SHAPES = ((1, 3, 64, 64), (1, 3, 67, 53))
# heavy variants (grformer >7 GiB with backward at 67x53; esc_large slow eager flex attention):
# smaller inputs, one needs padding
SMALL_SHAPES = ((1, 3, 32, 32), (1, 3, 35, 27))
HEAVY = {"grformer_medium", "grformer_large", "esc_large"}

# (key count, sha256 of key:shape:dtype) per variant, verified equal to the archs at
# c7a2e7a (before Phase A): existing checkpoints keep loading strictly
MANIFEST = {
    "eimn": (708, "bb0a0e7ddc1e0eb0"),
    "eimn_l": (708, "bb0a0e7ddc1e0eb0"),
    "eimn_a": (620, "ca23efab28f4ab1f"),
    "grformer": (844, "424e71467e451eeb"),
    "grformer_medium": (1262, "43233f41eac16b32"),
    "grformer_large": (2516, "8646fc1c6e81f868"),
    "krgn": (462, "0e13cec11ec021dd"),
    "esc": (462, "2380374e12fe03c3"),
    "esc_light": (284, "48a9ccfbc8156070"),
    "esc_large": (907, "2e6df819d7fcfe70"),
    "esc_fp": (455, "da93befc6f0985e6"),
    "omnisr": (316, "49aae9d686ca25f9"),
    "hit_srf": (882, "a61e1543ce7d0e1f"),
    "hit_srf_medium": (1324, "9405e44fd0ac03a8"),
    "hit_srf_large": (2632, "e97b28762f3536e0"),
}


def _build(name: str, dev: str) -> torch.nn.Module:
    torch.manual_seed(0)
    return resolve(TABLE, "neosr.archs", "test", name)(**ARCHS[name]).to(dev)


def _assert_all_grads(net: torch.nn.Module) -> None:
    # every trainable param must be reached (a detached branch breaks training and DDP)
    for n, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, n
            assert torch.isfinite(p.grad).all(), n


@pytest.fixture
def _env():
    # archs must not rely on torch.set_default_device("cuda"); no TF32 for tight checks
    assert torch.get_default_device().type == "cpu"
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _manifest(net):
    sd = net.state_dict()
    blob = "\n".join(f"{k}:{tuple(v.shape)}:{v.dtype}" for k, v in sd.items())
    return len(sd), hashlib.sha256(blob.encode()).hexdigest()[:16]


@pytest.mark.parametrize("name", MANIFEST)
def test_state_dict_manifest(name: str) -> None:
    kw = ARCHS.get(name, {"upscale": SCALE})  # hit_srf_medium/large: build only
    assert _manifest(resolve(TABLE, "neosr.archs", "test", name)(**kw)) == MANIFEST[name]


@pytest.mark.parametrize("name", list(ARCHS))
def test_forward_backward(name: str) -> None:
    dev = DEV
    net = _build(name, dev)
    for mode in (True, False):
        net.train(mode)
        for shape in SMALL_SHAPES if name in HEAVY else SHAPES:
            x = torch.rand(shape, device=dev)
            y = net(x)
            assert y.shape == (1, 3, shape[2] * SCALE, shape[3] * SCALE)
            assert torch.isfinite(y).all()
            net.zero_grad(set_to_none=True)
            y.mean().backward()
            _assert_all_grads(net)


@pytest.mark.parametrize("name", ["esc", "esc_fp"])
def test_esc_batch(name: str) -> None:
    dev = DEV
    # eval path used to hard-code batch 1 for the dynamic conv; per-sample kernels must
    # stay per sample, so make the (zero-initialised) kernel projection non-trivial
    net = _build(name, dev)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, esc_arch.ConvolutionalAttention):
                m.dwc_proj[-1].weight.normal_(0, 0.1)
                m.dwc_proj[-1].bias.normal_(0, 0.1)
    x = torch.rand(3, 3, 32, 32, device=dev)
    for mode in (True, False):
        net.train(mode)
        with torch.no_grad():
            ref = torch.cat([net(x[i : i + 1]) for i in range(len(x))])
            torch.testing.assert_close(net(x), ref, rtol=1e-4, atol=1e-4)
        net.zero_grad(set_to_none=True)
        net(x).mean().backward()
        _assert_all_grads(net)


def test_esc_convert_matches() -> None:
    dev = DEV
    net = _build("esc", dev).eval()
    x = torch.rand(1, 3, 40, 40, device=dev)
    with torch.no_grad():
        ref = net(x)
        net.convert()
        torch.testing.assert_close(net(x), ref, rtol=1e-4, atol=1e-4)


def test_esc_kernel_size() -> None:
    dev = DEV
    for fp in (False, True):
        net = esc_arch.esc(
            kernel_size=11,
            is_fp=fp,
            realsr=not fp,
            use_dysample=False,
            upscaling_factor=2,
        ).to(dev)
        for mode in (True, False):
            net.train(mode)
            with torch.no_grad():
                assert net(torch.rand(1, 3, 32, 32, device=dev)).shape[-1] == 64


def test_eimn_freeze() -> None:
    net = resolve(TABLE, "neosr.archs", "test", "eimn_a")(scale=2, freeze_param=True)
    trainable = {n.split(".")[0] for n, p in net.named_parameters() if p.requires_grad}
    assert trainable == {"tail"}
    assert net.tail[0].out_channels == 3 * 2 * 2


@pytest.mark.parametrize("dev", DEVS)
def test_hitsrf_small_input(dev: str) -> None:
    # windows up to 8x the base window used to break reflect padding on small inputs
    net = resolve(TABLE, "neosr.archs", "test", "hit_srf")(upscale=2).to(dev).eval()
    with torch.no_grad():
        for h, w in ((12, 15), (1, 12), (12, 1), (1, 1)):
            assert net(torch.rand(1, 3, h, w, device=dev)).shape == (1, 3, 2 * h, 2 * w)


@pytest.mark.parametrize("name", ["hit_srf_medium", "hit_srf_large"])
def test_hitsrf_heavy_build(name: str) -> None:
    # forward is too memory-heavy for a test; at least check they build and round-trip
    net = resolve(TABLE, "neosr.archs", "test", name)(upscale=SCALE)
    net.load_state_dict(net.state_dict(), strict=True)


@pytest.mark.parametrize("dev", DEVS)
def test_omnisr_dropsample(dev: str) -> None:
    m = omnisr_arch.Dropsample(0.5).train()
    x = torch.ones(8, 2, 3, 3, device=dev)
    y = m(x)
    assert y.shape == x.shape
    assert set(y.unique().tolist()) <= {0.0, 2.0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_grformer_half_mask() -> None:
    # 48x32 != img_size: the shift mask is rebuilt in forward and must follow the fp16 logits
    net = _build("grformer", "cuda").eval().half()
    with torch.no_grad():
        y = net(torch.rand(1, 3, 48, 32, device="cuda", dtype=torch.half))
    assert y.dtype == torch.half
    assert torch.isfinite(y).all()
    assert net.mean.dtype == torch.half
    assert "mean" not in net.state_dict()

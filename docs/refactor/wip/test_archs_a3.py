"""Regression tests for group A3 archs: EIMN, GRFormer, KRGN, ESC, OmniSR, HiT-SRF."""

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
from neosr.utils.registry import ARCH_REGISTRY

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
# heavy variants (>7 GiB with backward at 67x53): smaller inputs, one needs padding
SMALL_SHAPES = ((1, 3, 32, 32), (1, 3, 35, 27))
HEAVY = {"grformer_medium", "grformer_large"}


def _build(name: str, dev: str) -> torch.nn.Module:
    torch.manual_seed(0)
    return ARCH_REGISTRY.get(name)(**ARCHS[name]).to(dev)


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


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("name", list(ARCHS))
def test_forward_backward(name: str, dev: str) -> None:
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
            grads = [p.grad for p in net.parameters() if p.grad is not None]
            assert grads, "no gradients"
            assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("name", ["esc", "esc_fp"])
def test_esc_batch(name: str, dev: str) -> None:
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
        assert all(
            torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
        )


@pytest.mark.parametrize("dev", DEVS)
def test_esc_convert_matches(dev: str) -> None:
    net = _build("esc", dev).eval()
    x = torch.rand(1, 3, 40, 40, device=dev)
    with torch.no_grad():
        ref = net(x)
        net.convert()
        torch.testing.assert_close(net(x), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dev", DEVS)
def test_esc_kernel_size(dev: str) -> None:
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
    net = ARCH_REGISTRY.get("eimn_a")(scale=2, freeze_param=True)
    trainable = {n.split(".")[0] for n, p in net.named_parameters() if p.requires_grad}
    assert trainable == {"tail"}
    assert net.tail[0].out_channels == 3 * 2 * 2


@pytest.mark.parametrize("dev", DEVS)
def test_hitsrf_small_input(dev: str) -> None:
    # windows up to 8x the base window used to break reflect padding on small inputs
    net = ARCH_REGISTRY.get("hit_srf")(upscale=2).to(dev).eval()
    with torch.no_grad():
        for h, w in ((12, 15), (1, 12), (12, 1), (1, 1)):
            assert net(torch.rand(1, 3, h, w, device=dev)).shape == (1, 3, 2 * h, 2 * w)


@pytest.mark.parametrize("name", ["hit_srf_medium", "hit_srf_large"])
def test_hitsrf_heavy_build(name: str) -> None:
    # forward is too memory-heavy for a test; at least check they build and round-trip
    net = ARCH_REGISTRY.get(name)(upscale=SCALE)
    net.load_state_dict(net.state_dict(), strict=True)


@pytest.mark.parametrize("dev", DEVS)
def test_omnisr_dropsample(dev: str) -> None:
    m = omnisr_arch.Dropsample(0.5).train()
    x = torch.ones(8, 2, 3, 3, device=dev)
    y = m(x)
    assert y.shape == x.shape
    assert set(y.unique().tolist()) <= {0.0, 2.0}

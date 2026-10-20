import gc
import os

import pytest
import torch
import torch.nn.functional as F

DEV = os.environ.get("NEOSR_TEST_DEVICE", "cuda")  # "cpu" for a slow run on a busy GPU
if DEV == "cuda" and not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

import neosr.archs.dat_arch as dat_arch  # noqa: E402
import neosr.archs.drct_arch as drct_arch  # noqa: E402
import neosr.archs.hat_arch  # noqa: E402, F401
import neosr.archs.swinir_arch as swinir_arch  # noqa: E402
import neosr.archs.swinirdense_arch as swinirdense_arch  # noqa: E402
from neosr.utils.registry import ARCH_REGISTRY  # noqa: E402

SCALE = 4
VARIANTS = [
    "hat_s", "hat_m", "hat_l",
    "drct", "drct_s", "drct_l", "drct_xl",
    "swinir_small", "swinir_medium", "swinir_large",
    "swinirdense",
    "dat_s", "dat_m", "dat_2",
]  # fmt: skip
SMALL = ["hat_s", "drct_s", "swinir_small", "swinirdense", "dat_s"]
# the heaviest variants train with activation checkpointing to fit on a shared GPU
LOW_MEM = {"hat_l": {"use_checkpoint": True}, "drct_l": {"use_checkpoint": True},
           "drct_xl": {"use_checkpoint": True}, "dat_m": {"use_chk": True},
           "dat_2": {"use_chk": True}}  # fmt: skip


def build(name, **kw):
    torch.manual_seed(0)
    return ARCH_REGISTRY.get(name)(upscale=SCALE, **kw).to(DEV)


@pytest.fixture(autouse=True)
def _free_cuda():
    yield
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


def finite(t):
    return bool(torch.isfinite(t).all())


@pytest.mark.parametrize("name", VARIANTS)
def test_forward_backward(name):
    net = build(name, **LOW_MEM.get(name, {}))
    assert "mean" not in net.state_dict()  # non-persistent: checkpoint keys unchanged
    # DAT has BatchNorm after AdaptiveAvgPool2d(1): batch 1 cannot train (expected)
    b = 2 if name.startswith("dat") else 1
    for h, w in ((64, 64), (67, 53)):  # aligned and not a multiple of the window
        x = torch.rand(b, 3, h, w, device=DEV)
        net.train()
        y = net(x)
        assert y.shape == (b, 3, h * SCALE, w * SCALE)
        assert finite(y)
        y.mean().backward()
        for n, p in net.named_parameters():
            assert p.grad is not None, n
            assert finite(p.grad), n
        net.zero_grad(set_to_none=True)
        net.eval()
        with torch.no_grad():
            ye = net(x)
        assert ye.shape == y.shape
        assert finite(ye)


@pytest.mark.parametrize("name", SMALL)
def test_tiny_input_and_half_masks(name):
    net = build(name).eval()
    with torch.no_grad():
        # pad larger than the input -> replicate fallback
        assert net(torch.rand(1, 3, 7, 5, device=DEV)).shape == (1, 3, 28, 20)
        # fp16 weights + non-aligned size -> masks rebuilt in forward must match dtype
        y = net.half()(torch.rand(1, 3, 67, 53, device=DEV, dtype=torch.half))
    assert y.dtype == torch.half
    assert finite(y)
    assert y.shape == (1, 3, 268, 212)
    assert net.mean.dtype == torch.half  # buffer follows .half(), not the input


@pytest.mark.parametrize("name", ["hat_s", "drct_s", "swinir_small", "swinirdense"])
def test_checkpoint_grad_parity(name):
    x = torch.rand(
        1, 3, 37, 29, device=DEV
    )  # non-aligned: padded path is recomputed too
    grads = []
    for flag in (False, True):
        net = build(name, use_checkpoint=flag).train()
        torch.manual_seed(1)
        net(x).mean().backward()
        grads.append([p.grad for p in net.parameters()])
    for g0, g1 in zip(*grads, strict=True):
        torch.testing.assert_close(g0, g1, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", ["hat_s", "drct_s", "swinir_small", "swinirdense"])
def test_pad_crop_equivalence(name):
    net = build(name).eval()
    x = torch.rand(1, 3, 37, 29, device=DEV)
    ws = min(net.window_size, *net.patches_resolution)
    xp = F.pad(x, (0, (-29) % ws, 0, (-37) % ws), "reflect")
    with torch.no_grad():
        torch.testing.assert_close(net(x), net(xp)[..., : 37 * SCALE, : 29 * SCALE])


# reduced models: scale-1 crop, and img_size < window_size (blocks clamp their window)
SMALL_CFG = {"depths": (2,), "num_heads": (2,), "embed_dim": 16}


@pytest.mark.parametrize(
    "cls", [drct_arch.drct, swinir_arch.swinir, swinirdense_arch.swinirdense]
)
def test_reduced_models(cls):
    torch.manual_seed(0)
    net = cls(img_size=6, window_size=8, upscale=2, **SMALL_CFG).to(DEV).eval()
    with torch.no_grad():
        for h, w in ((6, 6), (12, 12), (7, 5)):
            assert net(torch.rand(1, 3, h, w, device=DEV)).shape == (1, 3, 2 * h, 2 * w)
    if cls is not drct_arch.drct:  # denoising mode: output/input ratio 1
        net = cls(upscale=1, upsampler="", **SMALL_CFG).to(DEV).eval()
        with torch.no_grad():
            assert net(torch.rand(1, 3, 37, 29, device=DEV)).shape == (1, 3, 37, 29)


@pytest.mark.parametrize("name", ["swinir_small", "swinirdense"])
def test_flash_attn_trains_qkv(name):
    net = build(name, flash_attn=True, drop_path_rate=0.0).train()  # no dropped blocks
    net(torch.rand(1, 3, 32, 32, device=DEV)).mean().backward()
    qkv = [p.grad for n, p in net.named_parameters() if ".attn.qkv." in n]
    assert qkv
    assert all(g is not None and finite(g) and g.abs().sum() > 0 for g in qkv)


def test_swinirdense_3conv():
    net = build("swinirdense", resi_connection="3conv")
    assert net(torch.rand(1, 3, 32, 32, device=DEV)).shape == (1, 3, 128, 128)


def test_dat_bad_window_idx_raises():
    with pytest.raises(ValueError, match="idx must be 0 or 1"):
        dat_arch.Spatial_Attention(dim=16, idx=2, num_heads=2)

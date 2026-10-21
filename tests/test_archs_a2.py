"""Group A2 arch checks: HAT, DRCT, SwinIR, SwinIR-dense, DAT (padding, masks, mean, checkpointing)."""

import gc
import hashlib

import pytest
import torch
import torch.nn.functional as F

from neosr.archs import dat_arch, drct_arch, hat_arch, swinir_arch, swinirdense_arch
from neosr.archs import ARCHS as TABLE
from neosr.utils.factory import resolve

_REGISTERED = (hat_arch,)  # imported to register
DEV = "cuda"
pytestmark = pytest.mark.usefixtures("_free_cuda")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

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

# (key count, sha256 of key:shape:dtype) per variant, verified equal to the archs at
# c7a2e7a (before Phase A): existing checkpoints keep loading strictly
MANIFEST = {
    "hat_s": (864, "8b1a2cc1ae04398e"),
    "hat_m": (864, "71116a295a5ffdc9"),
    "hat_l": (1710, "0096f41095674b61"),
    "drct": (508, "dee0705e73f4d7bd"),
    "drct_s": (344, "908f4fa793ab33c0"),
    "drct_l": (1000, "04f95d6995160d44"),
    "drct_xl": (1164, "292805d59d7bcce2"),
    "swinir_small": (366, "bcd59da3fbaf15d0"),
    "swinir_medium": (550, "10608a5fbb161e3e"),
    "swinir_large": (859, "107a5a0e7ffdc873"),
    "swinirdense": (728, "c30186011df6b98d"),
    "dat_s": (2116, "011711950fc90d23"),
    "dat_m": (2116, "43899235ecffff9a"),
    "dat_2": (2116, "529ab9e2f44adc9b"),
}


def build(name, **kw):
    torch.manual_seed(0)
    return resolve(TABLE, "neosr.archs", "test", name)(upscale=SCALE, **kw).to(DEV)


@pytest.fixture
def _free_cuda():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def finite(t):
    return bool(torch.isfinite(t).all())


def _manifest(net):
    sd = net.state_dict()
    blob = "\n".join(f"{k}:{tuple(v.shape)}:{v.dtype}" for k, v in sd.items())
    return len(sd), hashlib.sha256(blob.encode()).hexdigest()[:16]


@pytest.mark.parametrize("name", VARIANTS)
def test_state_dict_manifest(name):
    assert _manifest(resolve(TABLE, "neosr.archs", "test", name)(upscale=SCALE)) == MANIFEST[name]


@cuda
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


@cuda
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


@cuda
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


@cuda
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


@cuda
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


@cuda
@pytest.mark.parametrize("name", ["swinir_small", "swinirdense"])
def test_flash_attn_trains_qkv(name):
    net = build(name, flash_attn=True, drop_path_rate=0.0).train()  # no dropped blocks
    net(torch.rand(1, 3, 32, 32, device=DEV)).mean().backward()
    qkv = [p.grad for n, p in net.named_parameters() if ".attn.qkv." in n]
    assert qkv
    assert all(g is not None and finite(g) and g.abs().sum() > 0 for g in qkv)


@cuda
def test_swinirdense_3conv():
    # the one intended shape change: the first conv takes the 2*dim dense concat like 1conv;
    # its old dim-input weights could never run a forward, so no trained checkpoint has them
    net = build("swinirdense", resi_connection="3conv")
    assert net(torch.rand(1, 3, 32, 32, device=DEV)).shape == (1, 3, 128, 128)


def test_dat_bad_window_idx_raises():
    with pytest.raises(ValueError, match="idx must be 0 or 1"):
        dat_arch.Spatial_Attention(dim=16, idx=2, num_heads=2)

"""Stage-2 export fixes: the arch edits that make ONNX export dynamic keep eager outputs bitwise.

Each test patches the replaced code (verbatim, below) back in and compares full-model outputs on a
fixed seed at aligned, non-aligned and batch-2 sizes.
"""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from neosr.archs import (
    asid_arch,
    build_network,
    dctlsa_arch,
    ditn_arch,
    esc_arch,
    grformer_arch,
    lmlt_arch,
    mfghmoe_arch,
    microsr_arch,
    omnisr_arch,
)

SIZES = [(1, 3, 16, 16), (1, 3, 19, 23), (2, 3, 20, 20)]


# ---- the replaced code ----


def _old_check_image_size(self, x):
    _, _, h, w = x.size()
    ws = self.window_size
    m = -(-15 // ws)
    mod_pad_h = max(-(-h // ws), m) * ws - h
    mod_pad_w = max(-(-w // ws), m) * ws - w
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant", 0)


def _old_window_reverse(windows, window_size, h, w):
    wh, ww = (
        window_size if isinstance(window_size, (tuple, list)) else (window_size,) * 2
    )
    b = int(windows.shape[0] / (h * w / wh / ww))
    x = windows.view(b, h // wh, w // ww, wh, ww, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


def _old_lmlt_window_reverse(_self, windows, window_size, h, w):
    return _old_window_reverse(windows, window_size, h, w)


def _old_ca_forward(self, x, lk_filter):
    x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)
    bs = x1.shape[0]
    dynamic_kernel = self.dwc_proj(x1).reshape(-1, 1, self.sk_size, self.sk_size)
    x1_ = rearrange(x1, "b c h w -> 1 (b c) h w")
    x1_ = F.conv2d(
        x1_, dynamic_kernel, stride=1, padding=self.sk_size // 2, groups=bs * self.pdim
    )
    x1_ = rearrange(x1_, "1 (b c) h w -> b c h w", b=bs, c=self.pdim)
    x1 = F.conv2d(x1, lk_filter, stride=1, padding=lk_filter.shape[-1] // 2) + x1_
    return torch.cat([x1, x2], dim=1)


def _old_dca_forward(self, x, lk_channel, lk_spatial):
    x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)
    b, c = x1.shape[:2]
    kernel = self.proj(x1)
    kernel = rearrange(kernel, "b (c kh kw) 1 1 -> (b c) 1 kh kw", kh=3, kw=3)
    ks = lk_spatial.shape[-1]
    n_pad = (ks - self.dynamic_kernel_size) // 2
    kernel = F.pad(kernel, (n_pad, n_pad, n_pad, n_pad))
    x1 = F.conv2d(x1, lk_channel, padding=0)
    x1 = rearrange(x1, "b c h w -> 1 (b c) h w")
    lk_spatial = lk_spatial.repeat(b, 1, 1, 1)
    x1 = F.conv2d(x1, kernel + lk_spatial, padding=ks // 2, groups=b * c)
    x1 = rearrange(x1, "1 (b c) h w -> b c h w", b=b, c=c)
    return torch.cat([x1, x2], dim=1)


def _old_geo_ensemble(k):
    flips = [k, k.flip([3]), k.flip([2]), k.flip([2, 3])]
    r = torch.rot90(k, -1, [2, 3])
    flips += [r, r.flip([3]), r.flip([2]), r.flip([2, 3])]
    out = flips[0]
    for f in flips[1:]:
        out = out + f
    return out / 8


def _old_ufone_forward(self, x):
    B, C, H, W = x.data.shape
    p = self.patch_size
    f = x.view(B, C, H // p, p, W // p, p)
    f = f.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, p, p)
    f = self.ITLs(f)
    f = f.view(B, H // p, W // p, C, p, p)
    f = f.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return self.SALs(f)


ESC = [
    (esc_arch.ConvolutionalAttention, "forward", _old_ca_forward),
    (esc_arch.DecomposedConvolutionalAttention, "forward", _old_dca_forward),
]
LEGACY = {
    "asid": [(asid_arch.asid, "check_image_size", _old_check_image_size)],
    "omnisr": [(omnisr_arch.omnisr_net, "check_image_size", _old_check_image_size)],
    "esc_light": ESC,
    "esc_fp": ESC,
    "ditn": [(ditn_arch.UFONE, "forward", _old_ufone_forward)],
    "dctlsa": [(dctlsa_arch, "window_reverse", _old_window_reverse)],
    "grformer": [(grformer_arch, "window_reverse", _old_window_reverse)],
    "microsr_light": [(microsr_arch, "window_reverse", _old_window_reverse)],
    "mfghmoe": [(mfghmoe_arch, "window_reverse", _old_window_reverse)],
    "lmlt_tiny": [
        (lmlt_arch.downsample_vit, "window_reverse", _old_lmlt_window_reverse)
    ],
}


def _run(net: nn.Module, sizes) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    with torch.inference_mode():
        return [net(torch.rand(s, generator=g)) for s in sizes]


@pytest.mark.parametrize("name", list(LEGACY))
def test_export_fix_keeps_eager_outputs_bitwise(monkeypatch, name: str) -> None:
    # asid/omnisr: also below the 15 px floor, where the max() branch decides the padding
    sizes = [*SIZES, (1, 3, 5, 7)] if name in {"asid", "omnisr"} else SIZES
    torch.manual_seed(0)
    net = build_network({"type": name}, scale=2).eval()
    new = _run(net, sizes)
    for owner, attr, old in LEGACY[name]:
        monkeypatch.setattr(owner, attr, old)
    if isinstance(net, esc_arch.esc) and not net.is_fp:  # rot90 -> flip + transpose
        monkeypatch.setattr(net, "plk_func", _old_geo_ensemble)
    old = _run(net, sizes)
    for s, a, b in zip(sizes, new, old, strict=True):
        assert torch.equal(a, b), s


@pytest.mark.parametrize("ws", [4, 8, 16])
@pytest.mark.parametrize("size", [(1, 1), (5, 7), (15, 16), (17, 33)])
def test_ceil_padding_matches_the_old_formula(ws: int, size) -> None:
    class M(nn.Module):
        window_size = ws

    x = torch.zeros(1, 1, *size)
    new = asid_arch.asid.check_image_size(M(), x).shape
    assert new == omnisr_arch.omnisr_net.check_image_size(M(), x).shape
    assert new == _old_check_image_size(M(), x).shape

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

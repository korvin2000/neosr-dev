"""Stage 3 A7: NAFNet (1x restoration) with the official key names. CPU."""

import pytest
import torch

from neosr.archs import build_network
from neosr.archs.arch_util import LayerNorm2d
from neosr.config import ConfigError, load_config

TINY = {
    "type": "nafnet",
    "width": 8,
    "middle_blk_num": 1,
    "enc_blk_nums": [1, 2],
    "dec_blk_nums": [1, 1],
}


def _official(width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2)) -> dict[str, tuple]:
    """Key -> shape of the official NAFNet (megvii-research/NAFNet, NAFNet_arch.py), spelled out."""

    def block(p: str, c: int) -> dict[str, tuple]:
        dw, ffn = 2 * c, 2 * c
        return {
            f"{p}.conv1.weight": (dw, c, 1, 1), f"{p}.conv1.bias": (dw,),
            f"{p}.conv2.weight": (dw, 1, 3, 3), f"{p}.conv2.bias": (dw,),
            f"{p}.conv3.weight": (c, dw // 2, 1, 1), f"{p}.conv3.bias": (c,),
            f"{p}.sca.1.weight": (dw // 2, dw // 2, 1, 1), f"{p}.sca.1.bias": (dw // 2,),
            f"{p}.conv4.weight": (ffn, c, 1, 1), f"{p}.conv4.bias": (ffn,),
            f"{p}.conv5.weight": (c, ffn // 2, 1, 1), f"{p}.conv5.bias": (c,),
            f"{p}.norm1.weight": (c,), f"{p}.norm1.bias": (c,),
            f"{p}.norm2.weight": (c,), f"{p}.norm2.bias": (c,),
            f"{p}.beta": (1, c, 1, 1), f"{p}.gamma": (1, c, 1, 1),
        }  # fmt: skip

    out = {"intro.weight": (width, 3, 3, 3), "intro.bias": (width,)}
    c = width
    for i, n in enumerate(enc):
        for j in range(n):
            out |= block(f"encoders.{i}.{j}", c)
        out |= {f"downs.{i}.weight": (2 * c, c, 2, 2), f"downs.{i}.bias": (2 * c,)}
        c *= 2
    for j in range(mid):
        out |= block(f"middle_blks.{j}", c)
    for i, n in enumerate(dec):
        out[f"ups.{i}.0.weight"] = (2 * c, c, 1, 1)
        c //= 2
        for j in range(n):
            out |= block(f"decoders.{i}.{j}", c)
    return out | {"ending.weight": (3, width, 3, 3), "ending.bias": (3,)}


@pytest.mark.parametrize(("name", "width"), [("nafnet", 32), ("nafnet_l", 64)])
def test_official_state_dict_loads_strictly(name, width):
    net = build_network({"type": name}, scale=1)
    shapes = _official(width)
    assert {k: tuple(v.shape) for k, v in net.state_dict().items()} == shapes
    g = torch.Generator().manual_seed(0)
    sd = {k: torch.randn(s, generator=g) for k, s in shapes.items()}
    net.load_state_dict(sd, strict=True)
    if name == "nafnet":
        assert (
            sum(p.numel() for p in net.parameters()) == 29_159_715
        )  # the paper's 29.1M


def test_scale_must_be_one():
    with pytest.raises(ConfigError, match="1x restoration network"):
        build_network({"type": "nafnet"}, scale=2)


def test_any_size_in_same_size_out_and_trains():
    net = build_network(TINY, scale=1)
    x = torch.rand(2, 3, 13, 22, requires_grad=True)
    y = net(x)
    assert y.shape == x.shape  # padded to a multiple of 4 inside, cropped back
    y.mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters())


def test_layer_norm2d_is_the_official_formula():
    ln = LayerNorm2d(6, eps=1e-6)
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.uniform_(-0.5, 0.5)
    x = torch.randn(2, 6, 5, 4, generator=torch.Generator().manual_seed(1))
    mu = x.mean(1, keepdim=True)
    var = (x - mu).pow(2).mean(1, keepdim=True)
    ref = ln.weight.view(1, 6, 1, 1) * (x - mu) / (var + 1e-6).sqrt() + ln.bias.view(
        1, 6, 1, 1
    )
    torch.testing.assert_close(ln(x), ref)


def test_the_inference_option_file_is_valid():
    cfg = load_config("options/test_nafnet.toml", is_train=False, root=".")
    assert (cfg["scale"], cfg["network_g"]["type"]) == (1, "nafnet")


def test_dynamic_onnx_export_verifies(tmp_path):
    """No NO_ONNX / STATIC_ONLY row: the padding traces dynamically (verified at the MT-02 shapes)."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from neosr.cli import convert_args  # noqa: PLC0415
    from tests.test_entry import convert_py  # noqa: PLC0415

    argv = ["--input", "unused.pth", "-net", "nafnet", "-s", "1"]
    args = convert_args([*argv, "--output", str(tmp_path / "m.onnx")])
    # 4 levels pad to 16: the verification shapes 20x20 and 32x24 need different padding
    four = {
        "type": "nafnet",
        "width": 4,
        "middle_blk_num": 1,
        "enc_blk_nums": [1] * 4,
        "dec_blk_nums": [1] * 4,
    }
    convert_py.to_onnx(args, build_network(four, scale=1).eval())
    assert (tmp_path / "m_fp32.onnx").is_file()


def test_block_matches_the_official_forward_with_open_gates():
    """beta/gamma start at 0 (identity blocks): open them and compare with the official NAFBlock
    forward and its LayerNormFunction math, spelled out."""
    from neosr.archs.nafnet_arch import NAFBlock  # noqa: PLC0415

    torch.manual_seed(0)
    blk = NAFBlock(8).eval()
    with torch.no_grad():
        for p in blk.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    def ln(x, m):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return m.weight.view(1, -1, 1, 1) * (x - mu) / (
            var + 1e-6
        ).sqrt() + m.bias.view(1, -1, 1, 1)

    def gate(x):
        a, b = x.chunk(2, dim=1)
        return a * b

    inp = torch.randn(2, 8, 9, 7)
    x = gate(blk.conv2(blk.conv1(ln(inp, blk.norm1))))
    x = blk.conv3(x * blk.sca(x))
    y = inp + x * blk.beta
    ref = y + blk.conv5(gate(blk.conv4(ln(y, blk.norm2)))) * blk.gamma
    torch.testing.assert_close(blk(inp), ref)

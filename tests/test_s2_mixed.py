"""Stage-2 lane I6 (mixed archs, MX-01..MX-18): parity against the replaced code, gradient census, sizes."""

import time

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from neosr.archs import build_network
from neosr.archs import esc_arch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _build(name: str, scale: int = 4, seed: int = 0, **kw) -> nn.Module:
    torch.manual_seed(seed)
    return build_network({"type": name, **kw}, scale=scale)


def _assert_grads_close(a: nn.Module, b: nn.Module, rtol: float = 1e-4) -> None:
    # same parameters, same loss: gradients agree up to accumulation-order rounding
    for (n, p), q in zip(a.named_parameters(), b.parameters(), strict=True):
        assert p.grad is not None, n
        assert q.grad is not None, n
        torch.testing.assert_close(
            p.grad, q.grad, rtol=rtol, atol=rtol * q.grad.abs().max().item(), msg=n
        )


# ---------------------------------------------------------------- MX-01 ESC rpe index


def _old_esc_idxs(ws: int, heads: int) -> torch.Tensor:
    # the replaced WindowAttention.create_table_idxs (Python loop, (heads*N*N, 2) CPU tensor)
    out = []
    for head in range(heads):
        for h in range(ws**2):
            for w in range(ws**2):
                rel_h = w // ws - h // ws + ws - 1
                rel_w = w % ws - h % ws + ws - 1
                out.append((head, rel_h * (2 * ws - 1) + rel_w))
    return torch.tensor(out, dtype=torch.long)


@pytest.mark.parametrize(("ws", "heads"), [(4, 3), (8, 4)])
def test_esc_rpe_bias_bitwise(ws: int, heads: int) -> None:
    m = esc_arch.WindowAttention(16, ws, heads, F.scaled_dot_product_attention, "sdpa")
    idx = _old_esc_idxs(ws, heads)
    old = m.relative_position_bias[idx[:, 0], idx[:, 1]].reshape(1, heads, ws * ws, ws * ws)
    assert torch.equal(old, m.rpe_bias_of())
    assert m.rpe_idx.shape == (ws * ws, ws * ws)
    assert "rpe_idx" not in m.state_dict()  # non-persistent: checkpoints unchanged


def test_esc_index_is_a_device_buffer_and_builds_fast() -> None:
    t = time.perf_counter()
    net = _build("esc")
    assert time.perf_counter() - t < 5  # the Python loop took ~19 s
    net = net.to("meta")  # buffers follow .to(); the old attribute stayed on the CPU
    for m in net.modules():
        if isinstance(m, esc_arch.WindowAttention):
            assert m.rpe_idx.device.type == "meta"


# ---------------------------------------------------------------- MX-11 CATANet initted


class _NoHostSync:
    """Raise on the Python-level device->host reads (bool/item/tolist/nonzero) inside the block."""

    NAMES = ("__bool__", "item", "tolist", "nonzero")

    def __enter__(self):
        self.saved = {n: getattr(torch.Tensor, n) for n in self.NAMES}

        def boom(*_a, **_k):
            raise AssertionError("host sync")

        for n in self.NAMES:
            setattr(torch.Tensor, n, boom)
        return self

    def __exit__(self, *exc):
        for n, f in self.saved.items():
            setattr(torch.Tensor, n, f)


def _old_tab_forward(self, x):
    # catanet TAB.forward before MX-11 (python branch on the device bool `initted`)
    from einops import rearrange

    from neosr.archs.catanet_arch import center_iter

    _, _, h, w = x.shape
    x = rearrange(x, "b c h w->b (h w) c")
    residual = x
    x = self.norm(x)
    B, N, _ = x.shape
    idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)
    if not self.initted:
        pad_n = self.num_tokens - N % self.num_tokens
        paded_x = torch.cat((x, torch.flip(x[:, N - pad_n : N, :], dims=[-2])), dim=-2)
        x_means = torch.mean(
            rearrange(paded_x, "b (cnt n) c->cnt (b n) c", cnt=self.num_tokens), dim=-2
        ).detach()
    else:
        x_means = self.means.detach()
    if self.training:
        with torch.no_grad():
            for _ in range(self.n_iter - 1):
                x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
    k_global, v_global, x_means = self.irca_attn(x, x_means)
    with torch.no_grad():
        x_scores = torch.einsum(
            "b i c,j c->b i j", F.normalize(x, dim=-1), F.normalize(x_means, dim=-1)
        )
        idx = torch.argsort(torch.argmax(x_scores, dim=-1), dim=-1)
        idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)
    y = self.iasa_attn(x, idx_last, k_global, v_global)
    y = self.conv1x1(rearrange(y, "b (h w) c->b c h w", h=h).contiguous())
    x = residual + rearrange(y, "b c h w->b (h w) c")
    x = self.mlp(x, x_size=(h, w)) + x
    if self.training:
        with torch.no_grad():
            if not self.initted:
                self.means.data.copy_(x_means)
                self.initted.data.copy_(torch.tensor(True))  # noqa: FBT003
            else:
                self.means.data.mul_(self.ema_decay).add_(x_means, alpha=1 - self.ema_decay)
    return rearrange(x, "b (h w) c->b c h w", h=h)


def test_catanet_initted_bitwise_and_sync_free(monkeypatch) -> None:
    import copy

    from neosr.archs.catanet_arch import TAB

    new = _build("catanet", scale=2)
    old = copy.deepcopy(new)
    new_fwd = TAB.forward
    for step in range(3):
        x = torch.rand(2, 3, 24, 24, generator=torch.Generator().manual_seed(step))
        for mode in (True, True, False):
            new.train(mode), old.train(mode)
            monkeypatch.setattr(TAB, "forward", _old_tab_forward)
            torch.manual_seed(step)
            ya = old(x)
            monkeypatch.setattr(TAB, "forward", new_fwd)
            torch.manual_seed(step)
            with _NoHostSync():
                yb = new(x)
            assert torch.equal(ya, yb)
    for a, b in zip(old.buffers(), new.buffers(), strict=True):
        assert torch.equal(a, b)


# ---------------------------------------------------------------- MX-03/12/14/17/18 ASID


def _old_asid_attn(grid: bool):
    # Channel_Attention(_grid).forward before MX-03: no_grad SDPA with its default scale
    from einops import rearrange

    src, dst = (
        ("b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)", "b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)")
        if grid
        else ("b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)", "b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)")
    )

    def forward(self, x):
        _b, _c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = (rearrange(t, src, ph=self.ps, pw=self.ps, head=self.heads) for t in qkv)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        if self.flash_attn:
            with torch.no_grad():
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1) @ v
        out = rearrange(out, dst, h=h // self.ps, w=w // self.ps, ph=self.ps, pw=self.ps, head=self.heads)
        return self.project_out(out)

    return forward


def test_asid_legacy_checkpoint_keeps_its_forward(monkeypatch) -> None:
    from neosr.archs.asid_arch import Channel_Attention, Channel_Attention_grid

    old = _build("asid").eval()
    sd = old.state_dict()  # flash default: every temperature is exactly 1 (it never trained)
    x = torch.rand(1, 3, 64, 64)  # the training patch (= 8 * window) where grid scales agree
    with monkeypatch.context() as m:
        m.setattr(Channel_Attention, "forward", _old_asid_attn(grid=False))
        m.setattr(Channel_Attention_grid, "forward", _old_asid_attn(grid=True))
        with torch.no_grad():
            ref = old(x)
    new = _build("asid", seed=1).eval()
    new.load_state_dict(sd, strict=True)
    assert all(
        torch.equal(m.temperature, torch.full_like(m.temperature, 1 / 8))
        for m in new.modules()
        if hasattr(m, "temperature")
    )
    assert sd["block0.residual_layer.0.BlockC.cttn1.temperature"].eq(1).all()  # caller's dict intact
    with torch.no_grad():
        torch.testing.assert_close(new(x), ref, rtol=1e-5, atol=1e-5)
    # learned (!= 1) or flash_attn=False checkpoints load unchanged
    manual = _build("asid", flash_attn=False)
    manual.load_state_dict(sd)
    assert all(m.temperature.eq(1).all() for m in manual.modules() if hasattr(m, "temperature"))


@pytest.mark.parametrize("cls_name", ["Channel_Attention", "Channel_Attention_grid"])
@pytest.mark.parametrize("size", [64, 128])
def test_asid_sdpa_matches_manual_with_grads(cls_name: str, size: int) -> None:
    from neosr.archs import asid_arch

    cls = getattr(asid_arch, cls_name)
    torch.manual_seed(0)
    a = cls(16, heads=2, window_size=8, flash_attn=True)
    b = cls(16, heads=2, window_size=8, flash_attn=False)
    with torch.no_grad():
        a.temperature.uniform_(0.5, 4)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 16, size, size)
    ya, yb = a(x), b(x)
    torch.testing.assert_close(ya, yb, rtol=1e-5, atol=1e-5)  # one equation, any input size
    g = torch.randn_like(ya)
    (ya * g).sum().backward()
    (yb * g).sum().backward()
    _assert_grads_close(a, b)


def test_asid_layernorm2d_saved_tensors_and_grads() -> None:
    from neosr.archs.asid_arch import LayerNorm2d

    torch.manual_seed(0)
    m = LayerNorm2d(6)
    with torch.no_grad():
        m.weight.uniform_(0.5, 1.5), m.bias.normal_()
    x = torch.randn(2, 6, 5, 7, requires_grad=True)
    g = torch.randn(2, 6, 5, 7)
    gx, gw, gb = torch.autograd.grad((m(x) * g).sum(), (x, m.weight, m.bias))
    ref = F.layer_norm(x.permute(0, 2, 3, 1), (6,), m.weight, m.bias, 1e-6).permute(0, 3, 1, 2)
    rx, rw, rb = torch.autograd.grad((ref * g).sum(), (x, m.weight, m.bias))
    for a, b in ((gx, rx), (gw, rw), (gb, rb)):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    explain = torch._dynamo.explain(m)(x)
    assert explain.graph_break_count == 0


def test_asid_hygiene() -> None:
    from neosr.archs.asid_arch import Dropsample

    m = Dropsample(0.5).train()
    y = m(torch.ones(64, 2, 3, 3))
    assert set(y.unique().tolist()) <= {0.0, 2.0}
    with pytest.raises(ValueError, match="block_num"):
        _build("asid", block_num=2)
    net = _build("asid").eval()
    with torch.no_grad():
        for h, w in ((1, 1), (7, 5), (8, 8), (9, 12)):
            assert net(torch.rand(1, 3, h, w)).shape == (1, 3, 4 * h, 4 * w)

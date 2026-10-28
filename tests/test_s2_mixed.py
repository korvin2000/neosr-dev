"""Stage-2 lane I6 (mixed archs, MX-01..MX-18): parity against the replaced code, gradient census, sizes."""

import time

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from neosr.archs import build_network
from neosr.archs import esc_arch
from neosr.models.checkpoint import cpu_copy

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _build(name: str, scale: int = 4, seed: int = 0, **kw) -> nn.Module:
    torch.manual_seed(seed)
    return build_network({"type": name, **kw}, scale=scale)


def _assert_grads_close(a: nn.Module, b: nn.Module, rtol: float = 1e-4) -> None:
    # same parameters, same loss: gradients agree up to accumulation-order rounding
    for (n, p), q in zip(a.named_parameters(), b.parameters(), strict=True):
        if not p.requires_grad:
            continue
        assert p.grad is not None, n
        assert q.grad is not None, n
        torch.testing.assert_close(
            p.grad, q.grad, rtol=rtol, atol=rtol * q.grad.abs().max().item() + 1e-7, msg=n
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
    sd = cpu_copy(old.state_dict())  # a weight file; flash default: every t is exactly 1
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


# ---------------------------------------------------------------- MX-05/14 DITN


def _old_ditn_isa(self, x):
    # ISA.forward before MX-05: softmax(t q k^T) computed and discarded, 3-D SDPA default scale
    b, c, h, w = x.shape
    q, k, v = (t.transpose(-1, -2) for t in self.qkv(x.view(b, c, -1).transpose(-1, -2)).chunk(3, dim=-1))
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    return self.project_out(F.scaled_dot_product_attention(q, k, v).view(b, c, h, w))


def _spandrel_ditn_isa(self, x):
    # canonical DITN_Real ISA (spandrel / upstream "original" line)
    b, c, h, w = x.shape
    q, k, v = (t.transpose(-1, -2) for t in self.qkv(x.view(b, c, -1).transpose(-1, -2)).chunk(3, dim=-1))
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
    return self.project_out((attn @ v).view(b, c, h, w))


def test_ditn_matches_spandrel_and_keeps_legacy_forward(monkeypatch) -> None:
    from neosr.archs.ditn_arch import ISA

    old = _build("ditn").eval()
    sd = cpu_copy(old.state_dict())  # as read from a weight file (no module metadata)
    x = torch.rand(2, 3, 24, 40)
    with torch.no_grad(), monkeypatch.context() as m:
        m.setattr(ISA, "forward", _old_ditn_isa)
        ref_old = old(x)
    new = _build("ditn", seed=1).eval()
    new.load_state_dict(sd)  # t == 1 exactly -> 1/patch_size: the old forward
    with torch.no_grad():
        torch.testing.assert_close(new(x), ref_old, rtol=1e-5, atol=1e-5)
    # a trained temperature: the new forward is the canonical equation, gradients included
    torch.manual_seed(2)
    for mm in new.modules():
        if isinstance(mm, ISA):
            mm.temperature.data.uniform_(0.5, 3)
    canon = _build("ditn", seed=1).train()
    canon.load_state_dict(new.state_dict())
    new.train()
    y = new(x)
    with monkeypatch.context() as m:
        m.setattr(ISA, "forward", _spandrel_ditn_isa)
        y_ref = canon(x)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)
    g = torch.randn_like(y)
    (y * g).sum().backward()
    (y_ref * g).sum().backward()
    _assert_grads_close(new, canon)


@pytest.mark.parametrize("name", ["asid", "ditn"])
def test_legacy_temperature_remap_only_for_weight_files(name: str) -> None:
    import copy


    net = _build(name)
    ema = copy.deepcopy(net)
    temps = [m.temperature for m in ema.modules() if hasattr(m, "temperature")]
    ema.load_state_dict(net.state_dict())  # the EMA's first copy: module version 2, no remap
    assert temps and all(t.eq(1).all() for t in temps)
    ema.load_state_dict(cpu_copy(net.state_dict()))  # a weight file (no metadata): legacy
    assert all(t.eq(1 / 8).all() for t in temps)


# ---------------------------------------------------------------- MX-04/06 MicroSR


def _old_microsr_block(self, x, x_size, masks=None):
    # SwinTransformerBlock.forward before MX-06: a CPU-built mask per call, zero masks included
    from neosr.archs.microsr_arch import window_partition, window_reverse

    H, W = x_size
    B, _L, C = x.shape
    shortcut = x
    x = self.norm1(x).view(B, H, W, C)
    s = self.shift_size
    shifted_x = torch.roll(x, shifts=(-s, -s), dims=(1, 2)) if s > 0 else x
    x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size**2, C)
    if self.input_resolution == x_size:
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
    else:
        attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)
    x = torch.roll(shifted_x, shifts=(s, s), dims=(1, 2)) if s > 0 else shifted_x
    x = shortcut + self.drop_path(x.view(B, H * W, C))
    return x + self.drop_path(self.mlp(self.norm2(x)))


@pytest.mark.parametrize("patch_size", [32, 1])  # default 2x2 windows / the real 8x8 shifted ones
def test_microsr_masks_bitwise(monkeypatch, patch_size: int) -> None:
    from neosr.archs.microsr_arch import SwinTransformerBlock

    net = _build("microsr_light", patch_size=patch_size).eval()
    x = torch.rand(1, 3, 16, 24)  # aligned, != img_size: masks are rebuilt per forward
    built = []
    calc = SwinTransformerBlock.calculate_mask
    with torch.no_grad():
        with _NoHostSync(), monkeypatch.context() as m:
            m.setattr(SwinTransformerBlock, "calculate_mask", lambda s, *a: built.append(a) or calc(s, *a))
            y = net(x)
        assert len(built) == (1 if patch_size == 1 else 0)  # once per forward; none unshifted
        monkeypatch.setattr(SwinTransformerBlock, "forward", _old_microsr_block)
        assert torch.equal(y, net(x))


# ---------------------------------------------------------------- MX-04/07/08/09/10 HMA


def _old_hma_wa(self, x, rpi, mask=None):
    # hma WindowAttention.forward before MX-08 (explicit softmax probabilities)
    b_, n, c = x.shape
    qkv = x.reshape(b_, n, 3, self.num_heads, c // self.num_heads // 3).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q * self.scale) @ k.transpose(-2, -1)
    rpb = self.relative_position_bias_table[rpi.view(-1)].view(n, n, -1).permute(2, 0, 1)
    attn = attn + rpb.unsqueeze(0)
    if mask is not None:
        nw = mask.shape[0]
        attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, n, n)
    attn = self.attn_drop(attn.softmax(-1))
    return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(b_, n, c // 3)))


def _old_hma_affine(self, q, k, v, h, w):
    # hma AffineTransform.forward before MX-08/MX-09 (bias always computed, even when constant)
    attn = (q * self.scale) @ k.transpose(-2, -1)
    if self.position_bias:
        d = attn.device
        biases = torch.stack(
            torch.meshgrid(torch.arange(1 - h, h, device=d), torch.arange(1 - w, w, device=d), indexing="ij")
        )
        biases = biases.flatten(1).transpose(0, 1).float()
        coords = torch.stack(
            torch.meshgrid(torch.arange(h, device=d), torch.arange(w, device=d), indexing="ij")
        ).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += h - 1
        rel[:, :, 1] += w - 1
        rel[:, :, 0] *= 2 * w - 1
        pos = self.pos(biases)
        attn = attn + pos[rel.sum(-1).view(-1)].view(h * w, h * w, -1).permute(2, 0, 1).unsqueeze(0)
    return self.attn_drop(attn.softmax(-1)) @ v


def _hma_old(m):
    from neosr.archs.hma_arch import AffineTransform, WindowAttention

    m.setattr(WindowAttention, "forward", _old_hma_wa)
    m.setattr(AffineTransform, "forward", _old_hma_affine)


# reduced, same blocks; embed_dim 60 -> grid pos_dim 1 (the default), 96 -> pos_dim 3 (as hma_medium)
HMA_SMALL = {"depths": (2, 2), "num_heads": (6, 6), "drop_path_rate": 0.0}


@pytest.mark.parametrize("dim", [60, 96])
def test_hma_sdpa_parity_outputs_and_grads(monkeypatch, dim: int) -> None:
    from neosr.archs.hma_arch import AffineTransform

    new = _build("hma", embed_dim=dim, **HMA_SMALL).train()
    affine = [m for m in new.modules() if isinstance(m, AffineTransform)]
    assert all(m.pos_const == (dim == 60) for m in affine)
    if dim == 60:  # the skipped bias is a per-head constant: perturbing its weights changes nothing
        for m in affine:
            for p in m.pos.parameters():
                p.data.normal_()
    old = _build("hma", embed_dim=dim, **HMA_SMALL).train()
    old.load_state_dict(new.state_dict())
    x = torch.rand(2, 3, 32, 48)
    y = new(x)
    with monkeypatch.context() as m:
        _hma_old(m)
        y_old = old(x)
    torch.testing.assert_close(y, y_old, rtol=1e-4, atol=1e-5)
    g = torch.randn_like(y)
    (y * g).sum().backward()
    (y_old * g).sum().backward()
    _assert_grads_close(new, old)
    frozen = [n for n, p in new.named_parameters() if not p.requires_grad]
    assert all(".pos." in n for n in frozen)
    assert len(frozen) == (14 * len(affine) if dim == 60 else 0)  # frozen, keys kept


def test_hma_checkpoint_grad_parity() -> None:
    a = _build("hma", **HMA_SMALL).train()
    b = _build("hma", **HMA_SMALL, use_checkpoint=True).train()
    b.load_state_dict(a.state_dict())
    x = torch.rand(2, 3, 32, 32)
    a(x).square().mean().backward()
    b(x).square().mean().backward()
    _assert_grads_close(a, b, rtol=1e-5)


def test_hma_pad_is_identity_on_aligned_inputs(monkeypatch) -> None:
    net = _build("hma", **HMA_SMALL).eval()
    x = torch.rand(1, 3, 40, 24)
    with torch.no_grad():
        y = net(x)
        monkeypatch.setattr(type(net), "check_image_size", lambda self, t: t)
        assert torch.equal(y, net(x))


# ---------------------------------------------------------------- MX-02/04/07/08/10 MFGHMoE


def _ref_window_attn(q, k, v, bias, mask=None, scale=None, dropout_p=0.0):
    # the replaced attention core of HAB/OCAB (explicit fp32 probabilities)
    b_, h, n = q.shape[:3]
    attn = (q * scale) @ k.transpose(-2, -1) + bias.unsqueeze(0)
    if mask is not None:
        nw = mask.shape[0]
        attn = (attn.view(b_ // nw, nw, h, n, -1) + mask.unsqueeze(1).unsqueeze(0)).view(b_, h, n, -1)
    return F.dropout(attn.softmax(-1), dropout_p) @ v


def _old_dual_routing(self, x):
    # DualRouting.forward before MX-02: per-group boolean masks, constant one-hot
    from einops import rearrange

    B, C, H, W = x.shape
    x = rearrange(x, "b c h w -> (b h w) c")
    _, top = F.softmax(self.w_group_gate(x), dim=-1).topk(1, dim=-1)
    one_hot = torch.zeros(x.size(0), self.num_experts, device=x.device)
    for g in range(2):
        gm = (top == g).squeeze(-1)
        if gm.any():
            xg = x[gm] + self.group_embedding(torch.tensor(g, device=x.device)).expand(int(gm.sum()), -1)
            _, e = F.softmax(self.w_expert_gates[g](xg), dim=-1).topk(1, dim=-1)
            one_hot[gm, g * self.group_size + e.squeeze(-1)] = 1
    return rearrange(one_hot, "(b h w) n -> b h w n", b=B, h=H, w=W), None


def _old_hmoe(self, x, x_t):
    # HMoE.forward before MX-02: dense 16-expert stack times the one-hot
    from einops import rearrange

    B, _N, H, W = x.shape
    gate, _ = self.gate(x_t)
    outs = torch.stack([rearrange(e(x), "b c h w -> (b h w) c") for e in self.experts], dim=-1)
    out = (rearrange(gate, "b h w n -> (b h w) n").unsqueeze(-2) * outs).sum(-1)
    return rearrange(out, "(b h w) c -> b c h w", b=B, h=H, w=W)


def _mfghmoe_old(m):
    from neosr.archs import mfghmoe_arch as ma

    m.setattr(ma, "_window_sdpa", _ref_window_attn)
    m.setattr(ma.DualRouting, "forward", _old_dual_routing)
    m.setattr(ma.HMoE, "forward", _old_hmoe)


# reduced, same blocks: the default (6x6 HAB groups, 180 ch) is covered by the census below
MOE_SMALL = {"embed_dim": 48, "depths": (2, 2), "num_heads": (6, 6), "drop_path_rate": 0.0}


def test_mfghmoe_routing_matches_old_one_hot() -> None:
    from neosr.archs.mfghmoe_arch import DualRouting

    torch.manual_seed(0)
    r = DualRouting(64, 16)
    x = torch.randn(3, 64, 9, 11)
    idx, w = r(x)
    one_hot, _ = _old_dual_routing(r, x)
    assert torch.equal(one_hot.sum(-1), torch.ones(3, 9, 11))
    assert torch.equal(idx.squeeze(1), one_hot.argmax(-1))
    assert torch.equal(w, torch.ones_like(w))  # value-preserving straight-through weight
    assert w.requires_grad
    # d w / d p = 1, so the router receives the Switch-Transformer gradient <g, E_sel(x)>
    w.sum().backward()
    assert all(p.grad is not None for p in r.parameters())


def test_mfghmoe_parity_outputs_and_grads(monkeypatch) -> None:
    from neosr.archs.mfghmoe_arch import DualRouting

    new = _build("mfghmoe", **MOE_SMALL).train()
    old = _build("mfghmoe", **MOE_SMALL).train()
    old.load_state_dict(new.state_dict())
    x = torch.rand(2, 3, 32, 32)
    with _NoHostSync():
        new(x).sum().backward()  # the router and the routing-feature branch learn now
    reached = {n for n, p in new.named_parameters() if p.grad is not None}
    new.zero_grad(set_to_none=True)
    route = DualRouting.forward
    with monkeypatch.context() as m:  # without the straight-through gradient: the old backward
        m.setattr(DualRouting, "forward", lambda self, t: (lambda r: (r[0], r[1].detach()))(route(self, t)))
        y = new(x)
    with monkeypatch.context() as m:
        _mfghmoe_old(m)
        y_old = old(x)
    torch.testing.assert_close(y, y_old, rtol=1e-4, atol=1e-5)  # same routing, same forward
    g = torch.randn_like(y)
    (y * g).sum().backward()
    (y_old * g).sum().backward()
    assert sum(q.grad is None for q in old.parameters()) == 7 + 4 * 2 + 2  # router, mfg convs, aggr
    for (n, p), q in zip(new.named_parameters(), old.parameters(), strict=True):
        assert n in reached
        if q.grad is None:  # dead before MX-02: router, mfg conv stacks, aggr_conv
            assert n.startswith(("upsample.0.gate.", "mfg.", "aggr_conv.")), n
            continue
        torch.testing.assert_close(
            p.grad, q.grad, rtol=1e-4, atol=1e-4 * q.grad.abs().max().item() + 1e-7, msg=n
        )


def test_mfghmoe_checkpoint_grad_parity() -> None:
    a = _build("mfghmoe", **MOE_SMALL).train()
    b = _build("mfghmoe", **MOE_SMALL, use_checkpoint=True).train()
    b.load_state_dict(a.state_dict())
    x = torch.rand(1, 3, 32, 32)
    a(x).square().mean().backward()
    b(x).square().mean().backward()
    _assert_grads_close(a, b, rtol=1e-5)


# ---------------------------------------------------------------- MX-08/17 GRFormer


def _old_grsa(self, x, mask=None):
    # GRSA.forward before MX-08: explicit cosine-attention probabilities
    import math

    b_, n, c = x.shape
    x = x.reshape(b_, n, 2, c // 2).permute(2, 0, 1, 3).contiguous()

    def heads(t):
        return t.permute(1, 2, 0, 3).flatten(2).reshape(b_, n, self.num_heads, -1).permute(0, 2, 1, 3)

    k = heads(torch.stack((x[0] + self.k1(x[0]), x[1] + self.k2(x[1]))))
    q = heads(torch.stack((x[0] + self.q1(x[0]), x[1] + self.q2(x[1]))))
    v = heads(torch.stack((x[0] + self.v1(x[0]), x[1] + self.v2(x[1]))))
    attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
    attn = attn * torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
    table = self.ESRPB_MLP(self.relative_position_bias_table).view(-1, self.num_heads)
    rpb = table[self.relative_position_index.view(-1)].view(n, n, -1).permute(2, 0, 1)
    attn = attn + 16 * torch.sigmoid(rpb).unsqueeze(0)
    if mask is not None:
        nw = mask.shape[0]
        attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, n, n)
    x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(b_, n, c)
    x = x.reshape(b_, n, 2, c // 2).permute(2, 0, 1, 3)
    return torch.stack((self.proj1(x[0]), self.proj2(x[1]))).permute(1, 2, 0, 3).reshape(b_, n, c)


@pytest.mark.parametrize("size", [(64, 64), (32, 64)])  # buffer masks / masks rebuilt per forward
def test_grformer_sdpa_parity_outputs_and_grads(monkeypatch, size) -> None:
    from neosr.archs.grformer_arch import GRSA

    kw = {"depths": (4,), "num_heads": (3,), "drop_path_rate": 0.0}
    new = _build("grformer", **kw).train()
    old = _build("grformer", **kw).train()
    old.load_state_dict(new.state_dict())
    x = torch.rand(2, 3, *size)
    y = new(x)
    with monkeypatch.context() as m:
        m.setattr(GRSA, "forward", _old_grsa)
        y_old = old(x)
    torch.testing.assert_close(y, y_old, rtol=1e-4, atol=1e-5)
    g = torch.randn_like(y)
    (y * g).sum().backward()
    (y_old * g).sum().backward()
    _assert_grads_close(new, old)


# ---------------------------------------------------------------- MX-09/10 DCT


def _old_dct_wa(self, qkv, H, W, mask=None):
    # dct Window_Attention.forward before MX-09: the (constant) position bias always computed
    from neosr.archs.dct_arch import windows2img

    q, k, v = (self.im2win(t, H, W) for t in (qkv[0], qkv[1], qkv[2]))
    B, C = qkv[0].shape[0], qkv[0].shape[2]
    attn = (q * self.scale) @ k.transpose(-2, -1)
    if self.position_bias:
        pos = self.pos(self.rpe_biases)
        rpb = pos[self.relative_position_index.view(-1)].view(self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)
    N = attn.shape[3]
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = self.attn_drop(F.softmax(attn, dim=-1, dtype=attn.dtype))
    x = (attn @ v).transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
    return windows2img(x, self.H_sp, self.W_sp, H, W)


@pytest.mark.parametrize("size", [(64, 64), (32, 40)])  # img_size / off-size: masks per forward
def test_dct_constant_bias_skip_parity(monkeypatch, size) -> None:
    from neosr.archs.dct_arch import Adaptive_Window_Attention, Window_Attention

    new = _build("dct", drop_path_rate=0.0).train()
    wa = [m for m in new.modules() if isinstance(m, Window_Attention)]
    assert wa and all(m.pos_const for m in wa)  # the default: pos_dim 1 everywhere
    for m in wa:  # a per-head constant: perturbing its weights changes nothing
        for p in m.pos.parameters():
            p.data.normal_()
    old = _build("dct", drop_path_rate=0.0).train()
    old.load_state_dict(new.state_dict())
    built = []
    calc = Adaptive_Window_Attention.calculate_mask
    monkeypatch.setattr(
        Adaptive_Window_Attention, "calculate_mask", lambda s, *a: built.append(a) or calc(s, *a)
    )
    x = torch.rand(2, 3, *size)
    with _NoHostSync():
        y = new(x)
    assert len(built) == (0 if size == (64, 64) else 1)  # once per forward, on the input's device
    with monkeypatch.context() as m:
        m.setattr(Window_Attention, "forward", _old_dct_wa)
        y_old = old(x)
    torch.testing.assert_close(y, y_old, rtol=1e-4, atol=1e-5)
    g = torch.randn_like(y)
    (y * g).sum().backward()
    (y_old * g).sum().backward()
    _assert_grads_close(new, old)
    frozen = {n for n, p in new.named_parameters() if not p.requires_grad}
    assert frozen and all(".pos." in n for n in frozen)

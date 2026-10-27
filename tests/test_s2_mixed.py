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

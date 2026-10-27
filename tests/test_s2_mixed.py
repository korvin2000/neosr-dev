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

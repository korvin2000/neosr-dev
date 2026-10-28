"""Stage 2 review fixes (two Codex reviews of the merged branch): one regression test per finding."""

import torch

from neosr.losses.base import restoring_clamp
from neosr.losses.consistency_loss import consistency_loss


# ---------------------------------------------------------------------------- 1 consistency clamp
def test_restoring_clamp_with_bounds_gates_like_the_unit_clamp():
    x = torch.tensor([-0.5, -0.5, 0.0, 0.3, 0.3, 1.0, 1.5, 1.5], requires_grad=True)
    g = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    lo, hi = 1 / 255, 0.9
    y = restoring_clamp(x, lo, hi)
    assert torch.equal(y, x.detach().clamp(lo, hi))
    y.backward(g)
    assert x.grad.tolist() == [0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0]


def test_consistency_loss_passes_inward_gradients_of_out_of_range_outputs():
    torch.manual_seed(0)
    gt = torch.rand(1, 3, 32, 32) * 0.8 + 0.1
    raw = gt.clone()
    raw[..., :4, :4] = -0.3  # below range: descent must raise these
    raw[..., -4:, -4:] = 1.4  # above range: descent must lower these
    raw.requires_grad_()
    loss = consistency_loss(blur=False, cosim=False)
    loss(restoring_clamp(raw), gt).backward()
    g = raw.grad
    assert (g[..., :4, :4] <= 0).all() and g[..., :4, :4].any()
    assert (g[..., -4:, -4:] >= 0).all() and g[..., -4:, -4:].any()
    # in-range values: unchanged from the legacy hard clamp
    ref = gt.clone().requires_grad_()
    loss(ref.clamp(1 / 255, 1), gt.flip(-1)).backward()
    new = gt.clone().requires_grad_()
    loss(new, gt.flip(-1)).backward()
    assert torch.equal(ref.grad, new.grad)

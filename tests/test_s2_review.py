"""Stage 2 review fixes (two Codex reviews of the merged branch): one regression test per finding."""

import logging

import pytest
import torch

from neosr.config import ConfigError, load_config, validate_train
from neosr.losses.base import restoring_clamp
from neosr.losses.consistency_loss import consistency_loss
from neosr.optimizers import adamw_sf, adan, adan_sf
from neosr.optimizers.adan import clip_grad_norm_
from neosr.utils import get_root_logger
from tests.test_config import BASE, HEAD, write
from tests.test_model import batches, build, make_opt, run


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


# ---------------------------------------------------------------------------- 3 adan max_grad_norm
@pytest.mark.parametrize("make", [adan, adan_sf], ids=["adan", "adan_sf"])
def test_adan_clips_by_the_group_max_grad_norm(make):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16))
    q = torch.nn.Parameter(p.detach().clone())
    resumed = make([p], lr=1e-2, max_grad_norm=0.0)
    resumed.param_groups[0]["max_grad_norm"] = 0.5  # what a resume restores over the config
    reference = make([q], lr=1e-2, max_grad_norm=0.5)
    for i in range(3):
        g = torch.randn(16, generator=torch.Generator().manual_seed(i)) * 10
        p.grad, q.grad = g.clone(), g.clone()
        resumed.step()
        reference.step()
    assert torch.equal(p, q)
    for k, v in reference.state[q].items():
        assert torch.equal(resumed.state[p][k], v), k


def test_adan_clip_norm_spans_groups_and_skips_unclipped_groups():
    a, b = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
    opt = adan([{"params": [a], "max_grad_norm": 1.0}, {"params": [b]}], lr=0.0)
    a.grad, b.grad = torch.full((4,), 3.0), torch.full((4,), 4.0)
    clip_grad_norm_(opt.param_groups)
    norm = torch.tensor(10.0)  # sqrt(4 * 9 + 4 * 16): both groups count
    torch.testing.assert_close(a.grad, torch.full((4,), 3.0) / (norm + 1e-8))
    assert torch.equal(b.grad, torch.full((4,), 4.0))


# ---------------------------------------------------------------------------- 10 adamw_sf version
@pytest.mark.parametrize("foreach", [True, False])
def test_adamw_sf_step_bumps_the_parameter_version(foreach):
    p = torch.nn.Parameter(torch.ones(4))
    opt = adamw_sf([p], lr=1e-2, foreach=foreach)
    opt.train()
    p.grad = torch.ones(4)
    v = p._version
    opt.step()
    assert p._version > v
    v = p._version
    opt.eval()
    assert p._version > v


# ---------------------------------------------------------------------------- 4 run record
def test_resume_warns_about_global_objective_keys(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(get_root_logger(), "propagate", True)  # caplog listens on the root logger
    m = build(make_opt(tmp_path))
    run(m, batches(1))
    m.save()
    opt = make_opt(tmp_path)
    opt["path"]["resume_state"] = tmp_path / "training_states" / "1.state"
    opt["train"]["match_lq_colors"] = True
    with caplog.at_level(logging.WARNING, logger="neosr"):
        build(opt)
    assert "train.match_lq_colors: None -> True" in caplog.text
    assert "wavelet" not in caplog.text  # unset before and after: not a change


# ---------------------------------------------------------------------------- 5 mean/std
def test_validate_train_rejects_identity_normalisation_like_the_datasets(tmp_path):
    body = BASE.replace("batch_size = 2", "batch_size = 2\nmean = [0, 0, 0]\nstd = [1, 1, 1]", 1)
    with pytest.raises(ConfigError, match=r"\[datasets.train\] 'mean' is not supported"):
        validate_train(load_config(write(tmp_path, body, HEAD), is_train=True, root=tmp_path))

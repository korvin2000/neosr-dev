"""Stage 3 A4: the generic per-term schedule (start_iter / ramp_iters / end_iter on Loss), hinge GAN and
psnr_loss. CPU."""

import math

import pytest
import torch
from torch.nn import functional as F

from neosr.config import ConfigError
from neosr.losses import LOSSES, build_losses
from neosr.losses.gan_loss import gan_loss
from neosr.models.phase import Phase
from tests.test_gan import Critic, Forbidden, images
from tests.test_gan import ctx as gan_ctx

CPU = torch.device("cpu")


def _terms(**opts):
    train = {"pixel_opt": {"type": "L1Loss", "loss_weight": 1.0}} | opts
    return build_losses(train, scale=2, device=CPU, has_d="gan_opt" in opts)


# ------------------------------------------------------------------ the generic schedule
def test_any_term_takes_the_schedule_from_its_table():
    (pix, l1), _ = _terms(
        l1_opt={
            "type": "L1Loss",
            "loss_weight": 2.0,
            "start_iter": 2,
            "ramp_iters": 4,
            "end_iter": 8,
        }
    )
    assert [l1.active(it) for it in range(1, 10)] == [False] * 2 + [True] * 6 + [False]
    assert all(pix.active(it) for it in (1, 9, 10**6))  # the default: always on
    sr, gt = images(), images(seed=1)
    full = 2.0 * F.l1_loss(sr, gt)
    for it, share in ((3, 0.25), (5, 0.75), (6, 1.0), (8, 1.0)):
        value = l1.compute(gan_ctx(sr, gt, net_d=None, net_d_adv=None, it=it))
        if share == 1.0:
            assert torch.equal(value, full)  # full weight: bitwise unscaled
        else:
            torch.testing.assert_close(value, F.l1_loss(sr, gt) * share * 2.0)


@pytest.mark.parametrize(
    ("sched", "match"),
    [
        ({"start_iter": -1}, "start_iter=-1"),
        ({"ramp_iters": 1.5}, "ramp_iters=1.5"),
        ({"start_iter": 5, "end_iter": 5}, "end_iter > start_iter"),
        ({"end_iter": True}, "end_iter=True"),
    ],
)
def test_bad_schedules_are_config_errors(sched, match):
    with pytest.raises(ConfigError, match=rf"\[train\.l1_opt\] .*{match}"):
        _terms(l1_opt={"type": "L1Loss"} | sched)


def test_a_phase_whose_terms_all_ended_idles():
    (pix,), _ = build_losses(
        {"pixel_opt": {"type": "L1Loss", "end_iter": 3}},
        scale=2,
        device=CPU,
        has_d=False,
    )
    p = torch.nn.Parameter(torch.zeros(1))
    phase = Phase("g", [p], torch.optim.AdamW([p]), [pix], amp_dtype=torch.float16,
                  use_amp=False, accumulate=1, grad_clip=0.0)  # fmt: skip
    assert [phase.active(it) for it in (1, 3, 4)] == [True, True, False]


def test_gan_end_iter_stops_g_and_d_together():
    _, (d_term,) = _terms(gan_opt={"type": "gan_loss", "start_iter": 2, "end_iter": 5})
    assert [d_term.active(it) for it in range(1, 8)] == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert d_term.g.start_iter == 2  # the builder schedules G's term; D follows it


def test_gan_ctor_no_longer_takes_the_schedule():
    with pytest.raises(TypeError, match="start_iter"):
        gan_loss(start_iter=3)


# ------------------------------------------------------------------ hinge
def test_hinge_terms():
    critic = Critic().eval()
    sr, gt = images(), images(seed=1)
    g_term, d_term = gan_loss(gan_type="hinge").expand()
    g_term.loss_weight = 1.0
    g = g_term.compute(gan_ctx(sr, gt, net_d=Forbidden(), net_d_adv=critic))
    with torch.no_grad():
        real, fake = critic(gt), critic(sr)
    torch.testing.assert_close(g, -fake.mean())
    c = gan_ctx(sr, gt, net_d=critic, net_d_adv=Forbidden())
    d = d_term.compute(c)
    l_real, l_fake = F.relu(1 - real).mean(), F.relu(1 + fake).mean()
    torch.testing.assert_close(d, l_real + l_fake)
    torch.testing.assert_close(c.logs["l_d_real"], l_real)
    torch.testing.assert_close(c.logs["l_d_fake"], l_fake)


def test_hinge_takes_no_labels():
    with pytest.raises(ValueError, match="'hinge' takes no labels"):
        gan_loss(gan_type="hinge", real_label_val=0.9)


# ------------------------------------------------------------------ psnr_loss
def test_psnr_loss_is_minus_the_mean_psnr_in_db():
    assert LOSSES["psnr_loss"] == ("basic_loss", None)
    (term,), _ = build_losses(
        {"psnr_opt": {"type": "psnr_loss", "loss_weight": 0.5}},
        scale=2,
        device=CPU,
        has_d=False,
    )
    sr, gt = images(), images(seed=1)
    psnr = [
        -10 * math.log10(F.mse_loss(a, b).item()) for a, b in zip(sr, gt, strict=True)
    ]
    value = term.compute(gan_ctx(sr, gt, net_d=None, net_d_adv=None))
    assert value.dtype == torch.float32
    assert value.item() == pytest.approx(-0.5 * sum(psnr) / len(psnr), rel=1e-5)
    assert term.fp32  # outside autocast, fp32 inputs


def test_psnr_loss_reductions_and_options():
    from neosr.losses.basic_loss import psnr_loss  # noqa: PLC0415

    sr, gt = images(), images(seed=1)
    per = [
        10 * torch.log10(F.mse_loss(a, b) + 1e-8) for a, b in zip(sr, gt, strict=True)
    ]
    torch.testing.assert_close(psnr_loss(reduction="sum")(sr, gt), sum(per))
    assert psnr_loss()(gt, gt).item() == pytest.approx(-80.0)  # the eps floor (1e-8)
    for bad in ({"reduction": "none"}, {"eps": 0}, {"eps": math.inf}, {"eps": True}):
        with pytest.raises(ValueError, match="psnr_loss"):
            psnr_loss(**bad)


def test_falsy_non_numeric_end_iter_is_a_config_error():
    for bad in ("", []):
        with pytest.raises(ConfigError, match=r"\[train\.l1_opt\]"):
            _terms(l1_opt={"type": "L1Loss", "end_iter": bad})


def test_hinge_d_with_r1_r2_has_finite_penalty_gradients():
    critic = Critic()
    sr, gt = images(), images(seed=1)
    _, d_term = gan_loss(
        gan_type="hinge", r1_gamma=1.0, r2_gamma=0.5, reg_every=2
    ).expand()
    c = gan_ctx(
        sr, gt, net_d=critic, net_d_adv=Forbidden(), it=2
    )  # a regularised update
    value = d_term.compute(c)
    value.backward()
    assert torch.isfinite(value)
    assert c.logs["l_d_reg"] > 0
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in critic.parameters()
    )


# ------------------------------------------------------------------ resume across end_iter
@pytest.mark.parametrize("at", [2, 3])
def test_resume_is_exact_across_an_end_iter(tmp_path, at):
    """The GAN pair stops after update 3: resuming just before or at the boundary continues the
    uninterrupted run exactly (weights, D idle afterwards, optimizer moments, window logs)."""
    from tests.test_model import batches, build, clone, opt_tensors, same  # noqa: PLC0415
    from tests.test_s2_review import _gan_opt, _updates  # noqa: PLC0415

    def opt(tmp):
        o = _gan_opt(tmp)
        o["train"]["gan_opt"] |= {"start_iter": 0, "end_iter": 3}
        return o

    data = batches(12)
    a = build(opt(tmp_path / "a"))
    logs_a = _updates(a, data, 6)
    assert "l_g_gan" in logs_a[2] and "l_g_gan" not in logs_a[3]  # the boundary
    b = build(opt(tmp_path / "b"))
    _updates(b, data, at)
    b.save()
    o = opt(tmp_path / "b")
    o["path"]["resume_state"] = tmp_path / "b" / "training_states" / f"{at}.state"
    r = build(o, seed=5)
    r.restore_rng()
    assert _updates(r, data, 6 - at) == logs_a[at:]
    for x, y in ((a.net_g, r.net_g), (a.net_d, r.net_d)):
        assert same(clone(x), clone(y))
    for n in ("net_g", "net_d"):
        assert same(opt_tensors(a.phases[n].opt), opt_tensors(r.phases[n].opt))

"""GAN health log (DR-02, A-04 rec. 1): G's adversarial input-gradient norm and its ratio to the other
terms, D's sign accuracy and mean |logit|, measured on log updates only and without touching the
training numerics."""

import copy
import math
import warnings

import pytest
import torch
from torch.nn import functional as F

from neosr.losses.base import Context, restoring_clamp
from neosr.losses.gan_loss import gan_loss
from neosr.models import image as image_mod
from neosr.models.phase import LossMeter
from neosr.runtime import Runtime
from tests.test_model import batches, build, clone, make_opt, run, same

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
HEALTH = {"grad_sr_gan", "grad_sr_ratio", "out_d_acc", "out_d_abs"}
PATCH = {"type": "patchgan", "num_feat": 4, "num_layers": 1}
INVERTED = {"real_label_val": 0.0, "fake_label_val": 1.0}


def _opt(tmp, *, print_freq, d=None, accumulate=1, train=None, top=None, **gan):
    opt = make_opt(tmp, d=True, ema=0.9, accumulate=accumulate) | (top or {})
    opt["logger"] = {"print_freq": print_freq}
    opt["train"] |= train or {}
    opt["train"]["gan_opt"] |= gan
    if d is not None:
        opt["network_d"] = d
    return opt


def _opt_state(m):
    return [
        v.clone()
        for p in m.phases.values()
        for s in p.opt.state_dict()["state"].values()
        for v in s.values()
        if torch.is_tensor(v)
    ]


MSE = {"mse_opt": {"type": "MSELoss", "loss_weight": 0.5}}  # a third consumer of sr


@pytest.mark.parametrize(
    ("d", "gan", "accumulate", "train", "top"),
    [
        # None: the unet D with spectral norm, whose u/v buffers move on every train forward
        (None, {}, 1, None, None),
        (None, {}, 2, MSE, None),
        (None, {"gan_type": "rpgan", "r1_gamma": 1.0, "reg_every": 1}, 1, None, None),
        (
            PATCH | {"norm_type": "batch", "use_sn": False},
            {"gan_type": "hinge"},
            1,
            MSE,
            None,
        ),
        (
            PATCH | {"use_sn": False},
            {"gan_type": "mse", "ramp_iters": 3},
            1,
            None,
            None,
        ),
        # a compiled SN D: the hooks ride the one compiled backward (no retain_graph)
        (None, {}, 1, MSE, {"compile": True}),
    ],
)
def test_log_path_leaves_training_bitwise_unchanged(
    tmp_path, d, gan, accumulate, train, top
):
    bs = batches(4 * accumulate)
    on, off = (
        build(
            _opt(
                tmp_path,
                print_freq=f,
                d=d,
                accumulate=accumulate,
                train=train,
                top=top,
                **gan,
            )
        )
        for f in (1, 1000)
    )
    assert hasattr(on.net_d_adv, "_orig_mod") == bool(top)  # compiled D
    logs = {}
    for name, m in (("on", on), ("off", off)):
        torch.manual_seed(1)
        run(m, bs)
        logs[name] = m.get_current_log()
        logs[name, "rng"] = torch.get_rng_state()
    assert logs["on"].keys() >= HEALTH
    assert not HEALTH & logs["off"].keys()
    assert torch.equal(logs["on", "rng"], logs["off", "rng"])
    for net in ("net_g", "net_d"):  # parameters and buffers (BN statistics, SN u/v)
        assert same(clone(getattr(on, net)), clone(getattr(off, net)))
    assert same(clone(on.ema.module), clone(off.ema.module))
    assert same(_opt_state(on), _opt_state(off))
    shared = logs["off"].keys()  # the existing keys are unchanged too
    assert {k: logs["on"][k] for k in shared} == logs["off"]


def test_measured_on_log_updates_only(tmp_path):
    m = build(_opt(tmp_path, print_freq=2))
    run(m, batches(1))
    assert not HEALTH & m.get_current_log().keys()
    run(m, batches(1, seed=1))
    assert m.get_current_log().keys() >= HEALTH


def test_no_ratio_without_another_term(tmp_path):
    opt = _opt(tmp_path, print_freq=1, d=PATCH | {"use_sn": False})
    del opt["train"]["pixel_opt"]
    m = build(opt)
    run(m, batches(1))
    log = m.get_current_log()
    assert "grad_sr_gan" in log
    assert "grad_sr_ratio" not in log


@pytest.mark.parametrize(("accumulate", "scale"), [(1, None), (2, 2.0**10)])
def test_values_match_a_direct_computation(tmp_path, accumulate, scale):
    """A stateless D (no SN, no BN): its forwards before the update are the ones the update ran.
    `scale`: an fp16-style loss scale (a CPU GradScaler), divided out with 1/accumulate."""
    opt = _opt(
        tmp_path,
        print_freq=1,
        d=PATCH | {"use_sn": False},
        accumulate=accumulate,
        ramp_iters=4,
    )
    m = build(opt)
    if scale is not None:
        m.g.scaler = torch.amp.GradScaler("cpu", init_scale=scale)
    bs = batches(accumulate)
    g, d, pix = copy.deepcopy(m.net_g), copy.deepcopy(m.net_d), m.g.terms[0]
    run(m, bs)
    log = m.get_current_log()

    want = dict.fromkeys(HEALTH, 0.0)
    for b in bs:
        sr = restoring_clamp(g(b["lq"])).detach().requires_grad_()
        logits = d(sr)
        w = 0.1 * 1 / 4  # loss_weight x ramp at update 1 of 4: the term as applied
        adv = w * F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        (ga,) = torch.autograd.grad(adv, sr)
        (gr,) = torch.autograd.grad(pix.loss_weight * pix(sr, b["gt"]), sr)
        with torch.no_grad():
            real, fake = d(torch.cat([b["gt"], sr])).chunk(2)
        want["grad_sr_gan"] += ga.norm().item() / accumulate
        want["grad_sr_ratio"] += (ga.norm() / gr.norm()).item() / accumulate
        acc = torch.cat([real > 0, fake < 0]).float().mean()
        want["out_d_acc"] += acc.item() / accumulate
        want["out_d_abs"] += torch.cat([real, fake]).abs().mean().item() / accumulate
    for k, v in want.items():
        torch.testing.assert_close(log[k], v, msg=k)


def test_two_gan_slots_sum_into_the_adversarial_gradient(tmp_path):
    """Any slot may hold gan_loss: each slot's alias hook adds to the adversarial gradient."""
    opt = _opt(tmp_path, print_freq=1, d=PATCH | {"use_sn": False})
    opt["train"]["gan2_opt"] = {"type": "gan_loss", "loss_weight": 0.05}
    m = build(opt)
    b = batches(1)[0]
    g, d, pix = copy.deepcopy(m.net_g), copy.deepcopy(m.net_d), m.g.terms[0]
    run(m, [b])
    log = m.get_current_log()
    sr = restoring_clamp(g(b["lq"])).detach().requires_grad_()
    logits = d(sr)
    adv = 0.15 * F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    (ga,) = torch.autograd.grad(adv, sr)
    (gr,) = torch.autograd.grad(pix.loss_weight * pix(sr, b["gt"]), sr)
    torch.testing.assert_close(log["grad_sr_gan"], ga.norm().item())
    torch.testing.assert_close(log["grad_sr_ratio"], (ga.norm() / gr.norm()).item())


def test_health_values_are_diagnostics_never_fatal(tmp_path, monkeypatch):
    """An overflowing probe in an applied update: logged as inf/nan, training goes on (F3)."""
    monkeypatch.setattr(
        image_mod.GanHealth,
        "_adv",
        lambda self, grad: setattr(self, "grad", grad * math.inf),
    )
    m = build(_opt(tmp_path, print_freq=1))
    run(m, batches(1))
    assert m.g.skipped == 0
    log = m.get_current_log()  # does not raise
    assert log["grad_sr_gan"] == math.inf
    assert math.isnan(log["grad_sr_ratio"])
    assert not any(k.startswith("nonfinite") for k in log)


def test_loss_meter_notes_are_plain_means():
    meter = LossMeter()
    meter.note({"a": torch.tensor(1.0), "b": torch.tensor(math.nan)})
    meter.note({"a": torch.tensor(3.0), "b": torch.tensor(0.0)})
    meter.add({"l": torch.tensor(2.0)})
    out = meter.read()
    assert out["a"] == pytest.approx(2.0)
    assert math.isnan(out["b"])
    assert out["l"] == pytest.approx(2.0)
    assert meter.read() == {}


@pytest.mark.parametrize(
    ("gan_type", "labels", "real", "fake", "acc", "mean_abs"),
    [
        # logits, threshold 0: right 0.7 and -0.4
        ("bce", {}, [0.7, -0.2], [-0.4, 0.9], 0.5, (0.7 + 0.2 + 0.4 + 0.9) / 4),
        ("hinge", {}, [0.7, -0.2], [-0.4, 0.9], 0.5, (0.7 + 0.2 + 0.4 + 0.9) / 4),
        # regressed labels 1 / 0, threshold 0.5: z = 0.2, -0.3 | 0.1, -0.4
        ("mse", {}, [0.7, 0.2], [0.4, 0.9], 0.5, (0.2 + 0.3 + 0.1 + 0.4) / 4),
        ("huber", {}, [0.7, 0.2], [0.4, 0.9], 0.5, (0.2 + 0.3 + 0.1 + 0.4) / 4),
        # inverted labels: real low is right; z = 0.2, -0.3 | 0.1, -0.4
        ("mse", INVERTED, [0.3, 0.8], [0.6, 0.1], 0.5, (0.2 + 0.3 + 0.1 + 0.4) / 4),
        ("bce", INVERTED, [-0.7, 0.2], [0.4, -0.9], 0.5, (0.7 + 0.2 + 0.4 + 0.9) / 4),
        # paired real - fake: 1.1, -1.1
        ("rpgan", {}, [0.7, -0.2], [-0.4, 0.9], 0.5, 1.1),
    ],
)
def test_d_accuracy_per_gan_type(gan_type, labels, real, fake, acc, mean_abs):
    real, fake = torch.tensor(real), torch.tensor(fake)
    x = torch.zeros(1)
    ctx = Context(sr=x, gt=x, lq=x, it=1)
    d = gan_loss(gan_type=gan_type, **labels).expand()[1]
    d.adv_loss(real, fake, ctx)
    assert HEALTH.isdisjoint(ctx.logs)  # not a log update
    ctx.health = object()  # type: ignore[attr-defined]
    d.adv_loss(real, fake, ctx)
    torch.testing.assert_close(ctx.logs["out_d_acc"].item(), acc)
    torch.testing.assert_close(ctx.logs["out_d_abs"].item(), mean_abs)


@CUDA
@pytest.mark.parametrize("print_freq", [1, 1000])
def test_log_update_adds_no_host_sync(tmp_path, print_freq):
    opt = _opt(tmp_path, print_freq=print_freq, d=PATCH | {"norm_type": "batch"})
    opt |= {"use_amp": True, "bfloat16": True}
    m = build(opt, rt=Runtime(torch.device("cuda", 0), 0, 1, 0, False))
    bs = [{k: v.cuda() for k, v in b.items()} for b in batches(4)]
    run(m, bs[:3])
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            assert run(m, bs[3:]) == [True]
        finally:
            torch.cuda.set_sync_debug_mode(0)
    syncs = [w for w in caught if "called a synchronizing" in str(w.message)]
    assert len(syncs) == 2  # Phase.step's isfinite, G and D: as without the health log
    assert (m.get_current_log().keys() >= HEALTH) == (print_freq == 1)

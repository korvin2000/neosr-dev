"""Stage 2, trainer lane (docs/refactor/stage2/analysis/trainer.md, stability.md): clamp and clip
modes, the skip transaction, idle phases, resume reporting, config invariants, runtime flags."""

import copy
import logging
import math
import os
import time
import warnings
from pathlib import Path

import pytest
import torch
import torch.ao.quantization as tq
import torch.distributed as dist
from torch import nn
from torch.nn.utils import get_total_norm

from neosr import runtime
from neosr.config import ConfigError, load_config, validate_train
from neosr.models import image as image_mod
from neosr.models.base import base
from neosr.models.checkpoint import prune_exports
from neosr.models.image import Buffers, restoring_clamp
from neosr.models.phase import SPIKE_K, LossMeter, Phase, is_schedule_free
from neosr.optimizers import adan_sf
from neosr.runtime import Runtime
from neosr.utils.logger import DataWait, MessageLogger
from tests.test_config import BASE, HEAD, write
from tests.test_entry import data, logs, run_dir, states, toml, train_py  # noqa: F401
from tests.test_model import CPU, SF, batches, build, clone, loader, make_opt, run, same
from tests.test_runtime import fake_gpus, restore_torch_state  # noqa: F401

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class Log:
    def __init__(self) -> None:
        self.msgs: list[str] = []

    def warning(self, msg) -> None:
        self.msgs.append(msg)

    info = warning


# ---------------------------------------------------------------------------- T-01 clamp modes
def test_restoring_clamp_forward_and_backward_gating():
    x = torch.tensor([-0.5, -0.5, 0.3, 0.3, 1.5, 1.5], requires_grad=True)
    g = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    y = restoring_clamp(x)
    assert torch.equal(y, x.detach().clamp(0, 1))
    y.backward(g)
    # below 0: only g < 0 (descent raises x) passes; above 1: only g > 0; inside: everything
    assert x.grad.tolist() == [0.0, -1.0, 1.0, -1.0, 1.0, 0.0]


def test_all_below_range_output_receives_an_inward_gradient():
    raw = torch.full((1, 3, 4, 4), -2.0, requires_grad=True)
    gt = torch.full_like(raw, 0.5)
    (restoring_clamp(raw) - gt).abs().mean().backward()
    assert (raw.grad < 0).all()  # descent moves every pixel up, back into range
    hard = raw.detach().clone().requires_grad_()
    (hard.clamp(1 / 255, 1) - gt).abs().mean().backward()
    assert not hard.grad.any()  # the legacy clamp: no signal at all


def _sr_of_one_step(tmp, clamp, shift=0.0):
    opt = make_opt(tmp)
    if clamp is not None:
        opt["train"]["clamp"] = clamp
    m = build(opt)
    seen, net = {}, m.net_g
    m.net_g_fwd = lambda x: seen.setdefault("raw", net(x) + shift)
    orig = m.g.terms[0].compute
    m.g.terms[0].compute = lambda ctx: (
        seen.setdefault("sr", ctx.sr) is None or orig(ctx)
    )
    run(m, batches(1))
    return m, seen["raw"].detach(), seen["sr"].detach()


@pytest.mark.parametrize("clamp", [None, True, False, "restore"])
def test_clamp_modes_in_the_training_step(tmp_path, clamp):
    m, raw, sr = _sr_of_one_step(tmp_path, clamp)
    ref = {None: raw.clamp(1 / 255, 1), True: raw.clamp(1 / 255, 1), False: raw}
    assert torch.equal(
        sr, ref.get(clamp, raw.clamp(0, 1))
    )  # legacy default bit-identical


@pytest.mark.parametrize(("clamp", "learns"), [(True, False), ("restore", True)])
def test_out_of_range_generator_learns_only_with_restore(tmp_path, clamp, learns):
    m, raw, _ = _sr_of_one_step(tmp_path, clamp, shift=-10.0)
    assert (raw < 0).all()
    assert (m.get_current_log()["grad_norm_g"] > 0) == learns


# ---------------------------------------------------------------------------- T-02 clip modes
class Ctx:
    it = 1

    def __init__(self, x, factor) -> None:
        self.x, self.factor, self.logs = x, factor, {}

    def log(self, key, value) -> None:
        self.logs[key] = value.detach()


class Term:
    """Squared error times `factor`: the gradient norm scales linearly with it."""

    def __init__(self, net, active=lambda it: True) -> None:
        self.net, self._active = net, active

    def active(self, it) -> bool:
        return self._active(it)

    def compute(self, ctx):
        return ctx.factor * self.net(ctx.x).pow(2).mean()


def _phase(grad_clip, *, opt=None, net=None, **kw):
    torch.manual_seed(0)
    net = net or nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    opt = (opt or (lambda ps: torch.optim.SGD(ps, lr=0.0)))(list(net.parameters()))
    p = Phase(
        "g", list(net.parameters()), opt, [Term(net)], amp_dtype=torch.float32,
        use_amp=False, accumulate=1, grad_clip=grad_clip, **kw,
    )  # fmt: skip
    return net, opt, p


def _grads_seen(opt):
    seen, step = [], opt.step

    def record(*a, **k):
        seen.append(
            get_total_norm([p.grad for g in opt.param_groups for p in g["params"]])
        )
        return step(*a, **k)

    opt.step = record
    return seen


def _update(p, factor):
    p.backward(Ctx(torch.ones(2, 4), factor))
    return p.step()


def test_fixed_float_max_norm():
    _, opt, p = _phase(0.5)
    seen = _grads_seen(opt)
    assert _update(p, 100.0)
    assert seen[0].item() == pytest.approx(0.5, rel=1e-5)
    assert p.clipped.item() == 1.0


def test_spike_clip_ema_and_threshold():
    _, opt, p = _phase("spike")
    seen = _grads_seen(opt)
    assert _update(p, 1.0)  # the first applied update only seeds the EMA
    n1 = p.grad_norm.clone()
    assert p.clipped is None and torch.equal(p.norm_ema, n1)
    assert torch.equal(seen[0], n1)
    assert _update(p, 2.0)  # 2x < 3x: untouched
    assert torch.equal(seen[1], p.grad_norm) and p.clipped.item() == 0.0
    ema = p.norm_ema.clone()
    assert torch.allclose(ema, n1.lerp(p.grad_norm, 0.02))
    assert _update(p, 50.0)  # a x50 spike is cut to 3x the EMA
    assert seen[2].item() == pytest.approx(SPIKE_K * ema.item(), rel=1e-4)
    assert p.clipped.item() == 1.0
    # the EMA follows the applied (clipped) norm, not the spike
    assert torch.allclose(p.norm_ema, ema.lerp(SPIKE_K * ema, 0.02))
    before = p.norm_ema.clone()
    assert _update(p, math.nan) is False  # a skipped update leaves the EMA alone
    assert torch.equal(p.norm_ema, before)


def test_spike_state_roundtrip_and_old_state():
    _, _, a = _phase("spike", names=["0.weight", "0.bias", "1.weight", "1.bias"])
    _update(a, 1.0)
    sd = a.state_dict()
    assert torch.equal(sd["clip_ema"], a.norm_ema)
    _, _, b = _phase("spike", names=a.names)
    b.load_state_dict(copy.deepcopy(sd), Log())
    assert torch.equal(b.norm_ema, a.norm_ema)
    del sd[
        "clip_ema"
    ]  # saved before stage 2 / by a non-spike run: re-seeded at the next update
    _, _, c = _phase("spike", names=a.names)
    c.load_state_dict(sd, Log())
    assert c.norm_ema is None


@CUDA
def test_spike_clip_adds_no_host_sync():
    net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1)).cuda()
    _, _, p = _phase(
        "spike", net=net, opt=lambda ps: torch.optim.AdamW(ps, foreach=True)
    )
    x = torch.ones(2, 4, device="cuda")
    for f in (1.0, 2.0):
        p.backward(Ctx(x, f))
        p.step()
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            p.backward(Ctx(x, 50.0))
            assert p.step()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    assert len([w for w in caught if "called a synchronizing" in str(w.message)]) == 1


@pytest.mark.parametrize(
    ("clip", "logged"), [(True, True), (False, False), ("spike", True)]
)
def test_clipped_share_is_logged(tmp_path, clip, logged):
    opt = make_opt(tmp_path)
    opt["train"]["grad_clip"] = clip
    opt["train"]["pixel_opt"]["loss_weight"] = (
        1000.0  # the template regime: norms far above 1
    )
    m = build(opt)
    run(m, batches(3))
    log = m.get_current_log()
    assert ("clipped_g" in log) == logged
    if clip is True:
        assert log["clipped_g"] == 1.0  # the legacy max-norm 1.0 scales every update


# ---------------------------------------------------------------------------- ST-02 / T-04
def test_loss_meter_masks_and_counts():
    t, m = torch.tensor, LossMeter()
    m.add({"a": t(1.0), "b": t(math.inf)})
    m.commit(applied=False)  # a skipped update: masked and counted
    m.add({"a": t(3.0), "b": t(2.0)})
    m.commit(applied=True)
    assert m.read() == {"a": 2.0, "b": 2.0, "nonfinite_b": 1.0}
    m.add({"a": t(math.nan)})
    m.commit(applied=False)
    assert m.read() == {"nonfinite_a": 1.0}  # no finite value: no mean
    m.add({"a": t(math.nan)})
    m.commit(applied=True)  # finite gradients, non-finite objective: a broken loss
    with pytest.raises(FloatingPointError, match="'a'.*applied"):
        m.read()
    assert m.read() == {}


def _gan_bn(tmp, **kw):
    opt = make_opt(tmp, d=True, ema=0.9, **kw)
    # BatchNorm D: running statistics written by every forward
    opt["network_d"] = {
        "type": "patchgan",
        "num_feat": 4,
        "num_layers": 1,
        "norm_type": "batch",
        "use_sn": False,
    }
    return build(opt)


def test_transient_non_finite_batch_is_a_clean_skip(tmp_path):
    m = _gan_bn(tmp_path)
    data = batches(4)
    run(m, data[:2])
    m.get_current_log()
    g0, d0, e0 = clone(m.net_g), clone(m.net_d), clone(m.ema.module)
    bad = {k: v.clone() for k, v in data[2].items()}
    bad["lq"][0, 0, 0, 0] = math.nan  # one NaN pixel
    assert run(m, [bad]) == [True]
    assert m.g.skipped == 1 and m.d.skipped == 1
    # weights AND buffers (BN running stats) exactly as at the window start; EMA untouched
    assert same(g0, clone(m.net_g)) and same(d0, clone(m.net_d))
    assert same(e0, clone(m.ema.module))
    m.check_finite()  # no NaN reached the nets
    log = m.get_current_log()  # does not raise (ST-02)
    assert (log["skipped_g"], log["skipped_d"]) == (1, 1)
    assert log["nonfinite_l_g_total"] == 1 and log["nonfinite_l_g_pix"] == 1
    assert "l_g_pix" not in log and "grad_norm_g" not in log
    run(m, data[3:])  # training goes on
    log = m.get_current_log()
    assert log["skipped_g"] == 0 and not any(k.startswith("nonfinite") for k in log)
    assert not same(g0, clone(m.net_g))
    assert all(torch.isfinite(t).all() for t in clone(m.ema.module))


def test_skip_streak_still_stops_the_run(tmp_path):
    opt = make_opt(tmp_path)
    opt["train"]["max_skip_streak"] = 3
    m = build(opt)
    bad = batches(3)
    for b in bad:
        b["lq"][0, 0, 0, 0] = math.nan
    run(m, bad[:2])
    with pytest.raises(FloatingPointError, match="3 consecutive"):
        run(m, bad[2:])


def test_buffers_snapshot_and_restore():
    bn = nn.BatchNorm1d(3)
    b = Buffers(bn)
    b.save()
    bn(torch.randn(4, 3))
    b.restore()
    assert bn.running_mean.eq(0).all() and bn.running_var.eq(1).all()
    assert (
        bn.num_batches_tracked == 0
    )  # counters too: momentum=None BN averages with it
    bn(torch.randn(4, 3))
    b.save()  # the next window starts from the applied update's statistics
    want = bn.running_mean.clone()
    bn(torch.full((4, 3), math.nan))
    b.restore()
    assert torch.equal(bn.running_mean, want) and bn.num_batches_tracked == 1


def test_check_finite_accepts_an_observer_reset_but_not_nan(tmp_path):
    m = build(make_opt(tmp_path))
    m.net_g.add_module(
        "obs", tq.MovingAverageMinMaxObserver()
    )  # min/max = +-inf until observed
    m.check_finite()
    m.net_g.obs.min_val.fill_(math.nan)
    with pytest.raises(FloatingPointError):
        m.check_finite()
    m.net_g.obs.min_val.fill_(math.inf)
    m.net_g.register_buffer(
        "stat", torch.tensor(math.inf)
    )  # any other buffer must be finite
    with pytest.raises(FloatingPointError):
        m.check_finite()


# ---------------------------------------------------------------------------- Interface 1: idle phase
def test_idle_phase_does_nothing_but_the_clock():
    class Clock:
        n = 0

        def step(self):
            self.n += 1

    clock = Clock()
    net = nn.Linear(4, 1)
    _, opt, p = _phase(
        1.0, net=net, opt=lambda ps: torch.optim.AdamW(ps), schedulers=[clock]
    )
    p.terms = [Term(net, active=lambda it: it > 1)]
    calls = []
    net.register_forward_hook(lambda *_: calls.append(1))
    ctx = Ctx(torch.ones(2, 4), 1.0)
    p.backward(ctx)
    assert p.idle and not calls and not ctx.logs
    assert p.step() is None
    assert (p.skipped, p.streak, clock.n) == (0, 0, 1) and not opt.state
    ctx.it = 2
    assert _update_ctx(p, ctx) is True and calls and opt.state


def _update_ctx(p, ctx):
    p.backward(ctx)
    return p.step()


def test_idle_discriminator_in_the_model(tmp_path):
    m = build(make_opt(tmp_path, d=True, ema=0.9))
    for t in m.d.terms:
        t.active = lambda it: it > 2  # what I3's gan_opt.start_iter does
    calls = []
    m.net_d.register_forward_hook(lambda *_: calls.append(1))
    m.g.terms = [t for t in m.g.terms if "gan" not in t.log_key]  # G's GAN term too
    d0 = clone(m.net_d)
    run(m, batches(2))
    assert not calls and same(d0, clone(m.net_d)) and not m.d.opt.state
    log = m.get_current_log()
    assert log["skipped_d"] == 0 and "grad_norm_d" not in log and "l_d_total" not in log
    run(m, batches(1))
    assert calls and m.d.opt.state


# ---------------------------------------------------------------------------- T-10 unused parameters
class Dead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used, self.dead = nn.Linear(4, 1), nn.Linear(4, 1)

    def forward(self, x):
        return self.used(x)


def test_unused_parameters_warn_once(caplog):
    net = Dead()
    names = [n for n, _ in net.named_parameters()]
    _, _, p = _phase(1.0, net=net, names=names)
    with caplog.at_level(logging.WARNING, logger="neosr"):
        _update(p, 1.0)
        _update(p, 1.0)
    msgs = [r.getMessage() for r in caplog.records if "no gradient" in r.getMessage()]
    assert msgs == [
        "net_g: 2 of 4 trainable parameters got no gradient (first: dead.weight, dead.bias): "
        "they are never updated or weight-decayed"
    ]


def test_unused_parameters_under_strict_ddp_raise(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1
    )
    try:
        net = Dead()
        ddp = nn.parallel.DistributedDataParallel(net, gradient_as_bucket_view=True)
        _, _, p = _phase(1.0, net=net, strict_unused=True)
        p.terms = [Term(ddp)]
        p.backward(Ctx(torch.ones(2, 4), 1.0))
        with pytest.raises(ConfigError, match="find_unused_parameters = true"):
            p.step()
    finally:
        dist.destroy_process_group()


def test_compact_has_no_unused_parameters(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="neosr"):
        run(build(make_opt(tmp_path, d=True)), batches(1))
    assert not [r for r in caplog.records if "no gradient" in r.getMessage()]


# ---------------------------------------------------------------------------- Interface 2, O-09, T-08
def test_state_version_is_compared_like_the_type():
    _, _, a = _phase(1.0, names=["0.weight", "0.bias", "1.weight", "1.bias"])
    sd = a.state_dict()
    assert sd["optimizer"]["version"] == 0
    del sd["optimizer"]["version"]  # a state from before versioning loads as version 0
    _, _, b = _phase(1.0, names=a.names)
    b.load_state_dict(copy.deepcopy(sd), Log())
    sd["optimizer"]["version"] = 2
    with pytest.raises(ConfigError, match="state version 2 -> 0"):
        b.load_state_dict(sd, Log())


def test_is_schedule_free_reads_the_group_flag():
    ps = list(nn.Linear(2, 2).parameters())
    opt = adan_sf(ps, schedule_free=False)
    assert not is_schedule_free(opt)
    opt.param_groups[0]["schedule_free"] = True
    assert is_schedule_free(opt)


def test_resume_reports_restored_hyper_parameters():
    names = ["0.weight", "0.bias", "1.weight", "1.bias"]
    adamw = lambda lr: lambda ps: torch.optim.AdamW(ps, lr=lr, weight_decay=0.05)  # noqa: E731
    _, _, a = _phase(1.0, opt=adamw(1e-3), names=names)
    _update(a, 1.0)
    _, _, b = _phase(1.0, opt=adamw(1e-4), names=names)
    log = Log()
    b.load_state_dict(a.state_dict(), log)
    assert log.msgs == [
        "net_g: restored lr 0.001 from the state (config 0.0001); use pretrain_network_g to change it"
    ]
    assert b.opt.param_groups[0]["lr"] == 1e-3


def test_resume_warns_about_objective_changes(tmp_path, caplog):
    m = build(make_opt(tmp_path))
    run(m, batches(1))
    m.save()
    path = tmp_path / "training_states" / "1.state"
    opt = make_opt(tmp_path)
    opt["path"]["resume_state"] = path
    opt["train"]["pixel_opt"]["loss_weight"] = 5.0
    opt["datasets"]["train"]["accumulate"] = 2
    with caplog.at_level(logging.WARNING, logger="neosr"):
        build(opt)
    text = caplog.text
    assert "losses.pixel_opt" in text and "'loss_weight': 5.0" in text
    assert "data.accumulate: 1 -> 2" in text
    # a state from before the record grew: its missing keys are skipped, not reported
    st = torch.load(path, weights_only=True)
    st["run"] = {
        k: st["run"][k]
        for k in ("network_g", "network_d", "use_amp", "bfloat16", "ema", "world_size")
    }
    torch.save(st, path)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="neosr"):
        build(opt)
    assert "changed since" not in caplog.text


# ---------------------------------------------------------------------------- ST-10, T-16 lr
def test_linear_warmup_for_plain_optimizers_chains_and_resumes(tmp_path):
    opt = make_opt(tmp_path, optim={"type": "adamw", "lr": 1.0, "warmup_steps": 4})
    opt["train"]["scheduler"] = {"type": "MultiStepLR", "milestones": [6], "gamma": 0.5}
    m = build(opt)
    assert [type(s).__name__ for s in m.g.schedulers] == ["LinearLR", "MultiStepLR"]
    lrs = [m.current_lr()[0]]
    for _ in range(7):
        run(m, batches(1))
        lrs.append(m.current_lr()[0])
    assert lrs == pytest.approx([0.25, 0.4375, 0.625, 0.8125, 1, 1, 0.5, 0.5])
    m.save()
    again = make_opt(tmp_path, optim={"type": "adamw", "lr": 1.0})  # warmup removed
    again["train"]["scheduler"] = opt["train"]["scheduler"]
    again["path"]["resume_state"] = tmp_path / "training_states" / "7.state"
    with pytest.raises(
        ConfigError, match="scheduler #0 changed: LinearLR -> MultiStepLR"
    ):
        build(again)


def test_warmup_steps_is_validated_and_sf_keeps_its_own(tmp_path):
    with pytest.raises(ConfigError, match="warmup_steps must be an integer"):
        build(
            make_opt(tmp_path, optim={"type": "adan", "lr": 1e-3, "warmup_steps": -1})
        )
    m = build(make_opt(tmp_path, optim=SF | {"lr": 1e-2, "warmup_steps": 10}))
    assert m.g.schedulers == []
    run(m, batches(2))
    assert m.current_lr() == [pytest.approx(1e-2 * 2 / 10)]  # the SF internal warmup


# ---------------------------------------------------------------------------- D-03 plumbing
def test_aug_multi_prob_reaches_apply_augment(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        image_mod, "apply_augment", lambda gt, lq, **kw: seen.append(kw) or (gt, lq)
    )
    for ds, want in (({}, 0.3), ({"aug_multi_prob": 0.0}, 0.0)):
        opt = make_opt(tmp_path)
        opt["datasets"]["train"] |= {
            "augmentation": ["none", "mixup"],
            "aug_prob": [0.5, 0.5],
        } | ds
        build(opt).feed_data(batches(1)[0])
        assert seen.pop()["multi_prob"] == want


# ---------------------------------------------------------------------------- T-09 validate_train
def _cfg(tmp_path, body=BASE, head=HEAD):
    return load_config(write(tmp_path, body, head), is_train=True, root=tmp_path)


TRAIN = "[train.optim_g]"
BAD = [
    *[
        (f"[train]\nema = {v}\n{TRAIN}", "train.ema")
        for v in ("1.5", "1.0", "nan", "inf", "-0.5")
    ],
    ("[train]\nclamp = 'hard'\n" + TRAIN, "train.clamp"),
    *[
        (f"[train]\ngrad_clip = {v}\n{TRAIN}", "train.grad_clip")
        for v in ("0.0", "-1", "inf", "'spiky'")
    ],
]


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        *[(TRAIN, new, m) for new, m in BAD],
        ("[logger]", "[val]\nval_freq = 0\n[logger]", "val.val_freq"),
        ("[logger]", "[val]\nval_freq = -5\n[logger]", "val.val_freq"),
        (
            "[logger]",
            "[val.metrics.d]\ntype = 'calculate_dists'\nbetter = 'hgher'\n[logger]",
            "better",
        ),
        (
            "batch_size = 2",
            "batch_size = 2\nnum_worker_per_gpu = 'four'",
            "num_worker_per_gpu",
        ),
        (
            "batch_size = 2",
            "batch_size = 2\nnum_worker_per_gpu = -1",
            "num_worker_per_gpu",
        ),
        (
            "batch_size = 2",
            "batch_size = 2\ndataset_enlarge_ratio = 0",
            "dataset_enlarge_ratio",
        ),
        (
            "batch_size = 2",
            "batch_size = 2\nmean = [0.5, 0.5, 0.5]\nstd = [0.5, 0.5, 0.5]",
            "normalisation",
        ),
        ("batch_size = 2", "batch_size = 2\nmean = [0.0, 0.0, 0.0]", "normalisation"),
        ("batch_size = 2", "batch_size = 2\naug_multi_prob = 1.5", "aug_multi_prob"),
        ("total_iter = 10", "total_iter = 0", "logger.total_iter"),
        ("total_iter = 10", "total_iter = 10\nprint_freq = 0", "logger.print_freq"),
        ("save_checkpoint_freq = 5\n", "", "logger.save_checkpoint_freq"),
        (
            "[train.optim_g]",
            "[datasets.val_2]\ntype = 'paired'\nname = 'val'\n[train.optim_g]",
            "name 'val'",
        ),
    ],
)
def test_validate_train_rejects(tmp_path, old, new, match):
    assert old in BASE
    cfg = _cfg(tmp_path, BASE.replace(old, new, 1))
    with pytest.raises(ConfigError, match=match):
        validate_train(cfg)


@pytest.mark.parametrize(
    "extra",
    [
        "[train]\nema = 0\nclamp = 'restore'\ngrad_clip = 'spike'\n",
        "[train]\nema = 0.999\nclamp = false\ngrad_clip = 0.5\n",
        "[train]\ngrad_clip = 2\n",
    ],
)
def test_validate_train_accepts(tmp_path, extra):
    body = BASE.replace(TRAIN, extra + TRAIN).replace(
        "batch_size = 2",
        "batch_size = 2\nnum_worker_per_gpu = 'auto'\nmean = [0, 0, 0]\nstd = [1, 1, 1]",
    )
    validate_train(_cfg(tmp_path, body))


def test_val_dataset_name_defaults_to_its_phase(tmp_path):
    cfg = _cfg(tmp_path, BASE + "\n[datasets.val_1]\ntype = 'paired'\n")
    assert cfg["datasets"]["val_1"]["name"] == "val_1"
    assert "name" not in cfg["datasets"]["train"]
    validate_train(cfg)


@pytest.mark.parametrize(
    ("kind", "better"), [("calculate_dists", "lower"), ("calculate_psnr", "higher")]
)
def test_metric_direction_default(tmp_path, kind, better):
    m = build(make_opt(tmp_path))
    m.opt["val"]["metrics"] = {"x": {"type": kind}}
    for v in (0.5, 0.2, 0.9):
        m.update_best("valset", "x", v, 1)
    assert m.best_metric_results["valset"]["x"]["better"] == better
    assert m.best_metric_results["valset"]["x"]["val"] == (
        0.2 if better == "lower" else 0.9
    )


# ---------------------------------------------------------------------------- T-06, ST-09, use_amp
@pytest.mark.usefixtures("restore_torch_state", "fake_gpus")
def test_seed_alone_no_longer_switches_on_determinism(caplog):
    with caplog.at_level(logging.INFO, logger="neosr"):
        cfg = {"manual_seed": 7, "fast_matmul": True}
        rt = runtime.setup_runtime(cfg)
    assert rt.seed == 7 and cfg["deterministic"] is False
    assert (
        torch.backends.cudnn.benchmark
        and not torch.are_deterministic_algorithms_enabled()
    )
    assert not torch.utils.deterministic.fill_uninitialized_memory
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert "deterministic = true" in caplog.text
    # fast_matmul is TF32 only; fp16 accumulation is its own opt-in
    assert not torch.backends.cuda.matmul.allow_fp16_accumulation


@pytest.mark.usefixtures("restore_torch_state", "fake_gpus")
def test_deterministic_keeps_the_strict_branch_and_a_user_alloc_conf(monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:64")
    cfg = {"deterministic": True, "fp16_accumulation": True}
    runtime.setup_runtime(cfg)
    assert cfg["deterministic"] and torch.backends.cudnn.deterministic
    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:64"
    assert torch.backends.cuda.matmul.allow_fp16_accumulation


@pytest.mark.usefixtures("restore_torch_state", "fake_gpus")
def test_the_deprecated_alloc_conf_name_is_respected(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    runtime.setup_runtime({})
    assert (
        "PYTORCH_ALLOC_CONF" not in os.environ
    )  # it would override the user's setting


@pytest.mark.usefixtures("restore_torch_state")
def test_use_amp_without_cuda_is_a_config_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigError, match="use_amp needs CUDA"):
        runtime.setup_runtime({"use_amp": True, "is_train": True})
    assert (
        runtime.setup_runtime({"use_amp": True, "is_train": False}).device.type == "cpu"
    )


@CUDA
def test_fp16_scaler_grows_every_200_updates():
    ps = [nn.Parameter(torch.ones(1, device="cuda"))]
    p = Phase(
        "g", ps, torch.optim.SGD(ps), [], amp_dtype=torch.float16, use_amp=True,
        accumulate=1, grad_clip=0.0,
    )  # fmt: skip
    assert (p.scaler.get_scale(), p.scaler.get_growth_interval()) == (32.0, 200)


# ---------------------------------------------------------------------------- T-13, T-14 DDP
def _val_worker(rank: int, tmp: str, out: str) -> None:
    from datetime import timedelta  # noqa: PLC0415

    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp}/pg",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        m = build(
            make_opt(Path(tmp) / str(rank)),
            rt=Runtime(torch.device("cpu"), rank, 2, 0, True),
        )
        res = {"bucket_view": m.net_g_fwd.gradient_as_bucket_view}

        def boom(*_a):
            raise OSError("disk full")

        m.validate = boom
        t = time.monotonic()
        try:
            m.validation(loader(), 1, None, save_img=False)
        except Exception as e:  # noqa: BLE001
            res["raised"] = f"{type(e).__name__}: {e}"
        res["seconds"] = time.monotonic() - t
        torch.save(res, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_rank0_validation_failure_raises_on_every_rank(tmp_path):
    out = str(tmp_path / "res")
    torch.multiprocessing.spawn(
        _val_worker, args=(str(tmp_path), out), nprocs=2, join=True
    )
    r0, r1 = (torch.load(f"{out}.{r}") for r in (0, 1))
    assert r0["raised"] == "OSError: disk full"
    assert r1["raised"] == "RuntimeError: validation failed on rank 0"
    assert (
        r1["seconds"] < 30
    )  # not stranded at a barrier until the process-group timeout
    assert r0["bucket_view"] and r1["bucket_view"]


# ---------------------------------------------------------------------------- T-03, T-16, D-13 train.py
def test_validation_failure_saves_the_update_first(tmp_path, data, monkeypatch):  # noqa: F811
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []

    def boom(self, *_a, **_k):
        calls.append(self.update)
        raise OSError("val disk full")

    monkeypatch.setattr(base, "validation", boom)
    cfg = toml(
        tmp_path, data, total=4, save=4, val_freq=2
    )  # validation due at 2, save at 4
    assert train_py.main(["-opt", str(cfg)]) == 1
    assert calls == [2]
    assert states(tmp_path) == [2]  # the state of the failing update was kept
    assert "validation failed: saving update 2 first" in logs(tmp_path)


def test_loaders_are_built_before_the_model(tmp_path, data, monkeypatch):  # noqa: F811
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    order = []
    real_loaders, real_model = train_py.build_loaders, train_py.build_model
    monkeypatch.setattr(
        train_py,
        "build_loaders",
        lambda *a: order.append("loaders") or real_loaders(*a),
    )
    monkeypatch.setattr(
        train_py, "build_model", lambda *a: order.append("model") or real_model(*a)
    )
    assert train_py.main(["-opt", str(toml(tmp_path, data, total=2))]) == 0
    assert order == ["loaders", "model"]
    assert "it/s, data " in logs(tmp_path)  # D-13: the data-wait share next to it/s


def test_data_wait_share():
    w = DataWait()
    w(lambda: time.sleep(0.05))
    assert w.share() > 0.9
    time.sleep(0.02)
    assert w.share() < 0.5  # reset by the previous read


def test_message_logger_shows_the_data_share(caplog):
    ml = MessageLogger({"name": "x", "logger": {"total_iter": 10}}, None)
    with caplog.at_level(logging.INFO, logger="neosr"):
        ml({
            "epoch": 0,
            "iter": 1,
            "lrs": [1e-3],
            "time": 0.5,
            "data": 0.25,
            "l_g_pix": 1.0,
        })
    assert "[ performance: 2.000 it/s, data 25% ]" in caplog.text
    assert "data:" not in caplog.text


# ---------------------------------------------------------------------------- T-12 keep_exports
def test_prune_exports_keeps_the_newest_per_net_and_latest(tmp_path):
    for n in ("net_g", "net_d"):
        for u in (1, 2, 10, "latest"):
            (tmp_path / f"{n}_{u}.pth").write_bytes(b"x")
    (tmp_path / "net_g_best.pth").write_bytes(b"x")
    prune_exports(tmp_path, 0, ["net_g", "net_d"])
    assert len(list(tmp_path.iterdir())) == 9
    prune_exports(tmp_path, 2, ["net_g", "net_d"])
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "net_d_10.pth", "net_d_2.pth", "net_d_latest.pth",
        "net_g_10.pth", "net_g_2.pth", "net_g_best.pth", "net_g_latest.pth",
    ]  # fmt: skip


def test_keep_exports_in_save_and_size_logged_once(tmp_path, caplog):
    opt = make_opt(tmp_path)
    opt["logger"]["keep_exports"] = 1
    m = build(opt)
    with caplog.at_level(logging.INFO, logger="neosr"):
        for _ in range(3):
            run(m, batches(1))
            m.save()
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["net_g_3.pth"]
    assert caplog.text.count("one export set is") == 1


# ---------------------------------------------------------------------------- host syncs, CUDA
@CUDA
@pytest.mark.parametrize("clip", [True, "spike"])
def test_model_update_keeps_one_host_sync_per_phase(tmp_path, clip):
    """Buffer snapshot/restore, masked meter, restoring clamp and the spike clip add no sync."""
    opt = make_opt(tmp_path, d=True, ema=0.9)
    opt |= {"use_amp": True, "bfloat16": True}
    opt["network_d"] = {
        "type": "patchgan",
        "num_feat": 4,
        "num_layers": 1,
        "norm_type": "batch",
    }
    opt["train"] |= {"clamp": "restore", "grad_clip": clip}
    cuda = Runtime(torch.device("cuda", 0), 0, 1, 0, False)
    m = build(opt, rt=cuda)
    data = [{k: v.cuda() for k, v in b.items()} for b in batches(4)]
    run(m, data[:3])
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            assert run(m, data[3:]) == [True]
        finally:
            torch.cuda.set_sync_debug_mode(0)
    syncs = [w for w in caught if "called a synchronizing" in str(w.message)]
    assert len(syncs) == 2  # Phase.step's isfinite, G and D
    assert m.g.skipped == m.d.skipped == 0


# ---------------------------------------------------------------------------- end to end, CPU
def test_short_run_with_every_new_mode(tmp_path):
    opt = make_opt(
        tmp_path,
        d=True,
        ema=0.9,
        optim={"type": "adamw", "lr": 1e-3, "warmup_steps": 2},
    )
    opt["train"] |= {"clamp": "restore", "grad_clip": "spike"}
    m = build(opt)
    run(m, batches(4))
    log = m.get_current_log()
    assert log["clipped_g"] == pytest.approx(0.0, abs=0.5) and log["skipped_g"] == 0
    m.check_finite()
    m.save()
    st = torch.load(tmp_path / "training_states" / "4.state", weights_only=True)
    assert st["phases"]["net_g"]["clip_ema"] is not None
    assert st["run"]["train"]["clamp"] == "restore"
    r = make_opt(tmp_path, d=True, ema=0.9, optim=opt["train"]["optim_g"])
    r["train"] |= {"clamp": "restore", "grad_clip": "spike"}
    r["path"]["resume_state"] = tmp_path / "training_states" / "4.state"
    resumed = build(r)
    assert torch.equal(resumed.g.norm_ema, m.g.norm_ema)
    assert same(clone(resumed.net_g), clone(m.net_g))
    assert CPU.device.type == "cpu"

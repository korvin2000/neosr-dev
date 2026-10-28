"""Stage 3 A2: scheduler targeting (`train.scheduler.nets`), resume of a changed scheduler config
(rebuild + replay) and the scheduler range checks. CPU."""

import warnings

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR

from neosr.config import ConfigError, ConfigWarning, validate_train
from tests.test_model import SF, batches, build, make_opt, run
from tests.test_phase_state import Log, _adamw, _net, _phase, _roundtrip, _train


def _ms(*milestones):
    return (lambda o: MultiStepLR(o, milestones=list(milestones), gamma=0.5),)


def _lrs(phase, steps):
    out = []
    for _ in range(steps):
        _train(phase, 1)
        out.append(phase.opt.param_groups[0]["lr"])
    return out


# ------------------------------------------------------------------ Phase: resume semantics
def test_unchanged_scheduler_restores_exactly_and_quietly():
    a = _phase(_net(), _adamw, scheds=_ms(2))
    _train(a)
    b = _phase(_net(seed=1), _adamw, scheds=_ms(2))
    log = Log()
    b.load_state_dict(_roundtrip(a.state_dict()), log, update=3)
    assert not log.msgs
    assert b.schedulers[0].state_dict() == a.schedulers[0].state_dict()


@pytest.mark.parametrize(
    ("saved", "now"),
    [
        (_ms(2), _ms(5)),  # changed milestones
        ((), (lambda o: CosineAnnealingLR(o, 10),)),  # added
        (_ms(2), ()),  # removed: back to the base lr
        (
            _ms(2),
            (lambda o: LinearLR(o, 0.25, total_iters=4), *_ms(5)),
        ),  # chained warm-up
        (  # warm-up + cosine, T_max changed
            (
                lambda o: LinearLR(o, 0.5, total_iters=2),
                lambda o: CosineAnnealingLR(o, 4),
            ),
            (
                lambda o: LinearLR(o, 0.5, total_iters=2),
                lambda o: CosineAnnealingLR(o, 8),
            ),
        ),
    ],
)
def test_changed_scheduler_is_rebuilt_and_replayed_like_a_fresh_run(saved, now):
    a = _phase(_net(), _adamw, scheds=saved)
    _train(a)
    b = _phase(_net(seed=1), _adamw, scheds=now)
    log = Log()
    b.load_state_dict(_roundtrip(a.state_dict()), log, update=3)
    (msg,) = log.msgs
    assert "scheduler changed since the state" in msg
    assert "replayed to update 3" in msg
    ref = _phase(_net(), _adamw, scheds=now)  # the same config from update 0
    _train(ref)
    assert b.opt.param_groups[0]["lr"] == ref.opt.param_groups[0]["lr"]
    assert [s.state_dict() for s in b.schedulers] == [
        s.state_dict() for s in ref.schedulers
    ]
    assert _lrs(b, 4) == _lrs(ref, 4)


def test_equivalent_legacy_state_restores_exactly():
    """eta_min 0 (int, older torch) vs 0.0 and a saved `verbose` are the same configuration."""
    a = _phase(_net(), _adamw, scheds=(lambda o: CosineAnnealingLR(o, 10),))
    _train(a)
    sd = _roundtrip(a.state_dict())
    sd["schedulers"][0]["state"] |= {"eta_min": 0, "verbose": False}
    b = _phase(_net(seed=1), _adamw, scheds=(lambda o: CosineAnnealingLR(o, 10),))
    log = Log()
    b.load_state_dict(sd, log, update=3)
    assert not log.msgs
    assert b.schedulers[0].last_epoch == 3


def test_replay_starts_from_the_saved_base_lr():
    a = _phase(_net(), _adamw, scheds=_ms(2))
    _train(a)

    def lr2(ps):
        return torch.optim.AdamW(ps, lr=2e-2)

    b = _phase(_net(seed=1), lr2, scheds=_ms(5))
    log = Log()
    b.load_state_dict(_roundtrip(a.state_dict()), log, update=3)
    assert any("restored lr 0.01" in m for m in log.msgs)
    assert (
        b.opt.param_groups[0]["lr"] == 1e-2
    )  # saved 1e-2 wins, milestone 5 not reached
    assert b.schedulers[0].base_lrs == [1e-2]


def test_replay_defaults_to_the_saved_clock():
    a = _phase(_net(), _adamw, scheds=_ms(2))
    _train(a)
    b = _phase(_net(seed=1), _adamw, scheds=_ms(3))
    log = Log()
    b.load_state_dict(_roundtrip(a.state_dict()), log)
    assert "replayed to update 3" in log.msgs[0]
    assert b.schedulers[0].last_epoch == 3


# ------------------------------------------------------------------ trainer: nets targeting
def _gan(tmp_path, optim_g=None, **sched):
    opt = make_opt(tmp_path, d=True, optim=optim_g)
    opt["train"]["scheduler"] = {
        "type": "MultiStepLR",
        "milestones": [2],
        "gamma": 0.5,
        **sched,
    }
    return opt


def _kinds(phase):
    return [type(s).__name__ for s in phase.schedulers]


def test_scheduler_drives_every_plain_optimizer_by_default(tmp_path):
    m = build(_gan(tmp_path))
    assert _kinds(m.g) == _kinds(m.d) == ["MultiStepLR"]


def test_schedule_free_optimizers_are_left_out_by_default(tmp_path, caplog):
    with caplog.at_level("INFO", logger="neosr"):
        m = build(_gan(tmp_path, optim_g=SF))
    assert (_kinds(m.g), _kinds(m.d)) == ([], ["MultiStepLR"])
    assert "net_g (adan_sf) is schedule-free: not scheduled" in caplog.text


def test_explicit_nets_choose_the_optimizers(tmp_path):
    m = build(_gan(tmp_path, optim_g=SF, nets=["g"]))
    assert (_kinds(m.g), _kinds(m.d)) == (["MultiStepLR"], [])
    run(m, batches(3))
    assert m.d.opt.param_groups[0]["lr"] == 1e-4


def test_resume_with_a_changed_scheduler_replays(tmp_path, caplog):
    opt = _gan(tmp_path)
    opt["train"]["gan_opt"]["start_iter"] = (
        2  # D idles for 2 updates: its clock still runs
    )
    m = build(opt)
    run(m, batches(3))
    m.save()
    assert m.current_lr() == [pytest.approx(5e-4)]
    again = _gan(tmp_path, milestones=[5])
    again["path"]["resume_state"] = tmp_path / "training_states" / "3.state"
    with caplog.at_level("WARNING", logger="neosr"):
        r = build(again)
    assert r.current_lr() == [pytest.approx(1e-3)]
    assert r.d.opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert caplog.text.count("replayed to update 3") == 2  # G and D
    assert [s.last_epoch for s in (*r.g.schedulers, *r.d.schedulers)] == [3, 3]


def test_resume_with_changed_nets_drops_a_scheduler(tmp_path):
    m = build(_gan(tmp_path))
    run(m, batches(3))
    m.save()
    again = _gan(tmp_path, nets=["g"])  # D no longer scheduled: back to its base lr
    again["path"]["resume_state"] = tmp_path / "training_states" / "3.state"
    r = build(again)
    assert (_kinds(r.g), _kinds(r.d)) == (["MultiStepLR"], [])
    assert r.d.opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert r.current_lr() == [
        pytest.approx(5e-4)
    ]  # G restored exactly (milestone 2 passed)


def test_a_state_with_a_scheduled_sf_optimizer_resumes_under_the_new_default(
    tmp_path, caplog
):
    m = build(_gan(tmp_path, optim_g=SF, nets=["g", "d"]))  # SF G scheduled explicitly
    run(m, batches(3))
    m.save()
    assert m.g.opt.param_groups[0]["lr"] == pytest.approx(5e-3)
    again = _gan(tmp_path, optim_g=SF)  # default: SF G left out
    again["path"]["resume_state"] = tmp_path / "training_states" / "3.state"
    with caplog.at_level("WARNING", logger="neosr"):
        r = build(again)
    assert _kinds(r.g) == []
    assert r.g.opt.param_groups[0]["lr"] == pytest.approx(1e-2)
    assert "net_g: scheduler changed since the state: [MultiStepLR(" in caplog.text


# ------------------------------------------------------------------ validate_train
def _cfg(sched, d=False, total=100):
    cfg = {
        "scale": 2,
        "network_g": {"type": "compact"},
        "datasets": {"train": {"patch_size": 8, "batch_size": 1}},
        "train": {"optim_g": {"type": "adamw"}, "scheduler": sched},
        "logger": {"save_checkpoint_freq": 1, "total_iter": total},
    }
    if d:
        cfg["network_d"] = {"type": "unet"}
        cfg["train"]["optim_d"] = {"type": "adamw"}
    return cfg


@pytest.mark.parametrize(
    ("nets", "d"),
    [
        ([], False),
        (["g", "g"], False),
        (["d"], False),
        (["net_g"], True),
        ("g", False),
        ([["g"]], False),
    ],
)
def test_scheduler_nets_range_is_checked(nets, d):
    with pytest.raises(ConfigError, match=r"train\.scheduler\.nets"):
        validate_train(
            _cfg({"type": "multisteplr", "milestones": [5], "nets": nets}, d)
        )


def test_scheduler_nets_valid():
    validate_train(
        _cfg({"type": "multisteplr", "milestones": [5], "nets": ["g", "d"]}, True)
    )


@pytest.mark.parametrize(
    ("sched", "match"),
    [
        ({"type": "CosineAnnealing", "T_max": 50}, "T_max=50 < logger.total_iter=100"),
        (
            {"type": "MultiStepLR", "milestones": [50, 100]},
            "milestone 100 >= logger.total_iter=100 never affects",
        ),
    ],
)
def test_scheduler_horizon_warns(sched, match):
    with pytest.warns(ConfigWarning, match=match):
        validate_train(_cfg(sched))


def test_scheduler_inside_the_run_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        validate_train(_cfg({"type": "cosineannealing", "T_max": 100}))
        validate_train(_cfg({"type": "multisteplr", "milestones": [50, 90]}))

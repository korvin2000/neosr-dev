"""Stage-2 cross-lane contracts: the GAN start/ramp terms (losses lane) run through the trainer's idle
phases (trainer lane), and the optimizer state version is scoped to schedule-free mode."""

import torch

from tests.test_model import batches, build, make_opt, run


def _d_params(model):
    return [p.detach().clone() for p in model.net_d.parameters()]


def test_gan_start_iter_keeps_d_idle_then_trains_it(tmp_path):
    opt = make_opt(tmp_path, d=True)
    opt["train"]["gan_opt"] |= {"start_iter": 3, "ramp_iters": 2}
    model = build(opt)
    d0 = _d_params(model)
    for it in range(1, 4):  # updates 1..3: no D step, no D optimizer state, no adversarial G term
        run(model, batches(1, seed=it))
        assert model.update == it
        assert model.d.idle
        assert not model.d.opt.state
        assert all(torch.equal(a, b) for a, b in zip(d0, model.net_d.parameters(), strict=True))
        assert not any("gan" in k for k in model.get_current_log())
    run(model, batches(1, seed=4))  # update 4: D trains and G's term enters at weight 1/2
    assert not model.d.idle
    assert model.d.opt.state
    assert any(not torch.equal(a, b) for a, b in zip(d0, model.net_d.parameters(), strict=True))
    logs = model.get_current_log()
    assert any("gan" in k for k in logs)
    assert all(torch.isfinite(torch.tensor(v)) for v in logs.values() if isinstance(v, float))


def test_default_gan_is_never_idle(tmp_path):
    model = build(make_opt(tmp_path, d=True))
    run(model, batches(2))
    assert not model.d.idle
    assert model.d.opt.state


def test_state_version_only_for_schedule_free(tmp_path):
    sf = build(make_opt(tmp_path / "sf", optim={"type": "adan_sf", "lr": 1e-3}))
    plain = build(
        make_opt(tmp_path / "p", optim={"type": "adan_sf", "lr": 1e-3, "schedule_free": False})
    )
    assert sf.g.version == getattr(type(sf.g.opt), "STATE_VERSION", 0)
    assert plain.g.version == 0  # plain Adan states keep loading across the SF rewrite

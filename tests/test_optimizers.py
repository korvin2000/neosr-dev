import io

import pytest
import torch

from neosr.optimizers import adamw_sf, adamw_win, adan, adan_sf

LR = 1e-2


def _param(seed: int = 0) -> torch.nn.Parameter:
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(4, 3, generator=g))


def _grad(step: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(100 + step)
    return torch.randn(4, 3, generator=g)


def _run(opt, p, steps: int, start: int = 0) -> None:
    for i in range(start, start + steps):
        p.grad = _grad(i)
        opt.step()


FACTORIES = {
    "adan": lambda ps: adan(ps, lr=LR),
    "adan_sf": lambda ps: adan_sf(ps, lr=LR, warmup_steps=4),
    "adan_nosf": lambda ps: adan_sf(ps, lr=LR, warmup_steps=4, schedule_free=False),
    "adamw_sf": lambda ps: adamw_sf(ps, lr=LR, warmup_steps=4),
    "adamw_win": lambda ps: adamw_win(ps, lr=LR, acceleration_mode="win"),
    "adamw_win2": lambda ps: adamw_win(ps, lr=LR, acceleration_mode="win2"),
}


def test_adan_sf_warmup_scales_z_step():
    deltas = []
    for warmup in (0, 10):
        p = _param()
        p0 = p.detach().clone()
        opt = adan_sf([p], lr=LR, warmup_steps=warmup)
        _run(opt, p, 1)
        deltas.append(opt.state[p]["z"] - p0)
    torch.testing.assert_close(deltas[0], -LR * _grad(0))
    torch.testing.assert_close(deltas[1], deltas[0] / 10)


def test_adan_sf_warmup_scales_update_without_sf():
    deltas = []
    for warmup in (0, 10):
        p = _param()
        p0 = p.detach().clone()
        opt = adan_sf([p], lr=LR, warmup_steps=warmup, schedule_free=False)
        _run(opt, p, 1)
        deltas.append(p.detach() - p0)
    torch.testing.assert_close(deltas[1], deltas[0] / 10)


@pytest.mark.parametrize("mode", ["win", "win2"])
def test_adamw_win_lr_zero(mode):
    p = _param()
    p0 = p.detach().clone()
    opt = adamw_win([p], lr=0.0, acceleration_mode=mode)
    _run(opt, p, 3)
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p.detach(), p0)


def test_adamw_win2_requires_positive_reckless_step():
    with pytest.raises(ValueError):
        adamw_win([_param()], reckless_steps=(0.0, 8.0), acceleration_mode="win2")


@pytest.mark.parametrize("name", ["adan_sf", "adamw_sf"])
def test_sf_eval_train_round_trip(name):
    p = _param()
    opt = FACTORIES[name]([p])
    _run(opt, p, 5)
    y = p.detach().clone()
    opt.eval()
    assert not torch.allclose(p, y)
    with pytest.raises(ValueError):
        _run(opt, p, 1)
    opt.train()
    torch.testing.assert_close(p.detach(), y)


@pytest.mark.parametrize("name", list(FACTORIES))
def test_state_dict_round_trip(name):
    p = _param()
    opt = FACTORIES[name]([p])
    _run(opt, p, 3)

    q = torch.nn.Parameter(p.detach().clone())
    opt2 = FACTORIES[name]([q])
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)
    opt2.load_state_dict(torch.load(buf))

    _run(opt, p, 2, start=3)
    _run(opt2, q, 2, start=3)
    torch.testing.assert_close(q.detach(), p.detach())


@pytest.mark.parametrize("name", ["adan", "adan_sf", "adamw_win2"])
def test_closure_runs_with_grad(name):
    p = _param()
    opt = FACTORIES[name]([p])

    def closure():
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss.item() > 0


CLIP_FACTORIES = {
    "adan": lambda ps, **kw: adan(ps, lr=LR, **kw),
    "adan_single": lambda ps, **kw: adan(ps, lr=LR, foreach=False, **kw),
    "adan_sf": lambda ps, **kw: adan_sf(ps, lr=LR, **kw),
    "adamw_win2": lambda ps, **kw: adamw_win(ps, lr=LR, **kw),
}


@pytest.mark.parametrize("name", CLIP_FACTORIES)
def test_max_grad_norm_matches_explicit_clipping(name):
    max_norm, eps = 1.0, 1e-8
    p, q = _param(), _param()
    clipped = CLIP_FACTORIES[name]([p], max_grad_norm=max_norm, eps=eps)
    reference = CLIP_FACTORIES[name]([q], eps=eps)
    # the norm of _grad() is ~3.5: steps above and below the threshold
    for i, scale in enumerate((1e3, 0.1, 50.0, 1e-2)):
        grad = _grad(i) * scale
        p.grad = grad.clone()
        q.grad = grad * min(1.0, max_norm / (grad.norm().item() + eps))
        clipped.step()
        reference.step()
    for key, value in reference.state[q].items():
        torch.testing.assert_close(clipped.state[p][key], value)
    torch.testing.assert_close(p.detach(), q.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", CLIP_FACTORIES)
def test_max_grad_norm_does_not_sync(name):
    p = torch.nn.Parameter(_param().detach().cuda())
    opt = CLIP_FACTORIES[name]([p], max_grad_norm=1.0)
    # warm-up: lazy state init and first allocator use may sync, steady-state steps must not
    for i in range(2):
        p.grad = _grad(i).cuda() * 1e3
        opt.step()
    p.grad = _grad(2).cuda() * 1e3
    torch.cuda.set_sync_debug_mode("error")
    try:
        opt.step()
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_constructor_validation():
    with pytest.raises(TypeError):
        adan([_param()], lr=LR, warmup_step=10)
    with pytest.raises(TypeError):
        adan_sf([_param()], lr=LR, warmup_step=10)
    with pytest.raises(ValueError):
        adamw_sf([_param()], schedule_free=False)
    for opt in (adamw_sf, adan_sf):
        with pytest.raises(ValueError):
            opt([_param()], betas=(0.0, 0.92, 0.99)[: 2 if opt is adamw_sf else 3])


@pytest.mark.parametrize(
    ("opt", "kwargs", "group"),
    [
        (adamw_win, {"acceleration_mode": "win2"}, {"reckless_steps": (0.0, 8.0)}),
        (adamw_win, {"acceleration_mode": "win"}, {"acceleration_mode": "win2", "reckless_steps": (0.0, 8.0)}),
        (adamw_sf, {}, {"betas": (0.0, 0.99)}),
        (adamw_sf, {}, {"schedule_free": False}),
        (adan_sf, {}, {"betas": (0.0, 0.92, 0.99)}),
    ],
)
def test_group_options_are_validated(opt, kwargs, group):
    with pytest.raises(ValueError):
        opt([{"params": [_param()], **group}], lr=LR, **kwargs)
    optimizer = opt([_param()], lr=LR, **kwargs)
    with pytest.raises(ValueError):
        optimizer.add_param_group({"params": [_param(1)], **group})
    assert len(optimizer.param_groups) == 1


def test_adan_sf_group_override_of_schedule_free():
    p = _param()
    opt = adan_sf([{"params": [p], "schedule_free": True}], lr=LR, schedule_free=False)
    _run(opt, p, 2)
    assert "z" in opt.state[p]

    q = _param()
    opt = adan_sf([{"params": [q], "schedule_free": False}], lr=LR)
    _run(opt, q, 2)
    assert "z" not in opt.state[q]
    opt.eval()
    _run(opt, q, 1)
